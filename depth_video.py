#!/usr/bin/env python3
"""Offline video-to-relative-depth conversion with Depth Anything V2.

This script intentionally never contacts Hugging Face or any other service.  A
model has to be present in the local Transformers cache (or supplied as a local
directory with ``--model``).  Frames are decoded with OpenCV, inferred in small
batches, and encoded as a silent H.264 depth visualization.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

# Set offline/cache flags before importing transformers.  The default cache is
# the cache shipped alongside this script, making the tool portable/offline.
_HERE = Path(__file__).resolve().parent
_DEFAULT_CACHE = _HERE / "hf_cache"
if _DEFAULT_CACHE.is_dir():
    os.environ.setdefault("HF_HOME", str(_DEFAULT_CACHE))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402


DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _pick_cache(cache_dir: str | None) -> Path:
    if cache_dir:
        return Path(cache_dir).expanduser().resolve()
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]).expanduser().resolve()
    # For an installed package or PyInstaller executable, users commonly put
    # the supplied cache beside the command/current working directory.
    cwd_cache = Path.cwd() / "hf_cache"
    if _DEFAULT_CACHE.is_dir():
        return _DEFAULT_CACHE
    if cwd_cache.is_dir():
        return cwd_cache.resolve()
    return Path.home() / ".cache" / "huggingface"


def _load_pipeline(model: str, cache_dir: Path, device: str):
    """Load a local depth pipeline; fail clearly instead of attempting a download."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    # TRANSFORMERS_CACHE is still honored by older transformers releases.
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir))
    from transformers import pipeline

    model_ref: str | Path = Path(model).expanduser().resolve() if Path(model).expanduser().exists() else model
    requested_device = device
    if device == "auto":
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    pipe_device = 0 if device.startswith("cuda") else -1
    _log(f"Loading {model_ref!s} from local cache ({device}) …")
    try:
        result = pipeline(
            "depth-estimation",
            model=model_ref,
            device=pipe_device,
            model_kwargs={"local_files_only": True},
        )
    except Exception as exc:
        hint = (
            f"Could not load the local model {model!r}. Put the model in {cache_dir} "
            "or pass --model /path/to/model. Network downloads are disabled."
        )
        raise RuntimeError(hint) from exc
    if requested_device == "auto":
        _log(f"Using {device}")
    return result, device


def _prediction_to_array(prediction: Any) -> np.ndarray:
    raw = prediction["predicted_depth"] if isinstance(prediction, dict) else prediction
    if hasattr(raw, "detach"):
        raw = raw.detach().cpu().numpy()
    array = np.asarray(raw, dtype=np.float32).squeeze()
    if array.ndim != 2:
        raise ValueError(f"Depth model returned shape {array.shape}, expected a 2D map")
    return array


def _predict_batch(pipe: Any, bgr_frames: list[np.ndarray], width: int, height: int, batch_size: int) -> list[np.ndarray]:
    images = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in bgr_frames]
    outputs = pipe(images, batch_size=batch_size)
    if isinstance(outputs, dict):
        outputs = [outputs]
    depths: list[np.ndarray] = []
    for output in outputs:
        depth = _prediction_to_array(output)
        depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_CUBIC)
        depths.append(depth.astype(np.float32, copy=False))
    if len(depths) != len(bgr_frames):
        raise RuntimeError(f"Model returned {len(depths)} maps for {len(bgr_frames)} frames")
    return depths


def _normalization_range(depths: np.ndarray) -> tuple[float, float]:
    # Percentiles avoid a single hot pixel flattening the entire video.  A
    # small random-free stride bounds memory while remaining deterministic.
    flat = depths.reshape(-1)
    if flat.size > 2_000_000:
        flat = flat[:: max(1, flat.size // 2_000_000)]
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(flat, [1.0, 99.0]).astype(float)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(flat)), float(np.max(flat))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def _colorize(depth: np.ndarray, lo: float, hi: float, colormap: str) -> np.ndarray:
    normalized = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    gray = np.round(normalized * 255.0).astype(np.uint8)
    if colormap == "gray":
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    lut = {
        "turbo": cv2.COLORMAP_TURBO,
        "magma": cv2.COLORMAP_MAGMA,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "plasma": cv2.COLORMAP_PLASMA,
        "inferno": cv2.COLORMAP_INFERNO,
        "jet": cv2.COLORMAP_JET,
    }
    return cv2.applyColorMap(gray, lut[colormap])


def _video_info(path: Path) -> tuple[cv2.VideoCapture, float, int, int, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open input video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 24.0)
    if not np.isfinite(fps) or fps <= 0:
        fps = 24.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"Input has no readable video dimensions: {path}")
    return capture, fps, width, height, count


def convert_video(
    input_path: str | Path,
    output_path: str | Path,
    *,
    seconds: float | None = None,
    max_frames: int | None = None,
    model: str = DEFAULT_MODEL,
    cache_dir: str | Path | None = None,
    device: str = "auto",
    batch_size: int = 4,
    colormap: str = "gray",
    threads: int | None = None,
) -> dict[str, Any]:
    """Convert a video and return metadata.  All model inference is local."""
    source = Path(input_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source == destination:
        raise ValueError("Output path must differ from input path")
    if seconds is not None and seconds <= 0:
        raise ValueError("--seconds must be greater than zero")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("--max-frames must be greater than zero")
    if batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero")
    if colormap not in {"gray", "turbo", "magma", "viridis", "plasma", "inferno", "jet"}:
        raise ValueError(f"Unknown colormap: {colormap}")
    if threads:
        try:
            import torch

            torch.set_num_threads(threads)
        except Exception:
            pass

    cache = _pick_cache(str(cache_dir) if cache_dir else None)
    pipe, actual_device = _load_pipeline(model, cache, device)
    capture, fps, width, height, reported_count = _video_info(source)
    frame_limit = reported_count if reported_count > 0 else None
    if seconds is not None:
        frame_limit = min(frame_limit, int(round(seconds * fps))) if frame_limit else int(round(seconds * fps))
    if max_frames is not None:
        frame_limit = min(frame_limit, max_frames) if frame_limit else max_frames
    if not frame_limit:
        frame_limit = 10**9
    _log(f"Input: {source.name} ({width}×{height} @ {fps:.3f} fps); processing up to {frame_limit} frames")

    destination.parent.mkdir(parents=True, exist_ok=True)
    # Float16 keeps temporary storage manageable for long videos while being
    # more than sufficient for an 8-bit visualization.
    tmp = tempfile.NamedTemporaryFile(prefix="depth_", suffix=".dat", delete=False, dir=str(destination.parent))
    tmp_path = Path(tmp.name)
    tmp.close()
    depths: np.memmap | None = None
    processed = 0
    started = time.time()
    try:
        depths = np.memmap(tmp_path, dtype=np.float16, mode="w+", shape=(int(frame_limit), height, width))
        while processed < frame_limit:
            frames: list[np.ndarray] = []
            for _ in range(min(batch_size, frame_limit - processed)):
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(frame)
            if not frames:
                break
            predicted = _predict_batch(pipe, frames, width, height, batch_size)
            for depth in predicted:
                depths[processed] = depth
                processed += 1
            if processed == 1 or processed % max(batch_size * 5, 1) == 0 or processed >= frame_limit:
                elapsed = time.time() - started
                _log(f"Inferred {processed} frames ({processed / max(elapsed, 1e-6):.2f} frames/s)")
        capture.release()
        if processed == 0:
            raise RuntimeError("No decodable frames found in input")
        depths.flush()
        lo, hi = _normalization_range(depths[:processed])
        _log(f"Depth range p1={lo:.6f}, p99={hi:.6f}")
        # OpenCV's mp4v writer is available in the base environment.  Convert
        # to H.264 with ffmpeg when available for broad player compatibility.
        temp_video = destination.with_suffix(destination.suffix + ".mp4v.tmp.mp4")
        writer = cv2.VideoWriter(str(temp_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height), True)
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create output video: {temp_video}")
        for index in range(processed):
            writer.write(_colorize(np.asarray(depths[index], dtype=np.float32), lo, hi, colormap))
        writer.release()
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            command = [ffmpeg, "-y", "-loglevel", "error", "-i", str(temp_video), "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(destination)]
            try:
                subprocess.run(command, check=True)
                temp_video.unlink(missing_ok=True)
            except (subprocess.CalledProcessError, OSError):
                temp_video.replace(destination)
        else:
            temp_video.replace(destination)
    finally:
        capture.release()
        if depths is not None:
            del depths
        tmp_path.unlink(missing_ok=True)
    metadata: dict[str, Any] = {
        "input": str(source),
        "output": str(destination),
        "frames": processed,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_seconds": processed / fps,
        "model": model,
        "device": actual_device,
        "colormap": colormap,
        "normalization": {"p1": lo, "p99": hi},
        "offline": True,
    }
    metadata_path = destination.with_suffix(destination.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    _log(f"Wrote {destination} ({destination.stat().st_size:,} bytes)")
    _log(f"Metadata: {metadata_path}")
    return metadata


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline Depth Anything V2 video-to-depth converter")
    parser.add_argument("input", help="input video path")
    parser.add_argument("-o", "--output", help="output video path (default: <input>_depth.mp4)")
    parser.add_argument("--seconds", type=float, help="process only the first N seconds")
    parser.add_argument("--max-frames", type=int, help="process no more than this many frames")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="local model directory or cached model id")
    parser.add_argument("--cache-dir", help="Transformers/Hugging Face cache directory")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threads", type=int, help="PyTorch CPU thread count")
    parser.add_argument("--colormap", choices=("gray", "turbo", "magma", "viridis", "plasma", "inferno", "jet"), default="gray")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    source = Path(args.input)
    output = Path(args.output) if args.output else source.with_name(source.stem + "_depth.mp4")
    try:
        convert_video(
            source,
            output,
            seconds=args.seconds,
            max_frames=args.max_frames,
            model=args.model,
            cache_dir=args.cache_dir,
            device=args.device,
            batch_size=args.batch_size,
            colormap=args.colormap,
            threads=args.threads,
        )
    except KeyboardInterrupt:
        _log("Interrupted")
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
