#!/usr/bin/env python3
"""Offline video-to-relative-depth conversion with Depth Anything V2.

This script intentionally never contacts Hugging Face or any other service.  A
model has to be present in the local Transformers cache (or supplied as a local
directory with ``--model``).  Frames are decoded with OpenCV, inferred in small
batches and streamed straight into FFmpeg as a silent H.264 depth
visualization.  No per-frame depth cache is written, so long or 4K inputs need
no extra disk space or memory beyond a single batch.
"""

from __future__ import annotations

import argparse
import errno
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import warnings
from pathlib import Path
from typing import Any, Iterable

# The desktop app decodes our output as UTF-8.  PyInstaller ignores PYTHONUTF8
# and PYTHONIOENCODING, so without this Windows writes paths such as
# C:\Users\<中文用户名> in the ANSI code page and every log line is unreadable.
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore", category=FutureWarning)

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

# PyTorch already uses every physical core for inference.  OpenCV's own thread
# pool and x264's default threads only fight its spinning OpenMP workers, which
# measured ~45% slower end to end on an 8-core CPU.
cv2.setNumThreads(1)


DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
# Depth Anything V2 preprocessing: short side >= 518, both sides multiples of
# 14, ImageNet mean/std.  Matches the model's preprocessor_config.json.
MODEL_SIZE = 518
PATCH = 14
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# Frames spread across the clip used to find letterbox bars and the global
# depth range before the single streaming pass (fewer for short test clips).
SAMPLE_FRAMES = 32
MIN_SAMPLE_FRAMES = 8
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


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


def _load_model(model: str, cache_dir: Path, device: str):
    """Load the local depth model; fail clearly instead of attempting a download."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    import torch
    from transformers import DepthAnythingForDepthEstimation

    model_ref: str | Path = Path(model).expanduser().resolve() if Path(model).expanduser().exists() else model
    requested_device = device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    _log(f"Loading {model_ref!s} from local cache ({device}) …")
    try:
        net = DepthAnythingForDepthEstimation.from_pretrained(
            str(model_ref), cache_dir=str(cache_dir), local_files_only=True
        )
    except Exception as exc:
        raise RuntimeError(
            f"本地模型加载失败：{model_ref}。请重新安装完整版本（不会联网下载模型）。"
        ) from exc
    net.to(device).eval()
    if requested_device == "auto":
        _log(f"Using {device}")
    return net, torch, device


def _model_input_size(width: int, height: int) -> tuple[int, int]:
    scale = MODEL_SIZE / min(width, height)

    def fit(value: float) -> int:
        rounded = round(value / PATCH) * PATCH
        if rounded < MODEL_SIZE:
            rounded = math.ceil(value / PATCH) * PATCH
        return int(rounded)

    return fit(width * scale), fit(height * scale)


def _infer(net: Any, torch: Any, device: str, frames: list[np.ndarray]) -> np.ndarray:
    """Return relative depth maps at model resolution, shape (N, h, w)."""
    height, width = frames[0].shape[:2]
    size = _model_input_size(width, height)
    # INTER_AREA when shrinking avoids aliasing on 4K sources; the model
    # authors' INTER_CUBIC is used when enlarging small inputs.
    interpolation = cv2.INTER_AREA if size[0] < width else cv2.INTER_CUBIC
    batch = np.empty((len(frames), 3, size[1], size[0]), dtype=np.float32)
    for index, frame in enumerate(frames):
        rgb = cv2.cvtColor(cv2.resize(frame, size, interpolation=interpolation), cv2.COLOR_BGR2RGB)
        batch[index] = ((rgb.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)
    with torch.inference_mode():
        depth = net(pixel_values=torch.from_numpy(batch).to(device)).predicted_depth
    result = depth.float().cpu().numpy()
    if result.ndim != 3 or result.shape[0] != len(frames):
        raise RuntimeError(f"Depth model returned shape {result.shape} for {len(frames)} frames")
    return result


def _normalization_range(frame_ranges: list[tuple[float, float]]) -> tuple[float, float]:
    """One range for the whole video, so brightness never pumps between frames.

    Each sample contributes its own p1/p99 (a single hot pixel cannot flatten
    the video), then the quartiles across samples are used: a few frames with
    an object right at the lens no longer darken every other frame, they just
    saturate to white themselves.
    """
    ranges = np.array([pair for pair in frame_ranges if all(np.isfinite(pair))], dtype=np.float64)
    if ranges.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(ranges[:, 0], 25))
    hi = float(np.percentile(ranges[:, 1], 75))
    if hi <= lo:
        lo, hi = float(ranges[:, 0].min()), float(ranges[:, 1].max())
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


_COLORMAPS = {
    "turbo": cv2.COLORMAP_TURBO,
    "magma": cv2.COLORMAP_MAGMA,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "plasma": cv2.COLORMAP_PLASMA,
    "inferno": cv2.COLORMAP_INFERNO,
    "jet": cv2.COLORMAP_JET,
}


def _render(depth: np.ndarray, lo: float, hi: float, width: int, height: int, crop: tuple[int, int], colormap: str) -> np.ndarray:
    top, bottom = crop
    full = cv2.resize(depth, (width, bottom - top), interpolation=cv2.INTER_CUBIC)
    gray = np.clip((full - lo) * (255.0 / (hi - lo)), 0.0, 255.0)
    gray = np.round(gray).astype(np.uint8)
    if top or bottom != height:
        canvas = np.zeros((height, width), dtype=np.uint8)
        canvas[top:bottom, :] = gray
        gray = canvas
    if colormap == "gray":
        return gray
    return cv2.applyColorMap(gray, _COLORMAPS[colormap])


def _find_ffmpeg() -> str | None:
    """Prefer the encoder shipped beside the executable over PATH."""
    candidates: list[Path] = []
    configured = os.environ.get("FFMPEG_PATH")
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend(
        [
            Path(sys.executable).resolve().parent / "ffmpeg.exe",
            _HERE / "ffmpeg.exe",
            Path.cwd() / "ffmpeg.exe",
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return shutil.which("ffmpeg")


class _FfmpegWriter:
    """Pipe raw frames into FFmpeg (H.264, yuv420p, even dimensions)."""

    def __init__(self, ffmpeg: str, destination: Path, width: int, height: int, fps: float, channels: int):
        self._stderr = tempfile.TemporaryFile()
        command = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "gray" if channels == 1 else "bgr24",
            "-s", f"{width}x{height}", "-framerate", f"{fps:.6f}", "-i", "-",
            "-an", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-threads", "2",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination),
        ]
        self._process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._stderr, creationflags=_NO_WINDOW
        )

    def _failure(self) -> RuntimeError:
        code = self._process.wait()
        self._stderr.seek(0)
        detail = self._stderr.read().decode("utf-8", errors="replace").strip().splitlines()
        tail = " | ".join(detail[-3:]) or "无输出"
        return RuntimeError(f"FFmpeg 编码失败（退出码 {code}）：{tail}")

    def write(self, frame: np.ndarray) -> None:
        try:
            self._process.stdin.write(np.ascontiguousarray(frame).tobytes())
        except OSError as exc:
            raise self._failure() from exc

    def close(self) -> None:
        try:
            self._process.stdin.close()
        except OSError:
            pass
        if self._process.wait() != 0:
            raise self._failure()
        self._stderr.close()

    def abort(self) -> None:
        if self._process.poll() is None:
            self._process.kill()
            self._process.wait()
        self._stderr.close()


class _OpenCvWriter:
    """Fallback for source checkouts without FFmpeg (MPEG-4 Part 2)."""

    def __init__(self, destination: Path, width: int, height: int, fps: float, channels: int):
        self._writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height), True)
        if not self._writer.isOpened():
            raise RuntimeError(f"Cannot create output video: {destination}")
        self._channels = channels

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if self._channels == 1 else frame)

    def close(self) -> None:
        self._writer.release()

    def abort(self) -> None:
        self._writer.release()


def _open_video(path: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"无法读取输入视频（格式或编码不受支持，或文件已损坏）：{path}")
    return capture


def _sample_frames(path: Path, frame_limit: int, known_length: bool) -> list[np.ndarray]:
    """Read up to SAMPLE_FRAMES frames spread over the part being converted."""
    capture = _open_video(path)
    frames: list[np.ndarray] = []
    try:
        count = min(frame_limit, max(MIN_SAMPLE_FRAMES, min(SAMPLE_FRAMES, frame_limit // 10)))
        indices = sorted({int(round(value)) for value in np.linspace(0, frame_limit - 1, count)})
        dense = not known_length or frame_limit <= SAMPLE_FRAMES * 8
        position = 0
        for index in indices:
            if dense:
                # Short clips: decode sequentially, cheaper than seeking.
                while position < index and capture.grab():
                    position += 1
            else:
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            position += 1
            if ok:
                frames.append(frame)
            elif dense:
                break
    finally:
        capture.release()
    return frames


def _dark_rows(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return (gray.mean(axis=1) < 15.0) & (gray.std(axis=1) < 15.0)


def _detect_letterbox(frames: list[np.ndarray], height: int) -> tuple[int, int]:
    """Find cinematic letterbox bars that stay dark across sampled frames.

    The depth model should see the active picture area.  Feeding black bars to
    the model makes them compete in the normalization and produces bright/dirty
    bands in the exported depth video.  A row counts as bar when it is dark in
    most informative samples, so fade-to-black frames, dark scenes and
    subtitles burned into the bars do not break detection.
    """
    if height < 32:
        return 0, height
    votes = [rows for rows in (_dark_rows(frame) for frame in frames if frame.shape[0] == height) if rows.mean() < 0.9]
    if not votes:
        return 0, height
    dark = np.mean(votes, axis=0) >= 0.6
    top = 0
    while top < height // 3 and bool(dark[top]):
        top += 1
    bottom = height
    while bottom > (height * 2) // 3 and bool(dark[bottom - 1]):
        bottom -= 1
    if top < 8 or height - bottom < 8 or bottom - top < height * 0.5:
        return 0, height
    return top, bottom


def _frame_range(depth: np.ndarray) -> tuple[float, float]:
    values = depth[np.isfinite(depth)]
    if values.size == 0:
        return math.nan, math.nan
    lo, hi = np.percentile(values, [1.0, 99.0])
    return float(lo), float(hi)


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
        raise FileNotFoundError(errno.ENOENT, "找不到输入视频", str(source))
    if source == destination:
        raise ValueError("Output path must differ from input path")
    if seconds is not None and seconds <= 0:
        raise ValueError("--seconds must be greater than zero")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("--max-frames must be greater than zero")
    if batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero")
    if colormap != "gray" and colormap not in _COLORMAPS:
        raise ValueError(f"Unknown colormap: {colormap}")

    capture = _open_video(source)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 24.0)
    if not np.isfinite(fps) or fps <= 0:
        fps = 24.0
    reported_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    ok, first_frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"无法从输入视频解码出画面（格式或编码不受支持，或文件已损坏）：{source}")
    # Use the decoded frame, not container metadata: OpenCV applies rotation
    # tags, so phone videos may report swapped dimensions.
    height, width = first_frame.shape[:2]

    frame_limit = reported_count if reported_count > 0 else None
    if seconds is not None:
        frame_limit = min(frame_limit, int(round(seconds * fps))) if frame_limit else int(round(seconds * fps))
    if max_frames is not None:
        frame_limit = min(frame_limit, max_frames) if frame_limit else max_frames
    known_length = bool(frame_limit)
    if not frame_limit:
        frame_limit = 10**9
    frame_limit = max(1, frame_limit)

    if threads:
        try:
            import torch

            torch.set_num_threads(threads)
        except Exception:
            pass
    net, torch, actual_device = _load_model(model, _pick_cache(str(cache_dir) if cache_dir else None), device)

    samples = _sample_frames(source, frame_limit, known_length) or [first_frame]
    crop_top, crop_bottom = _detect_letterbox(samples, height)
    crop = (crop_top, crop_bottom)
    if crop_top or crop_bottom != height:
        _log(f"Detected letterbox rows: top={crop_top}, bottom={height - crop_bottom}; bars will be kept black")
    _log(f"Analyzing {len(samples)} sample frames for the depth range")
    frame_ranges: list[tuple[float, float]] = []
    for start in range(0, len(samples), batch_size):
        batch = [frame[crop_top:crop_bottom, :] for frame in samples[start : start + batch_size] if frame.shape[:2] == (height, width)]
        if batch:
            frame_ranges.extend(_frame_range(depth) for depth in _infer(net, torch, actual_device, batch))
    lo, hi = _normalization_range(frame_ranges)
    _log(f"Normalization range lo={lo:.6f}, hi={hi:.6f}")
    _log(f"Input: {source.name} ({width}×{height} @ {fps:.3f} fps); processing up to {frame_limit} frames")

    destination.parent.mkdir(parents=True, exist_ok=True)
    channels = 1 if colormap == "gray" else 3
    ffmpeg = _find_ffmpeg()
    writer: _FfmpegWriter | _OpenCvWriter = (
        _FfmpegWriter(ffmpeg, destination, width, height, fps, channels)
        if ffmpeg
        else _OpenCvWriter(destination, width, height, fps, channels)
    )
    capture = _open_video(source)
    processed = 0
    started = time.time()
    try:
        while processed < frame_limit:
            frames: list[np.ndarray] = []
            for _ in range(min(batch_size, frame_limit - processed)):
                ok, frame = capture.read()
                if not ok:
                    break
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                frames.append(frame)
            if not frames:
                break
            for depth in _infer(net, torch, actual_device, [frame[crop_top:crop_bottom, :] for frame in frames]):
                writer.write(_render(depth, lo, hi, width, height, crop, colormap))
                processed += 1
            if processed == len(frames) or processed % max(batch_size * 5, 1) == 0 or processed >= frame_limit:
                elapsed = time.time() - started
                _log(f"Inferred {processed} frames ({processed / max(elapsed, 1e-6):.2f} frames/s)")
        if processed == 0:
            raise RuntimeError("No decodable frames found in input")
        _log(f"Finalizing video ({processed} frames)")
        writer.close()
    except BaseException:
        writer.abort()
        destination.unlink(missing_ok=True)
        raise
    finally:
        capture.release()
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
        "letterbox": {"top": crop_top, "bottom": height - crop_bottom},
        "normalization": {"low": lo, "high": hi, "sample_frames": len(samples)},
        "offline": True,
    }
    metadata_path = destination.with_suffix(destination.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _log(f"Wrote {destination} ({destination.stat().st_size:,} bytes)")
    _log(f"Metadata: {metadata_path}")
    return metadata


def _explain(exc: BaseException) -> str:
    """One user-facing line; the traceback above it stays in the diagnostic log."""
    if isinstance(exc, MemoryError) or "not enough memory" in str(exc) or "DefaultCPUAllocator" in str(exc):
        return "内存不足：可关闭其他程序后重试，或先用“只转换前几秒”测试。"
    if isinstance(exc, OSError):
        if exc.errno == errno.ENOSPC or getattr(exc, "winerror", None) in (39, 112):
            return f"磁盘空间不足，无法写入：{exc.filename or '保存目录'}"
        if exc.errno in (errno.EACCES, errno.EPERM):
            return f"没有写入权限：{exc.filename or '保存目录'}"
    text = str(exc)
    if "No space left" in text:
        return f"磁盘空间不足：{text}"
    return text or exc.__class__.__name__


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
    parser.add_argument("--colormap", choices=("gray", *_COLORMAPS), default="gray")
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
    except BaseException as exc:
        traceback.print_exc(file=sys.stderr)
        print(f"error: {_explain(exc)}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
