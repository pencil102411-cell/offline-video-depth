# Offline Video Depth

`depth_video.py` converts a video to a frame-by-frame relative-depth video using
the locally cached **Depth Anything V2 Small** model. It is fully offline:
Hugging Face offline mode is enabled and model loading uses
`local_files_only=True`; there are no API calls and no video-generation model.

## Quick start (source checkout)

The workspace used for verification already contains `hf_cache/` with the model
files. From this directory:

```bash
python depth_video.py 099.mp4 --seconds 5 -o 099_depth_5s.mp4 --device cpu
```

The output is a silent H.264 video and a sidecar JSON metadata file:
`099_depth_5s.mp4` and `099_depth_5s.mp4.json`. The default output name is
`<input>_depth.mp4`. Gray is the default visualization; use
`--colormap turbo|magma|viridis|plasma|inferno|jet` for color.

Useful options:

```text
--seconds N       only process the first N seconds
--max-frames N    cap the number of frames
--batch-size N    inference batch size (4 is a good CPU default)
--device auto|cpu|cuda
--cache-dir PATH  local Transformers cache (defaults to ./hf_cache when present)
--model PATH      local model directory, or the cached model id
```

For a fresh environment, install runtime dependencies with `pip install -r
requirements.txt`. `requirements-lock.txt` records the versions used for the
verification run. The model weights are deliberately kept outside the Python
package; copy the `hf_cache/` directory to the target machine or pass its path
with `--cache-dir`. If the cache is missing, the command fails instead of
trying to download anything.

See [MODEL_CACHE.md](MODEL_CACHE.md) for the expected local cache layout.

## Minimal local UI

```bash
python depth_ui.py --host 127.0.0.1 --port 8765
```

Open <http://127.0.0.1:8765>. Enter a path on the same computer and click
**Convert locally**. The browser UI is stdlib-only and does not upload files.

## Install as a command

```bash
python -m pip install .
depth-video 099.mp4 --seconds 5
depth-video-ui
```

## Windows executable (PyInstaller)

On Windows, install Python 3.10+ and run in PowerShell:

```powershell
.\build_windows.ps1
# Optional: include the local model cache in the folder (large):
.\build_windows.ps1 -IncludeModelCache
```

The one-folder executables are `dist\depth-video\depth-video.exe` (CLI) and
`dist\depth-ui\depth-ui.exe` (local browser UI). Without
`-IncludeModelCache`, copy `hf_cache\` beside the executable and run, for
example:

```powershell
.\dist\depth-video\depth-video.exe C:\video\input.mp4 --seconds 5 --cache-dir .\hf_cache
```

`build.sh` provides the equivalent PyInstaller build on Linux/macOS. PyInstaller
is in `requirements-build.txt`; the model cache is external by default so the
executable stays manageable.

## Verification

The checked-in source was exercised against `099.mp4` (854×480, 24 fps):

```text
python depth_video.py 099.mp4 -o 099_depth_cli_5s.mp4 --seconds 5 --device cpu --cache-dir hf_cache
```

Result: 120 decoded/inferred frames, 5.000 seconds, H.264 854×480 output,
319,955 bytes. The output was decoded again with OpenCV to verify all 120
frames and non-constant depth values; metadata is in
`099_depth_cli_5s.mp4.json`.
