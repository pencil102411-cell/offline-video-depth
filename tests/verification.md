# Reproducible verification

The source workspace's `099.mp4` fixture is intentionally not copied into the
project tree (10.8 MB). To reproduce, place that file beside the project or
provide any local MP4 and run:

```bash
python depth_video.py /path/to/099.mp4 \
  --seconds 5 --device cpu --cache-dir /path/to/hf_cache \
  -o /tmp/099_depth_cli_5s.mp4
```

Recorded verification (2026-10-07): input 854×480 at 24 fps; 120 frames were
inferred; output duration 5.000 s, H.264 854×480, 319,955 bytes. OpenCV
decoded all 120 output frames and their means varied (non-constant depth).
The fixture SHA-256 was
`a8373f9cd92a017f8cff950687cf3f43ca81ad9d38771d9d0fbbfae068464428`.

`tests/fixtures/README.md` explains the deliberately external fixture policy.
