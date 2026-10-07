#!/usr/bin/env bash
set -euo pipefail
python -m pip install -r requirements-build.txt
python -m PyInstaller --noconfirm --clean depth_video.spec
python -m PyInstaller --noconfirm --clean depth_ui.spec
echo "Built dist/depth-video/depth-video and dist/depth-ui/depth-ui"
