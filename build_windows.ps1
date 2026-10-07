param(
  [switch]$IncludeModelCache
)
$ErrorActionPreference = "Stop"
python -m pip install -r requirements-build.txt
python -m PyInstaller --noconfirm --clean depth_video.spec
python -m PyInstaller --noconfirm --clean depth_ui.spec
if ($IncludeModelCache) {
  if (!(Test-Path "hf_cache")) { throw "hf_cache/ is missing; obtain the model before using -IncludeModelCache" }
  Copy-Item -Recurse -Force hf_cache dist\depth-video\hf_cache
  Copy-Item -Recurse -Force hf_cache dist\depth-ui\hf_cache
}
Write-Host "Built dist\depth-video\depth-video.exe"
Write-Host "Built dist\depth-ui\depth-ui.exe (serves http://127.0.0.1:8765)"
Write-Host "Copy hf_cache\ beside the executable (or pass --cache-dir) before running."
