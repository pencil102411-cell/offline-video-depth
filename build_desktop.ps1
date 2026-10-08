param(
  [switch]$SkipNpmInstall,
  [switch]$Debug
)

$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $project

$engine = Join-Path $project 'release\offline-video-depth\cli'
$required = @(
  (Join-Path $project 'release\engine-dist\depth-video\depth-video.exe'),
  (Join-Path $project 'release\engine-dist\depth-video\_internal'),
  (Join-Path $engine 'ffmpeg.exe'),
  (Join-Path $project 'hf_cache\models--depth-anything--Depth-Anything-V2-Small-hf\snapshots\5426e4f0f36572d16453bbda7a8389317b1bef99\model.safetensors')
)
foreach ($path in $required) {
  if (!(Test-Path -LiteralPath $path)) {
    throw "缺少安装包资源：$path。"
  }
}

if (!$SkipNpmInstall) {
  npm install --no-audit --no-fund
}

if ($Debug) {
  npm run tauri:build -- --debug
} else {
  npm run tauri:build
}

$installer = Get-ChildItem -LiteralPath (Join-Path $project 'src-tauri\target\release\bundle\nsis') -Filter '*-setup.exe' -File -ErrorAction SilentlyContinue | Select-Object -First 1
if ($installer) {
  Write-Host "安装包：$($installer.FullName)"
} else {
  Write-Warning 'Tauri 构建完成，但没有找到 NSIS 安装包。'
}
