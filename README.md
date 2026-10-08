# 离线深度视频

这是一个 Windows 本地桌面工具：选择普通视频，生成灰度深度视频。模型、推理和 FFmpeg 编码都在本机完成，不调用 API，不访问网络，也不需要 Python、端口或浏览器。

## 使用安装版

运行 `src-tauri/target/release/bundle/nsis/离线深度视频_0.3.0_x64-setup.exe`，安装后从桌面或开始菜单打开“离线深度视频”。

1. 点击“选择视频”，或把视频拖进窗口。
2. 确认自动生成的 `<原文件名>_depth.mp4` 保存位置。
3. 点击“开始转换”。窗口会显示模型加载、已处理帧数和编码状态；需要时可以取消。
4. 完成后点击“打开文件夹”查看灰度深度视频和 `.json` 元数据。同名文件已存在时自动另存为 `_depth (2).mp4`，不会覆盖。

转换失败时，界面显示引擎给出的具体原因和退出码，并可点击“打开诊断日志”查看完整记录（`%LOCALAPPDATA%\com.local.offline-video-depth\logs\last-conversion.log`）。

安装包把 `depth-video.exe`、Depth Anything V2 Small 本地模型缓存和 FFmpeg 一起放进资源目录，断网也可以使用。当前引擎为 CPU 版：推理分辨率固定为短边 518，所以速度只取决于帧数和画幅比，与视频是 720p 还是 4K 基本无关（8 核 CPU 约 0.6–0.9 帧/秒）。

转换是单遍流式的：先抽样少量帧确定黑边和全片统一的深度范围，然后逐帧推理并直接交给 FFmpeg 编码，不在磁盘或内存里缓存整段深度数据，长视频和 4K 视频不会因此耗尽空间。

## 从源码生成安装包

需要 Node.js、Rust、WebView2、Python 3.12（torch、transformers、opencv-python、PyInstaller）、`hf_cache/` 模型缓存和 `release/offline-video-depth/cli/ffmpeg.exe`。

先用 PyInstaller 生成目录版引擎（不要用单文件模式：单文件每次运行都要解压约 1 GB 到临时目录，取消转换时还会残留）：

```powershell
python -m PyInstaller depth_video.spec --noconfirm --distpath release/engine-dist --workpath build/engine-work
```

再在 PowerShell 中执行：

```powershell
.\build_desktop.ps1
```

脚本会安装前端依赖、运行 Vue 类型检查和构建，并调用 Tauri 生成 NSIS 安装器。安装器会把整个本地引擎目录映射到 `resources/engine`，不会启动 HTTP 服务。

前端单独检查：

```powershell
npm run check
cargo check --manifest-path src-tauri/Cargo.toml
```

## Python/CLI 开发入口

`depth_video.py` 仍可用于算法调试。它只使用本地 Transformers 缓存，网络下载已禁用：

```powershell
python depth_video.py input.mp4 --seconds 5 --device cpu --cache-dir hf_cache -o output_depth.mp4
```

默认输出为 H.264 灰度视频，并生成同名 `.json` 元数据。程序会识别上下电影黑边：黑边不送入深度模型，导出时保持黑色，避免归一化造成亮条带。

## 资源和许可

模型缓存的目录约定见 [MODEL_CACHE.md](MODEL_CACHE.md)。项目代码和第三方声明分别见 [LICENSE](LICENSE)、[NOTICE](NOTICE)。
