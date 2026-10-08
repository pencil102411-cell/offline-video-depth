use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fs;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};
use tauri::{AppHandle, Emitter, Manager, State};

#[derive(Default)]
struct ConversionState {
    child: Arc<Mutex<Option<Child>>>,
    cancelled: Arc<Mutex<bool>>,
    engine_log: Arc<Mutex<VecDeque<String>>>,
    active: Arc<Mutex<bool>>,
}

/// Engine output kept for the diagnostic log; long videos print ~1 line per
/// 20 frames, so this covers the whole run in practice.
const ENGINE_LOG_LINES: usize = 4000;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct RuntimeStatus {
    ready: bool,
    engine_path: Option<String>,
    model_cache: Option<String>,
    model_available: bool,
    message: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ConversionRequest {
    input: String,
    output: String,
    #[serde(default)]
    seconds: Option<f64>,
    #[serde(default)]
    colormap: Option<String>,
    #[serde(default)]
    device: Option<String>,
    #[serde(default)]
    batch_size: Option<u32>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct ConversionEvent {
    state: String,
    processed: Option<u64>,
    total: Option<u64>,
    percent: Option<f64>,
    message: Option<String>,
    output: Option<String>,
    success: bool,
    diagnostic: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ConversionStarted {
    task_id: String,
    output: String,
}

fn hidden(command: &mut Command) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000);
    }
}

fn resource_dir(app: &AppHandle) -> PathBuf {
    app.path()
        .resource_dir()
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".."))
}

fn engine_candidates(app: &AppHandle) -> Vec<PathBuf> {
    let resources = resource_dir(app);
    let mut candidates = vec![resources.join("engine-v2").join("depth-video.exe")];
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            candidates.push(parent.join("engine-v2").join("depth-video.exe"));
            candidates.push(
                parent
                    .join("resources")
                    .join("engine-v2")
                    .join("depth-video.exe"),
            );
        }
    }
    let dev = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("release")
        .join("offline-video-depth")
        .join("cli")
        .join("depth-video.exe");
    if cfg!(debug_assertions) {
        candidates.push(dev);
    }
    candidates
}

fn resolve_engine(app: &AppHandle) -> Option<PathBuf> {
    engine_candidates(app)
        .into_iter()
        .find(|path| path.is_file())
}

fn cache_for_engine(engine: &Path) -> PathBuf {
    engine
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("model")
}

fn cache_has_model(cache: &Path) -> bool {
    [
        "config.json",
        "preprocessor_config.json",
        "model.safetensors",
    ]
    .iter()
    .all(|name| cache.join(name).is_file())
        && fs::metadata(cache.join("model.safetensors"))
            .map(|meta| meta.len() == 99_173_660)
            .unwrap_or(false)
}

fn emit(app: &AppHandle, event_name: &str, event: &ConversionEvent) {
    if let Err(error) = app.emit(event_name, event) {
        log::warn!("无法发送转换事件 {event_name}: {error}");
    }
}

fn event(
    state: impl Into<String>,
    processed: Option<u64>,
    total: Option<u64>,
    message: Option<String>,
    output: Option<String>,
    success: bool,
) -> ConversionEvent {
    let state = state.into();
    let percent = match (processed, total) {
        _ if state == "completed" => Some(100.0),
        _ if state == "encoding" => Some(95.0),
        (Some(done), Some(all)) if all > 0 => Some((done as f64 / all as f64 * 95.0).min(95.0)),
        _ => None,
    };
    ConversionEvent {
        state,
        processed,
        total,
        percent,
        message,
        output,
        success,
        diagnostic: None,
    }
}

fn first_number_after(line: &str, marker: &str) -> Option<u64> {
    let start = line.find(marker)? + marker.len();
    line[start..]
        .split(|character: char| !character.is_ascii_digit())
        .find(|part| !part.is_empty())
        .and_then(|part| part.parse().ok())
}

fn handle_log_line(
    app: &AppHandle,
    line: &str,
    total: &Arc<Mutex<Option<u64>>>,
    engine_log: &Arc<Mutex<VecDeque<String>>>,
) {
    let line = line.trim();
    if line.is_empty() {
        return;
    }
    if let Ok(mut lines) = engine_log.lock() {
        if lines.len() >= ENGINE_LOG_LINES {
            lines.pop_front();
        }
        lines.push_back(line.to_string());
    }
    // Tracebacks and library warnings go to the diagnostic log only; the
    // final `error:` line reaches the UI through the finished event.
    if line.starts_with("warnings.warn(")
        || line.contains("Warning")
        || line.starts_with("Traceback")
        || line.starts_with("File \"")
        || line.starts_with("error:")
    {
        return;
    }
    if let Some(limit) = first_number_after(line, "processing up to") {
        if limit < 100_000_000 {
            if let Ok(mut value) = total.lock() {
                *value = Some(limit);
            }
        }
        emit(
            app,
            "conversion://progress",
            &event(
                "processing",
                Some(0),
                (limit < 100_000_000).then_some(limit),
                Some(line.to_string()),
                None,
                false,
            ),
        );
        return;
    }
    if let Some(done) = first_number_after(line, "Inferred") {
        let all = total.lock().ok().and_then(|value| *value);
        emit(
            app,
            "conversion://progress",
            &event(
                "processing",
                Some(done),
                all,
                Some(line.to_string()),
                None,
                false,
            ),
        );
        return;
    }
    let state = if line.starts_with("Loading") {
        "loading"
    } else if line.starts_with("Finalizing") {
        "encoding"
    } else {
        "processing"
    };
    emit(
        app,
        "conversion://progress",
        &event(
            state,
            None,
            total.lock().ok().and_then(|value| *value),
            Some(line.to_string()),
            None,
            false,
        ),
    );
}

fn spawn_reader<R: Read + Send + 'static>(
    reader: R,
    app: AppHandle,
    total: Arc<Mutex<Option<u64>>>,
    engine_log: Arc<Mutex<VecDeque<String>>>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        // Decode lossily: one non-UTF-8 byte (FFmpeg/OpenCV messages, GBK
        // paths) must not stop the reader, or the real error is lost and the
        // engine blocks on a full pipe.
        let mut reader = BufReader::new(reader);
        let mut buffer = Vec::new();
        loop {
            buffer.clear();
            match reader.read_until(b'\n', &mut buffer) {
                Ok(0) | Err(_) => break,
                Ok(_) => {
                    for line in String::from_utf8_lossy(&buffer).split('\r') {
                        handle_log_line(&app, line, &total, &engine_log);
                    }
                }
            }
        }
    })
}

/// The engine ends a failed run with one `error: ...` line meant for users.
fn failure_reason(lines: &VecDeque<String>, code: Option<i32>) -> String {
    if let Some(line) = lines.iter().rev().find_map(|line| line.strip_prefix("error:")) {
        return line.trim().to_string();
    }
    let recent = lines.iter().rev().take(30).cloned().collect::<Vec<_>>().join("\n");
    if recent.contains("MemoryError") || recent.contains("not enough memory") {
        "内存不足：可关闭其他程序后重试，或先用“只转换前几秒”测试".to_string()
    } else if code.is_none() {
        "本地引擎被系统或安全软件强制结束".to_string()
    } else {
        lines
            .back()
            .cloned()
            .unwrap_or_else(|| "本地引擎没有输出任何信息就退出了，可能被安全软件拦截".to_string())
    }
}

fn write_diagnostic(
    app: &AppHandle,
    header: &[(&str, String)],
    lines: &VecDeque<String>,
) -> Option<PathBuf> {
    let dir = app.path().app_log_dir().ok()?;
    fs::create_dir_all(&dir).ok()?;
    let path = dir.join("last-conversion.log");
    let mut text = String::new();
    for (key, value) in header {
        text.push_str(&format!("{key}: {value}\n"));
    }
    text.push_str("\n--- engine output ---\n");
    for line in lines {
        text.push_str(line);
        text.push('\n');
    }
    fs::write(&path, text).ok()?;
    Some(path)
}

/// `name_depth.mp4` -> `name_depth (2).mp4`, so a rerun never asks the user to
/// rename anything and never overwrites earlier work.
fn available_output(output: &Path) -> PathBuf {
    let taken = |path: &Path| path.exists() || path.with_extension("mp4.json").exists();
    if !taken(output) {
        return output.to_path_buf();
    }
    let stem = output
        .file_stem()
        .map(|value| value.to_string_lossy().to_string())
        .unwrap_or_else(|| "depth".to_string());
    (2..10_000)
        .map(|index| output.with_file_name(format!("{stem} ({index}).mp4")))
        .find(|candidate| !taken(candidate))
        .unwrap_or_else(|| output.to_path_buf())
}

#[tauri::command]
fn default_output_dir() -> Result<String, String> {
    let profile = std::env::var_os("USERPROFILE").ok_or_else(|| "无法读取用户目录".to_string())?;
    let path = PathBuf::from(profile).join("Downloads");
    fs::create_dir_all(&path).map_err(|error| format!("创建下载目录失败：{error}"))?;
    Ok(path.to_string_lossy().to_string())
}

#[tauri::command]
fn check_runtime(app: AppHandle) -> RuntimeStatus {
    let Some(engine) = resolve_engine(&app) else {
        return RuntimeStatus {
            ready: false,
            engine_path: None,
            model_cache: None,
            model_available: false,
            message: "未找到本地转换引擎，请重新安装完整版本".to_string(),
        };
    };
    let cache = cache_for_engine(&engine);
    let model_available = cache_has_model(&cache);
    let ffmpeg_available = engine
        .parent()
        .map(|path| path.join("ffmpeg.exe").is_file())
        .unwrap_or(false);
    RuntimeStatus {
        ready: model_available && ffmpeg_available,
        engine_path: Some(engine.to_string_lossy().to_string()),
        model_cache: Some(cache.to_string_lossy().to_string()),
        model_available,
        message: if model_available && ffmpeg_available {
            "本地引擎、模型和 FFmpeg 已就绪".to_string()
        } else if !model_available {
            "已找到引擎，但模型缓存不完整".to_string()
        } else {
            "缺少随包提供的 FFmpeg 编码器".to_string()
        },
    }
}

#[tauri::command]
fn start_conversion(
    app: AppHandle,
    state: State<'_, ConversionState>,
    request: ConversionRequest,
) -> Result<ConversionStarted, String> {
    let input = PathBuf::from(request.input.trim());
    let output = PathBuf::from(request.output.trim());
    if !input.is_file() {
        return Err(format!("找不到输入视频：{}", input.display()));
    }
    if output.as_os_str().is_empty() || input == output {
        return Err("输出路径无效，且不能覆盖输入文件".to_string());
    }
    if !output.is_absolute()
        || output
            .extension()
            .and_then(|s| s.to_str())
            .map(|s| s.to_lowercase())
            != Some("mp4".to_string())
    {
        return Err("请选择完整的 MP4 保存路径".to_string());
    }
    let output = available_output(&output);
    if let Some(seconds) = request.seconds {
        if !seconds.is_finite() || seconds <= 0.0 {
            return Err("时长必须大于 0 秒".to_string());
        }
    }
    let engine =
        resolve_engine(&app).ok_or_else(|| "未找到本地转换引擎，请重新安装完整版本".to_string())?;
    let cache = cache_for_engine(&engine);
    if !cache_has_model(&cache) {
        return Err("未找到随程序安装的模型缓存".to_string());
    }
    if !engine
        .parent()
        .map(|path| path.join("ffmpeg.exe").is_file())
        .unwrap_or(false)
    {
        return Err("未找到随程序安装的 FFmpeg 编码器".to_string());
    }
    {
        let mut active = state
            .active
            .lock()
            .map_err(|_| "转换状态不可用".to_string())?;
        if *active {
            return Err("已有转换任务正在运行".to_string());
        }
        let mut guard = state
            .child
            .lock()
            .map_err(|_| "转换状态不可用".to_string())?;
        if guard.is_some() {
            return Err("已有转换任务正在运行".to_string());
        }
        let parent = output.parent().ok_or("输出目录无效")?;
        fs::create_dir_all(parent).map_err(|e| format!("无法创建输出目录：{e}"))?;
        let job_id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| e.to_string())?
            .as_nanos();
        let job_dir = parent.join(format!(".offline-depth-{}-{job_id}", std::process::id()));
        fs::create_dir(&job_dir).map_err(|e| format!("无法写入保存目录：{e}"))?;
        let staged_output = job_dir.join("result.mp4");
        let app_cache = app
            .path()
            .app_cache_dir()
            .map_err(|e| e.to_string())?
            .join("huggingface");
        let mut command = Command::new(&engine);
        command.arg(&input).arg("--output").arg(&staged_output);
        command.arg("--model").arg(&cache);
        command.arg("--cache-dir").arg(&app_cache);
        command
            .env("PYTHONUTF8", "1")
            .env("PYTHONIOENCODING", "utf-8");
        command
            .env("HF_HUB_OFFLINE", "1")
            .env("TRANSFORMERS_OFFLINE", "1");
        command
            .arg("--device")
            .arg(request.device.as_deref().unwrap_or("auto"));
        command
            .arg("--colormap")
            .arg(request.colormap.as_deref().unwrap_or("gray"));
        if let Some(seconds) = request.seconds {
            command.arg("--seconds").arg(format!("{seconds}"));
        }
        if let Some(batch_size) = request.batch_size {
            if !(1..=32).contains(&batch_size) {
                return Err("批量大小必须在 1 到 32 之间".to_string());
            }
            command.arg("--batch-size").arg(batch_size.to_string());
        }
        if let Some(parent) = engine.parent() {
            command.current_dir(parent);
            let mut search_path = vec![parent.to_path_buf()];
            if let Some(existing) = std::env::var_os("PATH") {
                search_path.extend(std::env::split_paths(&existing));
            }
            if let Ok(joined) = std::env::join_paths(search_path) {
                command.env("PATH", joined);
            }
        }
        command
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        hidden(&mut command);
        let mut child = command.spawn().map_err(|error| {
            let _ = fs::remove_dir(&job_dir);
            format!("启动本地引擎失败：{error}")
        })?;
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        *state
            .cancelled
            .lock()
            .map_err(|_| "转换状态不可用".to_string())? = false;
        state
            .engine_log
            .lock()
            .map_err(|_| "转换状态不可用".to_string())?
            .clear();
        *guard = Some(child);
        *active = true;
        drop(guard);
        drop(active);

        // Publish the initial state before reader threads can forward the
        // engine's first log line, keeping the UI state order deterministic.
        let started = event(
            "started",
            Some(0),
            None,
            Some("已开始本地转换".to_string()),
            Some(output.to_string_lossy().to_string()),
            false,
        );
        emit(&app, "conversion://progress", &started);

        let total = Arc::new(Mutex::new(None));
        let mut readers = Vec::new();
        if let Some(reader) = stdout {
            readers.push(spawn_reader(
                reader,
                app.clone(),
                total.clone(),
                state.engine_log.clone(),
            ));
        }
        if let Some(reader) = stderr {
            readers.push(spawn_reader(
                reader,
                app.clone(),
                total.clone(),
                state.engine_log.clone(),
            ));
        }

        let child_slot = state.child.clone();
        let cancelled = state.cancelled.clone();
        let engine_log = state.engine_log.clone();
        let active = state.active.clone();
        let app_for_wait = app.clone();
        let output_for_wait = output.clone();
        let input_for_wait = input.clone();
        let engine_for_wait = engine.clone();
        let cache_for_wait = cache.clone();
        thread::spawn(move || {
            // Poll without holding the mutex while the process runs. This lets
            // the cancel command acquire the same lock and kill the child.
            let status = loop {
                let result = match child_slot.lock() {
                    Ok(mut slot) => match slot.as_mut() {
                        Some(child) => child.try_wait(),
                        None => Ok(None),
                    },
                    Err(_) => Err(std::io::Error::other("转换状态锁损坏")),
                };
                match result {
                    Ok(Some(status)) => {
                        if let Ok(mut slot) = child_slot.lock() {
                            slot.take();
                        }
                        break Some(status);
                    }
                    Ok(None) => thread::sleep(std::time::Duration::from_millis(120)),
                    Err(_) => break None,
                }
            };
            for reader in readers {
                let _ = reader.join();
            }
            let was_cancelled = cancelled.lock().map(|value| *value).unwrap_or(false);
            let lines = engine_log.lock().map(|value| value.clone()).unwrap_or_default();
            let code = status.as_ref().and_then(|value| value.code());
            if was_cancelled {
                let _ = fs::remove_dir_all(&job_dir);
                if let Ok(mut value) = active.lock() {
                    *value = false;
                }
                let cancelled_event = event(
                    "cancelled",
                    None,
                    None,
                    Some("已取消转换".to_string()),
                    Some(output_for_wait.to_string_lossy().to_string()),
                    false,
                );
                emit(&app_for_wait, "conversion://progress", &cancelled_event);
                emit(&app_for_wait, "conversion://finished", &cancelled_event);
                return;
            }
            let exited_ok = status.as_ref().is_some_and(|value| value.success());
            let publish_result = if exited_ok && staged_output.is_file() {
                publish_output(&staged_output, &output_for_wait)
            } else if exited_ok {
                Err("引擎报告完成，但没有生成视频文件".to_string())
            } else {
                Err(failure_reason(&lines, code))
            };
            let engine_dir = engine_for_wait.parent().unwrap_or(Path::new("."));
            let diagnostic = write_diagnostic(
                &app_for_wait,
                &[
                    (
                        "result",
                        match &publish_result {
                            Ok(()) => "completed".to_string(),
                            Err(message) => format!("failed: {message}"),
                        },
                    ),
                    (
                        "exit code",
                        code.map_or("none (killed)".to_string(), |value| value.to_string()),
                    ),
                    ("input", input_for_wait.display().to_string()),
                    ("output", output_for_wait.display().to_string()),
                    ("work dir", job_dir.display().to_string()),
                    ("engine", engine_for_wait.display().to_string()),
                    ("model ok", cache_has_model(&cache_for_wait).to_string()),
                    (
                        "ffmpeg ok",
                        engine_dir.join("ffmpeg.exe").is_file().to_string(),
                    ),
                ],
                &lines,
            );
            let _ = fs::remove_dir_all(&job_dir);
            if let Ok(mut value) = active.lock() {
                *value = false;
            }
            if publish_result.is_ok() {
                let done = event(
                    "completed",
                    None,
                    None,
                    Some("转换完成".to_string()),
                    Some(output_for_wait.to_string_lossy().to_string()),
                    true,
                );
                emit(&app_for_wait, "conversion://progress", &done);
                emit(&app_for_wait, "conversion://finished", &done);
            } else {
                let reason = publish_result.unwrap_err();
                let exit = code.map_or("被强制结束".to_string(), |value| format!("退出码 {value}"));
                let mut failed = event(
                    "failed",
                    None,
                    None,
                    Some(format!("转换失败：{reason}（引擎{exit}）")),
                    Some(output_for_wait.to_string_lossy().to_string()),
                    false,
                );
                failed.diagnostic = diagnostic.map(|path| path.to_string_lossy().to_string());
                emit(&app_for_wait, "conversion://error", &failed);
                emit(&app_for_wait, "conversion://finished", &failed);
            }
        });
    }
    let task_id = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_millis().to_string())
        .unwrap_or_else(|_| "conversion".to_string());
    Ok(ConversionStarted {
        task_id,
        output: output.to_string_lossy().to_string(),
    })
}

fn publish_output(staged: &Path, destination: &Path) -> Result<(), String> {
    if destination.exists() || destination.with_extension("mp4.json").exists() {
        return Err("输出文件名已被占用，请选择新的保存位置".to_string());
    }
    let metadata = staged.with_extension("mp4.json");
    let raw = fs::read(&metadata).map_err(|e| format!("读取结果信息失败：{e}"))?;
    let mut data: serde_json::Value = serde_json::from_slice(&raw).map_err(|e| e.to_string())?;
    data["output"] = serde_json::json!(destination.to_string_lossy());
    fs::write(
        &metadata,
        serde_json::to_vec_pretty(&data).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    fs::rename(staged, destination).map_err(|e| format!("保存视频失败：{e}"))?;
    fs::rename(metadata, destination.with_extension("mp4.json"))
        .map_err(|e| format!("视频已保存，保存结果信息失败：{e}"))?;
    Ok(())
}

#[tauri::command]
fn cancel_conversion(state: State<'_, ConversionState>) -> Result<bool, String> {
    let mut guard = state
        .child
        .lock()
        .map_err(|_| "转换状态不可用".to_string())?;
    let Some(child) = guard.as_mut() else {
        return Ok(false);
    };
    *state
        .cancelled
        .lock()
        .map_err(|_| "转换状态不可用".to_string())? = true;
    #[cfg(windows)]
    {
        let pid = child.id().to_string();
        let mut killer = Command::new("taskkill");
        killer.args(["/PID", &pid, "/T", "/F"]);
        hidden(&mut killer);
        let result = killer
            .status()
            .map_err(|error| format!("取消转换失败：{error}"))?;
        if !result.success() && child.try_wait().ok().flatten().is_none() {
            *state
                .cancelled
                .lock()
                .map_err(|_| "转换状态不可用".to_string())? = false;
            return Err("未能结束转换进程，请稍后重试".to_string());
        }
    }
    #[cfg(not(windows))]
    child
        .kill()
        .map_err(|error| format!("取消转换失败：{error}"))?;
    Ok(true)
}

#[tauri::command]
fn open_path(path: String) -> Result<(), String> {
    let target = PathBuf::from(path);
    if !target.exists() {
        return Err(format!("路径不存在：{}", target.display()));
    }
    #[cfg(windows)]
    {
        let mut command = Command::new("explorer.exe");
        if target.is_file() {
            command.arg(format!("/select,{}", target.display()));
        } else {
            command.arg(&target);
        }
        hidden(&mut command);
        command
            .spawn()
            .map_err(|error| format!("无法打开路径：{error}"))?;
        return Ok(());
    }
    #[cfg(not(windows))]
    {
        Command::new("xdg-open")
            .arg(&target)
            .spawn()
            .map_err(|error| format!("无法打开路径：{error}"))?;
        Ok(())
    }
}

pub fn run() {
    tauri::Builder::default()
        .manage(ConversionState::default())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_log::Builder::default().build())
        .on_window_event(|window, window_event| {
            if let tauri::WindowEvent::CloseRequested { api, .. } = window_event {
                let state = window.state::<ConversionState>();
                if state.active.lock().map(|value| *value).unwrap_or(false) {
                    api.prevent_close();
                    emit(
                        window.app_handle(),
                        "conversion://progress",
                        &event(
                            "processing",
                            None,
                            None,
                            Some("正在转换，请先点击取消转换，结束后再关闭窗口".to_string()),
                            None,
                            false,
                        ),
                    );
                }
            }
        })
        .invoke_handler(tauri::generate_handler![
            default_output_dir,
            check_runtime,
            start_conversion,
            cancel_conversion,
            open_path
        ])
        .run(tauri::generate_context!())
        .expect("启动离线深度视频失败");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lines(items: &[&str]) -> VecDeque<String> {
        items.iter().map(|line| line.to_string()).collect()
    }

    #[test]
    fn failure_reason_prefers_engine_error_line() {
        let log = lines(&[
            "Loading model",
            "Traceback (most recent call last):",
            "error: 磁盘空间不足，无法写入：D:/out.mp4",
            "[mov @ 0x1] moov atom not found",
        ]);
        assert_eq!(failure_reason(&log, Some(1)), "磁盘空间不足，无法写入：D:/out.mp4");
    }

    #[test]
    fn failure_reason_explains_silent_or_killed_engine() {
        assert!(failure_reason(&VecDeque::new(), Some(1)).contains("没有输出任何信息"));
        assert!(failure_reason(&lines(&["Inferred 20 frames"]), None).contains("强制结束"));
        assert!(failure_reason(&lines(&["MemoryError"]), Some(1)).contains("内存不足"));
    }

    #[test]
    fn available_output_never_reuses_a_taken_name() {
        let dir = std::env::temp_dir().join(format!("offline-depth-test-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let first = dir.join("片段_depth.mp4");
        assert_eq!(available_output(&first), first);
        fs::write(&first, b"x").unwrap();
        assert_eq!(available_output(&first), dir.join("片段_depth (2).mp4"));
        fs::write(dir.join("片段_depth (2).mp4.json"), b"{}").unwrap();
        assert_eq!(available_output(&first), dir.join("片段_depth (3).mp4"));
        fs::remove_dir_all(&dir).unwrap();
    }
}
