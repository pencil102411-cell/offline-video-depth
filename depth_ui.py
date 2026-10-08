#!/usr/bin/env python3
"""Tiny local browser UI for :mod:`depth_video` (stdlib only).

Run ``python depth_ui.py`` and open http://127.0.0.1:8765.  The UI accepts a
path on the same computer; it does not upload video or call a remote service.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import threading
import time
import traceback
import sys
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

from depth_video import DEFAULT_MODEL, convert_video


_JOB: dict[str, object] = {"state": "idle", "message": "Ready"}
_LOCK = threading.Lock()


def _set_job(**updates: object) -> None:
    with _LOCK:
        _JOB.update(updates)


def _run_job(values: dict[str, str]) -> None:
    try:
        source = Path(values["input"]).expanduser()
        output_text = values.get("output", "").strip()
        output = Path(output_text).expanduser() if output_text else source.with_name(source.stem + "_depth.mp4")
        seconds = float(values["seconds"]) if values.get("seconds", "").strip() else None
        _set_job(state="running", message=f"Processing {source} …", output=str(output))
        cache_text = values.get("cache_dir", "").strip()
        if not cache_text and getattr(sys, "frozen", False):
            bundled_cache = Path(sys.executable).resolve().parent / "hf_cache"
            if bundled_cache.is_dir():
                cache_text = str(bundled_cache)
        result = convert_video(
            source,
            output,
            seconds=seconds,
            model=values.get("model") or DEFAULT_MODEL,
            cache_dir=cache_text or None,
            device=values.get("device") or "auto",
            batch_size=int(values.get("batch_size") or 4),
            colormap=values.get("colormap") or "gray",
        )
        _set_job(state="done", message=f"Done: {result['output']}", result=result)
    except Exception as exc:  # report to the local page; no traceback to users
        _set_job(state="error", message=str(exc), traceback=traceback.format_exc())


class Handler(BaseHTTPRequestHandler):
    def _send(self, body: str, content_type: str = "text/html; charset=utf-8", status: int = 200) -> None:
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/status":
            with _LOCK:
                body = json.dumps(_JOB)
            self._send(body, "application/json; charset=utf-8")
            return
        self._send(PAGE)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/run":
            self._send("Not found", status=404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        values = {key: vals[-1] for key, vals in parse_qs(self.rfile.read(length).decode("utf-8")).items()}
        with _LOCK:
            running = _JOB.get("state") == "running"
        if running:
            self._send(json.dumps({"error": "A conversion is already running"}), "application/json", 409)
            return
        thread = threading.Thread(target=_run_job, args=(values,), daemon=True)
        thread.start()
        self._send(json.dumps({"state": "started"}), "application/json; charset=utf-8")

    def log_message(self, format: str, *args: object) -> None:
        print(f"[ui] {format % args}", flush=True)


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Local video depth</title>
<style>
:root{color-scheme:dark;font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;background:#10131a;color:#eef2f8}
*{box-sizing:border-box}body{margin:0;min-height:100vh;background:radial-gradient(circle at 12% 0%,#263655 0,#121722 42%,#0d1016 100%);font-size:16px}
main{width:min(900px,calc(100% - 40px));margin:0 auto;padding:52px 0 70px}h1{font-size:clamp(32px,5vw,52px);line-height:1.05;margin:12px 0;letter-spacing:-.04em}p{color:#aeb9cb;line-height:1.55}.intro{max-width:650px;font-size:17px;margin:0 0 28px}.card{border:1px solid #2a3447;background:rgba(22,28,40,.9);border-radius:18px;box-shadow:0 18px 60px #0005;padding:24px;max-width:760px}label{display:block;margin-top:16px;color:#cfd8e7;font-size:13px;font-weight:700}input,select{width:100%;box-sizing:border-box;padding:12px 13px;margin-top:7px;font:inherit;border:1px solid #354158;border-radius:10px;background:#101620;color:#eef2f8;outline:none}input:focus,select:focus{border-color:#79a7ff;box-shadow:0 0 0 3px #79a7ff22}button{margin-top:22px;width:100%;padding:13px 18px;font:700 15px inherit;border:0;border-radius:11px;background:linear-gradient(135deg,#8cc4ff,#9aafff);color:#08111f;cursor:pointer}button:hover{filter:brightness(1.08)}#status{white-space:pre-wrap;margin-top:18px;padding:15px;background:#0c111a;border:1px solid #29344a;border-radius:12px;min-height:70px;line-height:1.45;color:#d7e2f5}.eyebrow{font-size:12px;letter-spacing:.16em;text-transform:uppercase;color:#8da8d7;font-weight:700}.badge{display:inline-block;margin-top:6px;border-radius:999px;padding:6px 10px;background:#18352f;color:#8be1bd;font-size:12px;font-weight:700}
+</style>
</head><body><main><div class="eyebrow">Depth Studio · local edition</div><h1>Local video → depth</h1>
<p class="intro">Offline Depth Anything V2 Small processing that stays on this computer. No upload and no network calls.</p><div class="card">
<form id="form"><label>Input video path<input name="input" required placeholder="/path/to/video.mp4"></label>
<label>Output path (optional)<input name="output" placeholder="defaults to input_depth.mp4"></label>
<label>Seconds (optional)<input name="seconds" type="number" min="0.1" step="0.1" placeholder="whole video"></label>
<label>Colormap<select name="colormap"><option value="gray">gray</option><option>turbo</option><option>magma</option><option>viridis</option><option>plasma</option><option>inferno</option><option>jet</option></select></label>
<label>Cache directory (optional)<input name="cache_dir" placeholder="./hf_cache"></label>
<button type="submit">Convert locally</button></form><div id="status">Ready when you are.</div></div></main>
<script>const s=document.getElementById('status');document.getElementById('form').onsubmit=async e=>{e.preventDefault();s.textContent='Starting…';let r=await fetch('/run',{method:'POST',body:new URLSearchParams(new FormData(e.target))});s.textContent=JSON.stringify(await r.json(),null,2)};setInterval(async()=>{let r=await fetch('/status');let j=await r.json();if(j.state&&j.state!=='idle')s.textContent=j.message||JSON.stringify(j,null,2)},1500);</script>
</body></html>"""


_LOG_FILE: Path | None = None


def _append_log(message: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    if _LOG_FILE is not None:
        try:
            _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
            with _LOG_FILE.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except Exception:
            pass


def _show_fatal(message: str) -> None:
    _append_log("FATAL: " + message)
    try:
        from tkinter import Tk, messagebox
        root = Tk()
        root.withdraw()
        messagebox.showerror("Offline Video Depth", message)
        root.destroy()
    except Exception:
        pass


def _open_url_when_ready(url: str) -> None:
    deadline = time.monotonic() + 45.0
    status_url = url.rstrip("/") + "/status"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(status_url, timeout=1.0) as response:
                if response.status == 200:
                    opened = webbrowser.open(url, new=2)
                    if not opened and hasattr(os, "startfile"):
                        os.startfile(url)  # type: ignore[attr-defined]
                    _append_log(f"Browser launch requested: {url} (webbrowser={opened})")
                    return
        except Exception:
            time.sleep(0.25)
    _show_fatal(f"The local UI did not start within 45 seconds.\n\nOpen this address manually:\n{url}\n\nSee the launch log for details.")


def main() -> int:
    global _LOG_FILE
    parser = argparse.ArgumentParser(description="Local browser UI for offline depth conversion")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true", help="do not open the local UI in the default browser")
    parser.add_argument("--debug-log", type=Path, help="append startup diagnostics to this log file")
    args = parser.parse_args()
    _LOG_FILE = args.debug_log.expanduser().resolve() if args.debug_log else None
    url = f"http://{args.host}:{args.port}"
    try:
        server = ThreadingHTTPServer((args.host, args.port), Handler)
        _append_log(f"Server bound at {url}; executable={Path(__file__).resolve()}")
    except Exception as exc:
        _show_fatal(f"Could not start the local server at {url}: {exc}")
        return 1
    if not args.no_open:
        threading.Thread(target=_open_url_when_ready, args=(url,), daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        _show_fatal(f"The local server stopped unexpectedly: {exc}")
        return 1
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
