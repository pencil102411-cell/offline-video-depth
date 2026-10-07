#!/usr/bin/env python3
"""Tiny local browser UI for :mod:`depth_video` (stdlib only).

Run ``python depth_ui.py`` and open http://127.0.0.1:8765.  The UI accepts a
path on the same computer; it does not upload video or call a remote service.
"""

from __future__ import annotations

import argparse
import html
import json
import threading
import traceback
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
        result = convert_video(
            source,
            output,
            seconds=seconds,
            model=values.get("model") or DEFAULT_MODEL,
            cache_dir=values.get("cache_dir") or None,
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
<style>body{font:16px system-ui,sans-serif;max-width:760px;margin:40px auto;padding:0 20px;background:#f7f7f8;color:#202124}label{display:block;margin-top:14px;font-weight:600}input,select{width:100%;box-sizing:border-box;padding:9px;font:inherit;border:1px solid #bbb;border-radius:5px}button{margin-top:20px;padding:10px 18px;font:inherit;border:0;border-radius:5px;background:#2159d1;color:white;cursor:pointer}#status{white-space:pre-wrap;margin-top:22px;padding:12px;background:white;border-radius:5px;min-height:40px}small{color:#666}</style>
</head><body><h1>Local video → depth</h1>
<p><small>Runs on this computer with the cached Depth Anything V2 Small model. No upload and no network calls.</small></p>
<form id="form"><label>Input video path<input name="input" required placeholder="/path/to/video.mp4"></label>
<label>Output path (optional)<input name="output" placeholder="defaults to input_depth.mp4"></label>
<label>Seconds (optional)<input name="seconds" type="number" min="0.1" step="0.1" placeholder="whole video"></label>
<label>Colormap<select name="colormap"><option value="gray">gray</option><option>turbo</option><option>magma</option><option>viridis</option><option>plasma</option><option>inferno</option><option>jet</option></select></label>
<label>Cache directory (optional)<input name="cache_dir" placeholder="./hf_cache"></label>
<button type="submit">Convert locally</button></form><div id="status">Ready</div>
<script>const s=document.getElementById('status');document.getElementById('form').onsubmit=async e=>{e.preventDefault();s.textContent='Starting…';let r=await fetch('/run',{method:'POST',body:new URLSearchParams(new FormData(e.target))});s.textContent=JSON.stringify(await r.json(),null,2)};setInterval(async()=>{let r=await fetch('/status');let j=await r.json();if(j.state&&j.state!=='idle')s.textContent=j.message||JSON.stringify(j,null,2)},1500);</script>
</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Local browser UI for offline depth conversion")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Open http://{args.host}:{args.port} in your browser", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
