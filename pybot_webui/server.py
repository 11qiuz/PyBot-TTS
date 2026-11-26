"""Local ThreadingHTTPServer backend for PyBot-Larynx WebUI."""
from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import base64
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any
from urllib.parse import unquote, urlparse

from pybot_webui import STATIC_DIR
from pybot_webui.engine import LarynxStudioEngine
from pybot_webui.splitter import split_text

_AUDIO_ROUTE_RE = re.compile(r"^/api/audio/([A-Za-z0-9_-]+)\.wav$")
_RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")

_TK_PICKER_SCRIPT = (
    "import sys, tkinter as tk; from tkinter import filedialog; "
    "r = tk.Tk(); r.withdraw(); r.attributes('-topmost', True); "
    "p = filedialog.askdirectory(initialdir=sys.argv[1] if len(sys.argv) > 1 else '.', "
    "title='选择音频保存文件夹'); "
    "r.destroy(); print(p or '', end='')"
)


def pick_folder_dialog(initial_dir: str | None = None) -> str | None:
    """Open a native OS folder picker in an isolated subprocess (safe for worker threads)."""
    init_p = Path(initial_dir).expanduser() if initial_dir else Path.cwd()
    if not init_p.exists():
        init_p = Path.cwd()
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    res = subprocess.run(
        [sys.executable, "-c", _TK_PICKER_SCRIPT, str(init_p)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
        env=env,
        check=False,
    )
    chosen = (res.stdout or "").strip()
    return chosen or None


class LarynxHTTPServer(ThreadingHTTPServer):
    """Threaded HTTP server carrying the shared LarynxStudioEngine."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        engine: LarynxStudioEngine,
        *,
        quiet: bool = False,
    ) -> None:
        self.engine = engine
        self.quiet = quiet
        super().__init__(server_address, WebUIRequestHandler)


class WebUIRequestHandler(BaseHTTPRequestHandler):
    """REST + static handler for PyBot-Larynx WebUI."""

    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        srv: LarynxHTTPServer = self.server  # type: ignore[assignment]
        if not getattr(srv, "quiet", False):
            super().log_message(format, *args)

    def _send_bytes(
        self,
        status: int,
        data: bytes,
        content_type: str,
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            if extra_headers:
                for k, v in extra_headers.items():
                    self.send_header(k, v)
            self.end_headers()
            if data:
                self.wfile.write(data)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            pass

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_bytes(status, body, "application/json; charset=utf-8")

    def _read_json_body(self) -> dict[str, Any]:
        raw_len = self.headers.get("Content-Length")
        try:
            length = int(raw_len) if raw_len else 0
        except ValueError:
            raise ValueError("Invalid Content-Length")
        if length <= 0:
            return {}
        if length > 32 * 1024 * 1024:
            raise ValueError("Request body too large (>32 MB)")
        raw = self.rfile.read(length)
        doc = json.loads(raw.decode("utf-8"))
        if not isinstance(doc, dict):
            raise ValueError("JSON body must be an object")
        return doc

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        srv: LarynxHTTPServer = self.server  # type: ignore[assignment]

        if path in ("/", "/index.html"):
            index_file = STATIC_DIR / "index.html"
            if not index_file.exists():
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "index.html not found"})
                return
            html_bytes = index_file.read_text(encoding="utf-8").encode("utf-8")
            self._send_bytes(HTTPStatus.OK, html_bytes, "text/html; charset=utf-8")
            return

        if path == "/api/status":
            self._send_json(HTTPStatus.OK, srv.engine.status_dict())
            return

        if path == "/api/clone/profiles":
            self._send_json(HTTPStatus.OK, {"profiles": srv.engine.list_clone_profiles()})
            return

        m_audio = _AUDIO_ROUTE_RE.match(path)
        if m_audio:
            clip_id = m_audio.group(1)
            rec = srv.engine.get_clip(clip_id)
            if rec is None:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Clip not found: {clip_id}"})
                return
            wav_data = rec.wav_bytes
            total = len(wav_data)
            range_hdr = (self.headers.get("Range") or "").strip()
            if range_hdr:
                rm = _RANGE_RE.match(range_hdr)
                if not rm:
                    self._send_bytes(
                        HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE,
                        b"",
                        "audio/wav",
                        extra_headers={"Content-Range": f"bytes */{total}"},
                    )
                    return
                s_str, e_str = rm.group(1), rm.group(2)
                if s_str == "" and e_str == "":
                    start, end = 0, total - 1
                elif s_str == "":
                    suffix = int(e_str)
                    start = max(0, total - suffix)
                    end = total - 1
                else:
                    start = int(s_str)
                    end = int(e_str) if e_str else total - 1
                if start >= total or end < start:
                    self._send_bytes(
                        HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE,
                        b"",
                        "audio/wav",
                        extra_headers={"Content-Range": f"bytes */{total}"},
                    )
                    return
                end = min(end, total - 1)
                chunk = wav_data[start : end + 1]
                self._send_bytes(
                    HTTPStatus.PARTIAL_CONTENT,
                    chunk,
                    "audio/wav",
                    extra_headers={
                        "Accept-Ranges": "bytes",
                        "Content-Range": f"bytes {start}-{end}/{total}",
                    },
                )
                return

            self._send_bytes(
                HTTPStatus.OK,
                wav_data,
                "audio/wav",
                extra_headers={"Accept-Ranges": "bytes"},
            )
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown route: {path}"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        srv: LarynxHTTPServer = self.server  # type: ignore[assignment]

        try:
            body = self._read_json_body()
        except Exception as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return

        try:
            if path == "/api/split":
                segments = split_text(
                    str(body.get("text") or ""),
                    mode=str(body.get("mode") or "sentence"),
                    max_chars=int(body.get("max_chars", 80)),
                )
                self._send_json(HTTPStatus.OK, {"segments": segments})
                return

            if path == "/api/synthesize":
                rec = srv.engine.synthesize_segment(
                    str(body.get("text") or ""),
                    body.get("params") if isinstance(body.get("params"), dict) else None,
                    clip_id=body.get("clip_id"),
                )
                self._send_json(HTTPStatus.OK, {"clip": rec.to_meta_dict()})
                return

            if path == "/api/delete_clip":
                if body.get("clear_all"):
                    removed = srv.engine.clear_clips()
                    self._send_json(HTTPStatus.OK, {"ok": True, "removed": removed})
                else:
                    cid = str(body.get("clip_id") or "")
                    ok = srv.engine.delete_clip(cid)
                    self._send_json(HTTPStatus.OK, {"ok": ok})
                return

            if path == "/api/save":
                raw_ids = body.get("clip_ids") or []
                clip_ids = [str(x) for x in raw_ids]
                save_dir = str(body.get("save_dir") or srv.engine.default_save_dir)
                result = srv.engine.save_clips(
                    clip_ids,
                    save_dir,
                    merge=bool(body.get("merge", False)),
                    merge_gap_ms=float(body.get("merge_gap_ms", 280.0)),
                    merge_name=str(body.get("merge_name") or ""),
                )
                self._send_json(HTTPStatus.OK, result)
                return

            if path == "/api/pick_folder":
                chosen = pick_folder_dialog(body.get("initial_dir"))
                if chosen:
                    self._send_json(HTTPStatus.OK, {"path": chosen, "cancelled": False})
                else:
                    self._send_json(HTTPStatus.OK, {"path": None, "cancelled": True})
                return

            if path == "/api/open_folder":
                target = Path(str(body.get("save_dir") or srv.engine.default_save_dir)).expanduser()
                target.mkdir(parents=True, exist_ok=True)
                if hasattr(os, "startfile"):
                    os.startfile(str(target))
                    self._send_json(HTTPStatus.OK, {"ok": True, "path": str(target)})
                else:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "os.startfile 仅支持 Windows"})
                return

            if path == "/api/clone/extract":
                name = str(body.get("name") or "自定义音色").strip()
                raw_clips = body.get("clips_b64") or []
                if not isinstance(raw_clips, list) or len(raw_clips) == 0:
                    raise ValueError("请提供至少一段参考 WAV 音频。")
                wav_blobs: list[bytes] = []
                for item in raw_clips:
                    b64_s = str(item or "").strip()
                    if b64_s.startswith("data:") and "," in b64_s:
                        b64_s = b64_s.split(",", 1)[1]
                    if not b64_s:
                        continue
                    wav_blobs.append(base64.b64decode(b64_s))
                if not wav_blobs:
                    raise ValueError("参考音频内容为空。")
                save_flag = bool(body.get("save", True))
                prof = srv.engine.extract_and_save_clone_profile(name, wav_blobs, save_to_disk=save_flag)
                self._send_json(
                    HTTPStatus.OK,
                    {"profile": prof.summary_dict(), "profiles": srv.engine.list_clone_profiles()},
                )
                return

            if path == "/api/clone/delete":
                pid = str(body.get("profile_id") or "").strip()
                if not pid:
                    raise ValueError("缺少 profile_id")
                ok = srv.engine.delete_clone_profile(pid)
                self._send_json(HTTPStatus.OK, {"ok": ok, "profiles": srv.engine.list_clone_profiles()})
                return

            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown route: {path}"})
        except (ValueError, KeyError, FileNotFoundError, OSError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"{type(exc).__name__}: {exc}"})


def create_server(
    host: str = "127.0.0.1",
    port: int = 7860,
    *,
    engine: LarynxStudioEngine | None = None,
    quiet: bool = False,
) -> LarynxHTTPServer:
    """Create and configure a LarynxHTTPServer instance."""
    eng = engine or LarynxStudioEngine()
    return LarynxHTTPServer((host, int(port)), eng, quiet=quiet)
