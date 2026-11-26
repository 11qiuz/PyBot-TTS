"""CLI entry point for the PyBot-Larynx local WebUI voice studio."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import threading
import webbrowser

_LEGACY_DIR = Path(__file__).resolve().parent / "legacy"
if _LEGACY_DIR.is_dir() and str(_LEGACY_DIR) not in sys.path:
    sys.path.append(str(_LEGACY_DIR))

from pybot_webui import DEFAULT_SAVE_DIR
from pybot_webui.engine import LarynxStudioEngine
from pybot_webui.server import create_server


def _ensure_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def main(argv: list[str] | None = None) -> int:
    _ensure_utf8_console()
    parser = argparse.ArgumentParser(description="PyBot-Larynx 赛博声带本地 WebUI")
    parser.add_argument("--host", default="127.0.0.1", help="监听主机地址 (默认 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7860, help="监听端口 (默认 7860，0 表示随机端口)")
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR, help="默认音频保存文件夹")
    parser.add_argument("--no-browser", action="store_true", help="启动后不自动打开浏览器")
    args = parser.parse_args(argv)

    engine = LarynxStudioEngine(default_save_dir=args.save_dir)
    engine.start_loading_async()
    server = create_server(args.host, args.port, engine=engine, quiet=True)
    bound_host, bound_port = server.server_address[:2]
    url = f"http://{bound_host}:{bound_port}/"

    print("============================================================", flush=True)
    print("  PyBot-Larynx · 赛博声带 WebUI 已启动", flush=True)
    print(f"  访问地址: {url}", flush=True)
    print(f"  默认保存: {Path(args.save_dir).resolve()}", flush=True)
    print("  按 Ctrl+C 停止服务", flush=True)
    print("============================================================", flush=True)

    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n正在关闭 PyBot-Larynx WebUI...", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
