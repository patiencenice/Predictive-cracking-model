"""
在浏览器中打开预测界面。首次运行会自动在 models/ 下生成演示用 XGBoost 模型。

用法（在 SteelFiberCrackPredictor 目录下）:
  py run_web.py                    # 本机
  py run_web.py --lan              # 局域网可访问
  py run_web.py --port 8502        # 端口被占用时换端口
  py run_web.py --no-browser       # 仅启动服务
  py run_web.py --streamlit-browser  # 换由 Streamlit 弹浏览器

若提示找不到 streamlit:
  py -m pip install --user -r requirements.txt
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import threading
import time
import webbrowser


def _tcp_accepting(host: str, port: int, timeout: float = 1.5) -> bool:
    """用 TCP 探测端口是否已监听，避免系统代理导致 urllib 访问 localhost 失败。"""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _open_url(url: str) -> None:
    if sys.platform == "win32":
        # start 会走系统默认浏览器，部分环境下比 os.startfile / webbrowser 更可靠
        try:
            kw: dict = {}
            if hasattr(subprocess, "CREATE_NO_WINDOW"):
                kw["creationflags"] = subprocess.CREATE_NO_WINDOW
            subprocess.run(
                ["cmd", "/c", "start", "", url],
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                **kw,
            )
            return
        except OSError:
            pass
        try:
            os.startfile(url)  # noqa: S606
            return
        except OSError:
            pass
    webbrowser.open(url)


def _wait_then_open_browser(
    open_url: str,
    probe_host: str,
    port: int,
    timeout_sec: float,
    opened: list[bool],
) -> None:
    if opened[0]:
        return
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if _tcp_accepting(probe_host, port):
            if not opened[0]:
                opened[0] = True
                time.sleep(0.25)
                try:
                    _open_url(open_url)
                except Exception:
                    pass
            return
        time.sleep(0.35)
    print()
    print(f"在 {timeout_sec:.0f} 秒内未检测到端口 {port} 已监听（服务可能启动失败或极慢）。")
    print("请查看上方报错；或手动在浏览器尝试：")
    print(f"  {open_url}")
    print(f"  http://127.0.0.1:{port}/")
    print("若端口被占用，请执行: py run_web.py --port 8502")
    print()


def _print_lan_url(port: int) -> None:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.5)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        print(f"局域网访问（手机/其它电脑）: http://{ip}:{port}")
    except OSError:
        print(
            f"已监听 0.0.0.0；请在「网络和 Internet」中查看本机 IPv4 后手动访问 http://<本机IP>:{port}"
        )


def main() -> None:
    try:
        import streamlit  # noqa: F401
    except ImportError:
        print("未检测到 streamlit，请先执行：")
        print("  py -m pip install --user -r requirements.txt")
        raise SystemExit(1)

    parser = argparse.ArgumentParser(description="启动纤维混凝土开裂预测 Web（Streamlit）")
    parser.add_argument("--port", type=int, default=8501, help="端口，默认 8501")
    parser.add_argument(
        "--address",
        default="127.0.0.1",
        help="监听地址，默认 127.0.0.1（仅本机）",
    )
    parser.add_argument(
        "--lan",
        action="store_true",
        help="等价于监听 0.0.0.0，便于局域网访问",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="不自动打开浏览器",
    )
    parser.add_argument(
        "--streamlit-browser",
        action="store_true",
        help="改由 Streamlit 自动打开浏览器（本脚本不再执行 start/webbrowser）",
    )
    parser.add_argument(
        "--open-timeout",
        type=float,
        default=240.0,
        help="等待端口就绪并打开浏览器的最长秒数（首次加载可能较慢）",
    )
    args = parser.parse_args()
    if args.streamlit_browser and args.no_browser:
        print("--streamlit-browser 与 --no-browser 不能同时使用")
        raise SystemExit(2)
    address = "0.0.0.0" if args.lan else args.address
    port = int(args.port)

    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)

    # 必须用 127.0.0.1 打开页面：部分系统上 localhost 会走 ::1(IPv6)，而服务只绑在 IPv4 上会连不上
    open_url = f"http://127.0.0.1:{port}/"
    probe_host = "127.0.0.1"

    print()
    print(f"本机访问: {open_url}")
    if address == "0.0.0.0":
        _print_lan_url(port)
    print("须保持本窗口运行，关闭即停止服务。")
    if args.streamlit_browser:
        print("已启用 Streamlit 自带打开浏览器；若仍无窗口，请手动访问上方地址。")
    elif not args.no_browser:
        print("端口就绪后将自动打开浏览器（首次可能需数十秒）…")
    print()

    headless = "false" if args.streamlit_browser else "true"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app.py",
        "--server.address",
        address,
        "--server.port",
        str(port),
        "--server.headless",
        headless,
        "--browser.gatherUsageStats",
        "false",
    ]

    opened_flag: list[bool] = [False]
    if not args.no_browser and not args.streamlit_browser:
        threading.Thread(
            target=_wait_then_open_browser,
            args=(open_url, probe_host, port, args.open_timeout, opened_flag),
            daemon=True,
        ).start()

    proc = subprocess.Popen(cmd)
    try:
        code = proc.wait()
        raise SystemExit(code)
    except KeyboardInterrupt:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
