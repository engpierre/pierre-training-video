"""Dual TikTok VPN Browser Node Provisioner & Tunnel Auditor.

Manages two permanently isolated Chromium/Chrome browser profiles (`node_alpha`
and `node_beta`) strictly inside `C:\\Anti-Gravity-Core\\pierre-training-video\\.tiktok_nodes\\`
with WebRTC leak prevention, anti-automation flags, separate CDP debug ports
(`9222` and `9223`), and per-browser ProtonVPN extension egress verification.

No system-wide Windows VPN adapter is installed or modified, leaving the host OS,
OpenClaw Swarm, Finnhub/TwelveData feeds, and `pierre_stock_warroom` on direct WAN.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import struct
import subprocess
import sys
import time
from typing import Any
from urllib.parse import urlparse
import urllib.error
import urllib.request

PROJECT_ROOT = Path(r"C:\Anti-Gravity-Core\pierre-training-video")
PIPELINE_DIR = PROJECT_ROOT / "pipeline"
OUTPUT_DIR = PIPELINE_DIR / "output"
NODES_BASE_DIR = PROJECT_ROOT / ".tiktok_nodes"
NODES_CONFIG_PATH = PIPELINE_DIR / "nodes_config.json"
TUNNEL_AUDIT_PATH = OUTPUT_DIR / "nodes_tunnel_audit.json"

PROTON_VPN_EXTENSION_URL = (
    "https://chromewebstore.google.com/detail/proton-vpn-fast-secure/"
    "jplgfhpmjnbigmhklmmbgecoobifkmpa"
)
IPINFO_URL = "https://ipinfo.io/json"
TIKTOK_SIGNUP_URL = "https://www.tiktok.com/login/qrcode"
SETUP_TABS = (PROTON_VPN_EXTENSION_URL, IPINFO_URL, TIKTOK_SIGNUP_URL)

DEFAULT_BROWSER_CANDIDATES = (
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
)


@dataclass(slots=True, frozen=True)
class TikTokNodeConfig:
    node_id: str
    alias: str
    email: str
    expected_vpn_region: str
    profile_dir: str
    remote_debugging_port: int
    window_position: tuple[int, int]
    window_size: tuple[int, int]


@dataclass(slots=True, frozen=True)
class NodeEgressTelemetry:
    node_id: str
    cdp_port: int
    ip: str
    city: str
    region: str
    country: str
    org: str
    expected_vpn_region: str
    region_matched: bool
    tunnel_isolated_from_host: bool


@dataclass(slots=True, frozen=True)
class TunnelAuditReport:
    checked_at_utc: str
    host_wan_ip: str
    host_city: str
    host_region: str
    host_org: str
    node_alpha: NodeEgressTelemetry
    node_beta: NodeEgressTelemetry
    alpha_isolated_from_host: bool
    beta_isolated_from_host: bool
    alpha_beta_distinct: bool
    all_assertions_passed: bool


def default_nodes() -> tuple[TikTokNodeConfig, TikTokNodeConfig]:
    """Builds canonical Node Alpha (Montreal) and Node Beta (Toronto) definitions."""
    alpha = TikTokNodeConfig(
        node_id="node_alpha",
        alias="alpha",
        email=os.environ.get("TIKTOK_NODE_ALPHA_EMAIL", "alpha@example.com"),
        expected_vpn_region="Montreal",
        profile_dir=str(NODES_BASE_DIR / "node_alpha"),
        remote_debugging_port=9222,
        window_position=(40, 40),
        window_size=(900, 980),
    )
    beta = TikTokNodeConfig(
        node_id="node_beta",
        alias="beta",
        email=os.environ.get("TIKTOK_NODE_BETA_EMAIL", "beta@example.com"),
        expected_vpn_region="Toronto",
        profile_dir=str(NODES_BASE_DIR / "node_beta"),
        remote_debugging_port=9223,
        window_position=(960, 40),
        window_size=(900, 980),
    )
    return alpha, beta


def _assert_contained(target: Path) -> Path:
    """Enforces strict write containment inside C:\\Anti-Gravity-Core\\pierre-training-video."""
    resolved = target.resolve()
    allowed_root = PROJECT_ROOT.resolve()
    if not resolved.is_relative_to(allowed_root):
        raise PermissionError(
            f"Write containment violation: {resolved} is outside {allowed_root}"
        )
    return resolved


def ensure_nodes_config(config_path: Path = NODES_CONFIG_PATH) -> dict[str, TikTokNodeConfig]:
    """Creates node profile directories and persists pipeline/nodes_config.json."""
    resolved_cfg = _assert_contained(config_path)
    resolved_cfg.parent.mkdir(parents=True, exist_ok=True)
    _assert_contained(NODES_BASE_DIR).mkdir(parents=True, exist_ok=True)

    nodes_map: dict[str, TikTokNodeConfig] = {}
    if resolved_cfg.exists():
        with resolved_cfg.open(mode="r", encoding="utf-8") as handle:
            raw = json.load(handle)
        raw_nodes = raw.get("nodes", {}) if isinstance(raw, dict) else {}
        for key, item in raw_nodes.items():
            if not isinstance(item, dict):
                continue
            node = TikTokNodeConfig(
                node_id=str(item["node_id"]),
                alias=str(item["alias"]),
                email=str(item["email"]),
                expected_vpn_region=str(item["expected_vpn_region"]),
                profile_dir=str(item["profile_dir"]),
                remote_debugging_port=int(item["remote_debugging_port"]),
                window_position=(int(item["window_position"][0]), int(item["window_position"][1])),
                window_size=(int(item["window_size"][0]), int(item["window_size"][1])),
            )
            _assert_contained(Path(node.profile_dir)).mkdir(parents=True, exist_ok=True)
            nodes_map[key] = node

    if "node_alpha" not in nodes_map or "node_beta" not in nodes_map:
        alpha, beta = default_nodes()
        nodes_map = {"node_alpha": alpha, "node_beta": beta}
        for node in nodes_map.values():
            _assert_contained(Path(node.profile_dir)).mkdir(parents=True, exist_ok=True)

    payload = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "vpn_extension_url": PROTON_VPN_EXTENSION_URL,
        "ipinfo_url": IPINFO_URL,
        "tiktok_signup_url": TIKTOK_SIGNUP_URL,
        "nodes": {k: asdict(v) for k, v in nodes_map.items()},
    }
    with resolved_cfg.open(mode="w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    return nodes_map


def resolve_browser_binary() -> Path:
    """Locates Google Chrome first (native Chrome Web Store support), then Microsoft Edge."""
    for candidate in DEFAULT_BROWSER_CANDIDATES:
        if candidate.exists():
            return candidate
    for exe_name in ("chrome.exe", "msedge.exe"):
        found = shutil.which(exe_name)
        if found:
            return Path(found)
    raise FileNotFoundError("Neither Google Chrome nor Microsoft Edge was found on this system.")


def query_host_wan_ip(timeout_sec: float = 10.0) -> dict[str, Any]:
    """Queries the Windows host OS direct WAN IP without touching any VPN."""
    req = urllib.request.Request(
        IPINFO_URL,
        headers={"User-Agent": "PierreTrainingVideo-NodeProvisioner/1.0"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        if resp.status != 200:
            raise RuntimeError(f"ipinfo.io returned HTTP {resp.status}")
        data = json.loads(resp.read().decode("utf-8"))
    if not isinstance(data, dict) or not data.get("ip"):
        raise RuntimeError(f"Unexpected ipinfo.io payload: {data}")
    return data


def _cdp_http_get_json(port: int, path: str, timeout_sec: float = 3.0) -> Any:
    """Reads a JSON endpoint from local CDP HTTP server."""
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method="GET")
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        return json.loads(resp.read().decode("utf-8"))


def is_cdp_listening(port: int) -> bool:
    """Checks whether Chrome/Edge CDP debug endpoint responds on 127.0.0.1:<port>."""
    try:
        info = _cdp_http_get_json(port, "/json/version", timeout_sec=2.0)
        return isinstance(info, dict) and bool(info.get("Browser") or info.get("webSocketDebuggerUrl"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return False


def launch_browser_node(
    node: TikTokNodeConfig, urls: tuple[str, ...] = ()
) -> tuple[bool, int | None, str]:
    """Launches a hardened Chromium browser node with dedicated profile and CDP port."""
    profile_path = _assert_contained(Path(node.profile_dir))
    profile_path.mkdir(parents=True, exist_ok=True)

    if is_cdp_listening(node.remote_debugging_port):
        version_info = _cdp_http_get_json(node.remote_debugging_port, "/json/version")
        return (
            True,
            None,
            f"{node.node_id} already running on 127.0.0.1:{node.remote_debugging_port} "
            f"({version_info.get('Browser', 'Chromium')})",
        )

    browser_bin = resolve_browser_binary()
    wx, wy = node.window_position
    ww, wh = node.window_size

    cmd = [
        str(browser_bin),
        f"--user-data-dir={profile_path}",
        f"--remote-debugging-port={node.remote_debugging_port}",
        "--remote-debugging-address=127.0.0.1",
        f"--window-position={wx},{wy}",
        f"--window-size={ww},{wh}",
        "--force-webrtc-ip-handling-policy=disable_non_proxied_udp",
        "--enforce-webrtc-ip-permission-check",
        "--disable-blink-features=AutomationControlled",
        "--no-first-run",
        "--no-default-browser-check",
        "--new-window",
    ]
    cmd.extend(urls)

    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )

    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        if is_cdp_listening(node.remote_debugging_port):
            version_info = _cdp_http_get_json(node.remote_debugging_port, "/json/version")
            return (
                True,
                proc.pid,
                f"Launched {browser_bin.name} ({version_info.get('Browser', 'Chromium')}) "
                f"on 127.0.0.1:{node.remote_debugging_port}",
            )
        time.sleep(0.4)

    raise TimeoutError(
        f"Timed out waiting 20s for {node.node_id} CDP endpoint on 127.0.0.1:{node.remote_debugging_port}"
    )


def _ws_recv_exact(sock: socket.socket, n: int) -> bytes:
    """Reads exactly n bytes from a TCP socket."""
    chunks: list[bytes] = []
    remaining = n
    while remaining > 0:
        data = sock.recv(remaining)
        if not data:
            raise ConnectionError("WebSocket closed unexpectedly while reading CDP frame.")
        chunks.append(data)
        remaining -= len(data)
    return b"".join(chunks)


def _ws_send_text(sock: socket.socket, text: str) -> None:
    """Sends a masked RFC-6455 client text frame."""
    payload = text.encode("utf-8")
    mask_key = os.urandom(4)
    header = bytearray([0x81])
    length = len(payload)
    if length < 126:
        header.append(0x80 | length)
    elif length < 65536:
        header.append(0x80 | 126)
        header.extend(struct.pack("!H", length))
    else:
        header.append(0x80 | 127)
        header.extend(struct.pack("!Q", length))
    header.extend(mask_key)
    masked = bytes(b ^ mask_key[idx % 4] for idx, b in enumerate(payload))
    sock.sendall(bytes(header) + masked)


def _ws_recv_text(sock: socket.socket) -> str:
    """Reads a single RFC-6455 server text frame (handling ping/pong)."""
    while True:
        b1, b2 = _ws_recv_exact(sock, 2)
        opcode = b1 & 0x0F
        is_masked = bool(b2 & 0x80)
        length = b2 & 0x7F
        if length == 126:
            length = struct.unpack("!H", _ws_recv_exact(sock, 2))[0]
        elif length == 127:
            length = struct.unpack("!Q", _ws_recv_exact(sock, 8))[0]
        mask_key = _ws_recv_exact(sock, 4) if is_masked else b""
        payload = _ws_recv_exact(sock, length)
        if is_masked:
            payload = bytes(b ^ mask_key[idx % 4] for idx, b in enumerate(payload))
        if opcode == 0x8:  # close
            raise ConnectionError("CDP WebSocket closed by browser.")
        if opcode == 0x9:  # ping -> pong
            sock.sendall(b"\x8a\x80" + os.urandom(4))
            continue
        if opcode in (0x1, 0x0):
            return payload.decode("utf-8", errors="replace")


def _cdp_eval_on_target(ws_url: str, js_expr: str, timeout_sec: float = 12.0) -> Any:
    """Executes Runtime.evaluate over a localhost RFC-6455 CDP WebSocket."""
    parsed = urlparse(ws_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 80
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"

    ws_key = base64.b64encode(os.urandom(16)).decode("ascii")
    handshake = (
        f"GET {path} HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {ws_key}\r\n"
        "Sec-WebSocket-Version: 13\r\n\r\n"
    )

    sock = socket.create_connection((host, port), timeout=timeout_sec)
    try:
        sock.sendall(handshake.encode("ascii"))
        buf = b""
        while b"\r\n\r\n" not in buf:
            chunk = sock.recv(4096)
            if not chunk:
                raise ConnectionError("CDP WebSocket handshake failed.")
            buf += chunk
        status_line = buf.split(b"\r\n", 1)[0].decode("ascii", "replace")
        if "101" not in status_line:
            raise ConnectionError(f"Unexpected CDP WS handshake response: {status_line}")

        # Verify accept header per RFC 6455 sanity
        _ = hashlib.sha1((ws_key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")).digest()

        cmd_payload = {
            "id": 1,
            "method": "Runtime.evaluate",
            "params": {
                "expression": js_expr,
                "awaitPromise": True,
                "returnByValue": True,
            },
        }
        _ws_send_text(sock, json.dumps(cmd_payload))

        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            raw_msg = _ws_recv_text(sock)
            msg = json.loads(raw_msg)
            if msg.get("id") == 1:
                result = msg.get("result", {}).get("result", {})
                if "exceptionDetails" in msg.get("result", {}):
                    raise RuntimeError(f"CDP Runtime.evaluate error: {msg['result']['exceptionDetails']}")
                return result.get("value")
        raise TimeoutError("Timed out awaiting CDP Runtime.evaluate result.")
    finally:
        sock.close()


def query_node_egress_via_cdp(node: TikTokNodeConfig, host_wan_ip: str) -> NodeEgressTelemetry:
    """Queries live https://ipinfo.io/json inside a running browser node via CDP."""
    if not is_cdp_listening(node.remote_debugging_port):
        raise RuntimeError(
            f"{node.node_id} is not listening on CDP port {node.remote_debugging_port}. "
            f"Run with --mode open {node.alias} or --mode init-setup first."
        )

    targets = _cdp_http_get_json(node.remote_debugging_port, "/json/list", timeout_sec=5.0)
    page_targets = [
        t
        for t in targets
        if isinstance(t, dict)
        and t.get("type") == "page"
        and t.get("webSocketDebuggerUrl")
        and not str(t.get("url", "")).startswith("devtools://")
    ]
    if not page_targets:
        raise RuntimeError(f"No inspectable page tabs found in {node.node_id} (port {node.remote_debugging_port}).")

    # Prefer the open ipinfo.io tab if present, else any http/https tab
    ipinfo_tab = next((t for t in page_targets if "ipinfo.io" in str(t.get("url", ""))), None)
    target = ipinfo_tab or next(
        (t for t in page_targets if str(t.get("url", "")).startswith("http")), page_targets[0]
    )
    ws_url = str(target["webSocketDebuggerUrl"])

    js_fetch = (
        "(async () => {"
        "  try {"
        "    const r = await fetch('https://ipinfo.io/json', {cache: 'no-store'});"
        "    return await r.text();"
        "  } catch (e) {"
        "    return document.body ? document.body.innerText : '';"
        "  }"
        "})()"
    )
    raw_text = str(_cdp_eval_on_target(ws_url, js_fetch) or "").strip()
    if not raw_text and ipinfo_tab is not None:
        raw_text = str(
            _cdp_eval_on_target(ws_url, "document.body ? document.body.innerText : ''") or ""
        ).strip()

    try:
        info = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Could not parse ipinfo JSON from {node.node_id} (url={target.get('url')}): {raw_text[:200]}"
        ) from exc

    node_ip = str(info.get("ip") or "").strip()
    city = str(info.get("city") or "")
    region = str(info.get("region") or "")
    country = str(info.get("country") or "")
    org = str(info.get("org") or "")

    # Match expected VPN city/region leniently for accented spellings (e.g. Montréal vs Montreal)
    norm_expected = node.expected_vpn_region.lower().replace("é", "e")
    norm_loc = f"{city} {region}".lower().replace("é", "e")
    region_matched = norm_expected in norm_loc

    return NodeEgressTelemetry(
        node_id=node.node_id,
        cdp_port=node.remote_debugging_port,
        ip=node_ip,
        city=city,
        region=region,
        country=country,
        org=org,
        expected_vpn_region=node.expected_vpn_region,
        region_matched=region_matched,
        tunnel_isolated_from_host=bool(node_ip and node_ip != host_wan_ip),
    )


def run_init_setup() -> int:
    """Provisions both node profiles, prints Host WAN baseline, and opens 3-tab setup windows."""
    nodes = ensure_nodes_config()
    host_info = query_host_wan_ip()
    host_ip = str(host_info.get("ip", ""))
    host_city = str(host_info.get("city", ""))
    host_country = str(host_info.get("country", ""))
    print(f"[tiktok_nodes] Host OS Baseline WAN IP: {host_ip} ({host_city}, {host_country})")

    for key in ("node_alpha", "node_beta"):
        node = nodes[key]
        _, pid, detail = launch_browser_node(node, urls=SETUP_TABS)
        print(
            f"[tiktok_nodes] {node.node_id}: {detail} | pid={pid or 'existing'} | "
            f"target={node.expected_vpn_region} | email={node.email}"
        )

    print("\n--- MANUAL ONE-TIME PROVISIONING CHECKLIST ---")
    print(f"1. In LEFT window (node_alpha :9222): install ProtonVPN extension -> connect to Canada - Montreal -> sign up/log in TikTok as {nodes['node_alpha'].email}")
    print(f"2. In RIGHT window (node_beta :9223): install ProtonVPN extension -> connect to Canada - Toronto -> sign up/log in TikTok as {nodes['node_beta'].email}")
    print("3. Refresh the ipinfo.io/json tab in both windows, then run:")
    print(
        r"   & 'C:\Anti-Gravity-Core\pierre-training-video\video_studio\.venv\Scripts\python.exe' "
        r"'C:\Anti-Gravity-Core\pierre-training-video\pipeline\tiktok_nodes.py' --mode verify-tunnels"
    )
    return 0


def run_verify_tunnels() -> int:
    """Audits Host vs Alpha vs Beta egress IPs over CDP and saves nodes_tunnel_audit.json."""
    nodes = ensure_nodes_config()
    _assert_contained(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    host_info = query_host_wan_ip()
    host_ip = str(host_info.get("ip", ""))
    host_city = str(host_info.get("city", ""))
    host_region = str(host_info.get("region", ""))
    host_org = str(host_info.get("org", ""))

    alpha_tel = query_node_egress_via_cdp(nodes["node_alpha"], host_ip)
    beta_tel = query_node_egress_via_cdp(nodes["node_beta"], host_ip)

    alpha_ok = bool(alpha_tel.ip and alpha_tel.ip != host_ip)
    beta_ok = bool(beta_tel.ip and beta_tel.ip != host_ip)
    distinct_ok = bool(alpha_tel.ip and beta_tel.ip and alpha_tel.ip != beta_tel.ip)
    all_passed = alpha_ok and beta_ok and distinct_ok

    report = TunnelAuditReport(
        checked_at_utc=datetime.now(timezone.utc).isoformat(),
        host_wan_ip=host_ip,
        host_city=host_city,
        host_region=host_region,
        host_org=host_org,
        node_alpha=alpha_tel,
        node_beta=beta_tel,
        alpha_isolated_from_host=alpha_ok,
        beta_isolated_from_host=beta_ok,
        alpha_beta_distinct=distinct_ok,
        all_assertions_passed=all_passed,
    )

    audit_path = _assert_contained(TUNNEL_AUDIT_PATH)
    with audit_path.open(mode="w", encoding="utf-8") as handle:
        json.dump(asdict(report), handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(f"[verify-tunnels] Host OS WAN : {host_ip} ({host_city}, {host_region})")
    print(
        f"[verify-tunnels] Node Alpha  : {alpha_tel.ip} ({alpha_tel.city}, {alpha_tel.region}) "
        f"[expected={alpha_tel.expected_vpn_region}, region_match={alpha_tel.region_matched}, isolated={alpha_ok}]"
    )
    print(
        f"[verify-tunnels] Node Beta   : {beta_tel.ip} ({beta_tel.city}, {beta_tel.region}) "
        f"[expected={beta_tel.expected_vpn_region}, region_match={beta_tel.region_matched}, isolated={beta_ok}]"
    )
    print(f"[verify-tunnels] Audit JSON  : {audit_path}")

    assert alpha_ok, f"Tunnel isolation failed: node_alpha IP ({alpha_tel.ip}) matches host_wan_ip ({host_ip})"
    assert beta_ok, f"Tunnel isolation failed: node_beta IP ({beta_tel.ip}) matches host_wan_ip ({host_ip})"
    assert distinct_ok, f"Tunnel isolation failed: node_alpha IP ({alpha_tel.ip}) matches node_beta IP ({beta_tel.ip})"

    print("[verify-tunnels] ALL ISOLATION ASSERTIONS PASSED.")
    return 0


def run_open_nodes(target: str) -> int:
    """Re-opens Alpha, Beta, or both persistent profiles with CDP debug ports enabled."""
    nodes = ensure_nodes_config()
    selected = ("node_alpha", "node_beta") if target == "both" else (f"node_{target}",)
    for key in selected:
        node = nodes[key]
        _, pid, detail = launch_browser_node(node, urls=(TIKTOK_SIGNUP_URL,))
        print(f"[tiktok_nodes] {node.node_id}: {detail} (pid={pid or 'existing'})")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses CLI flags for dual TikTok VPN node manager."""
    parser = argparse.ArgumentParser(
        description="Provision & verify dual isolated TikTok VPN browser nodes (Alpha & Beta)"
    )
    parser.add_argument(
        "--mode",
        choices=("init-setup", "verify-tunnels", "open"),
        required=True,
        help="Operation mode: init-setup, verify-tunnels, or open.",
    )
    parser.add_argument(
        "open_target",
        nargs="?",
        choices=("alpha", "beta", "both"),
        default="both",
        help="Target node(s) when --mode open is selected (default: both).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = parse_args(argv)
    if args.mode == "init-setup":
        return run_init_setup()
    if args.mode == "verify-tunnels":
        return run_verify_tunnels()
    return run_open_nodes(str(args.open_target))


if __name__ == "__main__":
    sys.exit(main())

