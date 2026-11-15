"""Phase 5B: Compliance-Locked TikTok Studio Caption Builder & Publisher.

Validates the Phase 4 rendered 9:16 MP4, formats the deterministic financial
disclosure caption from `pipeline/latest_brief.json`, exports `tiktok_caption.txt`
and `tiktok_post_payload.json`, and supports `package-only`, `stage-draft`, or
`publish` modes using Playwright or a dedicated Edge/Chrome persistent profile
at `.tiktok_session/`.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any

PROJECT_ROOT = Path(r"C:\Anti-Gravity-Core\pierre-training-video")
DEFAULT_BRIEF_PATH = PROJECT_ROOT / "pipeline" / "latest_brief.json"
DEFAULT_RENDER_MANIFEST_PATH = PROJECT_ROOT / "pipeline" / "output" / "render_manifest.json"
DEFAULT_VIDEO_PATH = PROJECT_ROOT / "pipeline" / "output" / "tiktok_ready.mp4"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "pipeline" / "output"
DEFAULT_SESSION_DIR = PROJECT_ROOT / ".tiktok_session"
TIKTOK_STUDIO_UPLOAD_URL = "https://www.tiktok.com/tiktokstudio/upload"
MAX_TIKTOK_DURATION_SEC = 60.5
MIN_VIDEO_SIZE_BYTES = 1_000_000

WINGET_FFMPEG_BIN_DIR = (
    Path.home()
    / "AppData"
    / "Local"
    / "Microsoft"
    / "WinGet"
    / "Packages"
    / "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
    / "ffmpeg-9.0-full_build"
    / "bin"
)


@dataclass(slots=True, frozen=True)
class TikTokPostPayload:
    generated_at_utc: str
    upload_mode: str
    status: str
    run_date_iso: str
    run_date_badge: str
    t1_ticker: str
    t2_ticker: str
    t3_ticker: str
    t1_hashtag: str
    video_path: str
    video_size_bytes: int
    video_duration_sec: float
    caption_path: str
    caption_text: str
    tiktok_studio_url: str
    session_dir: str
    notes: str


def _resolve_ffprobe() -> str:
    """Locates ffprobe executable."""
    found = shutil.which("ffprobe")
    if found:
        return found
    candidate = WINGET_FFMPEG_BIN_DIR / "ffprobe.exe"
    if candidate.exists():
        return str(candidate)
    raise FileNotFoundError(f"ffprobe not found on PATH or at {candidate}")


def _fmt_price(value: Any) -> str:
    """Formats a numeric value cleanly as USD with comma separators and 2 decimals."""
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return "$0.00"


def _load_json(path: Path) -> dict[str, Any]:
    """Loads JSON object from disk with UTF-8 encoding."""
    if not path.exists():
        raise FileNotFoundError(f"Required input JSON missing: {path}")
    with path.open(mode="r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def validate_video_contract(video_path: Path) -> tuple[int, float]:
    """Validates TikTok MP4 existence, size (>1MB), and probed duration (<=60.5s)."""
    if not video_path.exists():
        raise FileNotFoundError(f"Rendered TikTok MP4 missing: {video_path}")
    size_bytes = video_path.stat().st_size
    if size_bytes <= MIN_VIDEO_SIZE_BYTES:
        raise ValueError(
            f"Video {video_path} is too small ({size_bytes} bytes <= {MIN_VIDEO_SIZE_BYTES} bytes)"
        )

    ffprobe_bin = _resolve_ffprobe()
    proc = subprocess.run(
        [
            ffprobe_bin,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {video_path}: {proc.stderr.strip()}")
    duration_sec = round(float(proc.stdout.strip()), 3)
    if duration_sec > MAX_TIKTOK_DURATION_SEC:
        raise ValueError(
            f"Video duration {duration_sec:.3f}s exceeds TikTok limit {MAX_TIKTOK_DURATION_SEC:.1f}s"
        )
    return size_bytes, duration_sec


def build_tiktok_caption(brief_data: dict[str, Any]) -> tuple[str, str]:
    """Deterministically formats the compliance-locked TikTok caption and hashtag."""
    run_date_badge = str(brief_data.get("run_date_badge") or "").strip() or "LIVE"
    t1 = brief_data.get("t1") if isinstance(brief_data.get("t1"), dict) else {}
    t2 = brief_data.get("t2") if isinstance(brief_data.get("t2"), dict) else {}
    t3 = brief_data.get("t3") if isinstance(brief_data.get("t3"), dict) else {}

    t1_symbol = str(t1.get("ticker") or "T1").strip().upper()
    t1_company = str(t1.get("company_name") or t1_symbol).strip()
    t2_symbol = str(t2.get("ticker") or "T2").strip().upper()
    t3_symbol = str(t3.get("ticker") or "T3").strip().upper()

    t1_hashtag = re.sub(r"[^A-Za-z0-9]", "", t1_symbol) or "WATCHTOWER"

    spot_str = _fmt_price(t1.get("spot_price", 0.0))
    atr_str = _fmt_price(t1.get("atr_14d", 0.0))
    atr_pct = float(t1.get("atr_14d_pct", 0.0))
    stop_str = _fmt_price(t1.get("dynamic_stop_2x_atr", 0.0))
    sma_str = _fmt_price(t1.get("sma_20d", 0.0))
    target_str = _fmt_price(t1.get("model_target_5d", 0.0))
    bias_str = str(t1.get("net_quant_bias") or "Neutral").strip()

    caption = "\n".join(
        [
            (
                f"[{run_date_badge}] Morning Quant Watchtower: "
                f"${t1_symbol} ({t1_company}) | Top Confluence: ${t1_symbol}, ${t2_symbol}, ${t3_symbol}"
            ),
            (
                f"Spot: {spot_str} | 14D ATR: {atr_str} ({atr_pct:.2f}%) | "
                f"2x ATR Stop: {stop_str} | 20D SMA: {sma_str} | "
                f"5D Chronos Target: {target_str} ({bias_str})"
            ),
            "Autonomous 16-Agent Swarm Telemetry — Educational & Research Only. Not Financial Advice.",
            f"#QuantTrading #MarketTelemetry #AlgoTrading #StockMarket #{t1_hashtag} #NotFinancialAdvice",
        ]
    )
    return caption, t1_hashtag


def _find_chromium_binary() -> str | None:
    """Locates system Microsoft Edge or Google Chrome executable on Windows."""
    candidates = [
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    for name in ("msedge.exe", "chrome.exe"):
        found = shutil.which(name)
        if found:
            return found
    return None


def _copy_to_clipboard(text: str) -> bool:
    """Copies UTF-8 caption text onto the Windows clipboard for quick paste."""
    if os.name != "nt":
        return False
    proc = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "Set-Clipboard -Value ([Console]::In.ReadToEnd())",
        ],
        input=text,
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=10,
        check=False,
    )
    return proc.returncode == 0


def execute_browser_upload(
    mode: str, video_path: Path, caption_text: str, session_dir: Path
) -> tuple[str, str]:
    """Stages or publishes the packaged video to TikTok Studio using Playwright or system Edge/Chrome."""
    session_dir.mkdir(parents=True, exist_ok=True)
    try:
        from playwright.sync_api import sync_playwright  # type: ignore[import-not-found]
    except ImportError:
        copied = _copy_to_clipboard(caption_text)
        browser_bin = _find_chromium_binary()
        if browser_bin is None:
            raise RuntimeError(
                "Neither playwright nor Edge/Chrome was found to stage TikTok Studio."
            )
        subprocess.Popen(
            [
                browser_bin,
                f"--user-data-dir={session_dir}",
                TIKTOK_STUDIO_UPLOAD_URL,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
        )
        note = (
            f"Playwright unavailable in active .venv; opened {Path(browser_bin).name} "
            f"with isolated profile {session_dir} at {TIKTOK_STUDIO_UPLOAD_URL}. "
            f"Clipboard caption ready={copied}; attach {video_path} manually."
        )
        return "BROWSER_OPENED_STAGED_MANUAL", note

    with sync_playwright() as p:
        launch_kwargs: dict[str, Any] = {
            "user_data_dir": str(session_dir),
            "headless": False,
        }
        browser_bin = _find_chromium_binary()
        if browser_bin and "msedge" in browser_bin.lower():
            launch_kwargs["channel"] = "msedge"
        elif browser_bin and "chrome" in browser_bin.lower():
            launch_kwargs["channel"] = "chrome"

        context = p.chromium.launch_persistent_context(**launch_kwargs)
        try:
            page = context.pages[0] if context.pages else context.new_page()
            page.goto(TIKTOK_STUDIO_UPLOAD_URL, wait_until="domcontentloaded", timeout=60000)

            file_input = page.locator('input[type="file"]').first
            file_input.wait_for(state="attached", timeout=45000)
            file_input.set_input_files(str(video_path))

            caption_box = page.locator('div[contenteditable="true"]').first
            caption_box.wait_for(state="visible", timeout=60000)
            caption_box.click()
            page.keyboard.press("Control+A")
            page.keyboard.press("Backspace")
            page.keyboard.insert_text(caption_text)

            if mode == "stage-draft":
                return (
                    "DRAFT_STAGED",
                    f"Video attached and caption filled at {TIKTOK_STUDIO_UPLOAD_URL}; stopped in stage-draft.",
                )

            post_btn = page.locator('button:has-text("Post")').first
            post_btn.wait_for(state="visible", timeout=120000)
            for _ in range(90):
                if post_btn.is_enabled():
                    break
                page.wait_for_timeout(1000)
            if not post_btn.is_enabled():
                raise RuntimeError("TikTok Studio Post button remained disabled after 90s.")
            post_btn.click()
            page.wait_for_timeout(5000)
            return ("PUBLISHED", f"Submitted video and caption via {TIKTOK_STUDIO_UPLOAD_URL}.")
        finally:
            context.close()


def package_or_upload(
    brief_path: Path,
    render_manifest_path: Path,
    video_path: Path,
    output_dir: Path,
    session_dir: Path,
    mode: str,
) -> TikTokPostPayload:
    """Validates video, generates caption/payload files, and handles upload mode."""
    allowed_root = PROJECT_ROOT.resolve()
    resolved_output = output_dir.resolve()
    resolved_session = session_dir.resolve()
    if not resolved_output.is_relative_to(allowed_root):
        raise PermissionError(f"Write containment violation: {resolved_output} outside {allowed_root}")
    if not resolved_session.is_relative_to(allowed_root):
        raise PermissionError(f"Write containment violation: {resolved_session} outside {allowed_root}")

    resolved_output.mkdir(parents=True, exist_ok=True)

    brief_data = _load_json(brief_path)
    render_manifest = _load_json(render_manifest_path)
    resolved_video = Path(str(render_manifest.get("video_output_path") or video_path))
    if video_path != DEFAULT_VIDEO_PATH:
        resolved_video = video_path

    video_size_bytes, video_duration_sec = validate_video_contract(resolved_video)
    caption_text, t1_hashtag = build_tiktok_caption(brief_data)

    caption_path = resolved_output / "tiktok_caption.txt"
    payload_path = resolved_output / "tiktok_post_payload.json"
    caption_path.write_text(caption_text + "\n", encoding="utf-8")

    t1_ticker = str(brief_data.get("t1", {}).get("ticker", "T1")).upper()
    t2_ticker = str(brief_data.get("t2", {}).get("ticker", "T2")).upper()
    t3_ticker = str(brief_data.get("t3", {}).get("ticker", "T3")).upper()
    run_date_iso = str(brief_data.get("run_date_iso") or "")
    run_date_badge = str(brief_data.get("run_date_badge") or "")

    if mode == "package-only":
        status = "PACKAGED_READY"
        notes = (
            "Validated MP4 and wrote compliance-locked caption + payload JSON. "
            "No browser session launched (--mode package-only)."
        )
    else:
        status, notes = execute_browser_upload(mode, resolved_video, caption_text, resolved_session)

    payload = TikTokPostPayload(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        upload_mode=mode,
        status=status,
        run_date_iso=run_date_iso,
        run_date_badge=run_date_badge,
        t1_ticker=t1_ticker,
        t2_ticker=t2_ticker,
        t3_ticker=t3_ticker,
        t1_hashtag=t1_hashtag,
        video_path=str(resolved_video),
        video_size_bytes=video_size_bytes,
        video_duration_sec=video_duration_sec,
        caption_path=str(caption_path),
        caption_text=caption_text,
        tiktok_studio_url=TIKTOK_STUDIO_UPLOAD_URL,
        session_dir=str(resolved_session),
        notes=notes,
    )

    with payload_path.open(mode="w", encoding="utf-8") as handle:
        json.dump(asdict(payload), handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses CLI flags for Phase 5B TikTok packager & uploader."""
    parser = argparse.ArgumentParser(
        description="Compliance-Locked TikTok Studio Caption Builder & Publisher (Phase 5B)"
    )
    parser.add_argument("--brief", type=Path, default=DEFAULT_BRIEF_PATH)
    parser.add_argument("--render-manifest", type=Path, default=DEFAULT_RENDER_MANIFEST_PATH)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--session-dir", type=Path, default=DEFAULT_SESSION_DIR)
    parser.add_argument(
        "--mode",
        choices=("package-only", "stage-draft", "publish"),
        default="package-only",
        help="Upload mode (default: package-only).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = parse_args(argv)
    payload = package_or_upload(
        brief_path=args.brief.resolve(),
        render_manifest_path=args.render_manifest.resolve(),
        video_path=args.video.resolve(),
        output_dir=args.output_dir.resolve(),
        session_dir=args.session_dir.resolve(),
        mode=str(args.mode),
    )
    print(f"[tiktok_upload] Status: {payload.status} (mode={payload.upload_mode})")
    print(f"[tiktok_upload] Caption: {payload.caption_path}")
    print(
        f"[tiktok_upload] Video: {payload.video_path} "
        f"({payload.video_duration_sec:.3f}s, {payload.video_size_bytes:,} bytes)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

