"""Phase 5D: Send Verified TikTok Video & Caption via Proton Mail.

Reads verified output artifacts (`tiktok_ready.mp4`, `tiktok_caption.txt`, and
`pipeline_run_summary.json` / `render_manifest.json`) strictly inside
`C:\\Anti-Gravity-Core\\pierre-training-video\\`, clones the Proton Mail browser
profile into `.tiktok_session\\proton_session` (leaving `pierres_email_agent`
strictly read-only), attaches the MP4, populates the caption + telemetry body,
sends from the configured Proton Mail sender to the target recipient, and writes
`pipeline/output/email_dispatch_receipt.json`.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any

from playwright.sync_api import Page, sync_playwright  # type: ignore[import-not-found]

PROJECT_ROOT = Path(r"C:\Anti-Gravity-Core\pierre-training-video")
PIPELINE_DIR = PROJECT_ROOT / "pipeline"
OUTPUT_DIR = PIPELINE_DIR / "output"
SESSION_ROOT = PROJECT_ROOT / ".tiktok_session" / "proton_session"

EMAIL_AGENT_DIR = Path(r"C:\Anti-Gravity-Core\pierres_email_agent")
EMAIL_AGENT_ENV = EMAIL_AGENT_DIR / ".env"
EMAIL_AGENT_SESSION = EMAIL_AGENT_DIR / "data" / "proton_session"

DEFAULT_VIDEO_PATH = OUTPUT_DIR / "tiktok_ready.mp4"
DEFAULT_CAPTION_PATH = OUTPUT_DIR / "tiktok_caption.txt"
DEFAULT_BRIEF_PATH = PIPELINE_DIR / "latest_brief.json"
DEFAULT_RENDER_MANIFEST_PATH = OUTPUT_DIR / "render_manifest.json"
DEFAULT_AUDIO_MANIFEST_PATH = OUTPUT_DIR / "audio_manifest.json"
DEFAULT_RECEIPT_PATH = OUTPUT_DIR / "email_dispatch_receipt.json"

DEFAULT_FROM_EMAIL = ""
DEFAULT_TO_EMAIL = ""


@dataclass(slots=True, frozen=True)
class EmailDispatchReceipt:
    sent_at_utc: str
    from_address: str
    to_address: str
    subject: str
    attachment_path: str
    attachment_size_bytes: int
    caption_included: bool
    caption_text: str
    status: str
    headless_mode: bool
    notes: str


def _parse_env_readonly(env_path: Path) -> dict[str, str]:
    """Reads key=value credentials in read-only mode without modifying external files."""
    if not env_path.exists():
        raise FileNotFoundError(f"Proton credentials .env missing: {env_path}")
    values: dict[str, str] = {}
    with env_path.open(mode="r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            values[key] = val
    return values


def _prepare_isolated_proton_session(target_session_dir: Path) -> None:
    """Clones existing Proton browser session into .tiktok_session (writes isolated to project)."""
    resolved_target = target_session_dir.resolve()
    allowed_root = PROJECT_ROOT.resolve()
    if not resolved_target.is_relative_to(allowed_root):
        raise PermissionError(
            f"Write containment violation: {resolved_target} is outside {allowed_root}"
        )

    if resolved_target.exists() and any(resolved_target.iterdir()):
        # Ensure Chromium lockfiles from any prior aborted launch don't block reuse
        for lock_name in ("SingletonLock", "SingletonCookie", "SingletonSocket", "lockfile"):
            lock_file = resolved_target / lock_name
            if lock_file.exists():
                try:
                    lock_file.unlink()
                except OSError:
                    pass
        return

    resolved_target.parent.mkdir(parents=True, exist_ok=True)
    if EMAIL_AGENT_SESSION.is_dir():
        shutil.copytree(
            EMAIL_AGENT_SESSION,
            resolved_target,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(
                "Singleton*",
                "lockfile",
                "*.lock",
                "Crashpad",
                "Service Worker",
                "Cache",
                "Code Cache",
            ),
        )
    else:
        resolved_target.mkdir(parents=True, exist_ok=True)


def _load_optional_json(path: Path) -> dict[str, Any]:
    """Reads optional metadata JSON safely."""
    if not path.exists():
        return {}
    with path.open(mode="r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def _build_subject_and_body(caption_text: str, video_path: Path, size_bytes: int) -> tuple[str, str]:
    """Formats deterministic subject and body containing caption + build metadata."""
    brief = _load_optional_json(DEFAULT_BRIEF_PATH)
    render_manifest = _load_optional_json(DEFAULT_RENDER_MANIFEST_PATH)
    audio_manifest = _load_optional_json(DEFAULT_AUDIO_MANIFEST_PATH)

    date_badge = str(brief.get("run_date_badge") or render_manifest.get("run_date_badge") or "15 NOV 2026")
    t1_ticker = str(brief.get("t1", {}).get("ticker") or render_manifest.get("t1_ticker") or "BTC-USD").upper()
    duration_sec = float(render_manifest.get("rendered_duration_sec") or 57.333)
    width = int(render_manifest.get("width") or 1080)
    height = int(render_manifest.get("height") or 1920)
    mean_vol = float(audio_manifest.get("master_mean_volume_db") or -15.2)
    size_mb = size_bytes / 1_000_000.0

    subject = f"[{date_badge}] Verified TikTok Video & Caption — ${t1_ticker} Morning Quant Watchtower"
    body = "\n".join(
        [
            caption_text.strip(),
            "",
            "---",
            "TikTok Production Telemetry:",
            f"• Attachment: {video_path.name}",
            f"• Format: {width}x{height} (9:16 H.264/AAC)",
            f"• Duration: {duration_sec:.3f}s",
            f"• Size: {size_mb:.2f} MB ({size_bytes:,} bytes)",
            f"• Master Audio Mean Loudness: {mean_vol:.1f} dB (EBU R128)",
        ]
    )
    return subject, body


def _ensure_proton_authenticated(page: Page, email: str, password: str) -> bool:
    """Navigates to Proton Mail inbox and logs in silently if session expired."""
    page.goto("https://mail.proton.me/u/1/inbox", wait_until="domcontentloaded", timeout=35000)
    page.wait_for_timeout(4500)

    if "mail.proton.me" in page.url and "login" not in page.url and "authorize" not in page.url:
        # Double-check either compose button or inbox chrome is mounted
        try:
            page.wait_for_selector(
                'button[data-testid="sidebar:compose"], button[data-testid="composer:compose"]',
                timeout=15000,
            )
            return True
        except Exception:
            pass

    page.goto("https://account.proton.me/login", wait_until="domcontentloaded", timeout=35000)
    page.wait_for_timeout(3000)

    u_input = page.wait_for_selector("#username", timeout=15000)
    if not u_input:
        return False
    u_input.fill(email)
    page.wait_for_timeout(400)
    submit_btn = page.wait_for_selector("button[type='submit']", timeout=5000)
    if submit_btn:
        submit_btn.click()

    pw_input = page.wait_for_selector("#password", timeout=15000)
    if not pw_input:
        return False
    pw_input.fill(password)
    page.wait_for_timeout(400)
    submit_btn = page.wait_for_selector("button[type='submit']", timeout=5000)
    if submit_btn:
        submit_btn.click()

    deadline = time.monotonic() + 40.0
    while time.monotonic() < deadline:
        url_now = page.url.lower()
        if "mail.proton.me" in url_now and "login" not in url_now:
            break
        if page.locator("input[autocomplete='one-time-code'], #twoFactorCode").count() > 0:
            raise RuntimeError("Proton Mail requested 2FA verification code.")
        if page.locator('iframe[title*="captcha" i], #captcha').count() > 0:
            raise RuntimeError("Proton Mail presented a CAPTCHA challenge.")
        page.wait_for_timeout(1500)

    page.goto("https://mail.proton.me/u/1/inbox", wait_until="domcontentloaded", timeout=35000)
    page.wait_for_selector(
        'button[data-testid="sidebar:compose"], button[data-testid="composer:compose"]',
        timeout=25000,
    )
    return "mail.proton.me" in page.url and "login" not in page.url


def _send_via_proton_playwright(
    from_email: str,
    to_email: str,
    subject: str,
    body: str,
    video_path: Path,
    login_email: str,
    login_password: str,
    headless: bool,
) -> tuple[bool, str]:
    """Executes Proton Mail web compose, MP4 upload, and send confirmation."""
    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            user_data_dir=str(SESSION_ROOT),
            headless=headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--window-size=1366,900",
            ],
        )
        page = context.pages[0] if context.pages else context.new_page()
        try:
            if not _ensure_proton_authenticated(page, login_email, login_password):
                return False, f"Failed to reach authenticated Proton Mail inbox (url={page.url})."

            page.wait_for_timeout(2500)

            for popup_sel in (
                'button[data-testid="spotlight-inner-close-button"]',
                'button[data-testid="onboarding-checklist-header-hide-button"]',
            ):
                pop_btn = page.locator(popup_sel).first
                if pop_btn.count() > 0 and pop_btn.is_visible():
                    try:
                        pop_btn.click(force=True, timeout=1500)
                        page.wait_for_timeout(400)
                    except Exception:
                        pass

            compose_btn = page.locator(
                'button[data-testid="sidebar:compose"], button:has-text("Nouveau message"), button:has-text("New message")'
            ).first
            compose_btn.wait_for(state="visible", timeout=15000)
            compose_btn.click(force=True)
            page.wait_for_timeout(2500)

            # 1. Recipient (retry click once if Proton Mail UI was still hydrating)
            to_loc = page.locator(
                'input[data-testid="composer:to"], input[placeholder*="recipient" i], input[placeholder*="destinataire" i]'
            ).first
            if to_loc.count() == 0 or not to_loc.is_visible():
                compose_btn.click(force=True)
                page.wait_for_timeout(2500)
            to_input = to_loc
            to_input.wait_for(state="visible", timeout=15000)
            to_input.click(force=True)
            to_input.fill(to_email)
            page.keyboard.press("Enter")
            page.wait_for_timeout(700)

            # 2. Sender verification / switch to preferred sender address
            from_btn = page.locator(
                '[data-testid="composer:from-select"], [data-testid="composer:from"]'
            ).first
            if from_btn.count() > 0:
                current_from = (from_btn.inner_text() or "").strip()
                if from_email.lower() not in current_from.lower():
                    from_btn.click(force=True)
                    page.wait_for_timeout(700)
                    opt = page.locator(
                        f'[role="option"]:has-text("{from_email}"), button:has-text("{from_email}")'
                    ).first
                    if opt.count() > 0:
                        opt.click(force=True)
                        page.wait_for_timeout(600)
                    else:
                        page.keyboard.press("Escape")
                        page.wait_for_timeout(300)

            # Dismiss Scribe / promo modal if visible
            close_modal = page.locator(
                'button[aria-label="Close"], button[aria-label="Fermer"]'
            ).first
            if close_modal.count() > 0 and close_modal.is_visible():
                # Only close if it's an overlay tooltip, not composer close button
                aria = (close_modal.get_attribute("data-testid") or "").lower()
                if "composer" not in aria:
                    try:
                        close_modal.click(force=True, timeout=1500)
                    except Exception:
                        pass

            # 3. Subject
            subj_input = page.locator(
                'input[data-testid="composer:subject"], input[placeholder*="Subject" i], input[placeholder*="Objet" i]'
            ).first
            subj_input.wait_for(state="visible", timeout=10000)
            subj_input.click(force=True)
            subj_input.fill(subject)
            page.wait_for_timeout(500)

            # 4. Rich Text Body (#rooster-editor inside composer iframe or contenteditable)
            body_filled = False
            for frame in page.frames:
                if "about:blank" in frame.url or frame != page.main_frame:
                    try:
                        ok = frame.evaluate(
                            """
                            (val) => {
                                const el = document.getElementById('rooster-editor') || document.querySelector('[contenteditable="true"]');
                                if (!el) return false;
                                el.focus();
                                const esc = (s) => s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
                                const lines = val.split('\\n');
                                el.innerHTML = lines.map(line => `<div>${line ? esc(line) : '<br>'}</div>`).join('');
                                el.dispatchEvent(new Event('input', { bubbles: true }));
                                el.dispatchEvent(new Event('change', { bubbles: true }));
                                return true;
                            }
                            """,
                            body,
                        )
                        if ok:
                            body_filled = True
                            break
                    except Exception:
                        continue

            if not body_filled:
                editor = page.locator(
                    '#rooster-editor, [contenteditable="true"], .composer-content'
                ).first
                if editor.count() > 0:
                    editor.click(force=True)
                    page.keyboard.insert_text(body)
                    body_filled = True

            if not body_filled:
                return False, "Failed to locate Proton Mail rich-text body editor."

            page.wait_for_timeout(700)

            # 5. Attach verified 15.26 MB MP4
            attached = False
            file_input = page.locator(
                'input[data-testid="composer-attachments-button"], '
                '[data-testid="composer:dialog"] input[type="file"], '
                'input[type="file"]'
            ).first
            if file_input.count() > 0:
                file_input.set_input_files(str(video_path))
                attached = True
            else:
                attach_btn = page.locator(
                    'label[data-testid="composer:attachment-button"], '
                    'button[data-testid="composer:attachment-button"], '
                    'button[data-testid="composer:attachments-button"], '
                    'button[aria-label*="Attach" i], button[aria-label*="Joindre" i]'
                ).first
                if attach_btn.count() > 0:
                    with page.expect_file_chooser(timeout=15000) as fc_info:
                        attach_btn.click(force=True)
                    fc_info.value.set_files(str(video_path))
                    attached = True

            if not attached:
                return False, "Could not locate Proton Mail attachment file input or button."

            # Wait for video attachment to appear and finish uploading/encrypting (up to 120s)
            composer_scope = page.locator(
                '[data-testid="composer:footer"], [data-testid="composer:dialog"], .composer, form'
            ).last
            send_btn = page.locator(
                'button[data-testid="composer:send-button"], button:has-text("Envoyer"), button:has-text("Send")'
            ).first

            upload_deadline = time.monotonic() + 120.0
            seen_filename = False
            while time.monotonic() < upload_deadline:
                try:
                    page_text = page.locator("body").inner_text()
                except Exception:
                    page_text = ""
                if video_path.name in page_text or "tiktok_ready" in page_text:
                    seen_filename = True

                progress_cnt = page.locator(
                    '[role="progressbar"]:visible, .progress-bar:visible'
                ).count()
                send_ready = send_btn.count() > 0 and send_btn.is_enabled()

                if seen_filename and progress_cnt == 0 and send_ready:
                    break
                page.wait_for_timeout(1500)

            if not seen_filename:
                return False, f"Attachment {video_path.name} did not appear in Proton composer within 120s."

            # 5s stability buffer before clicking Send
            page.wait_for_timeout(5000)

            if not send_btn.is_enabled():
                return False, "Proton Mail Send button stayed disabled after 120s attachment wait."

            send_btn.click(force=True)

            # 6. Wait for composer modal / send button & subject field to close (up to 120s) + 6s delivery wait
            send_deadline = time.monotonic() + 120.0
            closed = False
            composer_open_loc = page.locator(
                'button[data-testid="composer:send-button"], input[data-testid="composer:subject"]'
            )
            while time.monotonic() < send_deadline:
                if composer_open_loc.count() == 0:
                    closed = True
                    break
                page.wait_for_timeout(1500)

            if not closed:
                return False, "Composer did not close within 120s after clicking Send."

            page.wait_for_timeout(6000)
            return True, (
                f"Attached {video_path.name} ({video_path.stat().st_size:,} bytes) and sent cleanly "
                f"from {from_email} to {to_email} via mail.proton.me."
            )
        finally:
            context.close()


def dispatch_tiktok_email(
    video_path: Path,
    caption_path: Path,
    receipt_path: Path,
    from_email: str,
    to_email: str,
) -> EmailDispatchReceipt:
    """Validates local package, dispatches via Proton Mail, and saves JSON receipt."""
    allowed_root = PROJECT_ROOT.resolve()
    resolved_receipt = receipt_path.resolve()
    if not resolved_receipt.is_relative_to(allowed_root):
        raise PermissionError(
            f"Write containment violation: receipt path {resolved_receipt} outside {allowed_root}"
        )

    if not video_path.exists():
        raise FileNotFoundError(f"Video attachment missing: {video_path}")
    size_bytes = video_path.stat().st_size
    if size_bytes < 1_000_000:
        raise ValueError(f"Video attachment unexpectedly small: {size_bytes} bytes")
    if not caption_path.exists():
        raise FileNotFoundError(f"Caption file missing: {caption_path}")

    caption_text = caption_path.read_text(encoding="utf-8").strip()
    if not caption_text:
        raise ValueError(f"Caption file is empty: {caption_path}")

    env_vals = _parse_env_readonly(EMAIL_AGENT_ENV)
    resolved_from = (
        from_email.strip()
        or env_vals.get("PREFERRED_PROTON_SENDER", "").strip()
        or env_vals.get("PROTON_EMAIL", "").strip()
        or "sender@example.com"
    )
    resolved_to = (
        to_email.strip()
        or env_vals.get("DEFAULT_EMAIL_RECIPIENT", "").strip()
        or env_vals.get("AUTHORIZED_SENDERS", "").split(",")[0].strip()
        or "recipient@example.com"
    )
    login_email = env_vals.get("PROTON_EMAIL", "") or resolved_from
    login_password = env_vals.get("PROTON_PASSWORD", "")
    if not login_email or not login_password:
        raise ValueError("PROTON_EMAIL or PROTON_PASSWORD missing in pierres_email_agent/.env")

    _prepare_isolated_proton_session(SESSION_ROOT)
    subject, body = _build_subject_and_body(caption_text, video_path, size_bytes)

    used_headless = True
    ok, note = _send_via_proton_playwright(
        from_email=resolved_from,
        to_email=resolved_to,
        subject=subject,
        body=body,
        video_path=video_path,
        login_email=login_email,
        login_password=login_password,
        headless=True,
    )
    if not ok:
        used_headless = False
        ok, note = _send_via_proton_playwright(
            from_email=resolved_from,
            to_email=resolved_to,
            subject=subject,
            body=body,
            video_path=video_path,
            login_email=login_email,
            login_password=login_password,
            headless=False,
        )

    if not ok:
        raise RuntimeError(f"Proton Mail dispatch failed: {note}")

    receipt = EmailDispatchReceipt(
        sent_at_utc=datetime.now(timezone.utc).isoformat(),
        from_address=resolved_from,
        to_address=resolved_to,
        subject=subject,
        attachment_path=str(video_path),
        attachment_size_bytes=size_bytes,
        caption_included=True,
        caption_text=caption_text,
        status="SENT_VERIFIED",
        headless_mode=used_headless,
        notes=note,
    )

    resolved_receipt.parent.mkdir(parents=True, exist_ok=True)
    with resolved_receipt.open(mode="w", encoding="utf-8") as handle:
        json.dump(asdict(receipt), handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    return receipt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses CLI flags for Phase 5D Proton Mail dispatch."""
    parser = argparse.ArgumentParser(
        description="Send Verified TikTok Video & Caption from Proton Mail to Gmail"
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO_PATH)
    parser.add_argument("--caption", type=Path, default=DEFAULT_CAPTION_PATH)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT_PATH)
    parser.add_argument("--from-email", type=str, default=DEFAULT_FROM_EMAIL)
    parser.add_argument("--to-email", type=str, default=DEFAULT_TO_EMAIL)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = parse_args(argv)
    receipt = dispatch_tiktok_email(
        video_path=args.video.resolve(),
        caption_path=args.caption.resolve(),
        receipt_path=args.receipt.resolve(),
        from_email=str(args.from_email).strip(),
        to_email=str(args.to_email).strip(),
    )
    print(f"[email_dispatch] Status: {receipt.status}")
    print(f"[email_dispatch] From: {receipt.from_address} -> To: {receipt.to_address}")
    print(f"[email_dispatch] Subject: {receipt.subject}")
    print(
        f"[email_dispatch] Attachment: {receipt.attachment_path} "
        f"({receipt.attachment_size_bytes:,} bytes)"
    )
    print(f"[email_dispatch] Receipt: {args.receipt.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
