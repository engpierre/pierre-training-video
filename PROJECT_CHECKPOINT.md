# Project Checkpoint: `pierre-training-video`

**Status:** Saved & Cleanly Checkpointed (`email` production delivery verified; dual-node VPN browser system verified & parked).

---

## 1. Working Production Pipeline (`--upload-mode email`)

Single-command end-to-end daily briefing generator and Proton Mail delivery workflow:

```powershell
.\video_studio\.venv\Scripts\python.exe pipeline\run_daily_tiktok.py --upload-mode email
```

Optional overrides (all strictly contained inside `C:\Anti-Gravity-Core\pierre-training-video\`):
- `--ticker <SYMBOL>` (supports any of the 40 tracked assets in `portfolio_40.json` / `pierre_quant.db` / Vault Recons, e.g., `--ticker BTC-USD`)
- `--date YYYY-MM-DD` (override `America/Toronto` runtime date badge)
- `--max-duration 57.5` (transparent FFmpeg `atempo` target ceiling)
- `--rebuild-bg` (force regeneration of cached 9:16 War Room background assets)

### Stage Architecture (Phases 1–5D)
1. **Phase 1 (`pipeline/build_warroom_bg.py`):** Builds `1080x1920` (9:16) dark War Room composite background & looping backdrop (`assets/war_room_9x16_composite.png`, `video_studio/storage/local_videos/war_room_9x16.mp4`).
2. **Phase 2 (`pipeline/quant_reader.py`):** Reads `pierre_quant.db` (`?mode=ro`), `chronos_intel_buffer.json`, `portfolio_40.json`, and Obsidian Vault Recons to select `T1`, `T2`, `T3` and generate `pipeline/latest_brief.json` with a dynamic `America/Toronto` date badge.
3. **Phase 3 (`pipeline/audio_builder.py`):** Generates dual-host cloned voices (Julie & Pierre) via `tts_server.py` pinned to `GPU 1` (`cuda:1`), applies transparent `< 60s` `atempo` guard, normalizes EBU R128 loudness (`master_dialogue.mp3`), and guarantees `tts_server.py` termination in `finally:` so `GPU 1` returns to `0 MiB`.
4. **Phase 4 (`pipeline/render_video.py`):** Renders Option A/D hybrid animated HUD telemetry cards synced to Julie's handoff cue, composites hosts in the lower 9:16 frame, and burns `faster-whisper` forced-alignment captions (`GPU 1`) into `pipeline/output/tiktok_ready.mp4`.
5. **Phase 5 (`pipeline/run_daily_tiktok.py`, `pipeline/tiktok_upload.py`, `pipeline/email_tiktok_package.py`):**
   - Packages `tiktok_caption.txt` and `tiktok_post_payload.json` (`package-only`).
   - In `--upload-mode email`, launches `pierres_email_agent\.venv\Scripts\python.exe` against `pipeline/email_tiktok_package.py` to dispatch `tiktok_ready.mp4` + caption over Proton Mail Web (`mail.proton.me`) from `<PROTON_SENDER>` to `<GMAIL_RECIPIENT>` using `.tiktok_session/proton_session` (reads `pierres_email_agent/.env` strictly read-only) and writes `pipeline/output/email_dispatch_receipt.json` (`SENT_VERIFIED`).

---

## 2. Parked Multi-Node VPN Setup (`pipeline/tiktok_nodes.py`)

Multi-account isolated Chromium launcher provisioned and tunnel-verified (`nodes_tunnel_audit.json` recorded `all_assertions_passed: true`), currently shut down so no background browser processes or CDP ports (`9222`, `9223`) remain open:

- **`node_alpha` (`127.0.0.1:9222`):** Isolated profile `.tiktok_nodes/node_alpha` paired with ProtonVPN extension (Canada - Montreal exit).
- **`node_beta` (`127.0.0.1:9223`):** Isolated profile `.tiktok_nodes/node_beta` paired with ProtonVPN extension (Canada - Toronto exit).
- **Isolation Controls:** `--force-webrtc-ip-handling-policy=disable_non_proxied_udp`, isolated profile directories, distinct window placement/metadata, host Windows WAN untouched.
- **Configuration & Secrets:** Runtime node emails live in git-ignored `pipeline/nodes_config.json` (or `TIKTOK_NODE_ALPHA_EMAIL` / `TIKTOK_NODE_BETA_EMAIL` environment variables); tracked code uses `example.com` defaults.
- **Login UX:** Default TikTok target set to `https://www.tiktok.com/login/qrcode` for 1-scan mobile session transfer when resuming setup:
  - Setup / Launch: `.\video_studio\.venv\Scripts\python.exe pipeline\tiktok_nodes.py --mode init-setup`
  - 3-Way IP Audit: `.\video_studio\.venv\Scripts\python.exe pipeline\tiktok_nodes.py --mode verify-tunnels`

---

## 3. Non-Negotiable Isolation & Hygiene Guarantees
- **Write Containment:** All pipeline writes, sessions, and temporary outputs remain inside `C:\Anti-Gravity-Core\pierre-training-video\`.
- **Untouched External Subsystems:** `pierre_stock_warroom` (`:3888`), `.openclaw` daemons, `pierres_email_agent`, `pierre_quant.db`, and `chronos_intel_buffer.json` remain strictly read-only / unmodified.
- **Hardware Isolation:** `GPU 0` (`cuda:0`) remains at `0 MiB` studio allocation; `GPU 1` (`cuda:1`) returns to `0 MiB` after audio/Whisper stages exit.
