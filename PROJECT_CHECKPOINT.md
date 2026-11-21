# Project Checkpoint: `pierre-training-video`

**Status:** V2 Production Ready & Checkpointed (Animated HUD Spotlight Motion, Progressive `#0b111a` Trajectory Chart, Rotating Post-Opener Segments, `%`-Free FOMO Hook, 3-Day Cooldown, Automated Proton Mail Delivery, and Speech-Gated Lower-Third Pierre Avatar Overlay Verified).

---

## 1. Daily Morning Production SOP (2-Step Standard Operating Procedure)

Every morning before market open, execute the automated 2-step workflow:

### Step 1: Run War Room Pre-Market Scan
```powershell
Set-Location 'C:\Anti-Gravity-Core\pierre_stock_warroom'
python scripts\run_premarket_scan.py
```
*Refreshes dual-node spot prices, 20D SMAs, 14D ATR stops, and 16-bar TimesFM / Chronos-Bolt trajectories into `premarket_recon_output.json` and `chronos_intel_buffer.json`.*

### Step 2: Run Master TikTok Pipeline (Auto-Extract Top 3 Swarm Stocks & Deliver)
```powershell
Set-Location 'C:\Anti-Gravity-Core\pierre-training-video'
.\video_studio\.venv\Scripts\python.exe pipeline\run_daily_tiktok.py --upload-mode email
```
*Executes the full automated V2 pipeline:*
1. **Fresh Swarm Ingestion, 3-Day Cooldown & `-20%` Opacity Penalty (`extract_warroom_brief.py`):** Dynamically ranks the 40-asset universe from `premarket_recon_output.json` to select Slots #1 & #2 (Rapid Radar momentum setups) and Slot #3 (The Main Event anomaly/tension setup). Excludes tickers featured in the last 2 episodes unless overridden by an active `14D ATR` invalidation breach, and enforces a mandatory `-20%` confidence reduction when SEC Form 4 or Finnhub MSPR fields are `NULL`.
2. **Percentage-Free FOMO Opener & Rotating Segments (`generate_script.py`):** Generates dialogue well under the `57.5s` ceiling (`~141` words), opens with a percentage-free 3-state FOMO hook (`TARGET_HIT`, `ON_TRACK`, `STOPPED_OUT`), rotates middle segments by `episode_num % 3` (`catalyst_flow`, `trajectory_chart`, `trio_matrix`), validates against the compliance lexicon, and creates a 4-line SEO caption.
3. **Dual-Voice TTS Synthesis (`audio_builder.py`):** Neural audio generation on GPU 1 (`atempo=1.0400x` natural unhurried pacing) with EBU R128 loudness normalization (`~-15.5 dB`) and guaranteed server teardown.
4. **Broadcast V2 HUD Rendering (`render_video.py`):** Renders vertical `1080x1920` video with:
   - **Frame-0 3-Ticker Cover Card (`0.0s–3.5s`):** Centered `$T1 • $T2 • $T3` card + percentage-free status pill docking smoothly into the top radar banner at `3.5s`.
   - **Active Speaker Turn Motion:** Cross-fades the top radar banner with RMS-reactive border pulsing and highlights the active slot card (`LIVE AUDIT`) across `Pierre` and `Julie` turns via smooth scanline wipes.
   - **Rotating Opaque `#0b111a` Middle Segments (`18.0s–32.0s`):** Renders progressive 30fps draw-in of the `20D PRICE` cyan line, Agent 02 `14D ATR` dashed red floor, and `16-BAR CONE` (`trajectory_chart`), or switches cleanly to `trio_matrix` / `catalyst_flow` by `episode_num % 3` with zero HUD bleed-through.
   - Permanent bottom-left compliance watermark inside safe zones (`top >= 140px`, `bottom >= 320px`, `right >= 120px`).
5. **Episodic Ledger Lock (`episode_tracker.py`):** Automatically logs the 3-stock call into `pipeline/episode_ledger.json`.
6. **Proton Mail Dispatch (`email_tiktok_package.py`):** Transmits MP4 video and caption directly to `engpierre5@gmail.com`.

---

## 2. Command Line Overrides & Flags

- `--ticker <SYMBOL>`: Force a specific asset into Slot #3 as the Main Event (Slots #1 & #2 auto-filled with highest-scoring distinct tickers).
- `--dry-run`: Previews extraction, dialogue, and metadata without modifying `episode_ledger.json` or running full render.
- `--upload-mode <MODE>`: `package-only` (local package), `email` (Proton Mail dispatch to `engpierre5@gmail.com`), `stage-draft`, or `publish`.
- `--max-duration 57.5`: Target duration ceiling enforced by transparent FFmpeg `atempo`.
- `--bgm-volume 0.06`: Background music volume mix under dialogue.

---

## 3. Episodic Scorecard State Machine (`pipeline/episode_tracker.py`)

- **Ledger Storage:** `pipeline/episode_ledger.json` persists every recorded episode, entry spot, 14D ATR stop, 16-bar target, and directional bias.
- **Deterministic Grading:** `grade_previous_episode()` checks current market prices (via `pierre_quant.db ?mode=ro` or Yahoo chart API v8 fallback), computes exact delta percentage, and outputs spoken receipts and HUD badges.
- **Production State:** Cleanly holds **Episodes #01–#03** (`episode_ledger.json`) so the next live morning run registers as **Episode #04**.

---

## 4. Regulatory & Algorithmic Compliance Shield (CSA/AMF, SEC, TikTok Moderation)

1. **Layer 1 (Visual Watermark):** Permanent HUD badge rendered in safe zone (`y: 1550px`): `QUANT TELEMETRY LOG ONLY • NOT FINANCIAL ADVICE`.
2. **Layer 2 (Lexicon Filter):** Prohibits imperative advisory commands (`"buy now"`, `"sell now"`, `"you should"`, `"guaranteed"`).
3. **Layer 3 (Spoken Sign-Off):** Closes every episode with `"strictly our quant telemetry, never financial advice"`.
4. **Layer 4 (Caption Disclosure):** Appends `⚠️ Educational quant software telemetry & personal market log only. Not financial or investment advice.` to `tiktok_caption.txt`.

---

## 5. Non-Negotiable Isolation & Hardware Hygiene Guarantees

- **Write Containment:** All video, audio, brief, and ledger writes remain strictly inside `C:\Anti-Gravity-Core\pierre-training-video\`.
- **Untouched Subsystems:** `pierre_stock_warroom` (`:3888`), `.openclaw` daemons, `pierres_email_agent`, and `pierre_quant.db` remain strictly read-only.
- **Hardware Isolation:** `GPU 0` (`cuda:0`) remains at `0 MiB` studio compute; `GPU 1` (`cuda:1`) returns to `0 MiB` immediately upon TTS/Whisper completion.

---

## 6. Speech-Gated Lower-Third Avatar Overlay (MuseTalk v1.5)

- **Isolated Subprocess Engine (`pipeline/avatar_sync.py`):** Spawns MuseTalk v1.5 in `.venv_musetalk` pinned to `cuda:1` strictly **after** `tts_server.py` teardown, consuming pre-cached 9s latents and alpha matte from `training-video-studio`.
- **Lower-Frame Safe-Zone Placement:** Composited at `(x=580, y=1300)` scaled ~2x (`360x360` px, `<= 640px` height cap) with `180px` headroom below the subtitle deck, preserving watchtower metrics and compliance watermark.
- **Speech-Window Gating:** Hidden for the opening `1.0s` (`t < 1.0s`), active across Pierre's Part 1 dialogue (`1.0s <= t < t_julie_start`) with `0.20s` alpha fades, and hidden (`alpha = 0`) throughout Julie's Part 2 turn.
- **CLI Override:** Pass `--no-avatar` to `run_daily_tiktok.py` or `render_video.py` to skip lip-sync generation and render without overlay.
