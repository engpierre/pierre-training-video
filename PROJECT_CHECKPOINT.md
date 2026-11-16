# Project Checkpoint: `pierre-training-video`

**Status:** Production Ready & Checkpointed (3-Stock Swarm Radar, Episodic Scorecard State Machine, and Automated Proton Mail Delivery Verified).

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
*Executes the full automated pipeline:*
1. **Fresh Swarm Ingestion (`extract_warroom_brief.py`):** Dynamically ranks the 40-asset universe from `premarket_recon_output.json` to select Slots #1 & #2 (Rapid Radar momentum setups) and Slot #3 (The Main Event anomaly/tension setup).
2. **Ping-Pong Scripting (`generate_script.py`):** Generates 3-beat dialogue under the `57.5s` ceiling, grades prior episode's call, validates against compliance lexicon, and creates 4-line SEO caption.
3. **Dual-Voice TTS Synthesis (`audio_builder.py`):** Neural audio generation on GPU 1 with EBU R128 loudness normalization (-14 to -16 LUFS) and guaranteed server teardown.
4. **Broadcast HUD Rendering (`render_video.py`):** Renders vertical 1080x1920 video with Frame-0 3-Stock Radar Banner, active ticker highlight pill, and permanent bottom-left compliance watermark.
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
- **Production Seed:** Cleanly seeded with **Episode #01** (`BTC-USD` baseline) so the next live morning run registers as **Episode #02**.

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
