# Pierre Training & Short-Form Video Studio

Isolated local AI voice-cloning and short-form (`9:16` TikTok / `16:9` course) video production workspace. Runs strictly on-demand on **physical GPU 1 (`cuda:1`)** with loopback-only bindings (`127.0.0.1`), leaving **GPU 0 (`cuda:0`)**, trading systems, and daemon swarms untouched and consuming **0 MiB VRAM** when closed.

## Architecture & Ports

| Component | Stack | Compute / Device | Bind Address | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **Voice Studio UI** | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` (Gradio) | `cuda:1` (`bfloat16`, lazy load) | `http://127.0.0.1:7860` | Record/manage reference voice & synthesize clips |
| **OpenAI TTS Shim** | FastAPI (`POST /v1/audio/speech`) | `cuda:1` (`bfloat16`, FFmpeg MP3/WAV) | `http://127.0.0.1:7861/v1/audio/speech` | Serves cloned narration to MoneyPrinterTurbo |
| **Video Studio UI** | MoneyPrinterTurbo (Streamlit + MoviePy) | GPU 1 (`CUDA_VISIBLE_DEVICES=1`) for `faster-whisper` (`float16`) | `http://127.0.0.1:8501` | Script formatting, B-roll assembly, subtitles (`9:16` & `16:9`) |
| **Script LLM** | Local Ollama (`qwen3.8-hardened:27b` default) | Local Ollama server | `http://127.0.0.1:11434/v1` | Zero-cost local script generation |

## Directory Isolation

```text
pierre-training-video/
├── .gitignore                   # Blocks .venv/, config.toml, media (.wav/.mp3/.mp4), and weights
├── start_studio.ps1             # Manual on-demand launcher (no auto-start / no Task Scheduler)
├── voice_studio/
│   ├── .venv/                   # Isolated Python 3.11 + PyTorch cu128 + qwen-tts venv
│   ├── requirements.txt
│   ├── tts_server.py            # Gradio UI (:7860) + FastAPI OpenAI TTS endpoint (:7861)
│   ├── reference_voices/        # Drop pierre_ref.wav + pierre_ref.txt here (gitignored)
│   └── outputs/                 # Generated voiceover exports (gitignored)
└── video_studio/
    ├── .venv/                   # Isolated Python 3.11 + MoneyPrinterTurbo + faster-whisper venv
    ├── config.example.toml      # Tracked template configured for Ollama + Qwen3-TTS + CUDA Whisper
    ├── config.toml              # Local untracked runtime config (ffmpeg_path + endpoints)
    └── app/services/voice.py    # Wired to local OpenAI-compatible Qwen3-TTS shim
```

## Reference Voice Setup

1. Place a clean **3–15 second mono/stereo WAV sample** at:
   - `voice_studio/reference_voices/pierre_ref.wav`
2. Place the **exact verbatim transcript** alongside it at:
   - `voice_studio/reference_voices/pierre_ref.txt`
3. Alternatively, upload/record the reference sample directly inside the Gradio Voice Studio UI (`http://127.0.0.1:7860`) to save both files automatically.

> Note: Until `pierre_ref.wav` is present, `/v1/audio/speech` returns HTTP `412 Precondition Failed` with instructions instead of crashing.

## Manual Launch

Run from `C:\Anti-Gravity-Core\pierre-training-video\`:

```powershell
.\start_studio.ps1
```

This opens two console windows:
1. **Voice Studio & TTS API** (`http://127.0.0.1:7860` and `http://127.0.0.1:7861/v1/audio/speech`)
2. **MoneyPrinterTurbo Video Studio** (`http://127.0.0.1:8501`)

To shut down and return VRAM usage to **0 MiB**, simply close both console windows.

## TikTok (`9:16`) & Course (`16:9`) Workflow

1. Open **`http://127.0.0.1:8501`** in your browser.
2. Confirm **LLM Provider** is `ollama` (`qwen3.8-hardened:27b` or another local `ollama list` model).
3. Under **Speech / TTS**, choose **`chatterbox`** and voice **`chatterbox:pierre-clone-Auto-Male`** (or `chatterbox:pierre-clone-en-Male` / `chatterbox:pierre-clone-fr-Male`).
4. Under **Subtitles**, use **`whisper`** (`large-v3`, `cuda`, `float16`).
5. Pick aspect ratio **`9:16`** (vertical TikTok/Shorts) or **`16:9`** (course modules) and generate.

