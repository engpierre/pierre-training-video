"""
Pierre Voice Studio — Local Qwen3-TTS Server (`cuda:1`, bfloat16).

Exposes two loopback-only interfaces:
  1. Gradio Voice Studio UI (`http://127.0.0.1:7860`): record/upload 10s
     reference audio (`pierre_ref.wav` + `pierre_ref.txt`) and test EN/FR cloning.
  2. OpenAI-compatible REST TTS shim (`POST http://127.0.0.1:7861/v1/audio/speech`):
     consumed directly by `MoneyPrinterTurbo` so all generated videos speak
     with Pierre's cloned voice.
"""
from __future__ import annotations

import argparse
import io
import logging
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional, Tuple

import gradio as gr
import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [VoiceStudio]: %(message)s",
)
logger = logging.getLogger("VoiceStudio")

BASE_DIR = Path(__file__).resolve().parent
WINGET_FFMPEG_FALLBACK = (
    Path.home()
    / "AppData"
    / "Local"
    / "Microsoft"
    / "WinGet"
    / "Packages"
    / "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
    / "ffmpeg-9.0-full_build"
    / "bin"
    / "ffmpeg.exe"
)


@dataclass(slots=True, frozen=True)
class VoiceStudioConfig:
    """Immutable runtime configuration for local Qwen3-TTS studio."""

    model_id: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    preferred_device: str = "cuda:1"
    gradio_host: str = "127.0.0.1"
    gradio_port: int = 7860
    api_host: str = "127.0.0.1"
    api_port: int = 7861
    ref_dir: Path = BASE_DIR / "reference_voices"
    outputs_dir: Path = BASE_DIR / "outputs"
    ref_wav_name: str = "pierre_ref.wav"
    ref_txt_name: str = "pierre_ref.txt"

    @property
    def ref_wav_path(self) -> Path:
        return self.ref_dir / self.ref_wav_name

    @property
    def ref_txt_path(self) -> Path:
        return self.ref_dir / self.ref_txt_name


class OpenAISpeechRequest(BaseModel):
    """OpenAI `/v1/audio/speech` compatible payload contract."""

    model: str = Field(default="qwen3-tts", description="Model identifier")
    input: str = Field(..., min_length=1, max_length=20000, description="Text to synthesize")
    voice: str = Field(default="pierre-auto", description="Voice identifier or language hint")
    response_format: Literal["mp3", "wav"] = Field(default="wav")
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    language: Optional[str] = Field(default=None)


CFG = VoiceStudioConfig()
_MODEL_LOCK = threading.Lock()
_SYNTH_LOCK = threading.Lock()
_MODEL_INSTANCE: Any = None
_RESOLVED_DEVICE: Optional[str] = None


def resolve_ffmpeg_binary() -> Optional[str]:
    """Locate ffmpeg executable on PATH or default WinGet install."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    if WINGET_FFMPEG_FALLBACK.exists():
        return str(WINGET_FFMPEG_FALLBACK)
    return None


def resolve_compute_device(preferred: str = "cuda:1") -> str:
    """Pin Qwen3-TTS to physical GPU 1 (`cuda:1`) when dual GPUs are visible."""
    if not torch.cuda.is_available():
        return "cpu"
    count = torch.cuda.device_count()
    if preferred == "cuda:1":
        if count > 1:
            return "cuda:1"
        # If CUDA_VISIBLE_DEVICES=1 already masks GPU 0 externally, cuda:0 IS GPU 1.
        if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() == "1":
            return "cuda:0"
    if preferred.startswith("cuda:"):
        try:
            idx = int(preferred.split(":", 1)[1])
            if 0 <= idx < count:
                return preferred
        except ValueError:
            pass
    return "cuda:0"


def get_tts_model() -> Tuple[Any, str]:
    """Lazy-load `Qwen3-TTS-12Hz-1.7B-Base` onto target GPU on first synthesis."""
    global _MODEL_INSTANCE, _RESOLVED_DEVICE
    with _MODEL_LOCK:
        if _MODEL_INSTANCE is not None and _RESOLVED_DEVICE is not None:
            return _MODEL_INSTANCE, _RESOLVED_DEVICE

        from qwen_tts import Qwen3TTSModel

        target_device = resolve_compute_device(CFG.preferred_device)
        dtype = torch.bfloat16 if target_device.startswith("cuda") else torch.float32
        logger.info(
            "Loading %s on device=%s dtype=%s ...",
            CFG.model_id,
            target_device,
            dtype,
        )
        model = Qwen3TTSModel.from_pretrained(
            CFG.model_id,
            device_map=target_device,
            dtype=dtype,
        )
        _MODEL_INSTANCE = model
        _RESOLVED_DEVICE = target_device
        logger.info("Loaded Qwen3-TTS successfully on %s.", target_device)
        return _MODEL_INSTANCE, _RESOLVED_DEVICE


def infer_language(voice: str, explicit_language: Optional[str]) -> str:
    """Map voice preset suffixes or UI selection to Qwen3-TTS language names."""
    if explicit_language and explicit_language.strip():
        return explicit_language.strip()
    v = (voice or "").lower()
    if "-fr" in v or "french" in v:
        return "French"
    if "-en" in v or "english" in v:
        return "English"
    return "Auto"


def load_reference_voice(
    ref_wav_override: Optional[Path] = None,
    ref_txt_override: Optional[str] = None,
) -> Tuple[Path, str]:
    """Load saved reference audio clip and exact transcript."""
    wav_path = ref_wav_override or CFG.ref_wav_path
    if not wav_path.exists():
        raise FileNotFoundError(
            f"Reference audio missing at {wav_path}. "
            "Upload or record a 10s sample via http://127.0.0.1:7860 first."
        )

    if ref_txt_override is not None and ref_txt_override.strip():
        ref_text = ref_txt_override.strip()
    elif CFG.ref_txt_path.exists():
        ref_text = CFG.ref_txt_path.read_text(encoding="utf-8").strip()
    else:
        raise FileNotFoundError(
            f"Reference transcript missing at {CFG.ref_txt_path}. "
            "Save pierre_ref.txt alongside pierre_ref.wav via http://127.0.0.1:7860."
        )

    if not ref_text:
        raise ValueError("Reference transcript pierre_ref.txt is empty.")
    return wav_path, ref_text


def _encode_wav_bytes(waveform: np.ndarray, sample_rate: int) -> bytes:
    """Serialize float/int numpy audio array into clean 16-bit PCM WAV bytes."""
    arr = np.asarray(waveform, dtype=np.float32)
    arr = np.clip(arr, -1.0, 1.0)
    buf = io.BytesIO()
    sf.write(buf, arr, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _transcode_wav_with_ffmpeg(
    wav_bytes: bytes,
    output_format: Literal["wav", "mp3"],
    speed: float = 1.0,
) -> bytes:
    """Apply optional tempo scaling and transcode WAV bytes to MP3 via ffmpeg."""
    needs_speed = abs(speed - 1.0) > 0.02
    if output_format == "wav" and not needs_speed:
        return wav_bytes

    ffmpeg_bin = resolve_ffmpeg_binary()
    if not ffmpeg_bin:
        raise RuntimeError(
            "ffmpeg binary not found; cannot transcode audio to MP3 or adjust speed."
        )

    cmd = [ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error", "-f", "wav", "-i", "pipe:0"]
    if needs_speed:
        clamped_speed = max(0.5, min(2.0, float(speed)))
        cmd.extend(["-filter:a", f"atempo={clamped_speed:.3f}"])

    if output_format == "mp3":
        cmd.extend(["-codec:a", "libmp3lame", "-qscale:a", "2", "-f", "mp3", "pipe:1"])
    else:
        cmd.extend(["-codec:a", "pcm_s16le", "-f", "wav", "pipe:1"])

    proc = subprocess.run(
        cmd,
        input=wav_bytes,
        capture_output=True,
        timeout=90,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout:
        err_msg = (proc.stderr or b"").decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg transcoding failed (rc={proc.returncode}): {err_msg}")
    return proc.stdout


def synthesize_speech(
    text: str,
    language: str = "Auto",
    response_format: Literal["wav", "mp3"] = "wav",
    speed: float = 1.0,
    ref_wav_override: Optional[Path] = None,
    ref_txt_override: Optional[str] = None,
) -> Tuple[int, np.ndarray, bytes, Path]:
    """Generate zero-shot cloned speech via Qwen3-TTS and archive WAV output."""
    clean_text = (text or "").strip()
    if not clean_text:
        raise ValueError("Input text cannot be empty.")

    ref_wav_path, ref_text = load_reference_voice(ref_wav_override, ref_txt_override)
    CFG.outputs_dir.mkdir(parents=True, exist_ok=True)

    model, device = get_tts_model()
    with _SYNTH_LOCK:
        wavs, sample_rate = model.generate_voice_clone(
            text=clean_text,
            language=language,
            ref_audio=str(ref_wav_path),
            ref_text=ref_text,
        )

    waveform = np.asarray(wavs[0], dtype=np.float32)
    wav_bytes = _encode_wav_bytes(waveform, int(sample_rate))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    out_wav_path = CFG.outputs_dir / f"pierre_clone_{stamp}.wav"
    out_wav_path.write_bytes(wav_bytes)
    logger.info(
        "Synthesized %.2fs audio on %s -> %s",
        len(waveform) / max(1, int(sample_rate)),
        device,
        out_wav_path.name,
    )

    final_bytes = _transcode_wav_with_ffmpeg(wav_bytes, response_format, speed=speed)
    return int(sample_rate), waveform, final_bytes, out_wav_path


def save_reference_sample(audio_filepath: Optional[str], transcript: str) -> str:
    """Save uploaded/recorded reference WAV and transcript cleanly to reference_voices/."""
    if not audio_filepath or not Path(audio_filepath).exists():
        raise gr.Error("Record or upload a clean 10-second audio clip first.")
    clean_txt = (transcript or "").strip()
    if not clean_txt:
        raise gr.Error("Enter the exact spoken transcript matching your reference clip.")

    CFG.ref_dir.mkdir(parents=True, exist_ok=True)
    data, sr = sf.read(audio_filepath, dtype="float32")
    sf.write(str(CFG.ref_wav_path), data, sr, format="WAV", subtype="PCM_16")
    CFG.ref_txt_path.write_text(clean_txt, encoding="utf-8")
    return (
        f"Saved reference audio ({len(data) / max(1, sr):.1f}s @ {sr} Hz) to "
        f"{CFG.ref_wav_path} and transcript to {CFG.ref_txt_path}."
    )


def ui_generate_clone(text: str, language: str) -> Tuple[Tuple[int, np.ndarray], str]:
    """Gradio callback for Tab 2 zero-shot narration."""
    try:
        sr, waveform, _, out_path = synthesize_speech(
            text=text,
            language=language,
            response_format="wav",
        )
        return (sr, waveform), f"Generated & saved: {out_path}"
    except FileNotFoundError as exc:
        raise gr.Error(str(exc)) from exc
    except Exception as exc:
        logger.exception("Gradio synthesis failure")
        raise gr.Error(f"Synthesis failed: {exc}") from exc


# =====================================================================
# OpenAI-Compatible FastAPI Shim (`http://127.0.0.1:7861`)
# =====================================================================
api_app = FastAPI(title="Pierre Qwen3-TTS OpenAI Shim", version="1.0.0")


@api_app.get("/health")
def api_health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model": CFG.model_id,
        "target_device": resolve_compute_device(CFG.preferred_device),
        "model_loaded": _MODEL_INSTANCE is not None,
        "reference_ready": CFG.ref_wav_path.exists() and CFG.ref_txt_path.exists(),
        "ref_wav_path": str(CFG.ref_wav_path),
        "ffmpeg_ready": resolve_ffmpeg_binary() is not None,
    }


@api_app.get("/v1/audio/voices")
def api_list_voices() -> dict[str, list[str]]:
    return {
        "voices": [
            "pierre-clone-Auto-Male",
            "pierre-clone-en-Male",
            "pierre-clone-fr-Male",
        ]
    }


@api_app.post("/v1/audio/speech")
def api_create_speech(req: OpenAISpeechRequest) -> Response:
    try:
        language = infer_language(req.voice, req.language)
        _, _, audio_bytes, _ = synthesize_speech(
            text=req.input,
            language=language,
            response_format=req.response_format,
            speed=req.speed,
        )
        media_type = "audio/mpeg" if req.response_format == "mp3" else "audio/wav"
        return Response(content=audio_bytes, media_type=media_type)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=412, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("OpenAI speech endpoint failure")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def build_gradio_app() -> gr.Blocks:
    """Construct the local Qwen3-TTS Gradio Voice Studio UI (`127.0.0.1:7860`)."""
    with gr.Blocks(title="Pierre Voice Studio — Qwen3-TTS 1.7B") as demo:
        gr.Markdown(
            "# Pierre Voice Studio (`Qwen3-TTS-12Hz-1.7B-Base`)\n"
            f"* **Target GPU:** `{resolve_compute_device(CFG.preferred_device)}` (`bfloat16`) | "
            f"**OpenAI TTS API:** `http://{CFG.api_host}:{CFG.api_port}/v1/audio/speech`\n"
            f"* **Reference Slot:** `{CFG.ref_wav_path}` & `{CFG.ref_txt_path}`"
        )

        with gr.Tab("1. Record / Save Reference Voice"):
            ref_audio_in = gr.Audio(
                label="Record or Upload Clean 10s Reference Clip (WAV/MP3)",
                sources=["microphone", "upload"],
                type="filepath",
            )
            existing_txt = (
                CFG.ref_txt_path.read_text(encoding="utf-8")
                if CFG.ref_txt_path.exists()
                else ""
            )
            ref_text_in = gr.Textbox(
                label="Exact Spoken Transcript (pierre_ref.txt)",
                value=existing_txt,
                lines=3,
                placeholder="Type verbatim what is spoken in the reference clip...",
            )
            save_btn = gr.Button("Save Reference Voice", variant="primary")
            save_status = gr.Textbox(label="Reference Status", interactive=False)
            save_btn.click(
                fn=save_reference_sample,
                inputs=[ref_audio_in, ref_text_in],
                outputs=[save_status],
            )

        with gr.Tab("2. Clone & Narrate (EN / FR / Auto)"):
            synth_text = gr.Textbox(
                label="Narration Text",
                lines=5,
                placeholder="Enter English or French training narration...",
            )
            synth_lang = gr.Dropdown(
                label="Target Language",
                choices=["Auto", "English", "French"],
                value="Auto",
            )
            synth_btn = gr.Button("Generate Cloned Voice", variant="primary")
            synth_audio_out = gr.Audio(label="Generated Narration", type="numpy")
            synth_status = gr.Textbox(label="Saved Output Path", interactive=False)
            synth_btn.click(
                fn=ui_generate_clone,
                inputs=[synth_text, synth_lang],
                outputs=[synth_audio_out, synth_status],
            )

    return demo


def _run_api_server() -> None:
    uvicorn.run(
        api_app,
        host=CFG.api_host,
        port=CFG.api_port,
        log_level="info",
        access_log=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Pierre Qwen3-TTS Studio & OpenAI API Shim")
    parser.add_argument("--api-only", action="store_true", help="Serve FastAPI :7861 without Gradio UI")
    parser.add_argument("--preload", action="store_true", help="Load Qwen3-TTS weights on startup")
    args = parser.parse_args()

    CFG.ref_dir.mkdir(parents=True, exist_ok=True)
    CFG.outputs_dir.mkdir(parents=True, exist_ok=True)

    if args.preload:
        get_tts_model()

    if args.api_only:
        logger.info("Starting OpenAI-compatible TTS shim on http://%s:%s", CFG.api_host, CFG.api_port)
        _run_api_server()
        return

    api_thread = threading.Thread(target=_run_api_server, name="qwen-tts-api", daemon=True)
    api_thread.start()
    logger.info("Started OpenAI-compatible TTS shim on http://%s:%s/v1", CFG.api_host, CFG.api_port)

    # Tiny yield so uvicorn binds cleanly before Gradio prints startup banner
    time.sleep(0.3)
    logger.info("Starting Gradio Voice Studio on http://%s:%s", CFG.gradio_host, CFG.gradio_port)
    demo = build_gradio_app()
    demo.launch(
        server_name=CFG.gradio_host,
        server_port=CFG.gradio_port,
        share=False,
    )


if __name__ == "__main__":
    main()

