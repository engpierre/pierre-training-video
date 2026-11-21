"""Phase 3.5: Speech-Gated Lower-Frame Pierre Avatar Lip-Sync Engine.

Extracts Pierre's Part 1 speech window from the mastered audio track and invokes
the isolated MuseTalk v1.5 engine (.venv_musetalk on GPU 1 / cuda:1) using the
pre-cached 9s avatar latents and alpha matte in `training-video-studio`.

Runs strictly AFTER Phase 3 tts_server.py has stopped, and guarantees try...finally
process termination so GPU 1 returns to 0 MiB before Phase 4 render_video.py begins.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any
import urllib.error
import urllib.request

PROJECT_ROOT = Path(r"C:\Anti-Gravity-Core\pierre-training-video")
STUDIO_ROOT = Path(r"C:\Anti-Gravity-Core\training-video-studio")

MUSETALK_DIR = STUDIO_ROOT / "vendor" / "MuseTalk"
MUSETALK_PYTHON = STUDIO_ROOT / ".venv_musetalk" / "Scripts" / "python.exe"
AVATAR_BASE_MP4 = STUDIO_ROOT / "assets" / "avatar" / "avatar_base_9s.mp4"
AVATAR_MATTE_MP4 = STUDIO_ROOT / "assets" / "avatar" / "avatar_base_9s_matte.mp4"
AVATAR_CACHE_DIR = STUDIO_ROOT / "assets" / "avatar" / "cache"
MUSETALK_AVATAR_DIR = MUSETALK_DIR / "results" / "v15" / "avatars" / "pierre_avatar"

DEFAULT_AUDIO_PATH = PROJECT_ROOT / "pipeline" / "output" / "master_dialogue.wav"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "pipeline" / "output" / "audio_manifest.json"
DEFAULT_OUTPUT_MP4 = PROJECT_ROOT / "pipeline" / "output" / "pierre_avatar_synced.mp4"

MIN_GPU1_FREE_MIB = 6000
OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL_NAME = "qwen3.8-hardened:27b"
AVATAR_FPS = 25
AVATAR_BATCH_SIZE = 16


def _resolve_ffmpeg_bin() -> str:
    """Finds ffmpeg executable on PATH or via WinGet."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    winget_bin = (
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
    if winget_bin.exists():
        return str(winget_bin)
    return "ffmpeg"


def _query_gpu1_free_mib() -> int | None:
    """Queries free VRAM in MiB on physical GPU 1 via nvidia-smi."""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if proc.returncode != 0:
            return None
        for line in proc.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2 and parts[0] == "1":
                return int(parts[1])
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None
    return None


def _unload_ollama_model() -> None:
    """Gracefully unloads idle Ollama weights from VRAM via keep_alive: 0."""
    payload = json.dumps({"model": OLLAMA_MODEL_NAME, "keep_alive": 0}).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_GENERATE_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            resp.read()
    except (urllib.error.URLError, TimeoutError, OSError):
        pass


def ensure_gpu1_vram_headroom(min_free_mib: int = MIN_GPU1_FREE_MIB) -> None:
    """Verifies GPU 1 has >= min_free_mib free before MuseTalk launch."""
    free_mib = _query_gpu1_free_mib()
    if free_mib is None or free_mib >= min_free_mib:
        return

    _unload_ollama_model()
    time.sleep(2.0)
    post_free_mib = _query_gpu1_free_mib()
    if post_free_mib is not None and post_free_mib < min_free_mib:
        raise RuntimeError(
            f"Insufficient GPU 1 VRAM for MuseTalk v1.5: {post_free_mib} MiB free "
            f"(required >= {min_free_mib} MiB)."
        )


def _ensure_musetalk_cache_synced() -> None:
    """Ensures vendor/MuseTalk results directory has cached pierre_avatar artifacts."""
    required = ["coords.pkl", "latents.pt", "mask_coords.pkl", "avator_info.json"]
    if all((MUSETALK_AVATAR_DIR / f).exists() for f in required):
        return
    if not AVATAR_CACHE_DIR.exists():
        raise FileNotFoundError(
            f"Pre-computed avatar cache missing at {AVATAR_CACHE_DIR}."
        )
    MUSETALK_AVATAR_DIR.mkdir(parents=True, exist_ok=True)
    for fname in required:
        src = AVATAR_CACHE_DIR / fname
        if not src.exists():
            raise FileNotFoundError(f"Missing cached avatar artifact: {src}")
        shutil.copy2(src, MUSETALK_AVATAR_DIR / fname)
    for sub in ("full_imgs", "mask"):
        src_sub = AVATAR_CACHE_DIR / sub
        dst_sub = MUSETALK_AVATAR_DIR / sub
        if src_sub.exists() and not dst_sub.exists():
            shutil.copytree(src_sub, dst_sub)


def resolve_julie_start_sec(
    manifest_path: Path = DEFAULT_MANIFEST_PATH, fallback_sec: float = 34.0
) -> float:
    """Resolves Julie's start timestamp from audio_manifest.json."""
    if manifest_path.exists():
        try:
            with manifest_path.open(mode="r", encoding="utf-8") as handle:
                data = json.load(handle)
            if "julie_handoff_sec" in data:
                return float(data["julie_handoff_sec"])
            part1 = float(data.get("part1_sec", fallback_sec))
            handoff = float(data.get("handoff_sec", 0.25))
            return round(part1 + handoff, 3)
        except (OSError, ValueError):
            pass
    return fallback_sec


def extract_pierre_speech_slice(
    master_wav_path: Path,
    output_slice_wav: Path,
    duration_sec: float,
) -> Path:
    """Extracts Pierre's Part 1 speech window (t = 0.0s to julie_start_sec) via FFmpeg."""
    ffmpeg_bin = _resolve_ffmpeg_bin()
    output_slice_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(master_wav_path),
        "-t",
        f"{duration_sec:.3f}",
        "-c:a",
        "pcm_s16le",
        str(output_slice_wav),
    ]
    subprocess.run(cmd, check=True)
    return output_slice_wav


def render_pierre_avatar(
    master_wav_path: Path,
    output_mp4_path: Path,
    julie_start_sec: float | None = None,
    fps: int = AVATAR_FPS,
    batch_size: int = AVATAR_BATCH_SIZE,
) -> tuple[Path, float]:
    """Renders lip-synced Pierre avatar for Part 1 speech window on GPU 1 (cuda:1).

    Returns:
        (rendered_mp4_path, elapsed_seconds)
    """
    if not MUSETALK_PYTHON.exists():
        raise FileNotFoundError(f"MuseTalk Python missing: {MUSETALK_PYTHON}")
    if not MUSETALK_DIR.exists():
        raise FileNotFoundError(f"MuseTalk vendor dir missing: {MUSETALK_DIR}")
    if not master_wav_path.exists():
        raise FileNotFoundError(f"Audio file missing: {master_wav_path}")
    if not AVATAR_BASE_MP4.exists():
        raise FileNotFoundError(f"Base avatar video missing: {AVATAR_BASE_MP4}")

    if julie_start_sec is None:
        manifest_p = master_wav_path.parent / "audio_manifest.json"
        julie_start_sec = resolve_julie_start_sec(manifest_p)

    _ensure_musetalk_cache_synced()
    ensure_gpu1_vram_headroom(MIN_GPU1_FREE_MIB)

    output_mp4_path.parent.mkdir(parents=True, exist_ok=True)
    slice_wav = output_mp4_path.parent / "_tmp_pierre_part1_speech.wav"
    extract_pierre_speech_slice(master_wav_path, slice_wav, julie_start_sec)

    tmp_yaml = output_mp4_path.parent / "_tmp_tiktok_pierre_avatar.yaml"
    clip_key = "pierre_tiktok_part1"
    tmp_yaml.write_text(
        "\n".join(
            [
                "pierre_avatar:",
                "  preparation: False",
                "  bbox_shift: 0",
                f'  video_path: "{AVATAR_BASE_MP4.resolve().as_posix()}"',
                "  audio_clips:",
                f'    {clip_key}: "{slice_wav.resolve().as_posix()}"',
                "",
            ]
        ),
        encoding="utf-8",
    )

    runner_code = (
        "import sys, runpy, torch\n"
        "_orig_load = torch.load\n"
        "def _patched_load(*args, **kwargs):\n"
        "    if 'weights_only' not in kwargs:\n"
        "        kwargs['weights_only'] = False\n"
        "    return _orig_load(*args, **kwargs)\n"
        "torch.load = _patched_load\n"
        "import mmcv.utils.ext_loader as ext_loader\n"
        "class _DummyExt:\n"
        "    def __getattr__(self, name):\n"
        "        raise NotImplementedError(f'mmcv._ext.{name} unavailable in mmcv-lite')\n"
        "ext_loader.load_ext = lambda name, funcs=None: _DummyExt()\n"
        "sys.argv = [\n"
        "    'scripts/realtime_inference.py',\n"
        "    '--version', 'v15',\n"
        "    '--gpu_id', '0',\n"
        "    '--unet_config', './models/musetalkV15/musetalk.json',\n"
        "    '--unet_model_path', './models/musetalkV15/unet.pth',\n"
        "    '--vae_type', 'sd-vae',\n"
        "    '--whisper_dir', './models/whisper',\n"
        f"    '--inference_config', '{tmp_yaml.resolve().as_posix()}',\n"
        f"    '--batch_size', '{int(batch_size)}',\n"
        f"    '--fps', '{int(fps)}',\n"
        "]\n"
        "runpy.run_path('scripts/realtime_inference.py', run_name='__main__')\n"
    )

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "1"

    proc: subprocess.Popen[bytes] | None = None
    t0 = time.perf_counter()
    try:
        print(
            f"[avatar_sync] Spawning MuseTalk v1.5 on GPU 1 for Pierre Part 1 ({julie_start_sec:.2f}s)...",
            flush=True,
        )
        proc = subprocess.Popen(
            [str(MUSETALK_PYTHON), "-c", runner_code],
            cwd=str(MUSETALK_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        out_bytes, _ = proc.communicate(timeout=300)
        if proc.returncode != 0:
            tail = out_bytes.decode("utf-8", errors="replace")[-1500:]
            raise RuntimeError(
                f"MuseTalk avatar render failed (rc={proc.returncode}):\n{tail}"
            )
        elapsed_sec = round(time.perf_counter() - t0, 2)

        raw_vid = MUSETALK_AVATAR_DIR / "vid_output" / f"{clip_key}.mp4"
        if not raw_vid.exists() or raw_vid.stat().st_size == 0:
            raise RuntimeError(f"MuseTalk output video missing or empty: {raw_vid}")
        shutil.move(str(raw_vid), str(output_mp4_path))
        print(
            f"[avatar_sync] Render complete in {elapsed_sec:.2f}s -> {output_mp4_path}",
            flush=True,
        )
        return output_mp4_path, elapsed_sec
    finally:
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    proc.kill()
                except OSError:
                    pass
        for f in (tmp_yaml, slice_wav):
            if f.exists():
                try:
                    f.unlink()
                except OSError:
                    pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Speech-Gated Pierre Avatar Lip-Sync Renderer (Phase 3.5)"
    )
    parser.add_argument(
        "--audio",
        type=Path,
        default=DEFAULT_AUDIO_PATH,
        help="Path to master dialogue WAV or Part 1 speech WAV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_MP4,
        help="Output path for synced Pierre avatar MP4.",
    )
    parser.add_argument(
        "--julie-start",
        type=float,
        default=None,
        help="Speaking timestamp boundary where Julie starts (defaults to audio_manifest.json).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_path, elapsed = render_pierre_avatar(
        master_wav_path=args.audio.resolve(),
        output_mp4_path=args.output.resolve(),
        julie_start_sec=args.julie_start,
    )
    print(f"[avatar_sync] Successfully generated avatar: {out_path} ({elapsed}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

