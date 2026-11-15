"""Phase 3 Dual-Voice Audio Engine & EBU R128 Mastering Pipeline.

Synthesizes Part 1 (Pierre's cloned Qwen3-TTS voice via GPU 1 :7861) and Part 2
(Julie-Core's SAPI5 female voice written directly to disk), strips dead-air
silences, applies independent EBU R128 loudness normalization, inserts a 0.25s
handoff pause, and exports broadcast-leveled WAV + MP3 master dialogue alongside
an immutable audio telemetry manifest.
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
import time
from typing import Any
import urllib.error
import urllib.request

PROJECT_ROOT = Path(r"C:\Anti-Gravity-Core\pierre-training-video")
DEFAULT_BRIEF_PATH = PROJECT_ROOT / "pipeline" / "latest_brief.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "pipeline" / "output"
VOICE_STUDIO_PYTHON = PROJECT_ROOT / "voice_studio" / ".venv" / "Scripts" / "python.exe"
VOICE_STUDIO_SERVER = PROJECT_ROOT / "voice_studio" / "tts_server.py"

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

TTS_BASE_URL = "http://127.0.0.1:7861"
PART1_VOICE_ID = "pierre-clone-en-Male"
HANDOFF_PAUSE_SEC = 0.25
SAMPLE_RATE_HZ = 44100
CHANNELS = 2
EBUR128_TARGET_I = -14.0
EBUR128_TARGET_TP = -1.5
EBUR128_TARGET_LRA = 11.0
TARGET_MEAN_VOLUME_DB = -14.8
ALIMITER_CEILING_LINEAR = 0.84  # ~ -1.51 dB headroom before LAME MP3 encoding
DEFAULT_MAX_DURATION_SEC = 57.5
MIN_ATEMPO = 1.00
MAX_ATEMPO = 1.12

SILENCEREMOVE_FILTER = (
    "silenceremove=start_periods=1:start_duration=0.05:start_threshold=-45dB:"
    "stop_periods=-1:stop_duration=0.40:stop_threshold=-45dB"
)
LOUDNORM_FILTER = f"loudnorm=I={EBUR128_TARGET_I:.0f}:TP={EBUR128_TARGET_TP:.1f}:LRA={EBUR128_TARGET_LRA:.0f}"


@dataclass(slots=True, frozen=True)
class AudioTrackTelemetry:
    label: str
    voice_id: str
    raw_path: str
    raw_duration_sec: float
    trimmed_norm_duration_sec: float
    mean_volume_db: float
    max_volume_db: float


@dataclass(slots=True, frozen=True)
class AudioManifest:
    generated_at_utc: str
    source_brief_path: str
    t1_ticker: str
    part1_voice_id: str
    part2_voice_id: str
    part1_sec: float
    handoff_sec: float
    part2_sec: float
    julie_handoff_sec: float
    atempo_factor: float
    max_duration_target_sec: float
    total_sec: float
    master_wav_path: str
    master_mp3_path: str
    master_mean_volume_db: float
    master_max_volume_db: float
    ebur128_target_i_lufs: float
    ebur128_target_tp_db: float
    ebur128_target_lra_lu: float
    sample_rate_hz: int
    channels: int
    part1_telemetry: AudioTrackTelemetry
    part2_telemetry: AudioTrackTelemetry


def resolve_ffmpeg_tool(tool_name: str) -> str:
    """Finds ffmpeg.exe or ffprobe.exe on PATH or in the WinGet FFmpeg 9.0 directory."""
    found = shutil.which(tool_name)
    if found:
        return found
    candidate = WINGET_FFMPEG_BIN_DIR / f"{tool_name}.exe"
    if candidate.exists():
        return str(candidate)
    raise FileNotFoundError(f"Required binary '{tool_name}' not found on PATH or at {candidate}")


def read_brief_contract(brief_path: Path) -> tuple[str, str, str]:
    """Reads Pierre script, Julie script, and T1 ticker from Phase 2 JSON brief."""
    if not brief_path.exists():
        raise FileNotFoundError(f"Phase 2 brief missing: {brief_path}")
    with brief_path.open(mode="r", encoding="utf-8") as handle:
        data: dict[str, Any] = json.load(handle)

    pierre_script = str(data.get("pierre_script") or "").strip()
    julie_script = str(data.get("julie_script") or "").strip()
    t1_obj = data.get("t1") if isinstance(data.get("t1"), dict) else {}
    t1_ticker = str(t1_obj.get("ticker") or "UNKNOWN").upper()

    if not pierre_script or not julie_script:
        raise ValueError(f"Brief at {brief_path} is missing pierre_script or julie_script.")
    return pierre_script, julie_script, t1_ticker


def check_tts_health(timeout_sec: float = 5.0) -> bool:
    """Checks whether the GPU 1 Qwen3-TTS server responds 200 OK on /health."""
    req = urllib.request.Request(f"{TTS_BASE_URL}/health", method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            if resp.status != 200:
                return False
            payload = json.loads(resp.read().decode("utf-8"))
            return payload.get("status") == "ok"
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return False


def ensure_voice_studio_running() -> None:
    """Boots voice_studio/tts_server.py pinned to cuda:1 if /health is offline."""
    if check_tts_health():
        return

    if not VOICE_STUDIO_PYTHON.exists() or not VOICE_STUDIO_SERVER.exists():
        raise FileNotFoundError(
            f"Voice Studio runtime missing: {VOICE_STUDIO_PYTHON} / {VOICE_STUDIO_SERVER}"
        )

    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["QWEN_TTS_DEVICE"] = "cuda:1"
    ffmpeg_dir = str(WINGET_FFMPEG_BIN_DIR)
    if WINGET_FFMPEG_BIN_DIR.exists() and ffmpeg_dir not in env.get("PATH", ""):
        env["PATH"] = f"{ffmpeg_dir}{os.pathsep}{env.get('PATH', '')}"

    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    subprocess.Popen(
        [str(VOICE_STUDIO_PYTHON), str(VOICE_STUDIO_SERVER)],
        cwd=str(VOICE_STUDIO_SERVER.parent),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )

    deadline = time.monotonic() + 90.0
    while time.monotonic() < deadline:
        if check_tts_health(timeout_sec=3.0):
            return
        time.sleep(1.5)

    raise TimeoutError("Timed out after 90s waiting for http://127.0.0.1:7861/health")


def stop_voice_studio_server() -> list[int]:
    """Terminates any running voice_studio/tts_server.py processes so GPU 1 returns to 0 MiB."""
    ps_script = "\n".join(
        [
            "$ErrorActionPreference = 'SilentlyContinue'",
            "$stopped = New-Object System.Collections.Generic.List[int]",
            "$conns = Get-NetTCPConnection -LocalPort 7860,7861 -State Listen -ErrorAction SilentlyContinue",
            "if ($conns) {",
            "  foreach ($p in ($conns.OwningProcess | Sort-Object -Unique)) {",
            "    if ($p -and $p -ne 0) {",
            "      Stop-Process -Id $p -Force -ErrorAction SilentlyContinue",
            "      [void]$stopped.Add([int]$p)",
            "    }",
            "  }",
            "}",
            "$procs = Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' -and $_.CommandLine -like '*tts_server.py*' }",
            "foreach ($pr in $procs) {",
            "  $pidVal = [int]$pr.ProcessId",
            "  if ($pidVal -gt 0) {",
            "    Stop-Process -Id $pidVal -Force -ErrorAction SilentlyContinue",
            "    if (-not $stopped.Contains($pidVal)) { [void]$stopped.Add($pidVal) }",
            "  }",
            "}",
            "$stopped | ForEach-Object { Write-Output \"STOPPED_PID=$_\" }",
        ]
    )
    proc = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", ps_script],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    stopped_pids: list[int] = []
    for line in proc.stdout.splitlines():
        if line.startswith("STOPPED_PID="):
            try:
                stopped_pids.append(int(line.split("=", 1)[1].strip()))
            except ValueError:
                pass
    if stopped_pids:
        time.sleep(0.5)
    return stopped_pids


def synthesize_pierre_part1(pierre_script: str, output_wav: Path) -> None:
    """Requests Pierre cloned voice synthesis via Qwen3-TTS OpenAI-compatible shim."""
    ensure_voice_studio_running()
    payload = json.dumps(
        {
            "model": "qwen-tts",
            "voice": PART1_VOICE_ID,
            "input": pierre_script,
            "response_format": "wav",
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        f"{TTS_BASE_URL}/v1/audio/speech",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        if resp.status != 200:
            raise RuntimeError(f"Qwen3-TTS returned HTTP {resp.status}")
        audio_bytes = resp.read()

    if len(audio_bytes) < 64 or audio_bytes[:4] != b"RIFF":
        raise RuntimeError(f"Invalid WAV response from {TTS_BASE_URL}: {len(audio_bytes)} bytes")

    output_wav.write_bytes(audio_bytes)


def _select_com_female_voice(speaker: Any) -> str:
    """Matches Julie-Core preferred female English voice from SAPI5 COM tokens."""
    preferred_keywords = ("zira", "aria", "jenny")
    voices = speaker.GetVoices()
    fallback_en = None
    for idx in range(voices.Count):
        token = voices.Item(idx)
        desc = str(token.GetDescription() or "")
        desc_lower = desc.lower()
        if any(kw in desc_lower for kw in preferred_keywords):
            speaker.Voice = token
            return desc
        gender = str(token.GetAttribute("Gender") or "").lower()
        if "female" in gender:
            speaker.Voice = token
            return desc
        if fallback_en is None and "english" in desc_lower:
            fallback_en = (token, desc)
    if fallback_en is not None:
        speaker.Voice = fallback_en[0]
        return fallback_en[1]
    return str(speaker.Voice.GetDescription() or "SAPI5-Default")


def _synthesize_julie_via_powershell(julie_script: str, output_wav: Path) -> str:
    """Writes Julie SAPI5 WAV cleanly to disk using .NET SpeechSynthesizer."""
    ps_script = "\n".join(
        [
            "$ErrorActionPreference = 'Stop'",
            "Add-Type -AssemblyName System.Speech",
            "$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer",
            "$installed = $synth.GetInstalledVoices()",
            "$chosen = $null",
            "$preferred = @('Zira', 'Aria', 'Jenny')",
            "foreach ($kw in $preferred) {",
            "  $match = $installed | Where-Object { $_.Enabled -and $_.VoiceInfo.Name -like \"*$kw*\" } | Select-Object -First 1",
            "  if ($match) { $chosen = $match; break }",
            "}",
            "if (-not $chosen) {",
            "  $chosen = $installed | Where-Object { $_.Enabled -and $_.VoiceInfo.Gender -eq 'Female' -and $_.VoiceInfo.Culture.Name -like 'en-*' } | Select-Object -First 1",
            "}",
            "if (-not $chosen) {",
            "  $chosen = $installed | Where-Object { $_.Enabled -and $_.VoiceInfo.Culture.Name -like 'en-*' } | Select-Object -First 1",
            "}",
            "if ($chosen) { $synth.SelectVoice($chosen.VoiceInfo.Name) }",
            "$synth.Rate = 0",
            "$synth.Volume = 100",
            "$synth.SetOutputToWaveFile($env:JULIE_TTS_OUT)",
            "$synth.Speak($env:JULIE_TTS_TEXT)",
            "$voiceName = $synth.Voice.Name",
            "$synth.SetOutputToNull()",
            "$synth.Dispose()",
            "Write-Output \"SELECTED_VOICE=$voiceName\"",
        ]
    )
    env = os.environ.copy()
    env["JULIE_TTS_TEXT"] = julie_script
    env["JULIE_TTS_OUT"] = str(output_wav)

    proc = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", ps_script],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"PowerShell SAPI5 synthesis failed (rc={proc.returncode}): {proc.stderr.strip()}"
        )
    for line in proc.stdout.splitlines():
        if line.startswith("SELECTED_VOICE="):
            return line.split("=", 1)[1].strip()
    return "Windows-SAPI5-System.Speech"


def synthesize_julie_part2(julie_script: str, output_wav: Path) -> str:
    """Synthesizes Part 2 using Julie-Core SAPI5 voice selection directly to WAV."""
    try:
        import win32com.client  # type: ignore[import-not-found]
    except ImportError:
        voice_id = _synthesize_julie_via_powershell(julie_script, output_wav)
    else:
        stream = win32com.client.Dispatch("SAPI.SpFileStream")
        speaker = win32com.client.Dispatch("SAPI.SpVoice")
        voice_id = _select_com_female_voice(speaker)
        speaker.Rate = 0
        speaker.Volume = 100
        ssfm_create_for_write = 3
        stream.Open(str(output_wav), ssfm_create_for_write)
        try:
            speaker.AudioOutputStream = stream
            speaker.Speak(julie_script)
        finally:
            stream.Close()

    if not output_wav.exists() or output_wav.stat().st_size < 64:
        raise RuntimeError(f"Part 2 Julie WAV missing or too small: {output_wav}")
    return voice_id


def run_cmd(cmd: list[str], timeout_sec: int = 120) -> subprocess.CompletedProcess[str]:
    """Executes subprocess and raises detailed RuntimeError on non-zero exit."""
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_sec,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed (rc={proc.returncode}): {' '.join(cmd)}\n"
            f"STDERR:\n{proc.stderr.strip()}"
        )
    return proc


def probe_duration_sec(ffprobe_bin: str, media_path: Path) -> float:
    """Measures exact media duration in seconds using ffprobe."""
    proc = run_cmd(
        [
            ffprobe_bin,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(media_path),
        ],
        timeout_sec=30,
    )
    return round(float(proc.stdout.strip()), 3)


def measure_volumedetect(ffmpeg_bin: str, audio_path: Path) -> tuple[float, float]:
    """Extracts mean_volume and max_volume (dBFS) via FFmpeg volumedetect."""
    proc = run_cmd(
        [
            ffmpeg_bin,
            "-hide_banner",
            "-i",
            str(audio_path),
            "-af",
            "volumedetect",
            "-f",
            "null",
            "NUL",
        ],
        timeout_sec=30,
    )
    mean_match = re.search(r"mean_volume:\s*([-\d.]+)\s*dB", proc.stderr)
    max_match = re.search(r"max_volume:\s*([-\d.]+)\s*dB", proc.stderr)
    if not mean_match or not max_match:
        raise RuntimeError(f"Could not parse volumedetect output for {audio_path}:\n{proc.stderr}")
    return float(mean_match.group(1)), float(max_match.group(1))


def normalize_individual_track(ffmpeg_bin: str, raw_wav: Path, norm_wav: Path) -> None:
    """Applies dead-air removal and independent EBU R128 broadcast loudness normalization."""
    filter_chain = (
        f"{SILENCEREMOVE_FILTER},"
        f"{LOUDNORM_FILTER},"
        f"aresample={SAMPLE_RATE_HZ},"
        "aformat=sample_fmts=s16:channel_layouts=stereo"
    )
    run_cmd(
        [
            ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(raw_wav),
            "-af",
            filter_chain,
            "-ar",
            str(SAMPLE_RATE_HZ),
            "-ac",
            str(CHANNELS),
            "-c:a",
            "pcm_s16le",
            str(norm_wav),
        ]
    )


def compute_tempo_factor(raw_concat_sec: float, max_duration: float) -> float:
    """Computes transparent FFmpeg atempo multiplier to keep dialogue safely under max_duration."""
    if max_duration <= 0.0 or raw_concat_sec <= max_duration:
        return 1.0
    target_safe_sec = max(1.0, max_duration - 0.15)
    factor = max(MIN_ATEMPO, round(raw_concat_sec / target_safe_sec, 4))
    if factor <= MAX_ATEMPO:
        return factor
    # Allow up to 1.25x only if 1.12x still leaves total > max_duration
    if raw_concat_sec / MAX_ATEMPO <= max_duration:
        return MAX_ATEMPO
    return min(1.25, factor)


def build_master_audio(
    brief_path: Path, output_dir: Path, max_duration: float = DEFAULT_MAX_DURATION_SEC
) -> AudioManifest:
    """Runs full Phase 3 synthesis, FFmpeg 9.0 EBU R128 mastering, tempo guard, and telemetry export."""
    resolved_output = output_dir.resolve()
    allowed_root = PROJECT_ROOT.resolve()
    if not resolved_output.is_relative_to(allowed_root):
        raise PermissionError(
            f"Write containment violation: output dir {resolved_output} is outside {allowed_root}"
        )

    ffmpeg_bin = resolve_ffmpeg_tool("ffmpeg")
    ffprobe_bin = resolve_ffmpeg_tool("ffprobe")
    resolved_output.mkdir(parents=True, exist_ok=True)

    pierre_script, julie_script, t1_ticker = read_brief_contract(brief_path)

    part1_raw = resolved_output / "part1_pierre_raw.wav"
    part2_raw = resolved_output / "part2_julie_raw.wav"
    part1_norm = resolved_output / "_tmp_part1_norm.wav"
    part2_norm = resolved_output / "_tmp_part2_norm.wav"
    master_wav = resolved_output / "master_dialogue.wav"
    master_tmp_wav = resolved_output / "_tmp_master_polish.wav"
    master_mp3 = resolved_output / "master_dialogue.mp3"
    manifest_path = resolved_output / "audio_manifest.json"

    try:
        try:
            synthesize_pierre_part1(pierre_script, part1_raw)
        finally:
            stop_voice_studio_server()

        part2_voice_id = synthesize_julie_part2(julie_script, part2_raw)

        part1_raw_dur = probe_duration_sec(ffprobe_bin, part1_raw)
        part2_raw_dur = probe_duration_sec(ffprobe_bin, part2_raw)

        normalize_individual_track(ffmpeg_bin, part1_raw, part1_norm)
        normalize_individual_track(ffmpeg_bin, part2_raw, part2_norm)

        part1_sec = probe_duration_sec(ffprobe_bin, part1_norm)
        part2_sec = probe_duration_sec(ffprobe_bin, part2_norm)
        p1_mean, p1_max = measure_volumedetect(ffmpeg_bin, part1_norm)
        p2_mean, p2_max = measure_volumedetect(ffmpeg_bin, part2_norm)

        raw_concat_sec = part1_sec + HANDOFF_PAUSE_SEC + part2_sec
        atempo_factor = compute_tempo_factor(raw_concat_sec, max_duration)
        scaled_part1_sec = round(part1_sec / atempo_factor, 3)
        scaled_handoff_sec = round(HANDOFF_PAUSE_SEC / atempo_factor, 3)
        scaled_part2_sec = round(part2_sec / atempo_factor, 3)
        julie_handoff_sec = round(scaled_part1_sec + scaled_handoff_sec, 3)

        # Concatenate Part 1 -> 0.25s silence handoff -> Part 2 (+ transparent sub-60s atempo)
        atempo_stage = f",atempo={atempo_factor:.4f}" if atempo_factor > 1.0001 else ""
        filter_complex = (
            f"anullsrc=r={SAMPLE_RATE_HZ}:cl=stereo:d={HANDOFF_PAUSE_SEC:.2f}[silence];"
            f"[0:a][silence][1:a]concat=n=3:v=0:a=1{atempo_stage}[outa]"
        )
        run_cmd(
            [
                ffmpeg_bin,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(part1_norm),
                "-i",
                str(part2_norm),
                "-filter_complex",
                filter_complex,
                "-map",
                "[outa]",
                "-ar",
                str(SAMPLE_RATE_HZ),
                "-ac",
                str(CHANNELS),
                "-c:a",
                "pcm_s16le",
                str(master_wav),
            ]
        )

        # Calibrated mastering trim/limit pass guarantees mean_volume in [-16.5, -13.5] and max <= -1.0 dB
        wav_mean, wav_max = measure_volumedetect(ffmpeg_bin, master_wav)
        if wav_mean < -15.8 or wav_mean > -14.2 or wav_max > -1.3:
            delta_db = max(-8.0, min(8.0, round(TARGET_MEAN_VOLUME_DB - wav_mean, 2)))
            polish_filter = (
                f"volume={delta_db:+.2f}dB,"
                f"alimiter=limit={ALIMITER_CEILING_LINEAR:.3f}:attack=5:release=50:level=disabled"
            )
            run_cmd(
                [
                    ffmpeg_bin,
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(master_wav),
                    "-af",
                    polish_filter,
                    "-ar",
                    str(SAMPLE_RATE_HZ),
                    "-ac",
                    str(CHANNELS),
                    "-c:a",
                    "pcm_s16le",
                    str(master_tmp_wav),
                ]
            )
            master_tmp_wav.replace(master_wav)

        run_cmd(
            [
                ffmpeg_bin,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(master_wav),
                "-codec:a",
                "libmp3lame",
                "-b:a",
                "192k",
                "-ar",
                str(SAMPLE_RATE_HZ),
                "-ac",
                str(CHANNELS),
                str(master_mp3),
            ]
        )

        total_sec = probe_duration_sec(ffprobe_bin, master_wav)
        master_mean, master_max = measure_volumedetect(ffmpeg_bin, master_mp3)

        manifest = AudioManifest(
            generated_at_utc=datetime.now(timezone.utc).isoformat(),
            source_brief_path=str(brief_path),
            t1_ticker=t1_ticker,
            part1_voice_id=PART1_VOICE_ID,
            part2_voice_id=part2_voice_id,
            part1_sec=scaled_part1_sec,
            handoff_sec=scaled_handoff_sec,
            part2_sec=scaled_part2_sec,
            julie_handoff_sec=julie_handoff_sec,
            atempo_factor=atempo_factor,
            max_duration_target_sec=max_duration,
            total_sec=total_sec,
            master_wav_path=str(master_wav),
            master_mp3_path=str(master_mp3),
            master_mean_volume_db=master_mean,
            master_max_volume_db=master_max,
            ebur128_target_i_lufs=EBUR128_TARGET_I,
            ebur128_target_tp_db=EBUR128_TARGET_TP,
            ebur128_target_lra_lu=EBUR128_TARGET_LRA,
            sample_rate_hz=SAMPLE_RATE_HZ,
            channels=CHANNELS,
            part1_telemetry=AudioTrackTelemetry(
                label="Part 1 (Pierre Cloned)",
                voice_id=PART1_VOICE_ID,
                raw_path=str(part1_raw),
                raw_duration_sec=part1_raw_dur,
                trimmed_norm_duration_sec=scaled_part1_sec,
                mean_volume_db=p1_mean,
                max_volume_db=p1_max,
            ),
            part2_telemetry=AudioTrackTelemetry(
                label="Part 2 (Julie-Core SAPI5)",
                voice_id=part2_voice_id,
                raw_path=str(part2_raw),
                raw_duration_sec=part2_raw_dur,
                trimmed_norm_duration_sec=scaled_part2_sec,
                mean_volume_db=p2_mean,
                max_volume_db=p2_max,
            ),
        )

        with manifest_path.open(mode="w", encoding="utf-8") as handle:
            json.dump(asdict(manifest), handle, indent=2, ensure_ascii=False)
            handle.write("\n")

        return manifest
    finally:
        stop_voice_studio_server()
        for tmp_file in (part1_norm, part2_norm, master_tmp_wav):
            if tmp_file.exists():
                tmp_file.unlink()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses CLI flags for Phase 3 audio engine."""
    parser = argparse.ArgumentParser(
        description="Dual-Voice Audio Engine & EBU R128 Mastering (Phase 3)"
    )
    parser.add_argument(
        "--brief",
        type=Path,
        default=DEFAULT_BRIEF_PATH,
        help=f"Path to Phase 2 brief JSON (default: {DEFAULT_BRIEF_PATH}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output folder inside pierre-training-video (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=DEFAULT_MAX_DURATION_SEC,
        help=f"Target max master audio duration in seconds (default: {DEFAULT_MAX_DURATION_SEC}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = parse_args(argv)
    manifest = build_master_audio(
        brief_path=args.brief.resolve(),
        output_dir=args.output_dir.resolve(),
        max_duration=max(0.0, float(args.max_duration)),
    )
    print(f"[audio_builder] Master WAV: {manifest.master_wav_path}")
    print(f"[audio_builder] Master MP3: {manifest.master_mp3_path}")
    print(
        f"[audio_builder] Durations: Part1={manifest.part1_sec:.2f}s | "
        f"Handoff={manifest.handoff_sec:.2f}s | Part2={manifest.part2_sec:.2f}s | "
        f"Total={manifest.total_sec:.2f}s (atempo={manifest.atempo_factor:.4f}x, handoff={manifest.julie_handoff_sec:.3f}s)"
    )
    print(
        f"[audio_builder] MP3 Levels: mean_volume={manifest.master_mean_volume_db:.1f} dB, "
        f"max_volume={manifest.master_max_volume_db:.1f} dB"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

