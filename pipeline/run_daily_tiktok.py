"""Phase 5C: One-Command Master End-to-End Pipeline & TikTok Studio Orchestrator.

Runs all five stages sequentially inside `pierre-training-video` while keeping
`pierre_stock_warroom`, `.openclaw`, `pierre_quant.db`, and `chronos_intel_buffer.json`
strictly untouched / read-only:
  1. War Room 9:16 background asset check / optional rebuild (`build_warroom_bg.py`)
  2. Quant brief & dynamic date generation (`quant_reader.py`)
  3. Dual-voice TTS + EBU R128 mastering + sub-60s tempo guard (`audio_builder.py`)
  4. Option A/D HUD + GPU-1 Whisper subtitle compositing (`render_video.py`)
  5. Compliance-locked TikTok Studio caption packager / uploader (`tiktok_upload.py`)

Guarantees `voice_studio/tts_server.py` teardown in a `finally:` block so
`GPU 1` returns to 0 MiB and `GPU 0` maintains 0 MiB studio allocation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

from audio_builder import stop_voice_studio_server

PROJECT_ROOT = Path(r"C:\Anti-Gravity-Core\pierre-training-video")
PIPELINE_DIR = PROJECT_ROOT / "pipeline"
OUTPUT_DIR = PIPELINE_DIR / "output"
ASSETS_DIR = PROJECT_ROOT / "assets"

PYTHON_BIN = PROJECT_ROOT / "video_studio" / ".venv" / "Scripts" / "python.exe"
BUILD_BG_SCRIPT = PIPELINE_DIR / "build_warroom_bg.py"
QUANT_READER_SCRIPT = PIPELINE_DIR / "quant_reader.py"
AUDIO_BUILDER_SCRIPT = PIPELINE_DIR / "audio_builder.py"
RENDER_VIDEO_SCRIPT = PIPELINE_DIR / "render_video.py"
TIKTOK_UPLOAD_SCRIPT = PIPELINE_DIR / "tiktok_upload.py"
EMAIL_AGENT_PYTHON = Path(r"C:\Anti-Gravity-Core\pierres_email_agent\.venv\Scripts\python.exe")
EMAIL_DISPATCH_SCRIPT = PIPELINE_DIR / "email_tiktok_package.py"

WARROOM_COMPOSITE_PATH = ASSETS_DIR / "war_room_9x16_composite.png"
WARROOM_BG_VIDEO_PATH = (
    PROJECT_ROOT / "video_studio" / "storage" / "local_videos" / "war_room_9x16.mp4"
)

BRIEF_PATH = PIPELINE_DIR / "latest_brief.json"
AUDIO_MANIFEST_PATH = OUTPUT_DIR / "audio_manifest.json"
RENDER_MANIFEST_PATH = OUTPUT_DIR / "render_manifest.json"
TIKTOK_PAYLOAD_PATH = OUTPUT_DIR / "tiktok_post_payload.json"
TIKTOK_CAPTION_PATH = OUTPUT_DIR / "tiktok_caption.txt"
EMAIL_DISPATCH_RECEIPT_PATH = OUTPUT_DIR / "email_dispatch_receipt.json"
SUMMARY_PATH = OUTPUT_DIR / "pipeline_run_summary.json"


@dataclass(slots=True, frozen=True)
class PipelineRunSummary:
    generated_at_utc: str
    run_date_iso: str
    run_date_badge: str
    t1_ticker: str
    t2_ticker: str
    t3_ticker: str
    upload_mode: str
    bgm_volume: float
    max_duration_target_sec: float
    atempo_factor: float
    part1_sec: float
    handoff_sec: float
    part2_sec: float
    julie_handoff_sec: float
    final_video_duration_sec: float
    final_video_size_bytes: int
    master_mean_volume_db: float
    master_max_volume_db: float
    subtitle_card_count: int
    stage_timings_sec: dict[str, float]
    output_paths: dict[str, str]
    gpu_cleanup_status: dict[str, Any]


def _load_json(path: Path) -> dict[str, Any]:
    """Reads UTF-8 JSON dict from disk."""
    with path.open(mode="r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _run_stage(
    stage_name: str, cmd: list[str], env: dict[str, str], timeout_sec: int = 300
) -> float:
    """Runs a pipeline stage subprocess and returns elapsed seconds."""
    print(f"[orchestrator] >>> Stage: {stage_name}", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=False,
        timeout=timeout_sec,
        check=False,
    )
    elapsed = round(time.perf_counter() - t0, 3)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Stage '{stage_name}' failed after {elapsed:.2f}s (exit code {proc.returncode})"
        )
    print(f"[orchestrator] <<< Stage: {stage_name} done in {elapsed:.2f}s", flush=True)
    return elapsed


def query_gpu_cleanup_status(stopped_pids: list[int]) -> dict[str, Any]:
    """Queries nvidia-smi and active Qwen-TTS port bindings after cleanup."""
    gpus: dict[str, dict[str, Any]] = {}
    proc = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if proc.returncode == 0:
        for line in proc.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 4:
                idx, name, used_mib, total_mib = parts
                gpus[f"gpu{idx}"] = {
                    "name": name,
                    "memory_used_mib": int(used_mib),
                    "memory_total_mib": int(total_mib),
                }

    port_proc = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "(Get-NetTCPConnection -LocalPort 7860,7861 -State Listen -ErrorAction SilentlyContinue | Measure-Object).Count",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    try:
        active_tts_ports = int(port_proc.stdout.strip() or "0")
    except ValueError:
        active_tts_ports = -1

    return {
        "tts_server_stopped": active_tts_ports == 0,
        "stopped_tts_pids": stopped_pids,
        "active_tts_port_bindings": active_tts_ports,
        "gpus": gpus,
    }


def run_pipeline(
    ticker: str | None,
    date_override: str | None,
    bgm_volume: float,
    max_duration: float,
    upload_mode: str,
    rebuild_bg: bool,
) -> PipelineRunSummary:
    """Executes all Phase 1-5 stages cleanly and writes pipeline_run_summary.json."""
    if not PYTHON_BIN.exists():
        raise FileNotFoundError(f"Expected Video Studio Python missing: {PYTHON_BIN}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stage_env = os.environ.copy()
    stage_env["CUDA_VISIBLE_DEVICES"] = "1"
    stage_env["QWEN_TTS_DEVICE"] = "cuda:1"

    stage_timings: dict[str, float] = {}
    stopped_pids: list[int] = []

    try:
        # 1. Background 9:16 check / conditional rebuild
        if rebuild_bg or not WARROOM_COMPOSITE_PATH.exists() or not WARROOM_BG_VIDEO_PATH.exists():
            stage_timings["bg_build"] = _run_stage(
                "bg_build",
                [str(PYTHON_BIN), str(BUILD_BG_SCRIPT)],
                stage_env,
                timeout_sec=180,
            )
        else:
            stage_timings["bg_build"] = 0.0
            print("[orchestrator] --- Stage: bg_build skipped (cached assets verified)", flush=True)

        # 2. Phase 2 Quant Reader (Read-Only DB + Dynamic Date)
        qr_cmd = [str(PYTHON_BIN), str(QUANT_READER_SCRIPT)]
        if ticker:
            qr_cmd.extend(["--ticker", ticker])
        if date_override:
            qr_cmd.extend(["--date", date_override])
        stage_timings["quant_reader"] = _run_stage("quant_reader", qr_cmd, stage_env, timeout_sec=60)

        # 3. Phase 3 Audio Builder + guaranteed tts_server teardown
        ab_cmd = [
            str(PYTHON_BIN),
            str(AUDIO_BUILDER_SCRIPT),
            "--max-duration",
            f"{max_duration:.2f}",
        ]
        try:
            stage_timings["audio_builder"] = _run_stage(
                "audio_builder", ab_cmd, stage_env, timeout_sec=300
            )
        finally:
            stopped_pids.extend(stop_voice_studio_server())

        # 4. Phase 4 Video & Option A/D HUD Compositor (GPU 1 Whisper)
        rv_cmd = [
            str(PYTHON_BIN),
            str(RENDER_VIDEO_SCRIPT),
            "--bgm-volume",
            f"{bgm_volume:.4f}",
        ]
        stage_timings["render_video"] = _run_stage(
            "render_video", rv_cmd, stage_env, timeout_sec=300
        )

        brief_data = _load_json(BRIEF_PATH)
        t1_ticker = str(brief_data.get("t1", {}).get("ticker") or "T1").upper()
        t2_ticker = str(brief_data.get("t2", {}).get("ticker") or "T2").upper()
        t3_ticker = str(brief_data.get("t3", {}).get("ticker") or "T3").upper()

        main_mp4 = OUTPUT_DIR / "tiktok_ready.mp4"
        output_paths: dict[str, str] = {
            "brief_json": str(BRIEF_PATH),
            "master_wav": str(OUTPUT_DIR / "master_dialogue.wav"),
            "master_mp3": str(OUTPUT_DIR / "master_dialogue.mp3"),
            "master_srt": str(OUTPUT_DIR / "master_dialogue.srt"),
            "tiktok_ready_mp4": str(main_mp4),
            "tiktok_caption_txt": str(TIKTOK_CAPTION_PATH),
            "tiktok_post_payload_json": str(TIKTOK_PAYLOAD_PATH),
        }
        if t1_ticker == "BTC-USD" and main_mp4.exists():
            btc_copy = OUTPUT_DIR / "tiktok_ready_btc.mp4"
            shutil.copy2(main_mp4, btc_copy)
            output_paths["tiktok_ready_btc_mp4"] = str(btc_copy)

        # 5. Phase 5B TikTok Packager / Publisher & Optional Proton Email Dispatch
        tu_mode = "package-only" if upload_mode == "email" else upload_mode
        tu_cmd = [
            str(PYTHON_BIN),
            str(TIKTOK_UPLOAD_SCRIPT),
            "--mode",
            tu_mode,
        ]
        stage_timings["tiktok_upload"] = _run_stage(
            "tiktok_upload", tu_cmd, stage_env, timeout_sec=180
        )

        if upload_mode == "email":
            if not EMAIL_AGENT_PYTHON.exists():
                raise FileNotFoundError(
                    f"Expected Proton Email Agent Python missing: {EMAIL_AGENT_PYTHON}"
                )
            em_cmd = [
                str(EMAIL_AGENT_PYTHON),
                str(EMAIL_DISPATCH_SCRIPT),
            ]
            stage_timings["email_dispatch"] = _run_stage(
                "email_dispatch", em_cmd, stage_env, timeout_sec=300
            )
            output_paths["email_dispatch_receipt_json"] = str(EMAIL_DISPATCH_RECEIPT_PATH)
    finally:
        more_pids = stop_voice_studio_server()
        for pid in more_pids:
            if pid not in stopped_pids:
                stopped_pids.append(pid)

    audio_manifest = _load_json(AUDIO_MANIFEST_PATH)
    render_manifest = _load_json(RENDER_MANIFEST_PATH)
    tiktok_payload = _load_json(TIKTOK_PAYLOAD_PATH)
    gpu_status = query_gpu_cleanup_status(stopped_pids)

    summary = PipelineRunSummary(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        run_date_iso=str(brief_data.get("run_date_iso") or ""),
        run_date_badge=str(brief_data.get("run_date_badge") or ""),
        t1_ticker=t1_ticker,
        t2_ticker=t2_ticker,
        t3_ticker=t3_ticker,
        upload_mode=upload_mode,
        bgm_volume=bgm_volume,
        max_duration_target_sec=max_duration,
        atempo_factor=float(audio_manifest.get("atempo_factor", 1.0)),
        part1_sec=float(audio_manifest.get("part1_sec", 0.0)),
        handoff_sec=float(audio_manifest.get("handoff_sec", 0.0)),
        part2_sec=float(audio_manifest.get("part2_sec", 0.0)),
        julie_handoff_sec=float(render_manifest.get("julie_handoff_sec", 0.0)),
        final_video_duration_sec=float(render_manifest.get("rendered_duration_sec", 0.0)),
        final_video_size_bytes=int(tiktok_payload.get("video_size_bytes", 0)),
        master_mean_volume_db=float(audio_manifest.get("master_mean_volume_db", 0.0)),
        master_max_volume_db=float(audio_manifest.get("master_max_volume_db", 0.0)),
        subtitle_card_count=int(render_manifest.get("subtitle_card_count", 0)),
        stage_timings_sec=stage_timings,
        output_paths=output_paths,
        gpu_cleanup_status=gpu_status,
    )

    with SUMMARY_PATH.open(mode="w", encoding="utf-8") as handle:
        json.dump(asdict(summary), handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses CLI flags for Phase 5 master orchestrator."""
    parser = argparse.ArgumentParser(
        description="One-Command Master Orchestrator & TikTok Studio Publisher (Phase 5)"
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default=None,
        help="Optional 40-asset ticker override for T1 (defaults to Morning Top-1 owned ticker).",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Optional date override in YYYY-MM-DD (defaults to live America/Toronto date).",
    )
    parser.add_argument(
        "--bgm-volume",
        type=float,
        default=0.06,
        help="BGM mix volume under dialogue (default: 0.06).",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=57.5,
        help="Max dialogue duration target in seconds for transparent atempo guard (default: 57.5).",
    )
    parser.add_argument(
        "--upload-mode",
        choices=("package-only", "email", "stage-draft", "publish"),
        default="package-only",
        help="TikTok upload/delivery mode: package-only, email (Proton Mail dispatch), stage-draft, or publish (default: package-only).",
    )
    parser.add_argument(
        "--rebuild-bg",
        action="store_true",
        help="Force regeneration of war_room_9x16.mp4 via build_warroom_bg.py.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = parse_args(argv)
    summary = run_pipeline(
        ticker=args.ticker,
        date_override=args.date,
        bgm_volume=max(0.0, float(args.bgm_volume)),
        max_duration=max(0.0, float(args.max_duration)),
        upload_mode=str(args.upload_mode),
        rebuild_bg=bool(args.rebuild_bg),
    )
    gpu1_mib = summary.gpu_cleanup_status.get("gpus", {}).get("gpu1", {}).get("memory_used_mib", "?")
    gpu0_mib = summary.gpu_cleanup_status.get("gpus", {}).get("gpu0", {}).get("memory_used_mib", "?")
    print(
        f"[orchestrator] COMPLETE ({summary.run_date_badge} | T1={summary.t1_ticker}, "
        f"T2={summary.t2_ticker}, T3={summary.t3_ticker})"
    )
    print(
        f"[orchestrator] Duration={summary.final_video_duration_sec:.3f}s | "
        f"atempo={summary.atempo_factor:.4f}x | JulieHandoff={summary.julie_handoff_sec:.3f}s | "
        f"MeanVol={summary.master_mean_volume_db:.1f} dB"
    )
    print(
        f"[orchestrator] GPU0={gpu0_mib} MiB | GPU1={gpu1_mib} MiB | "
        f"TTS_Stopped={summary.gpu_cleanup_status.get('tts_server_stopped')}"
    )
    print(f"[orchestrator] Summary: {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

