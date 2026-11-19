"""Phase 4: Headless 9:16 Video & Subtitle Compositor.

Synchronizes the mastered Phase 3 dialogue audio (`master_dialogue.mp3`) with
the looped 9:16 War Room background (`war_room_9x16.mp4`) and GPU-1-aligned
`faster-whisper` short-form captions into `pipeline/output/tiktok_ready.mp4`.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any
import wave
from zoneinfo import ZoneInfo

import numpy as np

# Isolate all CUDA / ctranslate2 execution strictly onto GPU 1 (RTX 5060 Ti #1).
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

PROJECT_ROOT = Path(r"C:\Anti-Gravity-Core\pierre-training-video")
ASSETS_DIR = PROJECT_ROOT / "assets"
VIDEO_STUDIO_DIR = PROJECT_ROOT / "video_studio"
VOICE_STUDIO_TORCH_LIB = (
    PROJECT_ROOT / "voice_studio" / ".venv" / "Lib" / "site-packages" / "torch" / "lib"
)
DEFAULT_BRIEF_PATH = PROJECT_ROOT / "pipeline" / "latest_brief.json"
DEFAULT_AUDIO_PATH = PROJECT_ROOT / "pipeline" / "output" / "master_dialogue.mp3"
DEFAULT_BG_VIDEO_PATH = VIDEO_STUDIO_DIR / "storage" / "local_videos" / "war_room_9x16.mp4"
WARROOM_SOURCE_IMAGE_PATH = ASSETS_DIR / "war_room_source.png"
DEFAULT_SONGS_DIR = VIDEO_STUDIO_DIR / "resource" / "songs"
DEFAULT_FONT_PATH = VIDEO_STUDIO_DIR / "resource" / "fonts" / "BeVietnamPro-Bold.ttf"
FONT_MEDIUM_PATH = VIDEO_STUDIO_DIR / "resource" / "fonts" / "BeVietnamPro-Medium.ttf"
CONSOLAS_BOLD_PATH = Path(r"C:\Windows\Fonts\consolab.ttf")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "pipeline" / "output"

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

CANVAS_WIDTH = 1080
CANVAS_HEIGHT = 1920
TARGET_FPS = 30
VIDEO_CRF = 18
FONT_SIZE_PX = 52
SUBTITLE_Y_CENTER_PX = 990
BAND_HEIGHT_PX = 260
BAND_Y_TOP_PX = SUBTITLE_Y_CENTER_PX - (BAND_HEIGHT_PX // 2)  # 860..1120

# HUD Layout Coordinates (9:16 canvas = 1080x1920)
HEADER_BADGE_MASK_BOX = (545, 10, 1072, 66)
HEADER_BADGE_PILL_BOX = (552, 14, 1064, 62)

TICKER_Y_TOP = 78
TICKER_HEIGHT = 30
TICKER_LABEL_WIDTH = 96
TICKER_SCROLL_WIDTH = CANVAS_WIDTH - TICKER_LABEL_WIDTH  # 984 px
TICKER_SPEED_PX_PER_FRAME = 4  # 120 px/sec @ 30 fps

JULIE_CENTER_ABS = (639, 413)
JULIE_RING_BOX = (409, 183, 869, 643)  # 460x460 centered on (639, 413)
JULIE_STATUS_BOX = (455, 138, 972, 178)  # 517x40 inside julie_card top bar
JULIE_EQ_BOX = (409, 608, 869, 664)  # 460x56 bottom bar inside julie_card

LOWER_PANEL_BOX = (18, 1126, 1062, 1788)  # 1044x662 strictly below subtitles (860..1120) & above footer (1800..1920)
RADAR_BANNER_BOX = (40, 140, 1040, 196)  # 1000x56 px at top >= 140px safe zone
COMPLIANCE_WATERMARK_BOX = (40, 1550, 620, 1588)  # 580x38 px at bottom >= 320px safe zone (1588 <= 1600)
STAGE2_SWEEP_DURATION_SEC = 0.30
MAX_LINE_WIDTH_PX = 760
WHISPER_MODEL_ID = "Systran/faster-whisper-base"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = "float16"

_DLL_HANDLES: list[Any] = []


def _bootstrap_cuda_and_imports() -> Any:
    """Registers CUDA 12 / cuDNN 9 DLLs from voice_studio before importing faster-whisper."""
    video_studio_str = str(VIDEO_STUDIO_DIR)
    if video_studio_str not in sys.path:
        sys.path.insert(0, video_studio_str)

    try:
        from app.services.subtitle import _register_voice_studio_cuda_dlls  # type: ignore[import-not-found]

        _register_voice_studio_cuda_dlls()
    except ImportError:
        pass

    if os.name == "nt" and VOICE_STUDIO_TORCH_LIB.is_dir():
        torch_lib_str = str(VOICE_STUDIO_TORCH_LIB)
        cur_path = os.environ.get("PATH", "")
        if torch_lib_str not in cur_path.split(os.pathsep):
            os.environ["PATH"] = f"{torch_lib_str}{os.pathsep}{cur_path}"
        if hasattr(os, "add_dll_directory"):
            try:
                _DLL_HANDLES.append(os.add_dll_directory(torch_lib_str))
            except OSError:
                pass

    from faster_whisper import WhisperModel  # type: ignore[import-not-found]

    return WhisperModel


from PIL import Image, ImageDraw, ImageFont  # noqa: E402


@dataclass(slots=True, frozen=True)
class WordToken:
    text: str
    start: float
    end: float


@dataclass(slots=True, frozen=True)
class SubtitleCard:
    index: int
    start: float
    end: float
    text: str
    lines: tuple[str, ...]
    word_count: int


@dataclass(slots=True, frozen=True)
class RenderManifest:
    generated_at_utc: str
    video_output_path: str
    srt_output_path: str
    source_audio_path: str
    source_bg_video_path: str
    bgm_track_path: str | None
    bgm_volume: float
    whisper_model: str
    whisper_device: str
    whisper_compute_type: str
    cuda_visible_devices: str
    width: int
    height: int
    fps: int
    video_codec: str
    audio_codec: str
    pixel_format: str
    colorspace: str
    crf: int
    target_duration_sec: float
    rendered_duration_sec: float
    total_frames: int
    subtitle_card_count: int
    subtitle_y_center_px: int
    font_path: str
    font_size_px: int
    run_date_badge: str
    t1_ticker: str
    julie_handoff_sec: float


def resolve_ffmpeg_tool(tool_name: str) -> str:
    """Finds ffmpeg.exe or ffprobe.exe on PATH or in the WinGet FFmpeg 9.0 directory."""
    found = shutil.which(tool_name)
    if found:
        return found
    candidate = WINGET_FFMPEG_BIN_DIR / f"{tool_name}.exe"
    if candidate.exists():
        return str(candidate)
    raise FileNotFoundError(f"Required binary '{tool_name}' not found on PATH or at {candidate}")


def probe_media_info(ffprobe_bin: str, media_path: Path) -> dict[str, Any]:
    """Reads structured stream & format metadata via ffprobe."""
    proc = subprocess.run(
        [
            ffprobe_bin,
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            "-of",
            "json",
            str(media_path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {media_path}: {proc.stderr.strip()}")
    return json.loads(proc.stdout)


def load_brief_context(brief_path: Path) -> tuple[set[str], str]:
    """Reads brief JSON strictly read-only to extract tickers and Whisper prompt hints."""
    tickers: set[str] = set()
    spoken_pairs: list[str] = []
    if brief_path.exists():
        with brief_path.open(mode="r", encoding="utf-8") as handle:
            brief = json.load(handle)
        for key in ("t1", "t2", "t3"):
            item = brief.get(key)
            if isinstance(item, dict) and item.get("ticker"):
                sym = str(item["ticker"]).upper().lstrip("$")
                tickers.add(sym)
                short_name = str(item.get("spoken_short") or item.get("company_name") or sym)
                spoken_pairs.append(f"{short_name}, ticker {sym}")

    pair_text = ", ".join(spoken_pairs) if spoken_pairs else "AECOM, ticker ACM"
    prompt = (
        f"Pierre, Julie, Vault Recon, {pair_text}, "
        "14-day ATR, 2x ATR, invalidation stop, 20-day moving average, "
        "Chronos, Net quant bias, Bullish Convergence."
    )
    return tickers, prompt


def _split_trailing_punct(token: str) -> tuple[str, str]:
    match = re.match(r"^([^\.,!\?;:]+)([\.,!\?;:]*)$", token.strip())
    if not match:
        return token.strip(), ""
    return match.group(1), match.group(2)


def normalize_word_tokens(raw_words: list[WordToken], known_tickers: set[str]) -> list[WordToken]:
    """Applies deterministic lexical corrections for quant terms and ticker symbols."""
    merged: list[WordToken] = []
    idx = 0
    while idx < len(raw_words):
        curr = raw_words[idx]
        curr_text = curr.text.strip()
        if not curr_text:
            idx += 1
            continue

        # Merge "in" + "validation" -> "invalidation"
        if (
            idx + 1 < len(raw_words)
            and curr_text.lower() == "in"
            and raw_words[idx + 1].text.strip().lower().startswith("validation")
        ):
            nxt = raw_words[idx + 1]
            _, punct = _split_trailing_punct(nxt.text.strip())
            merged.append(WordToken(text=f"invalidation{punct}", start=curr.start, end=nxt.end))
            idx += 2
            continue

        merged.append(WordToken(text=curr_text, start=curr.start, end=curr.end))
        idx += 1

    # Single-token domain corrections
    corrected: list[WordToken] = []
    for tok in merged:
        text = re.sub(r"\$(\d+(?:\.\d+)?)%", r"\1%", tok.text)
        core, punct = _split_trailing_punct(text)
        core_lower = core.lower()

        if core_lower in ("e-com", "ecom", "aecom"):
            text = f"AECOM{punct}"
        elif core_lower in ("netquantbias", "netquant"):
            text = f"Net quant bias{punct}"
        elif core_lower in ("kronos", "chronos"):
            text = f"Chronos{punct}"
        elif core_lower == "atr":
            text = f"ATR{punct}"
        else:
            comma_match = re.match(r"^\$(\d{4,})(\.\d+)?([,\.\?!;:]*)$", text)
            if comma_match:
                int_part, dec_part, c_punct = comma_match.groups()
                text = f"${int(int_part):,}{dec_part or ''}{c_punct}"

        corrected.append(WordToken(text=text, start=tok.start, end=tok.end))

    # Contextual "ticker <SYM>" / spelled crypto ticker formatting
    final_tokens: list[WordToken] = []
    idx = 0
    while idx < len(corrected):
        tok = corrected[idx]
        core, punct = _split_trailing_punct(tok.text)
        if core.lower() == "ticker" and idx + 1 < len(corrected):
            nxt = corrected[idx + 1]
            nxt_core, nxt_punct = _split_trailing_punct(nxt.text)
            clean_sym = nxt_core.upper().lstrip("$")
            # Collapse spelled or multi-token crypto symbol (e.g. "BTC" "USD" or "B T C U S D") -> $BTC-USD
            crypto_matched = False
            for dash_sym in known_tickers:
                if "-" not in dash_sym:
                    continue
                compact_target = dash_sym.replace("-", "")
                accum = ""
                end_j = idx + 1
                last_punct = nxt_punct
                while end_j < len(corrected) and len(accum) < len(compact_target):
                    c_core, c_punct = _split_trailing_punct(corrected[end_j].text)
                    accum += c_core.upper().lstrip("$").replace("-", "")
                    last_punct = c_punct
                    if accum == compact_target:
                        final_tokens.append(WordToken(text="ticker", start=tok.start, end=tok.end))
                        final_tokens.append(
                            WordToken(
                                text=f"${dash_sym}{last_punct}",
                                start=nxt.start,
                                end=corrected[end_j].end,
                            )
                        )
                        idx = end_j + 1
                        crypto_matched = True
                        break
                    if not compact_target.startswith(accum):
                        break
                    end_j += 1
                if crypto_matched:
                    break
            if crypto_matched:
                continue

            if clean_sym in known_tickers:
                final_tokens.append(WordToken(text="ticker", start=tok.start, end=tok.end))
                final_tokens.append(WordToken(text=f"${clean_sym}{nxt_punct}", start=nxt.start, end=nxt.end))
                idx += 2
                continue
        final_tokens.append(tok)
        idx += 1

    return final_tokens


def transcribe_and_align(audio_path: Path, brief_path: Path) -> list[WordToken]:
    """Runs GPU-1 Faster-Whisper transcription with word-level forced timestamps."""
    known_tickers, initial_prompt = load_brief_context(brief_path)
    whisper_cls = _bootstrap_cuda_and_imports()
    model = whisper_cls(
        WHISPER_MODEL_ID,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE_TYPE,
    )
    segments, _ = model.transcribe(
        str(audio_path),
        beam_size=5,
        word_timestamps=True,
        vad_filter=False,
        initial_prompt=initial_prompt,
    )
    raw_words: list[WordToken] = []
    for seg in segments:
        for w in seg.words or []:
            raw_tok = str(w.word or "")
            stripped = raw_tok.strip()
            if not stripped:
                continue
            starts_contiguous = (not raw_tok[0].isspace()) or stripped.startswith((".", ",", "-", "%", "!", "?", ":"))
            if raw_words and starts_contiguous:
                prev = raw_words[-1]
                raw_words[-1] = WordToken(
                    text=f"{prev.text}{stripped}",
                    start=prev.start,
                    end=max(prev.end, float(w.end)),
                )
            else:
                raw_words.append(WordToken(text=stripped, start=float(w.start), end=float(w.end)))
    if not raw_words:
        raise RuntimeError(f"Faster-Whisper returned zero word timestamps for {audio_path}")
    return normalize_word_tokens(raw_words, known_tickers)


def wrap_card_lines(words: list[str], font: ImageFont.FreeTypeFont) -> tuple[str, ...]:
    """Wraps a 1-to-6-word subtitle card into 1 or 2 balanced lines within MAX_LINE_WIDTH_PX."""
    full_text = " ".join(words)
    bbox = font.getbbox(full_text, stroke_width=4)
    if (bbox[2] - bbox[0]) <= MAX_LINE_WIDTH_PX or len(words) == 1:
        return (full_text,)

    best_split = max(1, len(words) // 2)
    best_diff = 10**9
    for split_idx in range(1, len(words)):
        l1 = " ".join(words[:split_idx])
        l2 = " ".join(words[split_idx:])
        w1 = font.getbbox(l1, stroke_width=4)[2] - font.getbbox(l1, stroke_width=4)[0]
        w2 = font.getbbox(l2, stroke_width=4)[2] - font.getbbox(l2, stroke_width=4)[0]
        diff = abs(w1 - w2) + (0 if max(w1, w2) <= MAX_LINE_WIDTH_PX else 10000)
        if diff < best_diff:
            best_diff = diff
            best_split = split_idx

    return (" ".join(words[:best_split]), " ".join(words[best_split:]))


def build_subtitle_cards(
    tokens: list[WordToken],
    font: ImageFont.FreeTypeFont,
    total_duration_sec: float,
) -> list[SubtitleCard]:
    """Chunks aligned words into punchy 4-to-5-word short-form subtitle cards."""
    cards: list[SubtitleCard] = []
    buf: list[WordToken] = []
    max_words_soft = 5
    max_words_hard = 6

    def flush() -> None:
        if not buf:
            return
        start = max(0.0, buf[0].start)
        end = min(total_duration_sec, max(start + 0.25, buf[-1].end + 0.08))
        raw_texts = [t.text for t in buf]
        lines = wrap_card_lines(raw_texts, font)
        cards.append(
            SubtitleCard(
                index=len(cards) + 1,
                start=start,
                end=end,
                text="\n".join(lines),
                lines=lines,
                word_count=sum(len(t.text.split()) for t in buf),
            )
        )
        buf.clear()

    for idx, tok in enumerate(tokens):
        buf.append(tok)
        word_count = sum(len(t.text.split()) for t in buf)
        ends_sentence = tok.text.endswith((".", "!", "?", "—"))
        ends_clause = tok.text.endswith((",", ";"))
        has_gap = idx + 1 < len(tokens) and (tokens[idx + 1].start - tok.end) >= 0.32
        too_wide = len(" ".join(t.text for t in buf)) >= 40

        if (
            word_count >= max_words_hard
            or ends_sentence
            or has_gap
            or (word_count >= max_words_soft)
            or (ends_clause and word_count >= 3)
            or (too_wide and word_count >= 3)
        ):
            flush()
    flush()

    # Ensure strictly monotonic non-overlapping intervals
    cleaned: list[SubtitleCard] = []
    for idx, card in enumerate(cards):
        end = card.end
        if idx + 1 < len(cards):
            end = min(end, max(card.start + 0.15, cards[idx + 1].start - 0.01))
        cleaned.append(
            SubtitleCard(
                index=card.index,
                start=round(card.start, 3),
                end=round(end, 3),
                text=card.text,
                lines=card.lines,
                word_count=card.word_count,
            )
        )
    return cleaned


def _format_srt_ts(seconds: float) -> str:
    ms_total = max(0, int(round(seconds * 1000)))
    hrs, rem = divmod(ms_total, 3_600_000)
    mins, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{ms:03d}"


def write_srt(cards: list[SubtitleCard], srt_path: Path) -> None:
    """Exports standard UTF-8 SubRip (.srt) subtitle file."""
    blocks = [
        f"{card.index}\n{_format_srt_ts(card.start)} --> {_format_srt_ts(card.end)}\n{card.text}\n"
        for card in cards
    ]
    srt_path.write_text("\n".join(blocks), encoding="utf-8")


def render_card_overlay_band(card: SubtitleCard, font: ImageFont.FreeTypeFont) -> Image.Image:
    """Renders a 1080xBAND_HEIGHT_PX RGBA tactical subtitle card with stroke & backing box."""
    band = Image.new("RGBA", (CANVAS_WIDTH, BAND_HEIGHT_PX), (0, 0, 0, 0))
    draw = ImageDraw.Draw(band)

    line_gap = 12
    line_metrics: list[tuple[str, int, int, int]] = []
    max_w = 0
    total_h = 0
    for idx, line in enumerate(card.lines):
        bbox = draw.textbbox((0, 0), line, font=font, stroke_width=4)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        line_metrics.append((line, w, h, bbox[1]))
        max_w = max(max_w, w)
        total_h += h + (line_gap if idx > 0 else 0)

    pad_x = 34
    pad_y = 22
    box_w = min(CANVAS_WIDTH - 48, max_w + pad_x * 2)
    box_h = min(BAND_HEIGHT_PX - 12, total_h + pad_y * 2)
    box_left = (CANVAS_WIDTH - box_w) // 2
    box_top = (BAND_HEIGHT_PX - box_h) // 2

    draw.rounded_rectangle(
        (box_left, box_top, box_left + box_w, box_top + box_h),
        radius=18,
        fill=(6, 11, 20, 228),
        outline=(0, 229, 255, 220),
        width=2,
    )

    cursor_y = box_top + pad_y
    for line, w, h, top_offset in line_metrics:
        text_x = (CANVAS_WIDTH - w) // 2
        text_y = cursor_y - top_offset
        draw.text(
            (text_x, text_y),
            line,
            font=font,
            fill=(255, 255, 255, 255),
            stroke_fill=(6, 11, 20, 255),
            stroke_width=4,
        )
        cursor_y += h + line_gap

    return band


@dataclass(slots=True)
class HudFonts:
    bold_by_size: dict[int, ImageFont.FreeTypeFont]
    medium_by_size: dict[int, ImageFont.FreeTypeFont]
    mono_by_size: dict[int, ImageFont.FreeTypeFont]

    def bold(self, size: int) -> ImageFont.FreeTypeFont:
        if size not in self.bold_by_size:
            self.bold_by_size[size] = ImageFont.truetype(str(DEFAULT_FONT_PATH), size)
        return self.bold_by_size[size]

    def medium(self, size: int) -> ImageFont.FreeTypeFont:
        if size not in self.medium_by_size:
            font_path = FONT_MEDIUM_PATH if FONT_MEDIUM_PATH.exists() else DEFAULT_FONT_PATH
            self.medium_by_size[size] = ImageFont.truetype(str(font_path), size)
        return self.medium_by_size[size]

    def mono(self, size: int) -> ImageFont.FreeTypeFont:
        if size not in self.mono_by_size:
            font_path = CONSOLAS_BOLD_PATH if CONSOLAS_BOLD_PATH.exists() else DEFAULT_FONT_PATH
            self.mono_by_size[size] = ImageFont.truetype(str(font_path), size)
        return self.mono_by_size[size]


def _fmt_currency(value: Any) -> str:
    """Formats numeric values cleanly as USD with comma grouping."""
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return "$0.00"


def load_hud_context(
    brief_path: Path, audio_path: Path, target_duration_sec: float
) -> tuple[dict[str, Any], float, float, float]:
    """Loads brief payload and Part 1 -> Part 2 Julie handoff timestamps."""
    brief_data: dict[str, Any] = {}
    if brief_path.exists():
        with brief_path.open(mode="r", encoding="utf-8") as handle:
            brief_data = json.load(handle)

    manifest_path = audio_path.parent / "audio_manifest.json"
    part1_sec = round(target_duration_sec * 0.30, 3)
    handoff_sec = 0.25
    if manifest_path.exists():
        with manifest_path.open(mode="r", encoding="utf-8") as handle:
            am = json.load(handle)
        part1_sec = float(am.get("part1_sec", part1_sec))
        handoff_sec = float(am.get("handoff_sec", handoff_sec))

    julie_start_sec = part1_sec + handoff_sec
    part2_sec = max(1.0, target_duration_sec - julie_start_sec)
    return brief_data, julie_start_sec, handoff_sec, part2_sec


def compute_frame_rms(
    audio_path: Path, ffmpeg_bin: str, total_frames: int, fps: int
) -> list[float]:
    """Computes normalized 0.0..1.0 smoothed per-frame audio RMS envelope."""
    wav_path = audio_path.with_suffix(".wav")
    if wav_path.exists():
        with wave.open(str(wav_path), "rb") as wf:
            sample_rate = wf.getframerate()
            channels = wf.getnchannels()
            raw_pcm = wf.readframes(wf.getnframes())
        samples = np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float32)
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)
    else:
        sample_rate = 22050
        proc = subprocess.run(
            [
                ffmpeg_bin,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(audio_path),
                "-ac",
                "1",
                "-ar",
                str(sample_rate),
                "-f",
                "s16le",
                "pipe:1",
            ],
            capture_output=True,
            timeout=30,
            check=False,
        )
        if proc.returncode != 0:
            return [0.25] * total_frames
        samples = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32)

    if len(samples) == 0:
        return [0.2] * total_frames

    samples_per_frame = max(1.0, sample_rate / float(fps))
    raw_rms = np.zeros(total_frames, dtype=np.float32)
    n_samples = len(samples)
    for idx in range(total_frames):
        s0 = int(idx * samples_per_frame)
        s1 = min(n_samples, max(s0 + 1, int((idx + 1) * samples_per_frame)))
        if s0 < n_samples:
            chunk = samples[s0:s1]
            raw_rms[idx] = float(np.sqrt(np.mean(chunk * chunk)))

    ref = max(float(np.percentile(raw_rms, 95)), 1e-5)
    norm = np.clip(raw_rms / ref, 0.0, 1.0)
    smoothed: list[float] = []
    ema = 0.0
    for val in norm:
        ema = 0.55 * float(val) + 0.45 * ema
        smoothed.append(min(1.0, max(0.0, ema**0.85)))
    return smoothed


def build_top_badge_patch(
    run_date_badge: str, t1_ticker: str, fonts: HudFonts
) -> Image.Image:
    """Pre-renders Layer 1 top-right tactical Recon date + target pill."""
    x0, y0, x1, y1 = HEADER_BADGE_MASK_BOX
    w, h = x1 - x0, y1 - y0
    patch = Image.new("RGB", (w, h), (11, 16, 30))
    draw = ImageDraw.Draw(patch)

    px0 = HEADER_BADGE_PILL_BOX[0] - x0
    py0 = HEADER_BADGE_PILL_BOX[1] - y0
    px1 = HEADER_BADGE_PILL_BOX[2] - x0
    py1 = HEADER_BADGE_PILL_BOX[3] - y0
    draw.rounded_rectangle(
        (px0, py0, px1, py1),
        radius=12,
        fill=(8, 22, 42),
        outline=(0, 229, 255),
        width=2,
    )

    dot_cx, dot_cy = px0 + 20, (py0 + py1) // 2
    draw.ellipse((dot_cx - 5, dot_cy - 5, dot_cx + 5, dot_cy + 5), fill=(0, 229, 255))

    label = f"RECON // {run_date_badge}  •  TARGET: ${t1_ticker}"
    avail_w = (px1 - px0) - 46
    chosen_font = fonts.bold(18)
    for size in (18, 17, 16, 15, 14):
        candidate = fonts.bold(size)
        if candidate.getbbox(label)[2] <= avail_w:
            chosen_font = candidate
            break
    draw.text(
        (dot_cx + 16, (py0 + py1) // 2),
        label,
        font=chosen_font,
        fill=(226, 244, 255),
        anchor="lm",
    )
    return patch


def build_3stock_radar_banner(
    episode_id: int,
    t1: str,
    t2: str,
    t3: str,
    active_slot: int,
    fonts: HudFonts,
) -> Image.Image:
    """Pre-renders Frame-0 3-Stock Radar Banner (1000x56) respecting top >= 140px safe zone."""
    w = RADAR_BANNER_BOX[2] - RADAR_BANNER_BOX[0]  # 1000
    h = RADAR_BANNER_BOX[3] - RADAR_BANNER_BOX[1]  # 56
    banner = Image.new("RGBA", (w, h), (8, 16, 28, 235))
    draw = ImageDraw.Draw(banner)
    draw.rounded_rectangle((0, 0, w - 1, h - 1), radius=12, fill=(8, 16, 28, 235), outline=(0, 229, 255, 180), width=2)

    # Left: Episode #XX | 3-STOCK RADAR:
    draw.ellipse((16, 23, 26, 33), fill=(0, 229, 255))
    header_text = f"EPISODE #{episode_id:02d} | 3-STOCK RADAR:"
    draw.text((36, 28), header_text, font=fonts.bold(17), fill=(226, 244, 255), anchor="lm")

    # Right: 3 Ticker Pills
    pills = [
        (1, f"${t1.lstrip('$')}", (480, 8, 630, 48)),
        (2, f"${t2.lstrip('$')}", (645, 8, 795, 48)),
        (3, f"${t3.lstrip('$')}", (810, 8, 985, 48)),
    ]
    for slot_idx, pill_text, (px0, py0, px1, py1) in pills:
        is_active = (slot_idx == active_slot)
        if is_active:
            fill_col = (18, 48, 76, 255)
            out_col = (0, 229, 255, 255) if slot_idx != 3 else (251, 191, 36, 255)
            txt_col = (255, 255, 255, 255)
            fnt = fonts.bold(16)
            draw.rounded_rectangle((px0, py0, px1, py1), radius=8, fill=fill_col, outline=out_col, width=2)
            draw.ellipse((px0 + 12, (py0 + py1) // 2 - 4, px0 + 20, (py0 + py1) // 2 + 4), fill=out_col)
            draw.text(((px0 + px1) // 2 + 6, (py0 + py1) // 2), pill_text, font=fnt, fill=txt_col, anchor="mm")
        else:
            fill_col = (10, 18, 30, 200)
            out_col = (32, 64, 92, 160)
            txt_col = (148, 175, 200, 240)
            fnt = fonts.medium(15)
            draw.rounded_rectangle((px0, py0, px1, py1), radius=8, fill=fill_col, outline=out_col, width=1)
            draw.text(((px0 + px1) // 2, (py0 + py1) // 2), pill_text, font=fnt, fill=txt_col, anchor="mm")

    return banner


def build_compliance_watermark(fonts: HudFonts) -> Image.Image:
    """Pre-renders permanent bottom-left regulatory compliance badge (580x38)."""
    w = COMPLIANCE_WATERMARK_BOX[2] - COMPLIANCE_WATERMARK_BOX[0]  # 580
    h = COMPLIANCE_WATERMARK_BOX[3] - COMPLIANCE_WATERMARK_BOX[1]  # 38
    badge = Image.new("RGBA", (w, h), (12, 16, 28, 230))
    draw = ImageDraw.Draw(badge)
    draw.rounded_rectangle((0, 0, w - 1, h - 1), radius=8, fill=(12, 16, 28, 230), outline=(239, 68, 68, 160), width=1)

    # Caution indicator
    draw.ellipse((14, h // 2 - 5, 24, h // 2 + 5), fill=(245, 158, 11))
    watermark_text = "QUANT TELEMETRY LOG ONLY • NOT FINANCIAL ADVICE"
    draw.text((32, h // 2), watermark_text, font=fonts.bold(13), fill=(226, 244, 255), anchor="lm")
    return badge


def build_frame0_cover_card(
    t1: str,
    t2: str,
    t3: str,
    cover_pill_text: str,
    fonts: HudFonts,
) -> Image.Image:
    """Renders Frame-0 center 3-Ticker Cover Card (880x260) inside 1080x1920 safe zones."""
    w, h = 880, 260
    card = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(card)

    # Dark obsidian backdrop with glowing cyan/amber border
    draw.rounded_rectangle(
        (0, 18, w - 1, h - 1),
        radius=20,
        fill=(8, 16, 28, 242),
        outline=(0, 229, 255, 230),
        width=3,
    )

    # Top percentage-free status pill
    pill_clean = str(cover_pill_text or "LIVE TRACK RECORD").upper()
    pill_w = min(w - 60, max(380, len(pill_clean) * 14 + 64))
    px0 = (w - pill_w) // 2
    px1 = px0 + pill_w
    draw.rounded_rectangle(
        (px0, 0, px1, 42),
        radius=12,
        fill=(14, 42, 68, 255),
        outline=(251, 191, 36, 240),
        width=2,
    )
    draw.text((w // 2, 21), pill_clean, font=fonts.bold(18), fill=(254, 240, 138), anchor="mm")

    # Center 3-ticker symbols only ($T1 • $T2 • $T3)
    s1 = f"${t1.lstrip('$').upper()}"
    s2 = f"${t2.lstrip('$').upper()}"
    s3 = f"${t3.lstrip('$').upper()}"
    trio_line = f"{s1}   •   {s2}   •   {s3}"
    draw.text((w // 2, 135), trio_line, font=fonts.bold(54), fill=(255, 255, 255), anchor="mm")

    # Subtle sub-label
    draw.text(
        (w // 2, 215),
        "DUAL-GPU QUANT WATCHTOWER • 3-STOCK AUDIT",
        font=fonts.medium(16),
        fill=(56, 189, 248),
        anchor="mm",
    )
    return card


ROTATING_MODES_MAP: dict[int, str] = {
    0: "catalyst_flow",
    1: "trajectory_chart",
    2: "trio_matrix",
}
OPAQUE_SLATE_RGB = (11, 17, 26)  # #0b111a


def _make_opaque_segment_canvas(
    title_txt: str,
    accent_rgb: tuple[int, int, int],
    fonts: HudFonts,
) -> tuple[Image.Image, ImageDraw.ImageDraw, tuple[int, int, int, int]]:
    """Creates full LOWER_PANEL_BOX (1044x662) opaque #0b111a canvas with safe-zone inner card."""
    panel_w = LOWER_PANEL_BOX[2] - LOWER_PANEL_BOX[0]  # 1044
    panel_h = LOWER_PANEL_BOX[3] - LOWER_PANEL_BOX[1]  # 662
    canvas = Image.new("RGBA", (panel_w, panel_h), (*OPAQUE_SLATE_RGB, 255))
    draw = ImageDraw.Draw(canvas)

    # Safe-zone inner card: x=22..938 (canvas x=40..956 <= 960), y=8..412 (canvas y=1134..1538 < 1550 watermark)
    card_box = (22, 8, 938, 412)
    draw.rounded_rectangle(
        card_box,
        radius=16,
        fill=(*OPAQUE_SLATE_RGB, 255),
        outline=(*accent_rgb, 215),
        width=2,
    )
    draw.rectangle((card_box[0] + 16, card_box[1] + 10, card_box[0] + 22, card_box[1] + 34), fill=(*accent_rgb, 255))
    draw.text((card_box[0] + 32, card_box[1] + 22), title_txt, font=fonts.bold(15), fill=(226, 244, 255), anchor="lm")
    draw.line(
        [(card_box[0] + 14, card_box[1] + 42), (card_box[2] - 14, card_box[1] + 42)],
        fill=(30, 41, 59, 255),
        width=1,
    )
    return canvas, draw, card_box


def blend_panel_transition(
    panel_from: Image.Image,
    panel_to: Image.Image,
    frac: float,
    accent_rgb: tuple[int, int, int] = (0, 229, 255),
) -> Image.Image:
    """Smoothly cross-fades and wipes between two HUD cards across speaker/ticker turns."""
    p = min(1.0, max(0.0, frac))
    ease = p * p * (3.0 - 2.0 * p)
    from_rgb = panel_from.convert("RGB")
    to_rgb = panel_to.convert("RGB")
    blended = Image.blend(from_rgb, to_rgb, ease)
    if 0.02 < ease < 0.98:
        w, h = blended.size
        scan_x = max(2, min(w - 3, int(round(w * ease))))
        draw = ImageDraw.Draw(blended)
        draw.rectangle((max(0, scan_x - 6), 0, min(w - 1, scan_x + 2), h - 1), fill=accent_rgb)
        draw.rectangle((max(0, scan_x - 1), 0, min(w - 1, scan_x + 1), h - 1), fill=(236, 254, 255))
    return blended


def build_progressive_trajectory_layers(
    brief_data: dict[str, Any], fonts: HudFonts
) -> tuple[Image.Image, Image.Image, tuple[int, int, int, int]]:
    """Pre-renders fixed-axis base & full-curve #0b111a chart panels for 30fps progressive draw-in."""
    import io
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    t3 = brief_data.get("t3", {})
    tk3 = str(t3.get("ticker", "MAIN")).upper().lstrip("$")
    spot = float(t3.get("spot_price") or 50.0)
    stop = float(t3.get("dynamic_stop_2x_atr") or spot * 0.92)
    atr14 = float(t3.get("atr_14d") or spot * 0.03)
    target = float(t3.get("model_target_5d") or spot)

    hist_raw = t3.get("history_20d") or []
    if len(hist_raw) < 2:
        hist_20d = [spot * (0.98 + 0.02 * (i / 19.0)) for i in range(20)]
    else:
        hist_20d = [float(x) for x in hist_raw[-20:]]

    traj_raw = t3.get("trajectory_16b") or []
    if len(traj_raw) < 2:
        traj_16b = [spot + (target - spot) * ((i + 1) / 16.0) for i in range(16)]
    else:
        traj_16b = [float(x) for x in traj_raw[:16]]

    x_hist = list(range(-(len(hist_20d) - 1), 1))
    x_fc = list(range(0, len(traj_16b) + 1))
    y_fc = [hist_20d[-1]] + traj_16b
    cone_upper = [y_fc[0]] + [y_fc[i] + atr14 * 0.45 * math.sqrt(i) for i in range(1, len(y_fc))]
    cone_lower = [y_fc[0]] + [y_fc[i] - atr14 * 0.45 * math.sqrt(i) for i in range(1, len(y_fc))]

    all_y = hist_20d + [stop] + cone_upper + cone_lower
    raw_min, raw_max = min(all_y), max(all_y)
    pad = max(0.5, (raw_max - raw_min) * 0.14)
    y_min, y_max = raw_min - pad, raw_max + pad

    is_bullish = traj_16b[-1] >= hist_20d[-1]
    fc_color = "#10B981" if is_bullish else "#F59E0B"
    chart_w, chart_h = 892, 344
    dpi = 100

    def _render_fig(show_curves: bool) -> Image.Image:
        fig = Figure(figsize=(chart_w / dpi, chart_h / dpi), dpi=dpi, facecolor="#0b111a")
        ax = fig.add_subplot(111, facecolor="#0b111a")
        line_alpha = 1.0 if show_curves else 0.0
        fill_hist_alpha = 0.10 if show_curves else 0.0
        cone_alpha = 0.22 if show_curves else 0.0

        ax.plot(x_hist, hist_20d, color="#00E5FF", linewidth=2.6, alpha=line_alpha, label="20D PRICE")
        ax.fill_between(x_hist, hist_20d, y_min, color="#00E5FF", alpha=fill_hist_alpha)
        ax.fill_between(x_fc, cone_lower, cone_upper, color=fc_color, alpha=cone_alpha, label="16-BAR CONE")
        ax.plot(x_fc, y_fc, color=fc_color, linewidth=2.6, linestyle="-", alpha=line_alpha, label="16-BAR P50")
        ax.axhline(stop, color="#EF4444", linestyle="--", linewidth=2.0, alpha=line_alpha, label=f"14D ATR FLOOR ${stop:.2f}")
        ax.axvline(0, color="#64748B", linestyle=":", linewidth=1.2)

        ax.set_xlim(x_hist[0] - 0.5, x_fc[-1] + 0.5)
        ax.set_ylim(y_min, y_max)
        ax.tick_params(colors="#CBD5E1", labelsize=10)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("bottom", "left"):
            ax.spines[spine].set_color("#334155")
        ax.grid(color="#1E293B", linestyle="-", linewidth=0.7, alpha=0.85)
        leg = ax.legend(loc="upper left", frameon=False, labelcolor="#F1F5F9", fontsize=10, ncol=2)
        for handle in leg.legend_handles:
            try:
                handle.set_alpha(1.0)
            except Exception:
                pass
        fig.tight_layout(pad=0.6)

        canvas_agg = FigureCanvasAgg(fig)
        buf = io.BytesIO()
        canvas_agg.print_png(buf)
        buf.seek(0)
        return Image.open(buf).convert("RGBA")

    axes_img = _render_fig(show_curves=False)
    full_img = _render_fig(show_curves=True)

    title_txt = f"MAIN EVENT: ${tk3}  |  16-BAR CONE & 14D ATR"
    base_panel, _, card_box = _make_opaque_segment_canvas(title_txt, (0, 229, 255), fonts)
    full_panel, _, _ = _make_opaque_segment_canvas(title_txt, (0, 229, 255), fonts)

    paste_x, paste_y = card_box[0] + 12, card_box[1] + 50
    base_panel.paste(axes_img, (paste_x, paste_y), axes_img)
    full_panel.paste(full_img, (paste_x, paste_y), full_img)

    plot_box = (paste_x + 48, paste_y + 10, paste_x + chart_w - 10, paste_y + chart_h - 24)
    return base_panel.convert("RGB"), full_panel.convert("RGB"), plot_box


def render_progressive_chart_frame(
    base_panel: Image.Image,
    full_panel: Image.Image,
    plot_box: tuple[int, int, int, int],
    progress: float,
    haircut: bool,
    fonts: HudFonts,
) -> Image.Image:
    """Renders 30fps progressive draw-in across 20D price, 14D ATR floor, and 16-BAR CONE."""
    p = min(1.0, max(0.0, progress))
    draw_frac = min(1.0, p / 0.88)
    ease = 1.0 - (1.0 - draw_frac) ** 2

    frame_panel = base_panel.copy()
    x0, y0, x1, y1 = plot_box
    scan_x = max(x0 + 2, min(x1, int(round(x0 + (x1 - x0) * ease))))
    reveal_crop = full_panel.crop((x0, y0, scan_x, y1))
    frame_panel.paste(reveal_crop, (x0, y0))

    draw = ImageDraw.Draw(frame_panel)
    if ease < 0.99:
        cursor_col = (0, 229, 255) if ease < 0.54 else (16, 185, 129)
        draw.line([(scan_x, y0 + 2), (scan_x, y1 - 2)], fill=cursor_col, width=2)
        draw.ellipse((scan_x - 4, y0 + 6, scan_x + 4, y0 + 14), fill=(255, 255, 255))

    # Header telemetry status pill (strictly inside safe zone <= 938 right border)
    if ease < 0.54:
        phase_txt = "DRAWING 20D PRICE + 14D ATR..."
        pill_col = (0, 229, 255)
    elif ease < 0.99:
        phase_txt = "EXPANDING 16-BAR CONE..."
        pill_col = (16, 185, 129)
    else:
        phase_txt = "16-BAR CONE LOCKED [-20% NULL]" if haircut else "16-BAR CONE LOCKED"
        pill_col = (251, 191, 36) if haircut else (52, 211, 153)

    draw.rounded_rectangle((632, 16, 926, 44), radius=7, fill=(15, 23, 42), outline=pill_col, width=1)
    draw.text((779, 30), phase_txt, font=fonts.bold(12), fill=pill_col, anchor="mm")
    return frame_panel


def build_slot3_trajectory_chart(brief_data: dict[str, Any], fonts: HudFonts) -> Image.Image:
    """Compatibility wrapper returning locked full opaque #0b111a trajectory chart panel."""
    base_p, full_p, plot_b = build_progressive_trajectory_layers(brief_data, fonts)
    haircut = bool(brief_data.get("t3", {}).get("haircut_applied", False))
    return render_progressive_chart_frame(base_p, full_p, plot_b, 1.0, haircut, fonts).convert("RGBA")


def build_trio_matrix_panel(brief_data: dict[str, Any], fonts: HudFonts) -> Image.Image:
    """Renders side-by-side 3-ticker comparison matrix (08 velocity, 07 Z-score, 02 ATR stop)."""
    title_txt = "TRIO MATRIX  |  AGENT 08 VELOCITY • 07 Z-SCORE • 02 ATR STOP"
    panel, draw, card_box = _make_opaque_segment_canvas(title_txt, (56, 189, 248), fonts)

    slots = [
        ("SLOT #1 • RADAR", brief_data.get("t1", {}), (0, 229, 255)),
        ("SLOT #2 • RADAR", brief_data.get("t2", {}), (52, 211, 153)),
        ("SLOT #3 • MAIN", brief_data.get("t3", {}), (251, 191, 36)),
    ]

    inner_x0 = card_box[0] + 14
    col_w = 290
    gap = 10
    top_y = card_box[1] + 54
    bot_y = card_box[3] - 14

    for idx, (badge_lbl, slot_d, accent) in enumerate(slots):
        cx0 = inner_x0 + idx * (col_w + gap)
        cx1 = cx0 + col_w
        draw.rounded_rectangle(
            (cx0, top_y, cx1, bot_y),
            radius=12,
            fill=(15, 23, 42, 255),
            outline=(*accent, 190),
            width=2 if idx == 2 else 1,
        )
        tk = f"${str(slot_d.get('ticker', f'T{idx+1}')).upper().lstrip('$')}"
        spot = float(slot_d.get("spot_price") or 0.0)
        vel = float(slot_d.get("forecast_return_pct") or 0.0)
        z_val = float(slot_d.get("z_score_anomaly") or 0.0)
        atr_stop = float(slot_d.get("dynamic_stop_2x_atr") or 0.0)
        conv = float(slot_d.get("adjusted_confluence_score") or 0.0)

        draw.text((cx0 + 16, top_y + 18), badge_lbl, font=fonts.bold(12), fill=accent, anchor="lm")
        draw.text((cx0 + 16, top_y + 50), tk, font=fonts.bold(28), fill=(255, 255, 255), anchor="lm")
        draw.text((cx1 - 16, top_y + 50), f"${spot:.2f}", font=fonts.mono(22), fill=(226, 244, 255), anchor="rm")
        draw.line([(cx0 + 14, top_y + 78), (cx1 - 14, top_y + 78)], fill=(30, 41, 59), width=1)

        metrics = [
            ("08 VELOCITY", f"{vel:+6.2f}%", (52, 211, 153) if vel >= 0 else (248, 113, 113)),
            ("07 Z-SCORE", f"{z_val:+6.2f}σ", (251, 191, 36) if abs(z_val) >= 2.0 else (203, 213, 225)),
            ("02 ATR STOP", f"${atr_stop:>6.2f}", (248, 113, 113)),
            ("CONVICTION", f"{conv:>5.1f}%", accent),
        ]
        for m_idx, (m_lbl, m_val, m_col) in enumerate(metrics):
            my = top_y + 114 + m_idx * 56
            draw.text((cx0 + 16, my), m_lbl, font=fonts.medium(13), fill=(148, 163, 184), anchor="lm")
            draw.text((cx1 - 16, my), m_val, font=fonts.mono(18), fill=m_col, anchor="rm")

    return panel


def build_catalyst_flow_panel(brief_data: dict[str, Any], fonts: HudFonts) -> Image.Image:
    """Renders insider/institutional/news flow audit with explicit NULL blindspot markers."""
    title_txt = "CATALYST & BLINDSPOT AUDIT  |  SEC FORM 4 • FINNHUB MSPR • -20% RULE"
    panel, draw, card_box = _make_opaque_segment_canvas(title_txt, (251, 191, 36), fonts)

    rows = [
        ("SLOT #1", brief_data.get("t1", {})),
        ("SLOT #2", brief_data.get("t2", {})),
        ("MAIN EVENT", brief_data.get("t3", {})),
    ]
    rx0 = card_box[0] + 14
    rx1 = card_box[2] - 14
    start_y = card_box[1] + 54
    row_h = 104
    gap = 12

    for idx, (slot_lbl, slot_d) in enumerate(rows):
        ry0 = start_y + idx * (row_h + gap)
        ry1 = ry0 + row_h
        is_main = idx == 2
        draw.rounded_rectangle(
            (rx0, ry0, rx1, ry1),
            radius=12,
            fill=(15, 23, 42, 255),
            outline=(251, 191, 36, 210) if is_main else (51, 65, 85, 200),
            width=2 if is_main else 1,
        )
        tk = f"${str(slot_d.get('ticker', f'T{idx+1}')).upper().lstrip('$')}"
        mspr = slot_d.get("finnhub_mspr")
        sec_null = bool(slot_d.get("sec_form4_null", mspr is None))
        haircut = bool(slot_d.get("haircut_applied", sec_null or mspr is None))
        conv = float(slot_d.get("adjusted_confluence_score") or 0.0)
        stop = float(slot_d.get("dynamic_stop_2x_atr") or 0.0)
        bias = str(slot_d.get("net_quant_bias") or "NEUTRAL").replace("_", " ")

        mspr_txt = "MSPR: NULL" if mspr is None else f"MSPR: {float(mspr):+.2f}"
        sec_txt = "FORM 4: NULL" if sec_null else "FORM 4: ACTIVE"
        penalty_txt = "BLINDSPOT [-20% PENALTY]" if haircut else "VERIFIED FLOW"
        pen_col = (248, 113, 113) if haircut else (52, 211, 153)

        draw.text((rx0 + 18, ry0 + 26), f"{slot_lbl}  •  {tk}", font=fonts.bold(20), fill=(255, 255, 255), anchor="lm")
        draw.text((rx0 + 18, ry0 + 68), f"{sec_txt}   |   {mspr_txt}", font=fonts.mono(15), fill=(148, 163, 184), anchor="lm")
        draw.text((rx1 - 18, ry0 + 26), penalty_txt, font=fonts.bold(15), fill=pen_col, anchor="rm")
        draw.text(
            (rx1 - 18, ry0 + 68),
            f"{bias[:18]}  |  CONV {conv:.1f}%  |  STOP ${stop:.2f}",
            font=fonts.mono(14),
            fill=(203, 213, 225),
            anchor="rm",
        )

    return panel


def build_rotating_segment_panel(brief_data: dict[str, Any], fonts: HudFonts) -> Image.Image:
    """Selects middle rotating HUD segment (~16s-38s) by episode_num % 3."""
    ep_id = int(brief_data.get("episode_id", 4))
    mode = str(brief_data.get("rotation_mode") or ROTATING_MODES_MAP[ep_id % 3])
    if mode == "trio_matrix":
        return build_trio_matrix_panel(brief_data, fonts)
    if mode == "catalyst_flow":
        return build_catalyst_flow_panel(brief_data, fonts)
    return build_slot3_trajectory_chart(brief_data, fonts)


def build_ticker_assets(
    brief_data: dict[str, Any], fonts: HudFonts
) -> tuple[Image.Image, Image.Image]:
    """Builds Layer 2 static '» LIVE FEED' label and seamless 2160px scrolling strip."""
    if WARROOM_SOURCE_IMAGE_PATH.exists():
        with Image.open(WARROOM_SOURCE_IMAGE_PATH) as src_img:
            src_rgb = src_img.convert("RGB")
            label_crop = src_rgb.crop((100, 184, 195, 219)).resize(
                (TICKER_LABEL_WIDTH, TICKER_HEIGHT), Image.Resampling.LANCZOS
            )
            feed_crop = src_rgb.crop((195, 184, 1408, 219)).resize(
                (CANVAS_WIDTH, TICKER_HEIGHT), Image.Resampling.LANCZOS
            )
        draw_lbl = ImageDraw.Draw(label_crop)
        draw_lbl.line(
            [(TICKER_LABEL_WIDTH - 1, 0), (TICKER_LABEL_WIDTH - 1, TICKER_HEIGHT)],
            fill=(0, 160, 200),
            width=1,
        )
        tile = Image.new("RGB", (CANVAS_WIDTH * 2, TICKER_HEIGHT), (5, 9, 17))
        tile.paste(feed_crop, (0, 0))
        tile.paste(feed_crop, (CANVAS_WIDTH, 0))
        return label_crop, tile

    label_fb = Image.new("RGB", (TICKER_LABEL_WIDTH, TICKER_HEIGHT), (6, 16, 32))
    ImageDraw.Draw(label_fb).text(
        (8, TICKER_HEIGHT // 2), "» LIVE", font=fonts.bold(14), fill=(0, 229, 255), anchor="lm"
    )
    tile_fb = Image.new("RGB", (CANVAS_WIDTH * 2, TICKER_HEIGHT), (5, 9, 17))
    parts = [
        f"${brief_data.get(k, {}).get('ticker', k.upper())} {_fmt_currency(brief_data.get(k, {}).get('spot_price', 0.0))}"
        for k in ("t1", "t2", "t3")
    ]
    text_line = "   •   ".join(parts) + "   •   "
    draw_fb = ImageDraw.Draw(tile_fb)
    draw_fb.text((12, TICKER_HEIGHT // 2), text_line * 4, font=fonts.mono(15), fill=(165, 235, 255), anchor="lm")
    return label_fb, tile_fb


def build_subtitle_scrim() -> Image.Image:
    """Creates dark obsidian scrim (860..1120) so captions have clear separation."""
    scrim = Image.new("RGBA", (CANVAS_WIDTH, BAND_HEIGHT_PX), (6, 11, 20, 234))
    draw = ImageDraw.Draw(scrim)
    draw.line([(18, 0), (CANVAS_WIDTH - 18, 0)], fill=(0, 229, 255, 130), width=1)
    draw.line(
        [(18, BAND_HEIGHT_PX - 1), (CANVAS_WIDTH - 18, BAND_HEIGHT_PX - 1)],
        fill=(0, 229, 255, 130),
        width=1,
    )
    return scrim


def build_julie_status_badge(stage: int, fonts: HudFonts) -> Image.Image:
    """Pre-renders Layer 3 Julie HUD stage pill for top bar inside julie_card."""
    w = JULIE_STATUS_BOX[2] - JULIE_STATUS_BOX[0]
    h = JULIE_STATUS_BOX[3] - JULIE_STATUS_BOX[1]
    badge = Image.new("RGBA", (w, h), (10, 16, 27, 255))
    draw = ImageDraw.Draw(badge)

    if stage == 1:
        fill = (13, 24, 42, 255)
        outline = (56, 189, 248, 200)
        dot_color = (251, 191, 36, 255)
        text_color = (216, 236, 252, 255)
        label = "VOICE LINK // PIERRE (LEAD ENGINEER)"
    else:
        fill = (6, 28, 46, 255)
        outline = (0, 229, 255, 255)
        dot_color = (0, 255, 210, 255)
        text_color = (0, 242, 255, 255)
        label = "JULIE-CORE // ACTIVE TELEMETRY STREAM"

    draw.rounded_rectangle((2, 2, w - 3, h - 3), radius=10, fill=fill, outline=outline, width=2)
    draw.ellipse((16, h // 2 - 5, 26, h // 2 + 5), fill=dot_color)
    draw.text((36, h // 2), label, font=fonts.bold(16), fill=text_color, anchor="lm")
    return badge


def render_julie_rings_overlay(t_sec: float, rms: float, active: bool) -> Image.Image:
    """Draws Layer 3 concentric audio-reactive rings around Julie portrait center."""
    w = JULIE_RING_BOX[2] - JULIE_RING_BOX[0]
    h = JULIE_RING_BOX[3] - JULIE_RING_BOX[1]
    cx = JULIE_CENTER_ABS[0] - JULIE_RING_BOX[0]
    cy = JULIE_CENTER_ABS[1] - JULIE_RING_BOX[1]
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    if not active:
        breath = 0.5 + 0.5 * math.sin(2.0 * math.pi * 0.65 * t_sec)
        r_standby = 154.0 + 5.0 * breath + 4.0 * rms
        alpha_main = int(110 + 50 * breath)
        draw.ellipse(
            (cx - r_standby, cy - r_standby, cx + r_standby, cy + r_standby),
            outline=(56, 189, 248, alpha_main),
            width=3,
        )
        r_arc = r_standby + 16.0
        start_deg = (t_sec * 42.0) % 360.0
        draw.arc(
            (cx - r_arc, cy - r_arc, cx + r_arc, cy + r_arc),
            start=start_deg,
            end=start_deg + 85.0,
            fill=(0, 229, 255, 130),
            width=2,
        )
        draw.arc(
            (cx - r_arc, cy - r_arc, cx + r_arc, cy + r_arc),
            start=start_deg + 180.0,
            end=start_deg + 265.0,
            fill=(0, 229, 255, 130),
            width=2,
        )
        return overlay

    # Active Julie speech: 3 concentric RMS-pulsing cyan energy rings
    radii = (
        152.0 + 13.0 * rms,
        175.0 + 22.0 * rms,
        198.0 + 30.0 * rms,
    )
    alphas = (
        int(180 + 75 * rms),
        int(120 + 100 * rms),
        int(65 + 115 * rms),
    )
    widths = (4 if rms > 0.45 else 3, 3, 2)
    for radius, alpha, width in zip(radii, alphas, widths):
        draw.ellipse(
            (cx - radius, cy - radius, cx + radius, cy + radius),
            outline=(0, 229, 255, min(255, alpha)),
            width=width,
        )
    if rms > 0.55:
        r_peak = 149.0 + 8.0 * rms
        draw.ellipse(
            (cx - r_peak, cy - r_peak, cx + r_peak, cy + r_peak),
            outline=(210, 252, 255, int(180 * (rms - 0.5))),
            width=2,
        )
    return overlay


def render_julie_eq_overlay(t_sec: float, rms: float, active: bool) -> Image.Image:
    """Draws Layer 3 symmetric 24-bar cyan waveform equalizer at bottom of julie_card."""
    w = JULIE_EQ_BOX[2] - JULIE_EQ_BOX[0]
    h = JULIE_EQ_BOX[3] - JULIE_EQ_BOX[1]
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    # Mask underlying static 'IDLE // STANDBY' footer inside bottom strip of portrait panel
    draw.rounded_rectangle(
        (2, 2, w - 3, h - 3),
        radius=8,
        fill=(8, 16, 30, 240),
        outline=(0, 180, 220, 120) if active else (36, 78, 112, 100),
        width=1,
    )

    n_bars = 24
    bar_w = 10
    gap = 8
    total_w = n_bars * bar_w + (n_bars - 1) * gap
    x_start = (w - total_w) // 2
    y_bottom = h - 8

    for idx in range(n_bars):
        # Symmetric distance from center (0.0 at middle -> 1.0 at edges)
        sym_pos = abs((idx - (n_bars - 1) / 2.0) / ((n_bars - 1) / 2.0))
        envelope = 1.0 - 0.58 * (sym_pos**1.35)
        harmonic = 0.5 + 0.5 * math.sin(t_sec * 16.0 + idx * 0.72) * math.cos(
            t_sec * 7.5 - sym_pos * 3.2
        )
        if active:
            bar_h = int(4 + (h - 14) * min(1.0, (0.18 + 0.82 * rms) * envelope * (0.45 + 0.55 * harmonic)))
            bar_color = (0, 229, 255, 225)
            cap_color = (224, 252, 255, 255)
        else:
            ambient = 0.25 + 0.75 * (0.5 + 0.5 * math.sin(t_sec * 3.0 + idx * 0.5))
            bar_h = int(3 + 8 * envelope * ambient * (0.4 + 0.6 * rms))
            bar_color = (56, 189, 248, 120)
            cap_color = (147, 215, 250, 175)

        bx0 = x_start + idx * (bar_w + gap)
        bx1 = bx0 + bar_w
        by0 = max(4, y_bottom - bar_h)
        draw.rounded_rectangle((bx0, by0, bx1, y_bottom), radius=2, fill=bar_color)
        draw.rectangle((bx0, by0, bx1, min(y_bottom, by0 + 2)), fill=cap_color)
    return overlay


def build_stage1_lower_card(
    brief_data: dict[str, Any], fonts: HudFonts, active_slot: int = 1
) -> Image.Image:
    """Pre-renders Option D Stage 1 lower panel with dynamic per-slot active spotlight."""
    w = LOWER_PANEL_BOX[2] - LOWER_PANEL_BOX[0]  # 1044
    h = LOWER_PANEL_BOX[3] - LOWER_PANEL_BOX[1]  # 662
    panel = Image.new("RGB", (w, h), (6, 11, 20))
    draw = ImageDraw.Draw(panel)

    # Left Card: 8-Cluster Constellation Radar
    left_box = (0, 0, 444, h)
    draw.rounded_rectangle(left_box, radius=14, fill=(9, 17, 32), outline=(22, 78, 112), width=2)
    draw.rectangle((16, 0, 428, 3), fill=(0, 229, 255))
    draw.text(
        (20, 26),
        "8-CLUSTER CONSTELLATION RADAR",
        font=fonts.bold(18),
        fill=(0, 229, 255),
        anchor="lm",
    )
    draw.line([(16, 48), (428, 48)], fill=(20, 62, 92), width=1)

    if WARROOM_SOURCE_IMAGE_PATH.exists():
        with Image.open(WARROOM_SOURCE_IMAGE_PATH) as src_img:
            radar_crop = src_img.convert("RGB").crop((1347, 238, 1820, 619))
        radar_resized = radar_crop.resize((412, 332), Image.Resampling.LANCZOS)
        panel.paste(radar_resized, (16, 62))
        draw.rectangle((15, 61, 428, 394), outline=(26, 82, 118), width=1)
    else:
        draw.rounded_rectangle((16, 62, 428, 394), radius=10, fill=(10, 22, 40), outline=(26, 82, 118))

    # Left footer metadata pills
    active_key = f"t{max(1, min(3, active_slot))}"
    focus_asset = brief_data.get(active_key, brief_data.get("t1", {}))
    run_date_badge = str(brief_data.get("run_date_badge", "LIVE"))
    scan_ts = str(brief_data.get("morning_scan_timestamp", "")).split(" EST")[0]
    info_rows = [
        ("PORTFOLIO UNIVERSE", "40-ASSET MATRIX", (56, 189, 248)),
        ("RECON DATE", run_date_badge, (0, 229, 255)),
        ("PRIMARY FOCUS", f"${focus_asset.get('ticker', 'T1')}", (251, 191, 36) if active_slot == 3 else (16, 185, 129)),
        ("LEDGER MODE", "SQLITE READ-ONLY", (148, 163, 184)),
    ]
    y_cursor = 414
    for key_txt, val_txt, val_col in info_rows:
        draw.rounded_rectangle(
            (16, y_cursor, 428, y_cursor + 48),
            radius=8,
            fill=(11, 21, 39),
            outline=(22, 54, 82),
            width=1,
        )
        draw.text((28, y_cursor + 24), key_txt, font=fonts.bold(14), fill=(148, 163, 184), anchor="lm")
        draw.text((414, y_cursor + 24), val_txt, font=fonts.mono(15), fill=val_col, anchor="rm")
        y_cursor += 58
    if scan_ts:
        draw.text(
            (222, h - 16),
            f"SCAN LOCK: {scan_ts} EST",
            font=fonts.medium(13),
            fill=(100, 116, 139),
            anchor="mm",
        )

    # Right Card: Morning Watchtower // Top 3 Confluence
    rx0 = 458
    right_box = (rx0, 0, w - 1, h)
    draw.rounded_rectangle(right_box, radius=14, fill=(9, 17, 32), outline=(24, 96, 134), width=2)
    draw.rectangle((rx0 + 16, 0, w - 17, 3), fill=(0, 229, 255))
    draw.text(
        (rx0 + 20, 26),
        "MORNING WATCHTOWER // TOP 3 CONFLUENCE",
        font=fonts.bold(19),
        fill=(0, 229, 255),
        anchor="lm",
    )
    draw.line([(rx0 + 16, 48), (w - 17, 48)], fill=(20, 62, 92), width=1)

    card_y = 64
    card_h = 182
    card_gap = 16
    for idx, key in enumerate(("t1", "t2", "t3"), start=1):
        asset = brief_data.get(key, {})
        ticker = str(asset.get("ticker", f"T{idx}"))
        company = str(asset.get("company_name", ticker))
        if len(company) > 22:
            company = company[:21] + "…"
        spot = _fmt_currency(asset.get("spot_price", 0.0))
        atr_d = _fmt_currency(asset.get("atr_14d", 0.0))
        atr_p = float(asset.get("atr_14d_pct", 0.0))
        stop_d = _fmt_currency(asset.get("dynamic_stop_2x_atr", 0.0))
        tgt_d = _fmt_currency(asset.get("model_target_5d", 0.0))
        conf = float(asset.get("adjusted_confluence_score", 0.0))
        bias = str(asset.get("net_quant_bias", "Neutral"))
        haircut = bool(asset.get("haircut_applied", False))

        is_primary = (idx == active_slot)
        accent_col = (251, 191, 36) if idx == 3 else (0, 229, 255)
        card_fill = (15, 34, 62) if is_primary else (8, 14, 26)
        card_outline = accent_col if is_primary else (20, 48, 72)
        cx0, cy0, cx1, cy1 = rx0 + 16, card_y, w - 17, card_y + card_h

        draw.rounded_rectangle(
            (cx0, cy0, cx1, cy1), radius=12, fill=card_fill, outline=card_outline, width=3 if is_primary else 1
        )
        draw.rectangle(
            (cx0, cy0 + 14, cx0 + (7 if is_primary else 4), cy1 - 14),
            fill=accent_col if is_primary else (36, 76, 108),
        )

        # Row 1 inside asset card: Rank + Ticker + LIVE AUDIT Pill / Company + Confluence Pill
        rank_txt = f"T{idx}"
        draw.rounded_rectangle(
            (cx0 + 16, cy0 + 14, cx0 + 62, cy0 + 44),
            radius=6,
            fill=accent_col if is_primary else (20, 42, 68),
        )
        draw.text(
            (cx0 + 39, cy0 + 29),
            rank_txt,
            font=fonts.bold(16),
            fill=(6, 14, 26) if is_primary else (148, 175, 200),
            anchor="mm",
        )
        ticker_label = f"${ticker}"
        ticker_font = fonts.bold(23)
        draw.text(
            (cx0 + 74, cy0 + 29),
            ticker_label,
            font=ticker_font,
            fill=(255, 255, 255) if is_primary else (180, 198, 220),
            anchor="lm",
        )
        ticker_w = ticker_font.getbbox(ticker_label)[2]
        comp_x = cx0 + 74 + ticker_w + 12
        if is_primary:
            live_pill_w = 104
            draw.rounded_rectangle(
                (comp_x, cy0 + 17, comp_x + live_pill_w, cy0 + 41),
                radius=6,
                fill=(18, 52, 74) if idx != 3 else (64, 42, 10),
                outline=accent_col,
                width=1,
            )
            draw.text(
                (comp_x + live_pill_w // 2, cy0 + 29),
                "LIVE AUDIT",
                font=fonts.bold(12),
                fill=accent_col,
                anchor="mm",
            )
            comp_x += live_pill_w + 10
        conf_label = f"CONF {conf:.2f}%" + (" [-20%]" if haircut else "")
        conf_col = (245, 158, 11) if haircut else ((0, 229, 255) if conf > 0 else (148, 163, 184))
        conf_w = fonts.mono(15).getbbox(conf_label)[2]
        max_comp_w = max(32, (cx1 - 14 - conf_w - 12) - comp_x)
        comp_txt = company.upper()
        comp_font = fonts.medium(14)
        while len(comp_txt) > 3 and comp_font.getbbox(comp_txt)[2] > max_comp_w:
            comp_txt = comp_txt[:-2] + "…"
        if max_comp_w >= 48:
            draw.text(
                (comp_x, cy0 + 30),
                comp_txt,
                font=comp_font,
                fill=(196, 222, 248) if is_primary else (118, 145, 175),
                anchor="lm",
            )

        draw.text((cx1 - 14, cy0 + 29), conf_label, font=fonts.mono(15), fill=conf_col, anchor="rm")

        draw.line([(cx0 + 16, cy0 + 56), (cx1 - 16, cy0 + 56)], fill=(22, 54, 82), width=1)

        # Row 2: Spot + 14D ATR
        draw.text((cx0 + 18, cy0 + 76), "SPOT", font=fonts.bold(13), fill=(118, 145, 175), anchor="lm")
        draw.text((cx0 + 18, cy0 + 101), spot, font=fonts.mono(21), fill=(248, 250, 252), anchor="lm")

        draw.text((cx0 + 210, cy0 + 76), "14-DAY ATR", font=fonts.bold(13), fill=(118, 145, 175), anchor="lm")
        draw.text(
            (cx0 + 210, cy0 + 101),
            f"{atr_d} ({atr_p:.2f}%)",
            font=fonts.mono(19),
            fill=(56, 189, 248),
            anchor="lm",
        )

        # Row 3: Dynamic Stop + 5D Target + Bias
        draw.text((cx0 + 18, cy0 + 132), "2X STOP", font=fonts.bold(13), fill=(118, 145, 175), anchor="lm")
        draw.text((cx0 + 18, cy0 + 156), stop_d, font=fonts.mono(18), fill=(251, 146, 60), anchor="lm")

        draw.text((cx0 + 210, cy0 + 132), "5D MODEL", font=fonts.bold(13), fill=(118, 145, 175), anchor="lm")
        draw.text((cx0 + 210, cy0 + 156), tgt_d, font=fonts.mono(18), fill=(167, 243, 208), anchor="lm")

        bias_lower = bias.lower()
        if "bull" in bias_lower:
            bias_fill, bias_border, bias_txt = (8, 52, 42), (16, 185, 129), (110, 231, 183)
        elif "bear" in bias_lower:
            bias_fill, bias_border, bias_txt = (58, 18, 28), (244, 63, 94), (253, 164, 175)
        else:
            bias_fill, bias_border, bias_txt = (48, 36, 12), (245, 158, 11), (253, 224, 71)
        draw.rounded_rectangle(
            (cx1 - 210, cy0 + 132, cx1 - 14, cy0 + 168),
            radius=7,
            fill=bias_fill,
            outline=bias_border,
            width=1,
        )
        draw.text(
            (cx1 - 112, cy0 + 150),
            bias.upper(),
            font=fonts.bold(13),
            fill=bias_txt,
            anchor="mm",
        )

        card_y += card_h + card_gap

    return panel


def build_stage2_rows_and_shell(
    brief_data: dict[str, Any], fonts: HudFonts
) -> tuple[Image.Image, list[tuple[str, str, str, tuple[int, int, int]]]]:
    """Pre-renders Option D Stage 2 cyber terminal shell bound to Main Event (Slot #3)."""
    t3 = brief_data.get("t3", brief_data.get("t1", {}))
    ticker = str(t3.get("ticker", "T3"))
    company = str(t3.get("company_name", ticker)).upper()
    haircut = bool(
        t3.get("haircut_applied", False)
        or t3.get("sec_form4_null", False)
        or (t3.get("finnhub_mspr") is None)
    )

    w = LOWER_PANEL_BOX[2] - LOWER_PANEL_BOX[0]  # 1044
    h = LOWER_PANEL_BOX[3] - LOWER_PANEL_BOX[1]  # 662
    shell = Image.new("RGB", (w, h), (6, 13, 24))
    draw = ImageDraw.Draw(shell)

    draw.rounded_rectangle((0, 0, w - 1, h - 1), radius=14, fill=(6, 13, 24), outline=(0, 229, 255), width=2)
    draw.rectangle((18, 0, w - 19, 4), fill=(0, 229, 255))

    # Subtle scanline texture across terminal interior
    for sy in range(96, 588, 6):
        draw.line([(16, sy), (w - 17, sy)], fill=(9, 19, 34), width=1)

    # Header band
    draw.text(
        (28, 34),
        f"VAULT RECON // ${ticker}",
        font=fonts.bold(28),
        fill=(0, 229, 255),
        anchor="lm",
    )
    draw.text(
        (28, 66),
        f"{company}  •  DETERMINISTIC QUANT TELEMETRY",
        font=fonts.medium(15),
        fill=(148, 175, 205),
        anchor="lm",
    )

    if haircut:
        badge_txt = "[-20% HAIRCUT APPLIED]"
        b_fill, b_out, b_col = (54, 34, 8), (245, 158, 11), (253, 224, 71)
    else:
        badge_txt = "[DATA VERIFIED // 1.00x]"
        b_fill, b_out, b_col = (7, 42, 48), (0, 229, 255), (165, 243, 252)

    draw.rounded_rectangle((w - 300, 22, w - 24, 64), radius=9, fill=b_fill, outline=b_out, width=2)
    draw.text((w - 162, 43), badge_txt, font=fonts.mono(15), fill=b_col, anchor="mm")

    # Table column header bar
    draw.rounded_rectangle((22, 92, w - 23, 128), radius=6, fill=(10, 24, 44), outline=(22, 72, 106), width=1)
    draw.text((38, 110), "TELEMETRY FIELD", font=fonts.bold(14), fill=(110, 185, 225), anchor="lm")
    draw.text((430, 110), "QUANT READOUT", font=fonts.bold(14), fill=(110, 185, 225), anchor="lm")
    draw.text((w - 38, 110), "VERIFICATION", font=fonts.bold(14), fill=(110, 185, 225), anchor="rm")

    spot_price = float(t3.get("spot_price", 0.0))
    atr_14d = float(t3.get("atr_14d", 0.0))
    atr_pct = float(t3.get("atr_14d_pct", 0.0))
    stop_2x = float(t3.get("dynamic_stop_2x_atr", 0.0))
    sma_20d = float(t3.get("sma_20d", 0.0))
    target_16b = float(t3.get("model_target_5d", 0.0))
    conf_score = float(t3.get("adjusted_confluence_score", 0.0))
    net_bias = str(t3.get("net_quant_bias", "Neutral"))
    align_str = str(t3.get("sma_20d_alignment", "")).lower()
    sma_tag = "BELOW 20D SMA" if "below" in align_str or spot_price < sma_20d else "ABOVE 20D SMA"
    sma_col = (251, 191, 36) if "BELOW" in sma_tag else (52, 211, 153)

    target_src = str(t3.get("model_target_source", "chronos")).split(":")[-1].upper()
    conf_tag = "HAIRCUT -20%" if haircut else "SEC/FINNHUB VERIFIED"
    conf_col = (245, 158, 11) if haircut else (0, 229, 255)

    bias_lower = net_bias.lower()
    if "bull" in bias_lower:
        bias_col = (52, 211, 153)
    elif "bear" in bias_lower:
        bias_col = (251, 113, 133)
    else:
        bias_col = (253, 224, 71)

    rows: list[tuple[str, str, str, tuple[int, int, int]]] = [
        ("01  SPOT PRICE", _fmt_currency(spot_price), "LIVE CACHE // VERIFIED", (248, 250, 252)),
        ("02  14-DAY ATR", f"{_fmt_currency(atr_14d)} ({atr_pct:.2f}%)", "VOLATILITY BAND", (56, 189, 248)),
        ("03  DYNAMIC STOP (2X)", _fmt_currency(stop_2x), "RISK INVALIDATION", (251, 146, 60)),
        ("04  20-DAY SMA", _fmt_currency(sma_20d), sma_tag, sma_col),
        ("05  16-BAR TARGET", _fmt_currency(target_16b), f"MODEL // {target_src}", (167, 243, 208)),
        ("06  CONFLUENCE SCORE", f"{conf_score:.2f}%", conf_tag, conf_col),
        ("07  NET QUANT BIAS", net_bias.upper(), "CHRONOS + SWARM", bias_col),
    ]
    return shell, rows


def render_stage2_terminal_state(
    shell: Image.Image,
    rows: list[tuple[str, str, str, tuple[int, int, int]]],
    reveal_counts: tuple[int, ...],
    cursor_row: int,
    fonts: HudFonts,
) -> Image.Image:
    """Renders cached Stage 2 terminal image for a given progressive reveal state."""
    img = shell.copy()
    draw = ImageDraw.Draw(img)
    w = img.width
    y0 = 138
    row_h = 58
    row_gap = 6

    for idx, (label, val_txt, tag_txt, val_col) in enumerate(rows):
        ry0 = y0 + idx * (row_h + row_gap)
        ry1 = ry0 + row_h
        revealed = reveal_counts[idx]
        bg = (10, 22, 40) if idx % 2 == 0 else (8, 17, 32)
        border = (0, 180, 220) if revealed > 0 else (18, 44, 72)
        draw.rounded_rectangle((22, ry0, w - 23, ry1), radius=8, fill=bg, outline=border, width=1)
        if revealed <= 0:
            draw.text((38, (ry0 + ry1) // 2), f"{label}  ...", font=fonts.mono(16), fill=(52, 78, 108), anchor="lm")
            continue

        draw.rectangle((22, ry0 + 10, 27, ry1 - 10), fill=val_col)
        draw.text((40, (ry0 + ry1) // 2), label, font=fonts.bold(17), fill=(176, 204, 232), anchor="lm")

        shown_val = val_txt[:revealed]
        if idx == cursor_row and revealed < len(val_txt):
            shown_val += "▌"
        draw.text((430, (ry0 + ry1) // 2), shown_val, font=fonts.mono(22), fill=val_col, anchor="lm")

        if revealed >= len(val_txt):
            draw.rounded_rectangle(
                (w - 255, ry0 + 12, w - 34, ry1 - 12),
                radius=6,
                fill=(12, 32, 56),
                outline=(34, 105, 146),
                width=1,
            )
            draw.text((w - 144, (ry0 + ry1) // 2), tag_txt, font=fonts.mono(13), fill=(165, 228, 255), anchor="mm")

    return img


def stamp_stage2_progress_bar(
    panel_img: Image.Image, elapsed_stage2: float, part2_sec: float, fonts: HudFonts
) -> Image.Image:
    """Draws live Part 2 Julie telemetry sync bar at bottom of Stage 2 terminal."""
    out = panel_img.copy()
    draw = ImageDraw.Draw(out)
    w = out.width
    pct = min(1.0, max(0.0, elapsed_stage2 / max(1e-3, part2_sec)))
    bar_box = (22, 598, w - 23, 646)
    draw.rounded_rectangle(bar_box, radius=8, fill=(9, 20, 38), outline=(22, 86, 120), width=1)

    fill_right = bar_box[0] + 4 + int((bar_box[2] - bar_box[0] - 8) * pct)
    if fill_right > bar_box[0] + 6:
        draw.rounded_rectangle(
            (bar_box[0] + 3, bar_box[1] + 3, fill_right, bar_box[3] - 3),
            radius=6,
            fill=(0, 145, 185),
        )
    draw.text(
        (40, (bar_box[1] + bar_box[3]) // 2),
        f"JULIE RECON STREAM // SYNC {int(round(pct * 100)):02d}%",
        font=fonts.bold(15),
        fill=(240, 250, 255),
        anchor="lm",
    )
    draw.text(
        (w - 40, (bar_box[1] + bar_box[3]) // 2),
        f"{min(elapsed_stage2, part2_sec):04.1f}s / {part2_sec:04.1f}s",
        font=fonts.mono(15),
        fill=(210, 245, 255),
        anchor="rm",
    )
    return out


def decode_bg_loop_frames(
    ffmpeg_bin: str,
    bg_video_path: Path,
    top_badge_patch: Image.Image,
    ticker_label_patch: Image.Image,
    subtitle_scrim: Image.Image,
) -> list[Image.Image]:
    """Decodes looped 9:16 video frames and pre-bakes static Layer 1/2/scrim elements."""
    proc = subprocess.run(
        [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(bg_video_path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        capture_output=True,
        timeout=60,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Failed decoding {bg_video_path}: {proc.stderr.decode('utf-8', 'replace')}")

    frame_size = CANVAS_WIDTH * CANVAS_HEIGHT * 3
    raw = proc.stdout
    if len(raw) < frame_size or len(raw) % frame_size != 0:
        raise RuntimeError(
            f"Unexpected decoded frame byte size from {bg_video_path}: {len(raw)} vs {frame_size}"
        )

    num_frames = len(raw) // frame_size
    frames: list[Image.Image] = []
    scrim_box = (0, BAND_Y_TOP_PX, CANVAS_WIDTH, BAND_Y_TOP_PX + BAND_HEIGHT_PX)
    for idx in range(num_frames):
        offset = idx * frame_size
        img = Image.frombytes("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), raw[offset : offset + frame_size])
        img.paste(top_badge_patch, (HEADER_BADGE_MASK_BOX[0], HEADER_BADGE_MASK_BOX[1]))
        img.paste(ticker_label_patch, (0, TICKER_Y_TOP))
        band_bg = img.crop(scrim_box).convert("RGBA")
        band_scrimmed = Image.alpha_composite(band_bg, subtitle_scrim).convert("RGB")
        img.paste(band_scrimmed, (0, BAND_Y_TOP_PX))
        frames.append(img)

    return frames


def find_bgm_track(songs_dir: Path, bgm_volume: float) -> Path | None:
    """Selects first deterministic MP3 BGM track when bgm_volume > 0."""
    if bgm_volume <= 0.0 or not songs_dir.is_dir():
        return None
    tracks = sorted(songs_dir.glob("*.mp3"))
    return tracks[0] if tracks else None


def render_final_video(
    brief_path: Path,
    audio_path: Path,
    bg_video_path: Path,
    output_dir: Path,
    bgm_volume: float,
) -> RenderManifest:
    """Executes full Phase 4 subtitle alignment, Option A/D HUD compositing, and H.264 encode."""
    resolved_output = output_dir.resolve()
    allowed_root = PROJECT_ROOT.resolve()
    if not resolved_output.is_relative_to(allowed_root):
        raise PermissionError(
            f"Write containment violation: output dir {resolved_output} is outside {allowed_root}"
        )
    resolved_output.mkdir(parents=True, exist_ok=True)

    ffmpeg_bin = resolve_ffmpeg_tool("ffmpeg")
    ffprobe_bin = resolve_ffmpeg_tool("ffprobe")

    if not audio_path.exists():
        raise FileNotFoundError(f"Master dialogue audio missing: {audio_path}")
    if not bg_video_path.exists():
        raise FileNotFoundError(f"Background 9:16 video missing: {bg_video_path}")
    if not DEFAULT_FONT_PATH.exists():
        raise FileNotFoundError(f"Caption font missing: {DEFAULT_FONT_PATH}")

    audio_info = probe_media_info(ffprobe_bin, audio_path)
    target_duration_sec = round(float(audio_info["format"]["duration"]), 3)
    total_frames = max(1, int(round(target_duration_sec * TARGET_FPS)))

    brief_data, julie_start_sec, handoff_sec, part2_sec = load_hud_context(
        brief_path, audio_path, target_duration_sec
    )
    run_date_badge = str(
        brief_data.get("run_date_badge")
        or datetime.now(ZoneInfo("America/Toronto")).strftime("%d %b %Y").upper()
    )
    t1_ticker = str(brief_data.get("t1", {}).get("ticker", "T1"))

    hud_fonts = HudFonts(bold_by_size={}, medium_by_size={}, mono_by_size={})
    caption_font = hud_fonts.bold(FONT_SIZE_PX)
    tokens = transcribe_and_align(audio_path, brief_path)
    cards = build_subtitle_cards(tokens, caption_font, target_duration_sec)

    srt_path = resolved_output / "master_dialogue.srt"
    mp4_path = resolved_output / "tiktok_ready.mp4"
    manifest_path = resolved_output / "render_manifest.json"
    write_srt(cards, srt_path)

    card_bands = [render_card_overlay_band(card, caption_font) for card in cards]
    rms_env = compute_frame_rms(audio_path, ffmpeg_bin, total_frames, TARGET_FPS)

    top_badge_patch = build_top_badge_patch(run_date_badge, t1_ticker, hud_fonts)
    ticker_label_patch, ticker_tile = build_ticker_assets(brief_data, hud_fonts)
    subtitle_scrim = build_subtitle_scrim()
    julie_badge_s1 = build_julie_status_badge(1, hud_fonts)
    julie_badge_s2 = build_julie_status_badge(2, hud_fonts)
    stage1_card_s1 = build_stage1_lower_card(brief_data, hud_fonts, active_slot=1)
    stage1_card_s2 = build_stage1_lower_card(brief_data, hud_fonts, active_slot=2)
    stage1_card_s3 = build_stage1_lower_card(brief_data, hud_fonts, active_slot=3)
    stage2_shell, stage2_rows = build_stage2_rows_and_shell(brief_data, hud_fonts)

    ep_id = int(brief_data.get("episode_id", 2))
    t2_ticker = str(brief_data.get("t2", {}).get("ticker", "T2")).upper()
    t3_ticker = str(brief_data.get("t3", {}).get("ticker", "T3")).upper()
    rotation_mode = str(brief_data.get("rotation_mode") or ROTATING_MODES_MAP[ep_id % 3])
    haircut_t3 = bool(
        brief_data.get("t3", {}).get("haircut_applied", False)
        or brief_data.get("t3", {}).get("sec_form4_null", False)
        or (brief_data.get("t3", {}).get("finnhub_mspr") is None)
    )
    cover_pill_text = str(brief_data.get("cover_pill_text") or f"EPISODE #{ep_id:02d} | LIVE TRACK RECORD")
    frame0_cover_card = build_frame0_cover_card(t1_ticker, t2_ticker, t3_ticker, cover_pill_text, hud_fonts)
    if rotation_mode == "trajectory_chart":
        chart_base_panel, chart_full_panel, chart_plot_box = build_progressive_trajectory_layers(brief_data, hud_fonts)
        rotating_segment_panel = None
    else:
        chart_base_panel, chart_full_panel, chart_plot_box = None, None, (0, 0, 0, 0)
        rotating_segment_panel = build_rotating_segment_panel(brief_data, hud_fonts).convert("RGB")
    banner_s1 = build_3stock_radar_banner(ep_id, t1_ticker, t2_ticker, t3_ticker, 1, hud_fonts)
    banner_s2 = build_3stock_radar_banner(ep_id, t1_ticker, t2_ticker, t3_ticker, 2, hud_fonts)
    banner_s3 = build_3stock_radar_banner(ep_id, t1_ticker, t2_ticker, t3_ticker, 3, hud_fonts)
    compliance_watermark = build_compliance_watermark(hud_fonts)

    bg_frames = decode_bg_loop_frames(
        ffmpeg_bin, bg_video_path, top_badge_patch, ticker_label_patch, subtitle_scrim
    )
    n_bg = len(bg_frames)

    bgm_track = find_bgm_track(DEFAULT_SONGS_DIR, bgm_volume)
    cmd = [
        ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{CANVAS_WIDTH}x{CANVAS_HEIGHT}",
        "-r",
        str(TARGET_FPS),
        "-i",
        "pipe:0",
        "-i",
        str(audio_path),
    ]
    if bgm_track is not None:
        cmd.extend(
            [
                "-stream_loop",
                "-1",
                "-i",
                str(bgm_track),
                "-filter_complex",
                f"[2:a]volume={bgm_volume:.3f}[bgm];[1:a][bgm]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[outa]",
                "-map",
                "0:v:0",
                "-map",
                "[outa]",
            ]
        )
    else:
        cmd.extend(["-map", "0:v:0", "-map", "1:a:0"])

    cmd.extend(
        [
            "-c:v",
            "libx264",
            "-crf",
            str(VIDEO_CRF),
            "-preset",
            "fast",
            "-pix_fmt",
            "yuv420p",
            "-colorspace",
            "bt709",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-r",
            str(TARGET_FPS),
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "44100",
            "-ac",
            "2",
            "-t",
            f"{target_duration_sec:.3f}",
            "-movflags",
            "+faststart",
            str(mp4_path),
        ]
    )

    stage2_row_cache: dict[tuple[tuple[int, ...], int], Image.Image] = {}
    reveal_window_sec = max(6.0, min(21.0, part2_sec * 0.55) - STAGE2_SWEEP_DURATION_SEC)
    row_slot_sec = reveal_window_sec / float(len(stage2_rows))
    row_type_sec = min(2.2, row_slot_sec * 0.85)
    lower_w = LOWER_PANEL_BOX[2] - LOWER_PANEL_BOX[0]
    lower_h = LOWER_PANEL_BOX[3] - LOWER_PANEL_BOX[1]
    scrim_box = (0, BAND_Y_TOP_PX, CANVAS_WIDTH, BAND_Y_TOP_PX + BAND_HEIGHT_PX)

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    card_idx = 0
    num_cards = len(cards)
    try:
        for frame_idx in range(total_frames):
            t_sec = (frame_idx + 0.5) / TARGET_FPS
            rms = rms_env[frame_idx]
            active_julie = t_sec >= julie_start_sec
            while card_idx < num_cards and t_sec > cards[card_idx].end:
                card_idx += 1

            frame = bg_frames[frame_idx % n_bg].copy()

            # Layer 2: Scrolling live ticker strip (120 px/sec = 4 px/frame)
            scroll_x = (frame_idx * TICKER_SPEED_PX_PER_FRAME) % CANVAS_WIDTH
            ticker_slice = ticker_tile.crop((scroll_x, 0, scroll_x + TICKER_SCROLL_WIDTH, TICKER_HEIGHT))
            frame.paste(ticker_slice, (TICKER_LABEL_WIDTH, TICKER_Y_TOP))

            # Layer 3: Audio-reactive Julie HUD
            badge_img = julie_badge_s2 if active_julie else julie_badge_s1
            frame.paste(badge_img.convert("RGB"), (JULIE_STATUS_BOX[0], JULIE_STATUS_BOX[1]))

            rings_overlay = render_julie_rings_overlay(t_sec, rms, active_julie)
            rings_bg = frame.crop(JULIE_RING_BOX).convert("RGBA")
            frame.paste(
                Image.alpha_composite(rings_bg, rings_overlay).convert("RGB"),
                (JULIE_RING_BOX[0], JULIE_RING_BOX[1]),
            )

            eq_overlay = render_julie_eq_overlay(t_sec, rms, active_julie)
            eq_bg = frame.crop(JULIE_EQ_BOX).convert("RGBA")
            frame.paste(
                Image.alpha_composite(eq_bg, eq_overlay).convert("RGB"),
                (JULIE_EQ_BOX[0], JULIE_EQ_BOX[1]),
            )

            # Layer 4: Active Slot Spotlight, Smooth Transitions & Progressive 14s Chart Draw-In
            if t_sec < 10.5:
                lower_out = stage1_card_s1
            elif t_sec < 11.0:
                lower_out = blend_panel_transition(stage1_card_s1, stage1_card_s2, (t_sec - 10.5) / 0.5, (0, 229, 255))
            elif t_sec < 16.0:
                lower_out = stage1_card_s2
            elif t_sec < 16.5:
                lower_out = blend_panel_transition(stage1_card_s2, stage1_card_s3, (t_sec - 16.0) / 0.5, (251, 191, 36))
            elif t_sec < 18.0:
                lower_out = stage1_card_s3
            elif 18.0 <= t_sec <= 32.0:
                if rotation_mode == "trajectory_chart" and chart_base_panel is not None and chart_full_panel is not None:
                    chart_prog = (t_sec - 18.0) / 14.0
                    mid_panel = render_progressive_chart_frame(
                        chart_base_panel, chart_full_panel, chart_plot_box, chart_prog, haircut_t3, hud_fonts
                    )
                else:
                    assert rotating_segment_panel is not None
                    mid_panel = rotating_segment_panel

                if t_sec < 18.45:
                    lower_out = blend_panel_transition(stage1_card_s3, mid_panel, (t_sec - 18.0) / 0.45, (0, 229, 255))
                elif t_sec > 31.55:
                    lower_out = blend_panel_transition(mid_panel, stage1_card_s3, (t_sec - 31.55) / 0.45, (251, 191, 36))
                else:
                    lower_out = mid_panel
            elif not active_julie:
                lower_out = stage1_card_s3
            else:
                elapsed_s2 = t_sec - julie_start_sec
                elapsed_type = max(0.0, elapsed_s2 - STAGE2_SWEEP_DURATION_SEC)
                reveals: list[int] = []
                cursor_row = -1
                for r_idx, (_, val_txt, _, _) in enumerate(stage2_rows):
                    r_elapsed = elapsed_type - r_idx * row_slot_sec
                    full_len = len(val_txt)
                    if r_elapsed <= 0.0:
                        reveals.append(0)
                    elif r_elapsed >= row_type_sec:
                        reveals.append(full_len)
                    else:
                        frac = r_elapsed / max(0.1, row_type_sec)
                        chars = min(full_len, max(2, int(math.ceil(frac * full_len / 2.0) * 2)))
                        reveals.append(chars)
                        cursor_row = r_idx
                cache_key = (tuple(reveals), cursor_row)
                if cache_key not in stage2_row_cache:
                    stage2_row_cache[cache_key] = render_stage2_terminal_state(
                        stage2_shell, stage2_rows, tuple(reveals), cursor_row, hud_fonts
                    )
                stage2_panel = stamp_stage2_progress_bar(
                    stage2_row_cache[cache_key], elapsed_s2, part2_sec, hud_fonts
                )

                if elapsed_s2 < STAGE2_SWEEP_DURATION_SEC:
                    sweep_frac = min(1.0, max(0.0, elapsed_s2 / STAGE2_SWEEP_DURATION_SEC))
                    scan_x = max(1, min(lower_w - 1, int(lower_w * sweep_frac)))
                    blend = stage1_card_s3.copy()
                    blend.paste(stage2_panel.crop((0, 0, scan_x, lower_h)), (0, 0))
                    draw_beam = ImageDraw.Draw(blend)
                    beam_x0 = max(0, scan_x - 10)
                    beam_x1 = min(lower_w - 1, scan_x + 4)
                    draw_beam.rectangle((beam_x0, 0, beam_x1, lower_h - 1), fill=(0, 229, 255))
                    draw_beam.rectangle(
                        (max(0, scan_x - 2), 0, min(lower_w - 1, scan_x + 1), lower_h - 1),
                        fill=(235, 253, 255),
                    )
                    lower_out = blend
                else:
                    lower_out = stage2_panel

            frame.paste(lower_out, (LOWER_PANEL_BOX[0], LOWER_PANEL_BOX[1]))

            # Layer 5: Burned-in captions over darkened subtitle deck (860..1120)
            if card_idx < num_cards and cards[card_idx].start <= t_sec <= cards[card_idx].end:
                band_bg = frame.crop(scrim_box).convert("RGBA")
                composited = Image.alpha_composite(band_bg, card_bands[card_idx]).convert("RGB")
                frame.paste(composited, (0, BAND_Y_TOP_PX))

            # Layer 6: Frame-0 3-Ticker Cover Card (0.0s - 3.5s) -> Cross-Faded Active Radar Banner (>= 3.5s)
            if t_sec < 3.5:
                if t_sec < 2.8:
                    cx, cy, cw, ch = 60, 420, 880, 260
                    cover_img = frame0_cover_card
                else:
                    p_dock = min(1.0, max(0.0, (t_sec - 2.8) / 0.7))
                    ease = 1.0 - (1.0 - p_dock) ** 2
                    cx = int(round(60 + (RADAR_BANNER_BOX[0] - 60) * ease))
                    cy = int(round(420 + (RADAR_BANNER_BOX[1] - 420) * ease))
                    cw = int(round(880 + (1000 - 880) * ease))
                    ch = int(round(260 + (56 - 260) * ease))
                    cover_img = frame0_cover_card.resize((cw, ch), Image.Resampling.LANCZOS)
                cover_bg = frame.crop((cx, cy, cx + cw, cy + ch)).convert("RGBA")
                frame.paste(Image.alpha_composite(cover_bg, cover_img).convert("RGB"), (cx, cy))
            else:
                if t_sec < 10.5:
                    active_banner = banner_s1
                    slot_pulse_col = (0, 229, 255)
                elif t_sec < 11.0:
                    active_banner = Image.blend(banner_s1, banner_s2, (t_sec - 10.5) / 0.5)
                    slot_pulse_col = (0, 229, 255)
                elif t_sec < 16.0:
                    active_banner = banner_s2
                    slot_pulse_col = (0, 229, 255)
                elif t_sec < 16.5:
                    active_banner = Image.blend(banner_s2, banner_s3, (t_sec - 16.0) / 0.5)
                    slot_pulse_col = (251, 191, 36)
                else:
                    active_banner = banner_s3
                    slot_pulse_col = (251, 191, 36)
                banner_rgb = active_banner.convert("RGB")
                if rms > 0.35:
                    draw_ban = ImageDraw.Draw(banner_rgb)
                    draw_ban.rounded_rectangle(
                        (0, 0, banner_rgb.width - 1, banner_rgb.height - 1),
                        radius=12,
                        outline=slot_pulse_col,
                        width=3,
                    )
                frame.paste(banner_rgb, (RADAR_BANNER_BOX[0], RADAR_BANNER_BOX[1]))

            # Layer 7: Permanent Bottom-Left Compliance Watermark (safe zone bottom >= 320px)
            frame.paste(compliance_watermark.convert("RGB"), (COMPLIANCE_WATERMARK_BOX[0], COMPLIANCE_WATERMARK_BOX[1]))

            proc.stdin.write(frame.tobytes())
    finally:
        proc.stdin.close()

    stderr_bytes = proc.stderr.read() if proc.stderr else b""
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(
            f"FFmpeg composite encode failed (rc={rc}):\n{stderr_bytes.decode('utf-8', 'replace')}"
        )

    out_info = probe_media_info(ffprobe_bin, mp4_path)
    rendered_duration_sec = round(float(out_info["format"]["duration"]), 3)
    v_stream = next(s for s in out_info["streams"] if s["codec_type"] == "video")
    a_stream = next(s for s in out_info["streams"] if s["codec_type"] == "audio")

    manifest = RenderManifest(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        video_output_path=str(mp4_path),
        srt_output_path=str(srt_path),
        source_audio_path=str(audio_path),
        source_bg_video_path=str(bg_video_path),
        bgm_track_path=str(bgm_track) if bgm_track else None,
        bgm_volume=bgm_volume if bgm_track else 0.0,
        whisper_model=WHISPER_MODEL_ID,
        whisper_device=WHISPER_DEVICE,
        whisper_compute_type=WHISPER_COMPUTE_TYPE,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", "1"),
        width=int(v_stream["width"]),
        height=int(v_stream["height"]),
        fps=TARGET_FPS,
        video_codec=str(v_stream["codec_name"]),
        audio_codec=str(a_stream["codec_name"]),
        pixel_format=str(v_stream["pix_fmt"]),
        colorspace=str(v_stream.get("color_space") or "bt709"),
        crf=VIDEO_CRF,
        target_duration_sec=target_duration_sec,
        rendered_duration_sec=rendered_duration_sec,
        total_frames=total_frames,
        subtitle_card_count=len(cards),
        subtitle_y_center_px=SUBTITLE_Y_CENTER_PX,
        font_path=str(DEFAULT_FONT_PATH),
        font_size_px=FONT_SIZE_PX,
        run_date_badge=run_date_badge,
        t1_ticker=t1_ticker,
        julie_handoff_sec=round(julie_start_sec, 3),
    )
    with manifest_path.open(mode="w", encoding="utf-8") as handle:
        json.dump(asdict(manifest), handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    return manifest


def render_verification_frame(
    brief_path: Path,
    output_path: Path,
    slot: int = 1,
) -> Path:
    """Renders a single static verification frame (CPU-only, 0 MiB GPU compute)."""
    with brief_path.open("r", encoding="utf-8") as handle:
        brief_data = json.load(handle)
    hud_fonts = HudFonts(bold_by_size={}, medium_by_size={}, mono_by_size={})

    ep_id = int(brief_data.get("episode_id", 2))
    t1_tk = str(brief_data.get("t1", {}).get("ticker", "ACM"))
    t2_tk = str(brief_data.get("t2", {}).get("ticker", "NVO"))
    t3_tk = str(brief_data.get("t3", {}).get("ticker", "UBER"))
    run_date_badge = str(brief_data.get("run_date_badge", "LIVE"))

    if WARROOM_SOURCE_IMAGE_PATH.exists():
        with Image.open(WARROOM_SOURCE_IMAGE_PATH) as src_img:
            frame = src_img.convert("RGB").resize((CANVAS_WIDTH, CANVAS_HEIGHT), Image.Resampling.LANCZOS)
    else:
        frame = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), (6, 11, 20))

    top_badge_patch = build_top_badge_patch(run_date_badge, t1_tk, hud_fonts)
    frame.paste(top_badge_patch, (HEADER_BADGE_MASK_BOX[0], HEADER_BADGE_MASK_BOX[1]))

    ticker_label_patch, ticker_tile = build_ticker_assets(brief_data, hud_fonts)
    frame.paste(ticker_label_patch, (0, TICKER_Y_TOP))
    frame.paste(ticker_tile.crop((0, 0, TICKER_SCROLL_WIDTH, TICKER_HEIGHT)), (TICKER_LABEL_WIDTH, TICKER_Y_TOP))

    banner = build_3stock_radar_banner(ep_id, t1_tk, t2_tk, t3_tk, slot, hud_fonts)
    frame.paste(banner.convert("RGB"), (RADAR_BANNER_BOX[0], RADAR_BANNER_BOX[1]))

    stage1_lower = build_stage1_lower_card(brief_data, hud_fonts, active_slot=slot)
    frame.paste(stage1_lower, (LOWER_PANEL_BOX[0], LOWER_PANEL_BOX[1]))

    watermark = build_compliance_watermark(hud_fonts)
    frame.paste(watermark.convert("RGB"), (COMPLIANCE_WATERMARK_BOX[0], COMPLIANCE_WATERMARK_BOX[1]))

    subtitle_scrim = build_subtitle_scrim()
    band_bg = frame.crop((0, BAND_Y_TOP_PX, CANVAS_WIDTH, BAND_Y_TOP_PX + BAND_HEIGHT_PX)).convert("RGBA")
    scrimmed = Image.alpha_composite(band_bg, subtitle_scrim).convert("RGB")
    frame.paste(scrimmed, (0, BAND_Y_TOP_PX))

    caption_font = hud_fonts.bold(FONT_SIZE_PX)
    sample_text = f"AUDITING ${t1_tk} • ${t2_tk} • ${t3_tk}"
    draw_cap = ImageDraw.Draw(frame)
    draw_cap.text((CANVAS_WIDTH // 2, SUBTITLE_Y_CENTER_PX), sample_text, font=caption_font, fill=(255, 255, 255), anchor="mm")

    resolved_out = output_path.resolve()
    resolved_out.parent.mkdir(parents=True, exist_ok=True)
    frame.save(resolved_out)
    return resolved_out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses CLI flags for Phase 4 headless composite renderer."""
    parser = argparse.ArgumentParser(
        description="Headless 9:16 Video & Subtitle Compositor (Phase 4)"
    )
    parser.add_argument("--brief", type=Path, default=DEFAULT_BRIEF_PATH)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO_PATH)
    parser.add_argument("--bg-video", type=Path, default=DEFAULT_BG_VIDEO_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--bgm-volume",
        type=float,
        default=0.06,
        help="Subtle BGM mix level under mastered dialogue (0.0 disables BGM; default 0.06).",
    )
    parser.add_argument(
        "--test-frame",
        type=Path,
        default=None,
        help="Render a single CPU-only HUD verification frame to this path and exit.",
    )
    parser.add_argument(
        "--test-slot",
        type=int,
        default=1,
        help="Active pill slot for test-frame (1, 2, or 3; default: 1).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = parse_args(argv)
    if args.test_frame is not None:
        out_frame = render_verification_frame(
            brief_path=args.brief.resolve(),
            output_path=args.test_frame.resolve(),
            slot=args.test_slot,
        )
        print(f"[render_video] Verification frame rendered (CPU-only): {out_frame}")
        return 0

    manifest = render_final_video(
        brief_path=args.brief.resolve(),
        audio_path=args.audio.resolve(),
        bg_video_path=args.bg_video.resolve(),
        output_dir=args.output_dir.resolve(),
        bgm_volume=max(0.0, float(args.bgm_volume)),
    )
    print(f"[render_video] SRT: {manifest.srt_output_path} ({manifest.subtitle_card_count} cards)")
    print(
        f"[render_video] MP4: {manifest.video_output_path} "
        f"({manifest.width}x{manifest.height}, {manifest.video_codec}/{manifest.audio_codec}, "
        f"{manifest.rendered_duration_sec:.3f}s)"
    )
    print(f"[render_video] Manifest: {args.output_dir.resolve() / 'render_manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
