#!/usr/bin/env python3
"""pipeline/generate_script.py
====================================================================
Phase 2B: 3-Beat Ping-Pong Script Generator & Compliance Lexicon Shield.

Generates the broadcast script between Pierre and Julie following the
'tiktok-quant-growth' 5-rule protocol:
  - 0.0s–4.5s Hook: Pierre credibility, prior receipt scorecard, 3 tickers.
  - 4.5s–18.0s Beat 1: Rapid Radar on Stocks #1 & #2 (6-7s back-and-forth).
  - 18.0s–47.0s Beat 2: Main Event Deep Dive on Stock #3 (Model Clash,
    haircut flags, and Pierre's 14D ATR invalidation lock).
  - 47.0s–56.5s Beat 3: Vault Lock, compliance sign-off & binary comment question.
Validates dialogue against compliance lexicon and outputs tiktok_caption.txt.
====================================================================
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Any

from episode_tracker import DEFAULT_LEDGER_PATH, grade_previous_episode, load_ledger

PROJECT_ROOT = Path(r"C:\Anti-Gravity-Core\pierre-training-video")
PIPELINE_DIR = PROJECT_ROOT / "pipeline"
OUTPUT_DIR = PIPELINE_DIR / "output"
DEFAULT_BRIEF_PATH = PIPELINE_DIR / "latest_brief.json"
DEFAULT_CAPTION_PATH = OUTPUT_DIR / "tiktok_caption.txt"

ESTIMATED_WORDS_PER_SECOND = 3.05
MAX_VIDEO_DURATION_CEILING_SEC = 57.5

PROHIBITED_TERMS_MAP: dict[str, str] = {
    r"\bbuy now\b": "monitor for accumulation",
    r"\bsell now\b": "monitor for defensive trim",
    r"\byou should buy\b": "our quantitative models track",
    r"\byou should enter\b": "our models flag confluence",
    r"\byou should sell\b": "our system detects defensive rotation",
    r"\byou should\b": "systems telemetry indicates",
    r"\bguaranteed profit\b": "statistical confluence",
    r"\bguaranteed\b": "projected",
    r"\bsure thing\b": "high confluence setup",
    r"\bmoon\b": "upside breakout",
}


ROTATION_MODES: dict[int, str] = {
    0: "catalyst_flow",
    1: "trajectory_chart",
    2: "trio_matrix",
}


@dataclass(slots=True, frozen=True)
class DialogueScript:
    episode_id: int
    pierre_script: str
    julie_script: str
    pierre_words: int
    julie_words: int
    total_words: int
    estimated_duration_sec: float
    full_combined_transcript: str
    compliance_clean: bool
    hook_state: str
    cover_pill_text: str
    rotation_mode: str


def _fmt_price_spoken(val: float) -> str:
    """Formats float price into natural spoken English."""
    if val >= 1000.0:
        return f"{round(val):,} dollars".replace(",", " thousand ")
    dollars = int(val)
    cents = round((val - dollars) * 100)
    if cents == 0:
        return f"{dollars} dollars"
    return f"{dollars} dollars and {cents} cents"


def _fmt_pct_spoken(pct: float) -> str:
    """Formats percentage change into natural spoken words."""
    rounded = abs(round(pct, 1))
    int_part = int(rounded)
    dec_part = round((rounded - int_part) * 10)
    word = f"{int_part} point {dec_part}" if dec_part > 0 else f"{int_part}"
    return f"{word} percent"


def sanitize_compliance_lexicon(text: str) -> tuple[str, list[str]]:
    """Deterministic Lexicon Validator: replaces prohibited imperative financial terms."""
    violations: list[str] = []
    sanitized = text
    for pattern, replacement in PROHIBITED_TERMS_MAP.items():
        if re.search(pattern, sanitized, flags=re.IGNORECASE):
            violations.append(pattern)
            sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
    return sanitized, violations


def build_3beat_dialogue(brief_data: dict[str, Any], ledger_path: Path = DEFAULT_LEDGER_PATH) -> DialogueScript:
    """Composes the deterministic 3-beat dialogue script adhering to the 45s-57.5s ceiling."""
    ledger = load_ledger(ledger_path)
    episodes: list[dict[str, Any]] = ledger.get("episodes", [])
    next_id = int(episodes[-1].get("episode_id", 0)) + 1 if episodes else 1

    receipt = grade_previous_episode(ledger_path)

    t1 = brief_data.get("t1", {})
    t2 = brief_data.get("t2", {})
    t3 = brief_data.get("t3", {})

    tk1 = str(t1.get("ticker", "ACM")).upper()
    tk2 = str(t2.get("ticker", "NVO")).upper()
    tk3 = str(t3.get("ticker", "UBER")).upper()

    name1 = str(t1.get("spoken_short", tk1))
    name2 = str(t2.get("spoken_short", tk2))
    name3 = str(t3.get("spoken_short", tk3))

    spot1 = _fmt_price_spoken(float(t1.get("spot_price", 0.0)))
    spot2 = _fmt_price_spoken(float(t2.get("spot_price", 0.0)))
    spot3 = _fmt_price_spoken(float(t3.get("spot_price", 0.0)))

    delta1_pct = float(t1.get("sma_20d_delta_pct", 0.0))
    delta1_word = "above" if delta1_pct >= 0 else "below"
    delta1_str = f"{_fmt_pct_spoken(delta1_pct)} {delta1_word}"

    delta2_pct = float(t2.get("sma_20d_delta_pct", 0.0))
    delta2_word = "above" if delta2_pct >= 0 else "below"
    delta2_str = f"{_fmt_pct_spoken(delta2_pct)} {delta2_word}"

    stop3 = _fmt_price_spoken(float(t3.get("dynamic_stop_2x_atr", 0.0)))
    target3 = _fmt_price_spoken(float(t3.get("model_target_5d", 0.0)))
    haircut3 = bool(t3.get("haircut_applied", False))
    has_clash = bool(t3.get("has_model_clash", False))
    rotation_mode = str(brief_data.get("rotation_mode_override") or ROTATION_MODES[next_id % 3])

    # --- 0.0s–3.5s: Percentage-Free 3-State FOMO Opener (Pierre) ---
    prev_tk = str(receipt.graded_ticker or "UBER").upper().lstrip("$")
    raw_outcome = str(receipt.outcome or "IN_PROGRESS").upper()
    if raw_outcome == "TARGET_HIT":
        hook_state = "TARGET_HIT"
        cover_pill_text = "YESTERDAY'S CALL: VECTOR CONFIRMED"
        hook_sentence = (
            f"Morning, I'm Pierre, Senior Engineer. Yesterday our Dual-GPU AI nailed the ${prev_tk} "
            f"vector right on target—here are the three setups the models just flagged for today: "
            f"${tk1}, ${tk2}, and ${tk3}."
        )
    elif raw_outcome in ("STOP_TRIGGERED", "STOPPED_OUT"):
        hook_state = "STOPPED_OUT"
        cover_pill_text = "FULL AUDIT LOG | RISK FLOOR HELD"
        hook_sentence = (
            f"Morning, I'm Pierre, Senior Engineer. Yesterday's ${prev_tk} setup tagged our protective "
            f"A-T-R stop before the drop—saved by the risk floor. Today our AI is auditing "
            f"${tk1}, ${tk2}, and ${tk3}."
        )
    else:
        hook_state = "ON_TRACK"
        cover_pill_text = f"EPISODE #{next_id:02d} | LIVE TRACK RECORD"
        hook_sentence = (
            f"Morning, I'm Pierre, Senior Engineer. Yesterday's ${prev_tk} trajectory is playing out "
            f"right on schedule—and today's pre-market scan just locked three fresh setups: "
            f"${tk1}, ${tk2}, and ${tk3}."
        )

    # --- Beat 1: Rapid Radar on Stocks #1 & #2 ---
    pierre_beat1 = (
        f"Stock one: {name1}, {tk1}, at {spot1}, holding {delta1_str} its twenty-day average. "
        f"Stock two: {name2}, {tk2}, at {spot2}, {delta2_str} trendline."
    )

    if has_clash:
        anomaly_str = "Models flag a clash between Times-F-M and Chronos-Bolt."
    elif haircut3:
        anomaly_str = "SEC Form 4 flow is opaque, triggering a mandatory twenty percent haircut."
    else:
        anomaly_str = "Smart money flow converges with sixteen-bar neural momentum."

    # --- Beat 2: Rotating Main Event Segment (~16s-38s) ---
    if rotation_mode == "trio_matrix":
        julie_beat1 = f"Pulling up our Trio Matrix across {tk1}, {tk2}, and {tk3}."
        julie_beat2 = (
            f"Agent zero-eight velocity and Agent zero-seven Z-scores flag {name3}, ticker {tk3}, "
            f"at {spot3}. {anomaly_str}"
        )
        pierre_beat2 = f"Agent zero-two locks our fourteen-day A-T-R floor at {stop3}, targeting {target3}."
    elif rotation_mode == "catalyst_flow":
        julie_beat1 = (
            f"Scanning Catalyst Flow across all three setups, focusing on {name3}, ticker {tk3}, at {spot3}."
        )
        julie_beat2 = f"{anomaly_str} Institutional and sector flow remain under active surveillance."
        pierre_beat2 = (
            f"With blindspots priced in, our fourteen-day A-T-R floor locks at {stop3}, targeting {target3}."
        )
    else:
        julie_beat1 = (
            f"Neural models align bullish on both. Shifting compute to our sixteen-bar trajectory chart "
            f"on {name3}, ticker {tk3}."
        )
        julie_beat2 = f"Spot sits at {spot3}. {anomaly_str} The sixteen-bar cone projects toward {target3}."
        pierre_beat2 = f"Our fourteen-day A-T-R floor locks at {stop3}, protecting the downside."

    # --- Beat 3: Vault Lock & Compliance Sign-Off ---
    pierre_beat3 = (
        "All three vectors are logged for tomorrow's audit—strictly quant telemetry, never financial advice. "
        "Which setup moves first?"
    )

    pierre_full = f"{hook_sentence} {pierre_beat1} {pierre_beat2} {pierre_beat3}"
    julie_full = f"{julie_beat1} {julie_beat2}"

    # Apply compliance lexicon sanitizer
    pierre_clean, v_p = sanitize_compliance_lexicon(pierre_full)
    julie_clean, v_j = sanitize_compliance_lexicon(julie_full)

    p_words = len(pierre_clean.split())
    j_words = len(julie_clean.split())
    total_words = p_words + j_words
    est_duration = round(total_words / ESTIMATED_WORDS_PER_SECOND, 2)

    # Pacing safeguard: if exceeding 57.0s, trim filler phrases
    if est_duration > 57.0:
        pierre_clean = pierre_clean.replace("our quantitative models track", "we track")
        julie_clean = julie_clean.replace("Shifting compute to", "Focusing on")
        total_words = len(pierre_clean.split()) + len(julie_clean.split())
        est_duration = round(total_words / ESTIMATED_WORDS_PER_SECOND, 2)

    combined = f"PIERRE: {pierre_clean}\n\nJULIE: {julie_clean}"

    return DialogueScript(
        episode_id=next_id,
        pierre_script=pierre_clean,
        julie_script=julie_clean,
        pierre_words=p_words,
        julie_words=j_words,
        total_words=total_words,
        estimated_duration_sec=est_duration,
        full_combined_transcript=combined,
        compliance_clean=(len(v_p) == 0 and len(v_j) == 0),
        hook_state=hook_state,
        cover_pill_text=cover_pill_text,
        rotation_mode=rotation_mode,
    )


def generate_tiktok_caption(
    episode_id: int,
    t1: str,
    t2: str,
    t3: str,
    receipt_ticker: str,
    receipt_outcome: str,
    receipt_delta_pct: float,
) -> str:
    """Formats standardized 4-line search SEO & regulatory compliance caption."""
    clean1, clean2, clean3 = t1.lstrip("$"), t2.lstrip("$"), t3.lstrip("$")
    prev_clean = receipt_ticker.lstrip("$")

    line1 = f"[EP #{episode_id:02d}] Dual-GPU AI Audit: ${clean1} vs ${clean2} vs ${clean3} — Which Setup Breaks First?"
    line2 = f"Yesterday's EP #{episode_id - 1:02d} Call: ${prev_clean} {receipt_outcome} ({receipt_delta_pct:+.1f}%)."
    line3 = f"#{clean1} #{clean2} #{clean3} #quanttrading #swingtradingstrategies #stockmarket"
    line4 = "⚠️ Educational quant software telemetry & personal market log only. Not financial or investment advice."

    return "\n".join([line1, line2, line3, line4])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 3-Beat Ping-Pong Script & Caption")
    parser.add_argument("--brief", type=Path, default=DEFAULT_BRIEF_PATH)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--caption-out", type=Path, default=DEFAULT_CAPTION_PATH)
    parser.add_argument("--dry-run", action="store_true", help="Preview script and caption without modifying files")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.brief.exists():
        print(f"[generate_script] Error: Brief file {args.brief} not found. Run extract_warroom_brief.py first.")
        return 1

    with args.brief.open("r", encoding="utf-8") as handle:
        brief_data = json.load(handle)

    receipt = grade_previous_episode(args.ledger)
    script_obj = build_3beat_dialogue(brief_data, args.ledger)

    t1_tk = str(brief_data.get("t1", {}).get("ticker", "ACM"))
    t2_tk = str(brief_data.get("t2", {}).get("ticker", "NVO"))
    t3_tk = str(brief_data.get("t3", {}).get("ticker", "UBER"))

    caption_text = generate_tiktok_caption(
        episode_id=script_obj.episode_id,
        t1=t1_tk,
        t2=t2_tk,
        t3=t3_tk,
        receipt_ticker=receipt.graded_ticker,
        receipt_outcome=receipt.outcome,
        receipt_delta_pct=receipt.delta_pct,
    )

    print(f"[generate_script] === EPISODE #{script_obj.episode_id:02d} GENERATED SCRIPT ===")
    print(f"  Rotation Mode: {script_obj.rotation_mode} | Hook State: {script_obj.hook_state}")
    print(f"  Duration Budget: {script_obj.estimated_duration_sec:.2f}s ({script_obj.total_words} words)")
    print(f"  Pierre Words: {script_obj.pierre_words} | Julie Words: {script_obj.julie_words}")
    print(f"  Compliance Lexicon: {'CLEAN (Zero imperative advisory terms)' if script_obj.compliance_clean else 'SANITIZED'}")
    print("\n--- SCRIPT TRANSCRIPT ---")
    print(script_obj.full_combined_transcript)
    print("\n--- 4-LINE SEARCH SEO CAPTION ---")
    print(caption_text)

    if not args.dry_run:
        brief_data["pierre_script"] = script_obj.pierre_script
        brief_data["julie_script"] = script_obj.julie_script
        brief_data["pierre_word_count"] = script_obj.pierre_words
        brief_data["julie_word_count"] = script_obj.julie_words
        brief_data["total_word_count"] = script_obj.total_words
        brief_data["total_estimated_duration_sec"] = script_obj.estimated_duration_sec
        brief_data["episode_id"] = script_obj.episode_id
        brief_data["hook_state"] = script_obj.hook_state
        brief_data["cover_pill_text"] = script_obj.cover_pill_text
        brief_data["rotation_mode"] = script_obj.rotation_mode
        brief_data["caption_text"] = caption_text

        with args.brief.open("w", encoding="utf-8") as handle:
            json.dump(brief_data, handle, indent=2, ensure_ascii=False)
            handle.write("\n")

        args.caption_out.parent.mkdir(parents=True, exist_ok=True)
        with args.caption_out.open("w", encoding="utf-8") as handle:
            handle.write(caption_text + "\n")
        print(f"\n[generate_script] Saved updated dialogue to {args.brief} and caption to {args.caption_out}")
    else:
        print("\n[generate_script] DRY RUN: Files not modified.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
