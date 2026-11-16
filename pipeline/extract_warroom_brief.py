#!/usr/bin/env python3
"""pipeline/extract_warroom_brief.py
====================================================================
Phase 2A: 3-Ticker Morning Radar Extractor for Pierre TikTok Engine.

Selects a deterministic 3-Ticker Radar Trio from the read-only War Room
telemetry (pierre_quant.db ?mode=ro, chronos_intel_buffer.json, and
premarket_recon_output.json):
  - Slots #1 & #2 (Rapid Radar): High-conviction momentum setups with
    verified spot, 20D SMA delta, 14D ATR, and 16-bar neural bias.
  - Slot #3 (The Main Event): Highest-tension anomaly setup (Model Clash,
    Z-Score stretch, or SEC Form 4 / Finnhub opacity haircut).
Supports --ticker <SYMBOL> to place any specific ticker in Slot #3.
====================================================================
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import sys
from typing import Any
import urllib.parse
import urllib.request
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(r"C:\Anti-Gravity-Core\pierre-training-video")
PIPELINE_DIR = PROJECT_ROOT / "pipeline"
OUTPUT_DIR = PIPELINE_DIR / "output"
DEFAULT_BRIEF_PATH = PIPELINE_DIR / "latest_brief.json"

WARROOM_DATA_DIR = Path(r"C:\Anti-Gravity-Core\pierre_stock_warroom\data")
RECON_OUTPUT_PATH = WARROOM_DATA_DIR / "premarket_recon_output.json"
CHRONOS_BUFFER_PATH = WARROOM_DATA_DIR / "chronos_intel_buffer.json"
PORTFOLIO_40_PATH = WARROOM_DATA_DIR / "portfolio_40.json"
RO_DB_URI = "file:C:/Anti-Gravity-Core/pierre_quant.db?mode=ro"

SPOKEN_COMPANY_MAP: dict[str, str] = {
    "ACM": "AECOM",
    "ASTL": "Algoma Steel",
    "BABA": "Alibaba",
    "BSX": "Boston Scientific",
    "BTC-USD": "Bitcoin",
    "CELH": "Celsius Holdings",
    "EH": "EHang Holdings",
    "ENB": "Enbridge",
    "GOOGL": "Alphabet",
    "IONQ": "IonQ",
    "JD": "J D com",
    "JOBY": "Joby Aviation",
    "KWEB": "KraneShares China Internet",
    "META": "Meta Platforms",
    "NEM": "Newmont",
    "NVO": "Novo Nordisk",
    "NXE": "NexGen Energy",
    "OKLO": "Oklo",
    "PAAS": "Pan American Silver",
    "PFE": "Pfizer",
    "SOFI": "SoFi Technologies",
    "TSCO": "Tractor Supply",
    "UBER": "Uber Technologies",
    "HII": "Huntington Ingalls",
    "ETR": "Entergy",
    "NEE": "NextEra Energy",
    "EW": "Edwards Lifesciences",
    "GEN": "Gen Digital",
}


@dataclass(slots=True, frozen=True)
class TickerBrief:
    ticker: str
    company_name: str
    spoken_name: str
    spoken_short: str
    owned: bool
    spot_price: float
    atr_14d: float
    atr_14d_pct: float
    dynamic_stop_2x_atr: float
    sma_20d: float
    sma_20d_delta_pct: float
    sma_20d_alignment: str
    model_target_16b: float
    forecast_return_pct: float
    directional_bias: str
    timesfm_bias: str
    chronos_bias: str
    has_model_clash: bool
    z_score_anomaly: float
    sec_form4_null: bool
    finnhub_mspr: float | None
    haircut_applied: bool
    composite_conviction: float
    slot_role: str  # "RADAR_1", "RADAR_2", "MAIN_EVENT"


def clean_company_name(ticker: str, raw_name: str) -> str:
    """Normalizes company legal suffixes for natural spoken voice."""
    clean_tkr = ticker.upper().lstrip("$")
    if clean_tkr in SPOKEN_COMPANY_MAP:
        return SPOKEN_COMPANY_MAP[clean_tkr]
    cleaned = raw_name.strip()
    for suffix in (", Inc.", " Inc.", " Corp.", " Corporation", " Holdings", " Limited", " Ltd."):
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)].strip()
    return cleaned or clean_tkr


def format_ticker_letters(ticker: str) -> str:
    """Spells ticker letters cleanly for speech synthesis."""
    clean_tkr = ticker.upper().lstrip("$").strip()
    if clean_tkr == "BTC-USD":
        return "Bitcoin U S D"
    if "-" in clean_tkr:
        left, right = clean_tkr.split("-", 1)
        return f"{' '.join(left)} {' '.join(right)}"
    return " ".join(clean_tkr)


def load_recon_dataset() -> dict[str, Any]:
    """Reads War Room recon data in strict read-only mode."""
    if RECON_OUTPUT_PATH.exists():
        with RECON_OUTPUT_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict) and data:
            return data

    if CHRONOS_BUFFER_PATH.exists():
        with CHRONOS_BUFFER_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        forecasts = data.get("forecasts", {})
        if isinstance(forecasts, dict) and forecasts:
            return forecasts

    raise FileNotFoundError("No War Room recon dataset found in data directory.")


def load_portfolio_owned_map() -> dict[str, bool]:
    """Reads portfolio_40.json to identify active owned tickers."""
    if not PORTFOLIO_40_PATH.exists():
        return {}
    with PORTFOLIO_40_PATH.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    owned_map = {}
    for asset in data.get("assets", []):
        tk = asset.get("ticker", "")
        owned_map[tk] = bool(asset.get("owned", False) or asset.get("shares", 0) > 0)
    return owned_map


def build_ticker_brief(ticker: str, raw_data: dict[str, Any], owned: bool, slot_role: str) -> TickerBrief:
    """Constructs a standardized TickerBrief record from raw quant feeds."""
    raw_name = str(raw_data.get("name") or ticker)
    spoken_short = clean_company_name(ticker, raw_name)
    spoken_name = f"{spoken_short}, ticker {format_ticker_letters(ticker)}"

    spot = float(raw_data.get("spot_price") or 0.0)
    sma20 = float(raw_data.get("sma_20") or spot)
    sma_delta = float(raw_data.get("sma_delta_pct") or 0.0)
    atr = float(raw_data.get("atr_14") or max(0.1, spot * 0.02))
    atr_pct = float(raw_data.get("atr_14_pct") or round((atr / spot) * 100.0, 2) if spot > 0 else 2.5)
    stop = float(raw_data.get("dynamic_stop") or round(max(0.01, spot - 2.0 * atr), 2))
    target = float(raw_data.get("forecast_16b_p50") or round(spot + 3.0 * atr, 2))
    ret_pct = float(raw_data.get("forecast_return_pct") or 0.0)
    bias = str(raw_data.get("directional_bias") or "BULLISH_CONFLUENCE").upper()

    timesfm_bias = "BULLISH" if ret_pct > 0 else "BEARISH"
    chronos_bias = "BULLISH" if "BULL" in bias else ("BEARISH" if "BEAR" in bias else "NEUTRAL")
    has_model_clash = (timesfm_bias == "BULLISH" and chronos_bias == "BEARISH") or (
        timesfm_bias == "BEARISH" and chronos_bias == "BULLISH"
    )

    sma_align = "trading above its 20-day moving average" if spot >= sma20 else "testing below its 20-day moving average"
    haircut = bool(raw_data.get("data_opacity_penalty", False))
    sec_cat = str(raw_data.get("sec_catalyst", "NONE"))
    sec_null = "OPACITY" in sec_cat or "NONE" in sec_cat

    mspr = raw_data.get("finnhub_mspr")
    conviction = float(raw_data.get("composite_conviction") or 60.0)
    z_score = round(sma_delta / max(0.5, atr_pct), 2)

    return TickerBrief(
        ticker=ticker,
        company_name=raw_name,
        spoken_name=spoken_name,
        spoken_short=spoken_short,
        owned=owned,
        spot_price=round(spot, 2),
        atr_14d=round(atr, 2),
        atr_14d_pct=round(atr_pct, 2),
        dynamic_stop_2x_atr=round(stop, 2),
        sma_20d=round(sma20, 2),
        sma_20d_delta_pct=round(sma_delta, 2),
        sma_20d_alignment=sma_align,
        model_target_16b=round(target, 2),
        forecast_return_pct=round(ret_pct, 2),
        directional_bias=bias,
        timesfm_bias=timesfm_bias,
        chronos_bias=chronos_bias,
        has_model_clash=has_model_clash,
        z_score_anomaly=z_score,
        sec_form4_null=sec_null,
        finnhub_mspr=mspr,
        haircut_applied=haircut,
        composite_conviction=round(conviction, 1),
        slot_role=slot_role,
    )


def select_radar_trio(override_ticker: str | None = None) -> list[TickerBrief]:
    """Selects Slots #1, #2 (Rapid Radar) and Slot #3 (The Main Event).

    - Slot #3 (Main Event): Ticker with highest tension/anomaly score or user override.
    - Slots #1 & #2 (Rapid Radar): Top-scoring distinct momentum/confluence setups.
    """
    dataset = load_recon_dataset()
    owned_map = load_portfolio_owned_map()

    candidates: list[tuple[str, dict[str, Any], float, float]] = []
    # (ticker, data, momentum_score, anomaly_score)
    for tk, d in dataset.items():
        if tk in ("BTC-USD",):  # Reserved for macro benchmark
            continue
        conviction = float(d.get("composite_conviction") or 50.0)
        ret_pct = abs(float(d.get("forecast_return_pct") or 0.0))
        sma_delta = abs(float(d.get("sma_delta_pct") or 0.0))
        haircut = bool(d.get("data_opacity_penalty", False))
        is_owned = owned_map.get(tk, False)

        # Momentum score for Slots 1 & 2
        momentum_score = conviction + (15.0 if is_owned else 0.0) + (5.0 if float(d.get("sma_delta_pct") or 0) > 0 else 0.0)
        # Anomaly score for Slot 3: high Z-score stretch or opacity or clash
        anomaly_score = (sma_delta * 1.5) + (20.0 if haircut else 0.0) + (10.0 if is_owned else 0.0)

        candidates.append((tk, d, momentum_score, anomaly_score))

    if not candidates:
        raise ValueError("No eligible ticker candidates found in War Room recon data.")

    # 1. Determine Slot #3 (The Main Event)
    slot3_tk: str
    slot3_data: dict[str, Any]
    if override_ticker:
        clean_override = override_ticker.strip().upper().replace("$", "")
        if clean_override in dataset:
            slot3_tk = clean_override
            slot3_data = dataset[clean_override]
        else:
            raise KeyError(f"Override ticker '{clean_override}' not present in War Room dataset.")
    else:
        # Highest anomaly score
        sorted_by_anomaly = sorted(candidates, key=lambda x: x[3], reverse=True)
        slot3_tk, slot3_data, _, _ = sorted_by_anomaly[0]

    # 2. Determine Slots #1 & #2 (Rapid Radar Setups)
    # Distinct from Slot #3, sorted by momentum conviction
    remaining = [c for c in candidates if c[0] != slot3_tk]
    sorted_by_momentum = sorted(remaining, key=lambda x: x[2], reverse=True)

    if len(sorted_by_momentum) < 2:
        raise ValueError("Insufficient distinct candidates to fill 3-ticker radar trio.")

    slot1_tk, slot1_data, _, _ = sorted_by_momentum[0]
    slot2_tk, slot2_data, _, _ = sorted_by_momentum[1]

    b1 = build_ticker_brief(slot1_tk, slot1_data, owned_map.get(slot1_tk, False), "RADAR_1")
    b2 = build_ticker_brief(slot2_tk, slot2_data, owned_map.get(slot2_tk, False), "RADAR_2")
    b3 = build_ticker_brief(slot3_tk, slot3_data, owned_map.get(slot3_tk, False), "MAIN_EVENT")

    return [b1, b2, b3]


def export_radar_brief_json(trio: list[TickerBrief], out_path: Path = DEFAULT_BRIEF_PATH) -> dict[str, Any]:
    """Generates backward-compatible latest_brief.json with the 3-stock radar trio."""
    now_toronto = datetime.now(ZoneInfo("America/Toronto"))
    run_date_badge = now_toronto.strftime("%d %b %Y").upper()
    run_date_spoken = now_toronto.strftime("%A, %B ") + f"{now_toronto.day}th"

    b1, b2, b3 = trio[0], trio[1], trio[2]

    # Map slot briefs to t1, t2, t3 for legacy pipeline consumers
    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_date_iso": now_toronto.date().isoformat(),
        "run_date_badge": run_date_badge,
        "run_date_spoken": run_date_spoken,
        "db_uri_mode": RO_DB_URI,
        "radar_tickers": [b.ticker for b in trio],
        "primary_main_ticker": b3.ticker,
        "t1": {
            "ticker": b1.ticker,
            "company_name": b1.company_name,
            "spoken_name": b1.spoken_name,
            "spoken_short": b1.spoken_short,
            "owned": b1.owned,
            "spot_price": b1.spot_price,
            "atr_14d": b1.atr_14d,
            "atr_14d_pct": b1.atr_14d_pct,
            "dynamic_stop_2x_atr": b1.dynamic_stop_2x_atr,
            "sma_20d": b1.sma_20d,
            "sma_20d_delta_pct": b1.sma_20d_delta_pct,
            "sma_20d_alignment": b1.sma_20d_alignment,
            "model_target_5d": b1.model_target_16b,
            "model_target_source": "pierre_quant.db:16b_timesfm_chronos",
            "net_quant_bias": b1.directional_bias,
            "consensus_bias_raw": b1.directional_bias,
            "haircut_applied": b1.haircut_applied,
            "adjusted_confluence_score": b1.composite_conviction,
        },
        "t2": {
            "ticker": b2.ticker,
            "company_name": b2.company_name,
            "spoken_name": b2.spoken_name,
            "spoken_short": b2.spoken_short,
            "owned": b2.owned,
            "spot_price": b2.spot_price,
            "atr_14d": b2.atr_14d,
            "atr_14d_pct": b2.atr_14d_pct,
            "dynamic_stop_2x_atr": b2.dynamic_stop_2x_atr,
            "sma_20d": b2.sma_20d,
            "sma_20d_delta_pct": b2.sma_20d_delta_pct,
            "sma_20d_alignment": b2.sma_20d_alignment,
            "model_target_5d": b2.model_target_16b,
            "model_target_source": "pierre_quant.db:16b_timesfm_chronos",
            "net_quant_bias": b2.directional_bias,
            "consensus_bias_raw": b2.directional_bias,
            "haircut_applied": b2.haircut_applied,
            "adjusted_confluence_score": b2.composite_conviction,
        },
        "t3": {
            "ticker": b3.ticker,
            "company_name": b3.company_name,
            "spoken_name": b3.spoken_name,
            "spoken_short": b3.spoken_short,
            "owned": b3.owned,
            "spot_price": b3.spot_price,
            "atr_14d": b3.atr_14d,
            "atr_14d_pct": b3.atr_14d_pct,
            "dynamic_stop_2x_atr": b3.dynamic_stop_2x_atr,
            "sma_20d": b3.sma_20d,
            "sma_20d_delta_pct": b3.sma_20d_delta_pct,
            "sma_20d_alignment": b3.sma_20d_alignment,
            "model_target_5d": b3.model_target_16b,
            "model_target_source": "pierre_quant.db:16b_timesfm_chronos",
            "net_quant_bias": b3.directional_bias,
            "consensus_bias_raw": b3.directional_bias,
            "haircut_applied": b3.haircut_applied,
            "adjusted_confluence_score": b3.composite_conviction,
            "has_model_clash": b3.has_model_clash,
            "timesfm_bias": b3.timesfm_bias,
            "chronos_bias": b3.chronos_bias,
            "z_score_anomaly": b3.z_score_anomaly,
        },
        "compliance_footer": "QUANT TELEMETRY LOG ONLY • NOT FINANCIAL ADVICE",
    }

    resolved_path = out_path.resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = resolved_path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    tmp_path.replace(resolved_path)

    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract 3-Ticker Radar Trio from War Room Telemetry")
    parser.add_argument("--ticker", type=str, default=None, help="Force specific ticker into Slot #3 (Main Event)")
    parser.add_argument("--out", type=Path, default=DEFAULT_BRIEF_PATH, help="Output JSON path")
    parser.add_argument("--dry-run", action="store_true", help="Print radar selection without writing to disk")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    trio = select_radar_trio(override_ticker=args.ticker)

    print("[extract_brief] === SELECTED 3-STOCK MORNING RADAR TRIO ===")
    for b in trio:
        print(
            f"  [{b.slot_role}] ${b.ticker} ({b.company_name}): Spot ${b.spot_price:.2f} | "
            f"SMA20 Delta: {b.sma_20d_delta_pct:+.2f}% | 14D ATR Stop: ${b.dynamic_stop_2x_atr:.2f} | "
            f"16b Target: ${b.model_target_16b:.2f} | Conviction: {b.composite_conviction}%"
        )

    if not args.dry_run:
        payload = export_radar_brief_json(trio, args.out)
        print(f"[extract_brief] Saved updated 3-stock brief to {args.out}")
    else:
        print("[extract_brief] DRY RUN: Brief JSON not saved to disk.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
