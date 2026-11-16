#!/usr/bin/env python3
"""pipeline/episode_tracker.py
====================================================================
Episodic Scorecard State Machine for Pierre Quant TikTok Engine.

Tracks episodic history, grades prior episode price trajectories
deterministically against 16-bar targets and 2x ATR invalidation stops,
and generates standardized frame-0 HUD banners, spoken receipts,
and safe-zone captions.
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

# Root paths strictly contained within project boundaries
PROJECT_ROOT = Path(r"C:\Anti-Gravity-Core\pierre-training-video")
PIPELINE_DIR = PROJECT_ROOT / "pipeline"
DEFAULT_LEDGER_PATH = PIPELINE_DIR / "episode_ledger.json"
RO_DB_URI = "file:C:/Anti-Gravity-Core/pierre_quant.db?mode=ro"


@dataclass(slots=True, frozen=True)
class TickerCall:
    """Quantitative call parameters recorded for an asset in an episode."""
    ticker: str
    entry_spot: float
    atr_stop: float
    target_16bar: float
    directional_bias: str
    role: str = "MAIN_AUDIT"


@dataclass(slots=True, frozen=True)
class AuditReceipt:
    """Deterministic grading scorecard of a prior episode's call."""
    previous_episode_id: int
    graded_ticker: str
    entry_spot: float
    current_spot: float
    atr_stop: float
    target_16bar: float
    delta_pct: float
    outcome: str
    spoken_receipt: str
    hud_receipt_badge: str


@dataclass(slots=True, frozen=True)
class EpisodeRecord:
    """Full episodic entry stored in the persistent ledger."""
    episode_id: int
    recorded_at_utc: str
    date_badge: str
    primary_ticker: str
    radar_tickers: tuple[str, ...]
    calls: tuple[TickerCall, ...]
    receipt_graded: dict[str, Any] | None = None
    status: str = "LOCKED"


def _format_spoken_dollars(amount: float) -> str:
    """Converts a floating-point dollar figure into clean spoken English."""
    if amount >= 1000.0:
        k_val = round(amount)
        return f"{k_val:,} dollars".replace(",", " thousand ")
    if amount >= 1.0:
        dollars = int(amount)
        cents = round((amount - dollars) * 100)
        if cents == 0:
            return f"{dollars} dollars"
        return f"{dollars} dollars and {cents} cents"
    cents = round(amount * 100)
    return f"{cents} cents"


def fetch_deterministic_spot(ticker: str) -> float:
    """Queries current spot price using read-only SQLite or Yahoo Chart API v8.

    Strictly deterministic and does not use speculative LLM reasoning.
    """
    clean_ticker = ticker.strip().upper().replace("$", "")

    # 1. Primary: Try read-only SQLite database if available
    db_file = Path(r"C:\Anti-Gravity-Core\pierre_quant.db")
    if db_file.exists():
        try:
            conn = sqlite3.connect(RO_DB_URI, uri=True, timeout=5.0)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT spot_price FROM market_feed WHERE ticker = ? ORDER BY timestamp DESC LIMIT 1",
                (clean_ticker,),
            )
            row = cursor.fetchone()
            conn.close()
            if row and row[0] is not None and float(row[0]) > 0.0:
                return float(row[0])
        except (sqlite3.Error, ValueError, OSError):
            pass

    # 2. Corroboration / Fallback: Yahoo Finance Chart API v8 (zero-dependency)
    encoded = urllib.parse.quote(clean_ticker)
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded}?interval=1d&range=5d"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/129.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=8.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        res = data.get("chart", {}).get("result", [{}])[0]
        meta = res.get("meta", {})
        price = meta.get("regularMarketPrice")
        if price is not None and float(price) > 0.0:
            return float(price)
        closes = res.get("indicators", {}).get("quote", [{}])[0].get("close", [])
        valid_closes = [c for c in closes if c is not None and float(c) > 0.0]
        if valid_closes:
            return float(valid_closes[-1])
    except (urllib.error.URLError, json.JSONDecodeError, KeyError, IndexError, ValueError, OSError) as exc:
        raise RuntimeError(f"Failed to fetch deterministic spot for {clean_ticker}: {exc}") from exc

    raise RuntimeError(f"Spot price returned 0 or null for {clean_ticker}")


def load_ledger(ledger_path: Path = DEFAULT_LEDGER_PATH) -> dict[str, Any]:
    """Reads persistent episodic ledger JSON safely."""
    if not ledger_path.exists():
        raise FileNotFoundError(f"Ledger file missing at {ledger_path}")
    with ledger_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or "episodes" not in data:
        raise ValueError(f"Invalid ledger schema at {ledger_path}")
    return data


def grade_previous_episode(ledger_path: Path = DEFAULT_LEDGER_PATH) -> AuditReceipt:
    """Grades the most recent episode call against live market settlement.

    Calculates exact percentage return against entry spot, compares against
    the 16-bar target and 2x ATR invalidation stop, and builds broadcast receipts.
    """
    ledger = load_ledger(ledger_path)
    episodes: list[dict[str, Any]] = ledger.get("episodes", [])
    if not episodes:
        raise ValueError("Ledger has zero recorded episodes to grade.")

    last_ep = episodes[-1]
    ep_id = int(last_ep.get("episode_id", 1))
    calls_raw = last_ep.get("calls", [])
    if not calls_raw:
        raise ValueError(f"Episode #{ep_id} contains no recorded ticker calls.")

    # Select primary call
    main_call = next((c for c in calls_raw if c.get("role") == "MAIN_AUDIT"), calls_raw[0])
    raw_ticker = str(main_call.get("ticker", "BTC-USD")).upper()
    entry_spot = float(main_call.get("entry_spot", 0.0))
    atr_stop = float(main_call.get("atr_stop", 0.0))
    target_16bar = float(main_call.get("target_16bar", 0.0))
    bias = str(main_call.get("directional_bias", "NEUTRAL")).upper()

    if entry_spot <= 0.0:
        raise ValueError(f"Invalid entry spot for {raw_ticker}: {entry_spot}")

    current_spot = fetch_deterministic_spot(raw_ticker)
    delta_pct = ((current_spot - entry_spot) / entry_spot) * 100.0

    target_spoken = _format_spoken_dollars(target_16bar)
    clean_ticker = raw_ticker.replace("$", "")

    # Grade directional outcome
    if "BEAR" in bias:
        # Bearish trajectory expects lower prices
        if current_spot <= target_16bar:
            outcome = "TARGET_HIT"
            spoken_receipt = (
                f"Yesterday's {clean_ticker} short call hit our {target_spoken} target, "
                f"locking in a {abs(delta_pct):.1f} percent drop."
            )
            badge = f"EP #{ep_id:02d} RECEIPT: ${clean_ticker} TARGET HIT ({delta_pct:+.1f}%)"
        elif current_spot >= atr_stop and atr_stop > entry_spot:
            outcome = "STOP_TRIGGERED"
            spoken_receipt = (
                f"Yesterday's {clean_ticker} short stopped at the invalidation level "
                f"of {_format_spoken_dollars(atr_stop)}."
            )
            badge = f"EP #{ep_id:02d} RECEIPT: ${clean_ticker} STOPPED ({delta_pct:+.1f}%)"
        else:
            outcome = "IN_PROGRESS"
            spoken_receipt = (
                f"Yesterday's {clean_ticker} call is active at {_format_spoken_dollars(current_spot)}, "
                f"down {abs(delta_pct):.1f} percent toward target."
            )
            badge = f"EP #{ep_id:02d} RECEIPT: ${clean_ticker} ACTIVE ({delta_pct:+.1f}%)"
    else:
        # Bullish trajectory expects higher prices
        if current_spot >= target_16bar:
            outcome = "TARGET_HIT"
            spoken_receipt = (
                f"Yesterday's {clean_ticker} call hit our {target_spoken} target, "
                f"surging {delta_pct:+.1f} percent."
            )
            badge = f"EP #{ep_id:02d} RECEIPT: ${clean_ticker} TARGET HIT ({delta_pct:+.1f}%)"
        elif current_spot <= atr_stop and atr_stop < entry_spot:
            outcome = "STOP_TRIGGERED"
            spoken_receipt = (
                f"Yesterday's {clean_ticker} setup hit our invalidation floor at "
                f"{_format_spoken_dollars(atr_stop)}."
            )
            badge = f"EP #{ep_id:02d} RECEIPT: ${clean_ticker} STOPPED ({delta_pct:+.1f}%)"
        else:
            outcome = "IN_PROGRESS"
            spoken_receipt = (
                f"Yesterday's {clean_ticker} setup is holding at {_format_spoken_dollars(current_spot)}, "
                f"up {delta_pct:+.1f} percent toward target."
            )
            badge = f"EP #{ep_id:02d} RECEIPT: ${clean_ticker} ACTIVE ({delta_pct:+.1f}%)"

    return AuditReceipt(
        previous_episode_id=ep_id,
        graded_ticker=clean_ticker,
        entry_spot=round(entry_spot, 2),
        current_spot=round(current_spot, 2),
        atr_stop=round(atr_stop, 2),
        target_16bar=round(target_16bar, 2),
        delta_pct=round(delta_pct, 2),
        outcome=outcome,
        spoken_receipt=spoken_receipt,
        hud_receipt_badge=badge,
    )


def get_next_episode_metadata(
    t1: str,
    t2: str,
    t3: str,
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> dict[str, Any]:
    """Generates complete HUD strings, hook scripts, and captions for the next episode."""
    ledger = load_ledger(ledger_path)
    episodes: list[dict[str, Any]] = ledger.get("episodes", [])
    next_id = int(episodes[-1].get("episode_id", 0)) + 1 if episodes else 1

    clean_t1 = t1.strip().upper().replace("$", "")
    clean_t2 = t2.strip().upper().replace("$", "")
    clean_t3 = t3.strip().upper().replace("$", "")

    receipt = grade_previous_episode(ledger_path)

    # Frame-0 Persistent Top-Center HUD Header
    hud_header = f"EPISODE #{next_id:02d} | TODAY'S 3 STOCKS: ${clean_t1} • ${clean_t2} • ${clean_t3}"

    # 0.0s–4.5s Credibility + Open-Loop Spoken Hook
    hook_script = (
        f"Morning, I'm Pierre, Senior Engineer. Today my Dual-GPU AI War Room is auditing "
        f"${clean_t1}, ${clean_t2}, and ${clean_t3}—plus yesterday's ${receipt.graded_ticker} receipt. "
        f"Two setups look solid, and one just flashed a trap."
    )

    # 3-Tier Velocity Hashtags & Search SEO
    caption_line1 = (
        f"[EP #{next_id:02d}] Dual-GPU AI Audit: ${clean_t1} vs ${clean_t2} vs ${clean_t3} — "
        f"Which Setup Breaks First?"
    )
    caption_line2 = (
        f"Yesterday's EP #{receipt.previous_episode_id:02d} Call: "
        f"${receipt.graded_ticker} {receipt.outcome} ({receipt.delta_pct:+.1f}%)."
    )
    caption_line3 = (
        f"#{clean_t1} #{clean_t2} #{clean_t3} #quanttrading #swingtradingstrategies #stockmarket"
    )
    caption_line4 = (
        "⚠️ Educational quant software telemetry & personal market log only. Not financial or investment advice."
    )

    full_caption = "\n".join([caption_line1, caption_line2, caption_line3, caption_line4])

    return {
        "episode_id": next_id,
        "hud_header": hud_header,
        "hook_script": hook_script,
        "receipt": asdict(receipt),
        "caption": {
            "title_line": caption_line1,
            "scorecard_line": caption_line2,
            "hashtags_line": caption_line3,
            "disclosure_line": caption_line4,
            "full_text": full_caption,
        },
        "safe_zones": {
            "top_px": 140,
            "bottom_px": 320,
            "right_px": 120,
            "left_px": 40,
            "footer_badge": "QUANT TELEMETRY LOG ONLY • NOT FINANCIAL ADVICE",
        },
    }


def commit_episode_record(
    record: EpisodeRecord,
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> None:
    """Atomically commits a new EpisodeRecord into the persistent ledger."""
    resolved_path = ledger_path.resolve()
    allowed_root = PROJECT_ROOT.resolve()
    if not resolved_path.is_relative_to(allowed_root):
        raise PermissionError(f"Write containment violation: {resolved_path} outside {allowed_root}")

    ledger = load_ledger(resolved_path)
    episodes: list[dict[str, Any]] = ledger.get("episodes", [])

    # Check for duplicate episode_id
    if any(ep.get("episode_id") == record.episode_id for ep in episodes):
        raise ValueError(f"Episode #{record.episode_id} already exists in ledger.")

    ep_dict = {
        "episode_id": record.episode_id,
        "recorded_at_utc": record.recorded_at_utc,
        "date_badge": record.date_badge,
        "primary_ticker": record.primary_ticker,
        "radar_tickers": list(record.radar_tickers),
        "calls": [asdict(c) for c in record.calls],
        "receipt_graded": record.receipt_graded,
        "status": record.status,
    }
    episodes.append(ep_dict)
    ledger["episodes"] = episodes
    ledger["active_episode_id"] = record.episode_id
    ledger["last_updated_utc"] = datetime.now(timezone.utc).isoformat()

    tmp_path = resolved_path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(ledger, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    tmp_path.replace(resolved_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI Argument Parser."""
    parser = argparse.ArgumentParser(
        description="Episodic Scorecard State Machine for Pierre Quant TikTok Engine"
    )
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--dry-run", action="store_true", help="Print audit and next episode plan without committing.")
    parser.add_argument("--t1", type=str, default="ACM", help="Ticker 1 (Rapid Radar)")
    parser.add_argument("--t2", type=str, default="UBER", help="Ticker 2 (Rapid Radar)")
    parser.add_argument("--t3", type=str, default="BSX", help="Ticker 3 (Main Event Audit)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI Entrypoint."""
    args = parse_args(argv)
    try:
        next_meta = get_next_episode_metadata(
            t1=args.t1,
            t2=args.t2,
            t3=args.t3,
            ledger_path=args.ledger,
        )
    except Exception as exc:
        print(f"[episode_tracker] ERROR: {exc}", file=sys.stderr)
        return 1

    print("[episode_tracker] === PRIOR EPISODE GRADING RECEIPT ===")
    r = next_meta["receipt"]
    print(f"  Previous Episode: #{r['previous_episode_id']:02d} ({r['graded_ticker']})")
    print(f"  Entry Spot: ${r['entry_spot']:.2f} | Current Spot: ${r['current_spot']:.2f} | Delta: {r['delta_pct']:+.2f}%")
    print(f"  Target: ${r['target_16bar']:.2f} | Invalidation Stop: ${r['atr_stop']:.2f}")
    print(f"  Outcome: {r['outcome']}")
    print(f"  HUD Badge: {r['hud_receipt_badge']}")
    print(f"  Spoken Line: \"{r['spoken_receipt']}\"")
    print()
    print(f"[episode_tracker] === NEXT EPISODE METADATA (EPISODE #{next_meta['episode_id']:02d}) ===")
    print(f"  Frame-0 HUD Header: {next_meta['hud_header']}")
    print(f"  Open-Loop Hook: \"{next_meta['hook_script']}\"")
    print()
    print("[episode_tracker] === 4-LINE SEARCH SEO CAPTION ===")
    print(next_meta["caption"]["full_text"])

    if args.dry_run:
        print("\n[episode_tracker] DRY RUN COMPLETE: 0 ledger writes executed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
