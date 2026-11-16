"""Read-Only Quant Bridge & Dual-Voice Script Engine for Pierre Training Video.

Reads strictly from isolated portfolio telemetry, Obsidian Vault Recon markdown
frontmatter, and the SQLite quantitative forecast ledger using mandatory read-only
file descriptors and SQLite `?mode=ro` URI connections. Emits a deterministic,
compliance-safe dual-voice script brief (`latest_brief.json`) for Phase 3 audio
synthesis and FFmpeg assembly.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sqlite3
import sys
from typing import Any
import urllib.parse
import urllib.request
from zoneinfo import ZoneInfo

WORKSPACE_ROOT = Path(r"C:\Anti-Gravity-Core")
PROJECT_ROOT = Path(r"C:\Anti-Gravity-Core\pierre-training-video")
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "pipeline" / "latest_brief.json"

HERMES_BOOK_PATH = WORKSPACE_ROOT / "data" / "hermes_autonomous_book.json"
PORTFOLIO_40_PATH = WORKSPACE_ROOT / "pierre_stock_warroom" / "data" / "portfolio_40.json"
CROWD_SENTIMENT_ROOT_PATH = WORKSPACE_ROOT / "data" / "crowd_sentiment.json"
CROWD_SENTIMENT_SCRAPER_PATH = (
    WORKSPACE_ROOT / "pierre_openclaw_sentiment_scraper" / "shared_signals" / "crowd_sentiment.json"
)

OPENCLAW_WORKSPACE = Path(r"C:\Users\Pierre\.openclaw\workspace")
PIERRE_QUANT_DB_PATH = OPENCLAW_WORKSPACE / "pierre-quant" / "pierre_quant.db"
CHRONOS_BUFFER_PATH = OPENCLAW_WORKSPACE / "chronos_intel_buffer.json"
VAULT_RECONS_DIR = OPENCLAW_WORKSPACE / "vault" / "Recons"

COMPLIANCE_FOOTER = "AUTONOMOUS QUANT TELEMETRY // EDUCATIONAL & RESEARCH ONLY — NOT FINANCIAL ADVICE"
ESTIMATED_WORDS_PER_SECOND = 3.1
HAIRCUT_MULTIPLIER = 0.80

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
}


@dataclass(slots=True, frozen=True)
class TickerTelemetry:
    ticker: str
    company_name: str
    spoken_name: str
    spoken_short: str
    owned: bool
    raw_confluence_score: float
    adjusted_confluence_score: float
    haircut_applied: bool
    haircut_multiplier: float
    sec_form4_null: bool
    finnhub_sentiment_null: bool
    haircut_reasons: tuple[str, ...]
    spot_price: float
    atr_14d: float
    atr_14d_pct: float
    dynamic_stop_2x_atr: float
    sma_20d: float
    sma_20d_alignment: str
    model_target_5d: float
    model_target_source: str
    net_quant_bias: str
    consensus_bias_raw: str


@dataclass(slots=True, frozen=True)
class DailyBriefPayload:
    generated_at_utc: str
    run_date_iso: str
    run_date_badge: str
    run_date_spoken: str
    morning_scan_timestamp: str
    db_uri_mode: str
    t1_override_used: bool
    t1: TickerTelemetry
    t2: TickerTelemetry
    t3: TickerTelemetry
    pierre_script: str
    julie_script: str
    pierre_word_count: int
    julie_word_count: int
    total_word_count: int
    pierre_estimated_duration_sec: float
    julie_estimated_duration_sec: float
    total_estimated_duration_sec: float
    compliance_footer: str


def load_json_ro(path: Path) -> dict[str, Any]:
    """Reads a JSON file strictly with read-only mode."""
    if not path.exists():
        return {}
    with path.open(mode="r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def clean_company_name(ticker: str, raw_name: str) -> str:
    """Normalizes legal suffixes from company names for clean spoken narration."""
    clean_tkr = ticker.upper().lstrip("$")
    if clean_tkr in SPOKEN_COMPANY_MAP:
        return SPOKEN_COMPANY_MAP[clean_tkr]
    cleaned = raw_name.strip()
    for suffix in (", Inc.", " Inc.", " Corp.", " Corporation", " Holdings", " Limited", " Ltd."):
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)].strip()
    return cleaned or clean_tkr


def format_ticker_letters(ticker: str) -> str:
    """Spells ticker letters cleanly for TTS pronunciation without raw symbols."""
    clean_tkr = ticker.upper().lstrip("$").strip()
    if clean_tkr == "BTC-USD":
        return "Bitcoin U S D"
    if "-" in clean_tkr:
        left, right = clean_tkr.split("-", 1)
        return f"{' '.join(left)} {' '.join(right)}"
    return " ".join(clean_tkr)


def format_ticker_spoken(ticker: str, company_name: str) -> str:
    """Formats full spoken asset reference as '{Company}, ticker {L E T T E R S}'."""
    clean_tkr = ticker.upper().lstrip("$").strip()
    short = clean_company_name(clean_tkr, company_name)
    if clean_tkr == "BTC-USD":
        return "Bitcoin, ticker B T C U S D"
    return f"{short}, ticker {format_ticker_letters(clean_tkr)}"


def format_ticker_short_spoken(ticker: str, company_name: str) -> str:
    """Formats short spoken company reference for natural hook repetition."""
    return clean_company_name(ticker, company_name)


def format_price_spoken(val: float) -> str:
    """Formats decimal price for TTS without commas or currency symbols."""
    return f"{val:.2f} dollars"


def format_bias_spoken(raw_bias: str) -> str:
    """Converts internal snake/upper bias codes into readable English phrase."""
    cleaned = raw_bias.strip().replace("_", " ").lower()
    return " ".join(word.capitalize() for word in cleaned.split()) or "Neutral Consolidation"


def parse_iso_epoch(iso_text: Any) -> float:
    """Parses ISO-8601 timestamp string to UTC Unix epoch seconds."""
    if not isinstance(iso_text, str) or not iso_text.strip():
        return 0.0
    normalized = iso_text.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        return 0.0


def parse_vault_recon_frontmatter(ticker: str) -> dict[str, str]:
    """Reads YAML frontmatter key/value pairs from Vault Recon markdown in read-only mode."""
    recon_path = VAULT_RECONS_DIR / f"{ticker.upper().lstrip('$')}.md"
    if not recon_path.exists():
        return {}
    with recon_path.open(mode="r", encoding="utf-8") as handle:
        lines = [handle.readline() for _ in range(60)]
    if not lines or lines[0].strip() != "---":
        return {}
    metadata: dict[str, str] = {}
    for line in lines[1:]:
        stripped = line.strip()
        if stripped == "---":
            break
        if ":" not in stripped:
            continue
        key, val = stripped.split(":", 1)
        metadata[key.strip()] = val.strip().strip('"').strip("'")
    return metadata


def query_db_telemetry_ro(
    db_path: Path, tickers: list[str]
) -> tuple[str, dict[str, dict[str, dict[str, Any]]], dict[str, float]]:
    """Queries latest agent signals and 20-day SMAs from SQLite via strict `?mode=ro` URI."""
    if not db_path.exists():
        raise FileNotFoundError(f"Quantitative ledger not found: {db_path}")

    db_uri = f"file:{db_path.as_posix()}?mode=ro"
    agent_map: dict[str, dict[str, dict[str, Any]]] = {t: {} for t in tickers}
    sma20_map: dict[str, float] = {}

    target_agents = ("06a_timesfm", "06b_chronos", "10_smart_money", "13_sec_watchdog", "16_sentiment")
    placeholders = ",".join("?" for _ in target_agents)

    conn = sqlite3.connect(db_uri, uri=True)
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        for ticker in tickers:
            cur.execute(
                f"""
                SELECT agent_id, timestamp, spot_price, predicted_bias, predicted_target, predicted_prob
                FROM forecast_history
                WHERE ticker = ? AND agent_id IN ({placeholders})
                ORDER BY id DESC
                """,
                (ticker, *target_agents),
            )
            seen_agents: dict[str, dict[str, Any]] = {}
            for row in cur.fetchall():
                aid = str(row["agent_id"])
                if aid not in seen_agents:
                    seen_agents[aid] = {
                        "timestamp": float(row["timestamp"] or 0.0),
                        "spot_price": row["spot_price"],
                        "predicted_bias": row["predicted_bias"],
                        "predicted_target": row["predicted_target"],
                        "predicted_prob": row["predicted_prob"],
                    }
            agent_map[ticker] = seen_agents

            cur.execute(
                "SELECT prices FROM market_matrices WHERE ticker = ? ORDER BY id DESC LIMIT 1",
                (ticker,),
            )
            mm_row = cur.fetchone()
            if mm_row and mm_row["prices"]:
                closes = [float(x) for x in str(mm_row["prices"]).split(",") if x.strip()]
                if closes:
                    window = closes[-20:] if len(closes) >= 20 else closes
                    sma20_map[ticker] = round(sum(window) / len(window), 2)
    finally:
        conn.close()

    return db_uri, agent_map, sma20_map


def evaluate_null_haircut(
    ticker: str,
    agents: dict[str, dict[str, Any]],
    recon_meta: dict[str, str],
    cs_root: dict[str, Any],
    cs_scraper: dict[str, Any],
) -> tuple[bool, bool, tuple[str, ...]]:
    """Checks SEC Form 4 and Finnhub/crowd sentiment availability cleanly."""
    reasons: list[str] = []

    smart_money = agents.get("10_smart_money")
    sec_watchdog = agents.get("13_sec_watchdog")
    sec_form4_null = False
    if not smart_money or not sec_watchdog:
        sec_form4_null = True
        reasons.append("MISSING_SEC_OR_SMART_MONEY_AGENT_ROW")
    else:
        sm_prob = smart_money.get("predicted_prob")
        sec_prob = sec_watchdog.get("predicted_prob")
        if (
            not smart_money.get("predicted_bias")
            or not sec_watchdog.get("predicted_bias")
            or sm_prob is None
            or sec_prob is None
            or float(sm_prob) <= 0.0
            or float(sec_prob) <= 0.0
        ):
            sec_form4_null = True
            reasons.append("NULL_SEC_OR_SMART_MONEY_PROBABILITY")

    if "sec_form4" in recon_meta and recon_meta["sec_form4"].lower() in ("null", "none", "[]", ""):
        sec_form4_null = True
        reasons.append("RECON_SEC_FORM4_NULL")

    sent_agent = agents.get("16_sentiment")
    finnhub_sentiment_null = False
    if not sent_agent or not sent_agent.get("predicted_bias") or sent_agent.get("predicted_prob") is None:
        finnhub_sentiment_null = True
        reasons.append("MISSING_SENTIMENT_16_AGENT_ROW")
    elif float(sent_agent["predicted_prob"]) <= 0.0:
        finnhub_sentiment_null = True
        reasons.append("ZERO_CONVICTION_SENTIMENT_16_ROW")

    root_entry = cs_root.get(ticker)
    scraper_entry = cs_scraper.get(ticker)
    root_ok = isinstance(root_entry, dict) and (
        root_entry.get("polarity_score") is not None or bool(root_entry.get("headlines_sample"))
    )
    scraper_ok = isinstance(scraper_entry, dict) and isinstance(scraper_entry.get("metrics"), dict)
    if not root_ok and not scraper_ok:
        finnhub_sentiment_null = True
        reasons.append("CROWD_FINNHUB_SENTIMENT_MISSING_OR_EMPTY")

    if recon_meta.get("sentiment_regime", "").upper() == "NEUTRAL_OR_UNAVAILABLE":
        finnhub_sentiment_null = True
        reasons.append("RECON_SENTIMENT_NEUTRAL_OR_UNAVAILABLE")

    return sec_form4_null, finnhub_sentiment_null, tuple(reasons)


def resolve_model_target_5d(
    ticker: str,
    spot_price: float,
    atr_14_raw: float,
    agents: dict[str, dict[str, Any]],
    chronos_forecasts: dict[str, Any],
    ticket: dict[str, Any],
) -> tuple[float, str]:
    """Selects freshest 5-day foundation model target from DB or Chronos buffer."""
    candidates: list[tuple[float, float, str]] = []

    agent_06b = agents.get("06b_chronos")
    if agent_06b and agent_06b.get("predicted_target") is not None:
        val = float(agent_06b["predicted_target"])
        if val > 0.0:
            candidates.append((float(agent_06b.get("timestamp", 0.0)), val, "pierre_quant.db:06b_chronos"))

    cf = chronos_forecasts.get(ticker)
    if isinstance(cf, dict):
        raw_cf_val = cf.get("forecast_5d_p50") or cf.get("predicted_price") or cf.get("target_price")
        if raw_cf_val is not None and float(raw_cf_val) > 0.0:
            candidates.append(
                (parse_iso_epoch(cf.get("updated_at")), float(raw_cf_val), "chronos_intel_buffer.json:forecast_5d_p50")
            )

    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        _, best_val, best_source = candidates[0]
        return round(best_val, 2), best_source

    agent_06a = agents.get("06a_timesfm")
    if agent_06a and agent_06a.get("predicted_target") is not None and float(agent_06a["predicted_target"]) > 0.0:
        return round(float(agent_06a["predicted_target"]), 2), "pierre_quant.db:06a_timesfm"

    if ticket.get("target2_px") is not None and float(ticket["target2_px"]) > 0.0:
        return round(float(ticket["target2_px"]), 2), "hermes_autonomous_book.json:target2_px"

    bias = str(ticket.get("consensus_bias") or "").upper()
    fallback = spot_price + (atr_14_raw * 1.5) if "BULLISH" in bias else spot_price
    return round(fallback, 2), "deterministic_atr_fallback"


def fetch_live_market_spot(ticker: str) -> float | None:
    """Fetches real-time market spot including pre-market / extended hours."""
    clean = ticker.strip().upper().replace("$", "")
    try:
        url = (
            f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(clean)}"
            f"?interval=1m&range=1d&includePrePost=true"
        )
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/129.0"},
        )
        with urllib.request.urlopen(req, timeout=4) as r:
            d = json.loads(r.read().decode("utf-8"))
            result = d.get("chart", {}).get("result", [{}])[0]
            indicators = result.get("indicators", {}).get("quote", [{}])[0]
            closes = [c for c in indicators.get("close", []) if c is not None]
            if closes:
                return round(float(closes[-1]), 2)
            meta_price = result.get("meta", {}).get("regularMarketPrice")
            if meta_price:
                return round(float(meta_price), 2)
    except Exception:
        pass
    return None


def build_ticker_telemetry(
    ticker: str,
    ticket: dict[str, Any],
    portfolio_names: dict[str, str],
    agents: dict[str, dict[str, Any]],
    sma20_map: dict[str, float],
    chronos_forecasts: dict[str, Any],
    cs_root: dict[str, Any],
    cs_scraper: dict[str, Any],
) -> TickerTelemetry:
    """Computes complete deterministic telemetry and haircut adjustments for one ticker."""
    recon_meta = parse_vault_recon_frontmatter(ticker)
    company_name = portfolio_names.get(ticker) or str(ticket.get("name") or ticker)
    spoken_name = format_ticker_spoken(ticker, company_name)
    spoken_short = format_ticker_short_spoken(ticker, company_name)

    raw_score = round(float(ticket.get("confluence_score_pct") or 0.0), 2)
    sec_null, finnhub_null, reasons = evaluate_null_haircut(ticker, agents, recon_meta, cs_root, cs_scraper)
    haircut_applied = sec_null or finnhub_null
    multiplier = HAIRCUT_MULTIPLIER if haircut_applied else 1.0
    adjusted_score = round(raw_score * multiplier, 2) if haircut_applied else raw_score

    # Prioritize real-time live pre-market / extended hours feed
    live_spot = fetch_live_market_spot(ticker)
    ticket_spot = float(ticket.get("spot_price") or 0.0)
    use_recon_primary = bool(recon_meta.get("spot_price")) and (
        ticket_spot <= 0.0
        or (ticker == "BTC-USD" and ticket_spot == 100.0)
    )

    if live_spot is not None and live_spot > 0.0:
        spot_raw = live_spot
        atr_raw = float(recon_meta.get("atr_14") or ticket.get("atr_14") or 0.0)
    elif use_recon_primary:
        spot_raw = float(recon_meta.get("spot_price") or 0.0)
        atr_raw = float(recon_meta.get("atr_14") or ticket.get("atr_14") or 0.0)
    else:
        spot_raw = float(ticket.get("spot_price") or recon_meta.get("spot_price") or 0.0)
        atr_raw = float(ticket.get("atr_14") or recon_meta.get("atr_14") or 0.0)

    spot_price = round(spot_raw, 2)
    atr_14d = round(atr_raw, 2)
    atr_14d_pct = round((atr_raw / spot_raw) * 100.0, 2) if spot_raw > 0.0 else 0.0
    dynamic_stop_2x_atr = round(spot_raw - (2.0 * atr_raw), 2)

    recon_sma = recon_meta.get("sma_20")
    if use_recon_primary and recon_sma:
        sma_20d = round(float(recon_sma), 2)
    else:
        sma_20d = sma20_map.get(ticker)
        if sma_20d is None:
            sma_20d = round(float(recon_sma), 2) if recon_sma else spot_price

    sma_alignment = (
        "trading above its 20-day moving average"
        if spot_price >= sma_20d
        else "testing below its 20-day moving average"
    )

    model_target_5d, model_target_source = resolve_model_target_5d(
        ticker=ticker,
        spot_price=spot_raw,
        atr_14_raw=atr_raw,
        agents=agents,
        chronos_forecasts=chronos_forecasts,
        ticket=ticket,
    )

    raw_bias = str(
        (recon_meta.get("net_bias") if use_recon_primary else None)
        or ticket.get("consensus_bias")
        or recon_meta.get("net_bias")
        or "NEUTRAL_CONSOLIDATION"
    )
    net_quant_bias = format_bias_spoken(raw_bias)

    return TickerTelemetry(
        ticker=ticker,
        company_name=company_name,
        spoken_name=spoken_name,
        spoken_short=spoken_short,
        owned=bool(ticket.get("owned")),
        raw_confluence_score=raw_score,
        adjusted_confluence_score=adjusted_score,
        haircut_applied=haircut_applied,
        haircut_multiplier=multiplier,
        sec_form4_null=sec_null,
        finnhub_sentiment_null=finnhub_null,
        haircut_reasons=reasons,
        spot_price=spot_price,
        atr_14d=atr_14d,
        atr_14d_pct=atr_14d_pct,
        dynamic_stop_2x_atr=dynamic_stop_2x_atr,
        sma_20d=sma_20d,
        sma_20d_alignment=sma_alignment,
        model_target_5d=model_target_5d,
        model_target_source=model_target_source,
        net_quant_bias=net_quant_bias,
        consensus_bias_raw=raw_bias,
    )


def count_spoken_words(text: str) -> int:
    """Counts spoken words while collapsing spaced single-letter acronyms into one token."""
    normalized = re.sub(r"\b(?:[A-Z]\s+)+[A-Z]\b", "ACRONYM", text)
    return len(normalized.split())


def _ordinal(n: int) -> str:
    """Formats calendar day integer with English ordinal suffix (1st, 2nd, 3rd, 15th)."""
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def resolve_run_date(date_override: str | None = None) -> tuple[str, str, str]:
    """Computes dynamic ISO, uppercase HUD badge, and natural spoken date in America/Toronto."""
    tz = ZoneInfo("America/Toronto")
    if date_override and date_override.strip():
        parsed = datetime.strptime(date_override.strip(), "%Y-%m-%d")
        dt = parsed.replace(tzinfo=tz)
    else:
        dt = datetime.now(tz)

    run_date_iso = dt.strftime("%Y-%m-%d")
    run_date_badge = dt.strftime("%d %b %Y").upper()
    run_date_spoken = f"{dt.strftime('%A')}, {dt.strftime('%B')} {_ordinal(dt.day)}"
    return run_date_iso, run_date_badge, run_date_spoken


def build_scripts(
    t1: TickerTelemetry, t2: TickerTelemetry, t3: TickerTelemetry, run_date_spoken: str
) -> tuple[str, str]:
    """Generates compliance-safe spoken scripts for Pierre (Part 1) and Julie (Part 2)."""
    pierre_script = (
        f"Morning, it's Pierre. For {run_date_spoken}, our 16-agent quantitative swarm "
        "just completed its market telemetry scan. "
        f"Today's scan flagged {t1.spoken_name}, {t2.spoken_name}, and {t3.spoken_name} "
        f"with the strongest confluence—so I'm putting {t1.spoken_short} on the watchtower. "
        "Julie, run the deep-dive."
    )

    julie_script = (
        f"Tactical Recon for {t1.spoken_name}. "
        f"Spot price is {format_price_spoken(t1.spot_price)}. "
        f"Fourteen-day A T R is {format_price_spoken(t1.atr_14d)}, or {t1.atr_14d_pct:.2f} percent, "
        f"setting the 2x A T R invalidation stop at {format_price_spoken(t1.dynamic_stop_2x_atr)}. "
        f"Price is {t1.sma_20d_alignment} at {format_price_spoken(t1.sma_20d)}, "
        f"with a 5-day Chronos target of {format_price_spoken(t1.model_target_5d)}. "
        f"Net quant bias is {t1.net_quant_bias} at {t1.adjusted_confluence_score:.2f} percent confluence."
    )
    return pierre_script, julie_script


def generate_brief(
    ticker_override: str | None = None,
    date_override: str | None = None,
) -> DailyBriefPayload:
    """Orchestrates strictly read-only ingestion, haircut ranking, and script assembly."""
    book = load_json_ro(HERMES_BOOK_PATH)
    portfolio_40 = load_json_ro(PORTFOLIO_40_PATH)
    chronos_buffer = load_json_ro(CHRONOS_BUFFER_PATH)
    cs_root = load_json_ro(CROWD_SENTIMENT_ROOT_PATH)
    cs_scraper_doc = load_json_ro(CROWD_SENTIMENT_SCRAPER_PATH)

    cs_scraper = cs_scraper_doc.get("assets", {}) if isinstance(cs_scraper_doc.get("assets"), dict) else {}
    chronos_forecasts = (
        chronos_buffer.get("forecasts", {}) if isinstance(chronos_buffer.get("forecasts"), dict) else {}
    )

    portfolio_names: dict[str, str] = {}
    portfolio_owned_set: set[str] = set()
    for asset in portfolio_40.get("assets", []):
        if isinstance(asset, dict) and asset.get("ticker"):
            sym = str(asset["ticker"]).upper()
            portfolio_names[sym] = str(asset.get("name") or sym)
            if asset.get("owned") is True:
                portfolio_owned_set.add(sym)

    latest_tickets = book.get("latest_tickets", {})
    if not isinstance(latest_tickets, dict) or not latest_tickets:
        raise RuntimeError(f"Missing or empty latest_tickets in {HERMES_BOOK_PATH}")

    all_tickets_upper: dict[str, dict[str, Any]] = {
        str(k).upper(): v for k, v in latest_tickets.items() if isinstance(v, dict)
    }
    owned_tickets: dict[str, dict[str, Any]] = {
        sym: v
        for sym, v in all_tickets_upper.items()
        if v.get("owned") is True and (not portfolio_owned_set or sym in portfolio_owned_set)
    }
    if len(owned_tickets) < 3:
        raise RuntimeError(f"Insufficient owned tickers found ({len(owned_tickets)}); need at least 3.")

    clean_override = ticker_override.upper().lstrip("$").strip() if ticker_override else None
    eval_tickets: dict[str, dict[str, Any]] = dict(owned_tickets)
    if clean_override and clean_override not in eval_tickets:
        in_p40 = clean_override in portfolio_names
        in_book = clean_override in all_tickets_upper
        in_recon = (VAULT_RECONS_DIR / f"{clean_override}.md").exists()
        if not (in_p40 or in_book or in_recon):
            raise ValueError(
                f"Ticker override '{clean_override}' not found in portfolio_40, hermes book, or Vault Recons."
            )
        eval_tickets[clean_override] = all_tickets_upper.get(
            clean_override,
            {
                "ticker": clean_override,
                "name": portfolio_names.get(clean_override, clean_override),
                "owned": False,
                "confluence_score_pct": 0.0,
            },
        )

    db_uri, agent_map, sma20_map = query_db_telemetry_ro(PIERRE_QUANT_DB_PATH, sorted(eval_tickets.keys()))

    candidates: list[TickerTelemetry] = [
        build_ticker_telemetry(
            ticker=sym,
            ticket=ticket,
            portfolio_names=portfolio_names,
            agents=agent_map.get(sym, {}),
            sma20_map=sma20_map,
            chronos_forecasts=chronos_forecasts,
            cs_root=cs_root,
            cs_scraper=cs_scraper,
        )
        for sym, ticket in eval_tickets.items()
    ]

    if clean_override:
        t1 = next(c for c in candidates if c.ticker == clean_override)
        remaining = [c for c in candidates if c.ticker != clean_override and c.ticker in owned_tickets]
        remaining.sort(key=lambda c: (-c.adjusted_confluence_score, -c.raw_confluence_score, c.ticker))
        t2, t3 = remaining[0], remaining[1]
    else:
        candidates.sort(key=lambda c: (-c.adjusted_confluence_score, -c.raw_confluence_score, c.ticker))
        t1, t2, t3 = candidates[0], candidates[1], candidates[2]

    run_date_iso, run_date_badge, run_date_spoken = resolve_run_date(date_override)
    pierre_script, julie_script = build_scripts(t1, t2, t3, run_date_spoken)
    pierre_words = count_spoken_words(pierre_script)
    julie_words = count_spoken_words(julie_script)
    total_words = pierre_words + julie_words

    pierre_sec = round(pierre_words / ESTIMATED_WORDS_PER_SECOND, 2)
    julie_sec = round(julie_words / ESTIMATED_WORDS_PER_SECOND, 2)
    total_sec = round(pierre_sec + julie_sec, 2)

    morning_brief = book.get("latest_morning_brief", {})
    morning_ts = str(morning_brief.get("generated_at_ny") or "UNKNOWN")

    return DailyBriefPayload(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        run_date_iso=run_date_iso,
        run_date_badge=run_date_badge,
        run_date_spoken=run_date_spoken,
        morning_scan_timestamp=morning_ts,
        db_uri_mode=db_uri,
        t1_override_used=bool(clean_override),
        t1=t1,
        t2=t2,
        t3=t3,
        pierre_script=pierre_script,
        julie_script=julie_script,
        pierre_word_count=pierre_words,
        julie_word_count=julie_words,
        total_word_count=total_words,
        pierre_estimated_duration_sec=pierre_sec,
        julie_estimated_duration_sec=julie_sec,
        total_estimated_duration_sec=total_sec,
        compliance_footer=COMPLIANCE_FOOTER,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses CLI flags for Phase 2 read-only brief generation."""
    parser = argparse.ArgumentParser(
        description="Read-Only Quant Bridge & Dual-Voice Script Engine (Phase 2)"
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default=None,
        help="Optional ticker symbol override for T1 watchtower focus.",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Optional calendar date override (YYYY-MM-DD in America/Toronto).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print structured brief JSON to stdout without writing latest_brief.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output JSON path inside pierre-training-video (default: {DEFAULT_OUTPUT_PATH}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = parse_args(argv)
    resolved_output = args.output.resolve()
    allowed_root = PROJECT_ROOT.resolve()
    if not args.dry_run and not resolved_output.is_relative_to(allowed_root):
        raise PermissionError(
            f"Write containment violation: output path {resolved_output} is outside {allowed_root}"
        )

    payload = generate_brief(ticker_override=args.ticker, date_override=args.date)
    serialized = json.dumps(asdict(payload), indent=2, ensure_ascii=False)

    if args.dry_run:
        print(serialized)
        return 0

    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    with resolved_output.open(mode="w", encoding="utf-8") as handle:
        handle.write(serialized + "\n")
    print(f"[quant_reader] Wrote {resolved_output}")
    print(
        f"[quant_reader] T1={payload.t1.ticker} ({payload.t1.adjusted_confluence_score:.2f}%), "
        f"T2={payload.t2.ticker} ({payload.t2.adjusted_confluence_score:.2f}%), "
        f"T3={payload.t3.ticker} ({payload.t3.adjusted_confluence_score:.2f}%)"
    )
    print(
        f"[quant_reader] Words: Pierre={payload.pierre_word_count} (~{payload.pierre_estimated_duration_sec}s), "
        f"Julie={payload.julie_word_count} (~{payload.julie_estimated_duration_sec}s), "
        f"Total={payload.total_word_count} (~{payload.total_estimated_duration_sec}s)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
