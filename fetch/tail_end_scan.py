"""Scan Gamma markets and list tail-end markets that are unique to their event.

Usage example
-------------
python3 tail_end_scan.py --limit 1000 --time-share 0.5 --lead-time-days 90 --min-duration-days 30

The script relies on helper routines in ``tail_end_func`` and will hit the public
Gamma and CLOB APIs. Adjust the ``--limit`` argument if you only want to inspect
recent markets.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, Iterable, List

import pandas as pd

from fetch.tail_end_func import (
    YES_INDEX,
    add_event_uniqueness_flags,
    fetch_market,
    fetch_market_prices_history,
    fetch_markets,
    first_time_above_threshold,
    is_tailend_by_time_share,
    is_tailend_far_from_resolution,
    share_above_threshold,
    share_time_near_level,
)


DEFAULT_LEVEL = 0.90
DEFAULT_TOLERANCE = 0.02


def _collect_tailend_markets(
    markets: pd.DataFrame,
    level: float,
    tolerance: float,
    min_share: float,
    lead_time_days: int,
    min_duration_days: int,
    fidelity: int,
) -> List[Dict[str, Any]]:
    """Return tail-end market records with diagnostics."""
    candidates = markets[markets["is_single_event_market"]].copy()
    results: List[Dict[str, Any]] = []

    for _, market in candidates.iterrows():
        market_id = str(market["id"])
        try:
            prices = fetch_market_prices_history(market_id, YES_INDEX, fidelity=fidelity)
        except Exception as exc:  # network or parsing errors
            print(f"Failed to fetch prices for {market_id}: {exc}", file=sys.stderr)
            continue

        if prices.empty:
            continue

        share_near = share_time_near_level(prices, level=level, tolerance=tolerance)
        share_high = share_above_threshold(prices, threshold=level)
        time_share_flag = share_near >= min_share
        early_flag = is_tailend_far_from_resolution(
            market,
            prices,
            threshold=level,
            lead_time_days=lead_time_days,
            min_duration_days=min_duration_days,
            tolerance=tolerance,
        )

        if not (time_share_flag or early_flag):
            continue

        first_high_ts = first_time_above_threshold(prices, threshold=level)
        results.append(
            {
                "id": market_id,
                "question": market.get("question"),
                "primary_event_key": market.get("primary_event_key"),
                "share_near_level": share_near,
                "share_above_level": share_high,
                "time_share_flag": time_share_flag,
                "early_high_flag": early_flag,
                "first_high_ts": first_high_ts.isoformat() if pd.notna(first_high_ts) else None,
            }
        )

    return results


def find_tailend_markets(
    limit: int,
    offset: int,
    level: float,
    tolerance: float,
    min_share: float,
    lead_time_days: int,
    min_duration_days: int,
    fidelity: int,
) -> pd.DataFrame:
    """High-level wrapper returning a DataFrame of tail-end markets."""
    markets = fetch_markets(size=limit, offset=offset)
    if markets.empty:
        return pd.DataFrame()

    annotated = add_event_uniqueness_flags(markets)
    records = _collect_tailend_markets(
        annotated,
        level=level,
        tolerance=tolerance,
        min_share=min_share,
        lead_time_days=lead_time_days,
        min_duration_days=min_duration_days,
        fidelity=fidelity,
    )
    return pd.DataFrame.from_records(records)


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="Number of markets to fetch (0 => all pages)")
    parser.add_argument("--offset", type=int, default=500, help="Initial offset for pagination")
    parser.add_argument("--level", type=float, default=DEFAULT_LEVEL, help="Tail-end probability level (e.g., 0.9)")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help="Tolerance around the level for the time-share rule",
    )
    parser.add_argument("--time-share", type=float, default=0.5, help="Minimum fraction of time near the level")
    parser.add_argument(
        "--lead-time-days",
        type=int,
        default=90,
        help="Minimum days before resolution for the early-high rule",
    )
    parser.add_argument(
        "--min-duration-days",
        type=int,
        default=30,
        help="Minimum days spent above the level before the lead-time cutoff",
    )
    parser.add_argument(
        "--fidelity",
        type=int,
        default=1440,
        help="Sampling fidelity passed to the prices-history endpoint",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])

    tailend_df = find_tailend_markets(
        limit=args.limit,
        offset=args.offset,
        level=args.level,
        tolerance=args.tolerance,
        min_share=args.time_share,
        lead_time_days=args.lead_time_days,
        min_duration_days=args.min_duration_days,
        fidelity=args.fidelity,
    )

    if tailend_df.empty:
        print("No tail-end unique markets found with the current settings.")
        return 0

    display_cols = [
        "id",
        "question",
        "primary_event_key",
        "share_near_level",
        "share_above_level",
        "time_share_flag",
        "early_high_flag",
        "first_high_ts",
    ]
    tailend_df = tailend_df[display_cols]
    pd.set_option("display.max_rows", None)
    print(tailend_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
