from __future__ import annotations
import sys
from pathlib import Path

from dataclasses import dataclass
import sys
from typing import Dict, Iterable, List, Mapping, Optional, Sequence
if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd


def _tailend_start_time(
    token_prices: pd.DataFrame,
    *,
    threshold: float,
    required_pct: float,
) -> Optional[pd.Timestamp]:
    """
    Earliest timestamp where the remaining suffix of the series satisfies the tail-end condition.
    Returns None if no such point exists.
    """
    df = token_prices.copy()
    df["t"] = pd.to_datetime(df["t"], unit="s", utc=True, errors="coerce")
    df["p"] = pd.to_numeric(df["p"], errors="coerce")
    df = df.dropna(subset=["t", "p"]).sort_values("t").reset_index(drop=True)
    if df.empty:
        return None

    cond = (df["p"] >= threshold).to_numpy(dtype=bool)
    n = cond.size
    # suffix_mean[i] = mean(cond[i:])
    suffix_true = np.cumsum(cond[::-1])[::-1]
    suffix_mean = suffix_true / np.arange(n, 0, -1)

    eligible = np.where((suffix_mean >= required_pct) & cond)[0]
    if eligible.size == 0:
        return None
    return pd.Timestamp(df.loc[int(eligible[0]), "t"])


def volatility_main() -> None:
    # Heavy imports are local to keep `aggregate-trades` startup fast and memory-safe.
    import matplotlib.pyplot as plt

    from fetch.filtering import filter_by_duration
    from fetch.tail_end_func import find_tailend_markets_by_merged_prices, find_tailend_prices
    from fetch.volatility import (
        AssetVolatilityResult,
        VolatilityResult,
        asset_volatility_from_rolling_series,
        compute_average_tailend_volatility,
        compute_log_returns,
        compute_rolling_volatility,
        plot_reference_volatility_lines,
    )

    mode = 1
    annualize = False
    DIAGNOSTICS = True

    from config import PROJECT_ROOT
    from utils import read_markets_csv, read_prices_csv
    WINDOW = 20
    DAYS = 20
    markets_b = read_markets_csv(PROJECT_ROOT / "data/binary_markets.csv")
    markets_c = read_markets_csv(PROJECT_ROOT / "data/categorical_markets_all.csv")
    prices_b = read_prices_csv(PROJECT_ROOT / "data/prices_binary_all.csv")
    prices_c = read_prices_csv(PROJECT_ROOT / "data/prices_categorical_all.csv")

    markets_orig = pd.concat([markets_b, markets_c], ignore_index=True)
    prices = pd.concat([prices_b, prices_c], ignore_index=True)
    markets_orig = filter_by_duration(markets_orig, DAYS)
    tailend_markets = find_tailend_markets_by_merged_prices(markets_orig, prices, threshold=0.9, percent=0.5)
    if mode == 1:
        threshold = 0.9
        required_pct = 0.5
        tailend_prices = find_tailend_prices(markets=tailend_markets, prices=prices)
        tailend_prices["t"] = pd.to_datetime(tailend_prices["t"], unit="s", utc=True, errors="coerce")
        tailend_prices["p"] = pd.to_numeric(tailend_prices["p"], errors="coerce")
        tailend_prices["market_id"] = pd.to_numeric(tailend_prices["market_id"], errors="coerce").astype("Int64")
        tailend_prices = tailend_prices.dropna(subset=["market_id", "t", "p"]).sort_values(
            ["market_id", "t"]
        )

        start_times = (
            tailend_prices.groupby("market_id", sort=False)[["t", "p"]]
            .apply(lambda df: _tailend_start_time(df, threshold=threshold, required_pct=required_pct))
            .dropna()
            .rename("tailend_start")
            .reset_index()
        )

        # From here on, only use prices after each market's tail-end start time.
        prices_post = tailend_prices.merge(start_times, on="market_id", how="inner")
        prices_post = prices_post[prices_post["t"] >= prices_post["tailend_start"]]

        returns = compute_log_returns(prices_post[["market_id", "t", "p"]])
        rolling = compute_rolling_volatility(
            returns,
            window=WINDOW,
            freq_per_year=365,
            annualize=True,
        )
        avg_post = (
            rolling.groupby("market_id")["rolling_vol"]
            .mean()
            .reset_index()
            .rename(columns={"rolling_vol": "avg_annualized_vol_post_tailend"})
        )

        n_post = prices_post.groupby("market_id").size().reset_index(name="n_prices_post_tailend")

        summary = (
            start_times.merge(avg_post, on="market_id", how="inner")
            .merge(n_post, on="market_id", how="left")
            .sort_values("avg_annualized_vol_post_tailend", ascending=False)
        )
        out_path = "liquidation.csv"
        summary.to_csv(out_path, index=False)
        print(f"Wrote {len(summary)} markets to {out_path}")

        try:
            tailend_ids = pd.to_numeric(tailend_markets["id"], errors="coerce").dropna().astype(int).tolist()
            avg_vol = compute_average_tailend_volatility(
                prices_post,
                tailend_ids,
                window=WINDOW,
                smoothing=7,
                annualize=True,
                freq_per_year=365,
            )
            tailend_mean_vol = float(pd.to_numeric(avg_vol["rolling_vol"], errors="coerce").dropna().mean())
            tailend_avg_result = VolatilityResult(
                market_id=-1,
                prices=pd.DataFrame(columns=["market_id", "t", "p"]),
                returns=pd.DataFrame(columns=["market_id", "t", "log_return"]),
                rolling_volatility=avg_vol,
            )

            returns_orig = compute_log_returns(tailend_prices[["market_id", "t", "p"]])
            rolling_orig = compute_rolling_volatility(
                returns_orig,
                window=WINDOW,
                freq_per_year=365,
                annualize=True,
            )
            avg_orig_ts = (
                rolling_orig.assign(day=pd.to_datetime(rolling_orig["t"]).dt.floor("d"))
                .groupby("day", sort=True)["rolling_vol"]
                .mean()
                .reset_index()
                .rename(columns={"day": "t"})
            )
            avg_orig_ts["rolling_vol"] = avg_orig_ts["rolling_vol"].rolling(window=7, min_periods=1).mean()
            orig_mean_vol = float(pd.to_numeric(avg_orig_ts["rolling_vol"], errors="coerce").dropna().mean())

            if DIAGNOSTICS:
                def _print_spike_diagnostics(rolling_df: pd.DataFrame, *, name: str) -> None:
                    df = rolling_df[["market_id", "t", "rolling_vol"]].copy()
                    df["t"] = pd.to_datetime(df["t"], utc=True, errors="coerce")
                    df["rolling_vol"] = pd.to_numeric(df["rolling_vol"], errors="coerce")
                    df = df.dropna(subset=["t", "rolling_vol"]).reset_index(drop=True)
                    df["day"] = df["t"].dt.floor("d")

                    diag_start = pd.Timestamp("2024-01-01", tz="UTC")
                    diag_end = pd.Timestamp("2024-02-01", tz="UTC")
                    window_df = df[(df["day"] >= diag_start) & (df["day"] <= diag_end)]
                    if window_df.empty:
                        print(f"[diag] {name}: no data in diagnostic window")
                        return

                    daily_mean = window_df.groupby("day", sort=True)["rolling_vol"].mean()
                    print(f"\n[diag] {name}: daily mean volatility ({diag_start.date()}..{diag_end.date()})")
                    for day, mean_vol in daily_mean.sort_index().items():
                        day_df = window_df[window_df["day"] == day]
                        top_markets = (
                            day_df.groupby("market_id")["rolling_vol"]
                            .mean()
                            .sort_values(ascending=False)
                            .head(5)
                        )
                        markets_str = ", ".join(f"{int(mid)}:{v:.3f}" for mid, v in top_markets.items())
                        print(f"  {day.date()} mean={mean_vol:.3f} top_markets=[{markets_str}]")

                _print_spike_diagnostics(rolling_orig, name="original (all tailend markets)")
                _print_spike_diagnostics(rolling, name="post-tailend (suffix only)")

            orig_ref: AssetVolatilityResult = asset_volatility_from_rolling_series(
                avg_orig_ts,
                symbol="ORIG",
                label=f"Original avg rolling volatility (mean={orig_mean_vol:.3f})",
            )

            plot_reference_volatility_lines(
                references=[orig_ref],
                tailend_result=tailend_avg_result,
                tailend_label=f"Volatility after entering tail-end stage (mean={tailend_mean_vol:.3f})",
                title=(f"Comparison between overall tail-end rolling volatility "
                        f"and rolling volatility after the market enters the tail-end stage "
                        f"(window={WINDOW})"
                    ),
            )
            plt.show()
        except ImportError as exc:
            print(f"Skipping reference volatility plot (missing dependency): {exc}")
        except ValueError as exc:
            print(f"Skipping reference volatility plot: {exc}")
   
   

@dataclass(slots=True)
class MarketAgg:
    total_trades: int = 0
    total_volume: float = 0.0
    yes_volume: float = 0.0
    no_volume: float = 0.0
    traders: set[str] | None = None


@dataclass(slots=True)
class TraderAgg:
    total_trades: int = 0
    total_volume: float = 0.0
    yes_trades: int = 0
    no_trades: int = 0


@dataclass(slots=True)
class TraderGroupAgg:
    """
    Per-trader aggregates split by market group.

    This enables hypothesis-driven comparisons (e.g., tail-end vs non-tail-end)
    without storing raw trades: we keep only (trader, group) totals.
    """

    total_trades: int = 0
    total_volume: float = 0.0
    yes_trades: int = 0
    no_trades: int = 0


def _read_csv_chunks(
    csv_path: Path,
    *,
    usecols: list[str],
    chunksize: int,
    use_pyarrow: bool | None,
) -> Iterable[pd.DataFrame]:
    if use_pyarrow is None:
        try:
            import pyarrow  # noqa: F401

            use_pyarrow = True
        except Exception:
            use_pyarrow = False

    read_kwargs: dict = {
        "filepath_or_buffer": csv_path,
        "usecols": usecols,
        "chunksize": int(chunksize),
        "encoding": "utf-8",
    }

    if use_pyarrow:
        read_kwargs.update({"engine": "pyarrow", "dtype_backend": "pyarrow"})
    else:
        read_kwargs.update(
            {
                "engine": "c",
                "dtype": {c: "string" for c in usecols if c not in ("size", "price", "timestamp")},
            }
        )

    yield from pd.read_csv(**read_kwargs)


def aggregate_market_trades_csv(
    csv_path: str | Path,
    *,
    condition_ids: Sequence[str],
    chunksize: int = 1_000_000,
    volume_mode: str = "size",
    use_pyarrow: bool | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Memory-safe trade aggregation for very large CSVs.

    Key memory properties:
    - Uses `pd.read_csv(..., chunksize=...)` to stream rows.
    - Keeps only small accumulator dicts/sets in memory (no raw trades).
    - Restricts columns via `usecols` to reduce per-chunk footprint.

    Parameters
    ----------
    csv_path:
        Path to `market_trades.csv` (UTF-8, header row).
    condition_ids:
        List of market conditionIds to keep (e.g., 5 markets).
    chunksize:
        Number of CSV rows per chunk.
    volume_mode:
        "size" => traded volume = sum(size)
        "notional" => traded volume = sum(size * price)
    use_pyarrow:
        If True, uses pandas' pyarrow engine/backend when available (often lower memory).

    Returns
    -------
    (market_df, trader_df):
        Small DataFrames with final aggregates.
    """

    csv_path = Path(csv_path)
    target = {str(x) for x in condition_ids}
    if not target:
        raise ValueError("condition_ids must be non-empty")

    if volume_mode not in ("size", "notional"):
        raise ValueError("volume_mode must be 'size' or 'notional'")

    usecols = ["proxyWallet", "conditionId", "outcome", "size", "price"]
    # Stream chunks for memory safety (never loads the full file).

    market_acc: dict[str, MarketAgg] = {}
    trader_acc: dict[str, TraderAgg] = {}

    for chunk in _read_csv_chunks(csv_path, usecols=usecols, chunksize=chunksize, use_pyarrow=use_pyarrow):
        # Filter first to minimize processing and keep the rest of the chunk discardable.
        chunk = chunk[chunk["conditionId"].astype(str).isin(target)]
        if chunk.empty:
            continue

        # Robust coercion (handles occasional bad rows without blowing up the stream).
        chunk["proxyWallet"] = chunk["proxyWallet"].astype(str)
        chunk["conditionId"] = chunk["conditionId"].astype(str)
        chunk["outcome"] = chunk["outcome"].astype(str)
        chunk["size"] = pd.to_numeric(chunk["size"], errors="coerce")
        chunk["price"] = pd.to_numeric(chunk["price"], errors="coerce")
        chunk = chunk.dropna(subset=["proxyWallet", "conditionId", "outcome", "size"])
        if chunk.empty:
            continue

        # Compute per-row volume once for the chunk to keep the inner loop minimal.
        if volume_mode == "size":
            chunk["_vol"] = chunk["size"].astype("float64")
        else:
            chunk["_vol"] = (chunk["size"] * chunk["price"]).astype("float64")

        # Iterate rows without materializing extra objects; update only dict accumulators.
        # This avoids holding any raw trade history in memory.
        for wallet, condition_id, outcome, vol in chunk[["proxyWallet", "conditionId", "outcome", "_vol"]].itertuples(
            index=False, name=None
        ):
            if not (isinstance(vol, (int, float)) and np.isfinite(vol)):
                continue

            outcome_norm = str(outcome).strip().lower()
            is_yes = outcome_norm in ("yes", "y", "true", "1")

            m = market_acc.get(condition_id)
            if m is None:
                m = MarketAgg(traders=set())
                market_acc[condition_id] = m

            m.total_trades += 1
            m.total_volume += float(vol)
            if is_yes:
                m.yes_volume += float(vol)
            else:
                m.no_volume += float(vol)
            assert m.traders is not None
            m.traders.add(wallet)

            t = trader_acc.get(wallet)
            if t is None:
                t = TraderAgg()
                trader_acc[wallet] = t
            t.total_trades += 1
            t.total_volume += float(vol)
            if is_yes:
                t.yes_trades += 1
            else:
                t.no_trades += 1

        # Explicitly drop the temporary column to keep peak chunk memory lower.
        chunk.drop(columns=["_vol"], inplace=True, errors="ignore")

    market_rows: list[dict] = []
    for condition_id, m in market_acc.items():
        unique_traders = len(m.traders or ())
        market_rows.append(
            {
                "conditionId": condition_id,
                "total_trades": int(m.total_trades),
                "total_volume": float(m.total_volume),
                "yes_volume": float(m.yes_volume),
                "no_volume": float(m.no_volume),
                "unique_traders": int(unique_traders),
            }
        )

    trader_rows: list[dict] = []
    for wallet, t in trader_acc.items():
        trader_rows.append(
            {
                "proxyWallet": wallet,
                "total_trades": int(t.total_trades),
                "total_volume": float(t.total_volume),
                "yes_trades": int(t.yes_trades),
                "no_trades": int(t.no_trades),
            }
        )

    market_df = pd.DataFrame(market_rows).sort_values(["total_volume", "total_trades"], ascending=False)
    trader_df = pd.DataFrame(trader_rows).sort_values(["total_volume", "total_trades"], ascending=False)
    return market_df.reset_index(drop=True), trader_df.reset_index(drop=True)


def aggregate_market_trades_by_group_csv(
    csv_path: str | Path,
    *,
    condition_id_to_group: Mapping[str, str],
    chunksize: int = 1_000_000,
    volume_mode: str = "size",
    use_pyarrow: bool | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stream CSV and compute per-(group, trader) aggregates (plus per-market aggregates).

    Memory safety:
    - No trade-level retention; only small dict accumulators are kept.
    - Optional day-bucketed volume per group is stored as a small dict keyed by (group, day).
    """

    csv_path = Path(csv_path)
    mapping = {str(k): str(v) for k, v in condition_id_to_group.items()}
    target = set(mapping.keys())
    if not target:
        raise ValueError("condition_id_to_group must be non-empty")

    if volume_mode not in ("size", "notional"):
        raise ValueError("volume_mode must be 'size' or 'notional'")

    usecols = ["proxyWallet", "conditionId", "outcome", "size", "price", "timestamp", "side"]
    market_acc: dict[str, MarketAgg] = {}
    trader_group_acc: dict[tuple[str, str], TraderGroupAgg] = {}
    group_day_volume: dict[tuple[str, int], float] = {}

    for chunk in _read_csv_chunks(csv_path, usecols=usecols, chunksize=chunksize, use_pyarrow=use_pyarrow):
        chunk = chunk[chunk["conditionId"].astype(str).isin(target)]
        if chunk.empty:
            continue

        chunk["proxyWallet"] = chunk["proxyWallet"].astype(str)
        chunk["conditionId"] = chunk["conditionId"].astype(str)
        chunk["outcome"] = chunk["outcome"].astype(str)
        chunk["size"] = pd.to_numeric(chunk["size"], errors="coerce")
        chunk["price"] = pd.to_numeric(chunk["price"], errors="coerce")
        chunk["timestamp"] = pd.to_numeric(chunk["timestamp"], errors="coerce")
        chunk = chunk.dropna(subset=["proxyWallet", "conditionId", "outcome", "size", "timestamp"])
        if chunk.empty:
            continue

        if volume_mode == "size":
            chunk["_vol"] = chunk["size"].astype("float64")
        else:
            chunk["_vol"] = (chunk["size"] * chunk["price"]).astype("float64")

        chunk["_day"] = (chunk["timestamp"].astype("int64") // 86_400).astype("int64")

        for wallet, condition_id, outcome, vol, day in chunk[
            ["proxyWallet", "conditionId", "outcome", "_vol", "_day"]
        ].itertuples(index=False, name=None):
            if not (isinstance(vol, (int, float)) and np.isfinite(vol)):
                continue
            group = mapping.get(condition_id)
            if group is None:
                continue

            outcome_norm = str(outcome).strip().lower()
            is_yes = outcome_norm in ("yes", "y", "true", "1")

            m = market_acc.get(condition_id)
            if m is None:
                m = MarketAgg(traders=set())
                market_acc[condition_id] = m
            m.total_trades += 1
            m.total_volume += float(vol)
            if is_yes:
                m.yes_volume += float(vol)
            else:
                m.no_volume += float(vol)
            assert m.traders is not None
            m.traders.add(wallet)

            key = (group, wallet)
            tg = trader_group_acc.get(key)
            if tg is None:
                tg = TraderGroupAgg()
                trader_group_acc[key] = tg
            tg.total_trades += 1
            tg.total_volume += float(vol)
            if is_yes:
                tg.yes_trades += 1
            else:
                tg.no_trades += 1

            gd_key = (group, int(day))
            group_day_volume[gd_key] = group_day_volume.get(gd_key, 0.0) + float(vol)

        chunk.drop(columns=["_vol", "_day"], inplace=True, errors="ignore")

    market_rows: list[dict] = []
    for condition_id, m in market_acc.items():
        market_rows.append(
            {
                "conditionId": condition_id,
                "group": mapping.get(condition_id, "UNKNOWN"),
                "total_trades": int(m.total_trades),
                "total_volume": float(m.total_volume),
                "yes_volume": float(m.yes_volume),
                "no_volume": float(m.no_volume),
                "unique_traders": int(len(m.traders or ())),
            }
        )
    market_df = pd.DataFrame(market_rows).sort_values(["group", "total_volume"], ascending=[True, False]).reset_index(
        drop=True
    )

    trader_rows: list[dict] = []
    for (group, wallet), tg in trader_group_acc.items():
        total = tg.yes_trades + tg.no_trades
        yes_rate = (tg.yes_trades / total) if total else np.nan
        trader_rows.append(
            {
                "group": group,
                "proxyWallet": wallet,
                "total_trades": int(tg.total_trades),
                "total_volume": float(tg.total_volume),
                "yes_trades": int(tg.yes_trades),
                "no_trades": int(tg.no_trades),
                "yes_trade_rate": float(yes_rate) if np.isfinite(yes_rate) else np.nan,
            }
        )

    trader_df = pd.DataFrame(trader_rows).sort_values(["group", "total_volume"], ascending=[True, False]).reset_index(
        drop=True
    )

    group_daily_volume_df = pd.DataFrame(columns=["group", "day", "daily_volume"])
    if group_day_volume:
        gd_rows = [{"group": g, "day": d, "daily_volume": v} for (g, d), v in group_day_volume.items()]
        group_daily_volume_df = pd.DataFrame(gd_rows).sort_values(["group", "day"]).reset_index(drop=True)

    return market_df, trader_df, group_daily_volume_df


def _gini(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    if np.any(x < 0):
        return np.nan
    s = x.sum()
    if s <= 0:
        return 0.0
    x_sorted = np.sort(x)
    n = x_sorted.size
    index = np.arange(1, n + 1, dtype=float)
    return float((2.0 * (index * x_sorted).sum() / (n * s)) - (n + 1.0) / n)


def _hhi_from_shares(shares: np.ndarray) -> float:
    s = np.asarray(shares, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return np.nan
    return float(np.square(s).sum())


def _welch_ttest(x: np.ndarray, y: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    nx, ny = x.size, y.size
    if nx < 2 or ny < 2:
        return {"t": np.nan, "df": np.nan, "nx": int(nx), "ny": int(ny)}
    mx, my = float(x.mean()), float(y.mean())
    vx, vy = float(x.var(ddof=1)), float(y.var(ddof=1))
    denom = np.sqrt(vx / nx + vy / ny)
    if denom == 0:
        return {"t": np.nan, "df": np.nan, "nx": int(nx), "ny": int(ny)}
    t = (mx - my) / denom
    df_num = (vx / nx + vy / ny) ** 2
    df_den = (vx * vx) / (nx * nx * (nx - 1)) + (vy * vy) / (ny * ny * (ny - 1))
    df = df_num / df_den if df_den > 0 else np.nan
    return {"t": float(t), "df": float(df), "nx": int(nx), "ny": int(ny), "mean_x": mx, "mean_y": my}


def _ks_statistic(x: np.ndarray, y: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=float)[np.isfinite(x)])
    y = np.sort(np.asarray(y, dtype=float)[np.isfinite(y)])
    if x.size == 0 or y.size == 0:
        return np.nan
    all_vals = np.sort(np.concatenate([x, y]))
    cdf_x = np.searchsorted(x, all_vals, side="right") / x.size
    cdf_y = np.searchsorted(y, all_vals, side="right") / y.size
    return float(np.max(np.abs(cdf_x - cdf_y)))


def _permutation_pvalue(
    x: np.ndarray,
    y: np.ndarray,
    *,
    stat_fn,
    n_perm: int = 2_000,
    seed: int = 0,
) -> float:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return np.nan
    observed = float(stat_fn(x, y))
    pooled = np.concatenate([x, y])
    n = x.size
    count = 0
    for _ in range(int(n_perm)):
        rng.shuffle(pooled)
        xs = pooled[:n]
        ys = pooled[n:]
        if float(stat_fn(xs, ys)) >= observed:
            count += 1
    return float((count + 1) / (n_perm + 1))


def _summarize_distribution(series: pd.Series, *, name: str) -> pd.DataFrame:
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return pd.DataFrame([{"metric": name, "n": 0}])
    q = s.quantile([0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99]).to_dict()
    return pd.DataFrame(
        [
            {
                "metric": name,
                "n": int(s.size),
                "mean": float(s.mean()),
                "std": float(s.std(ddof=1)),
                "min": float(s.min()),
                "p01": float(q.get(0.01)),
                "p05": float(q.get(0.05)),
                "p10": float(q.get(0.1)),
                "p50": float(q.get(0.5)),
                "p90": float(q.get(0.9)),
                "p95": float(q.get(0.95)),
                "p99": float(q.get(0.99)),
                "max": float(s.max()),
            }
        ]
    )


def analyze_trader_behavior(
    trader_by_group: pd.DataFrame,
    *,
    group_daily_volume: pd.DataFrame | None = None,
    heavy_tail_log: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Thesis-oriented descriptive statistics + hypothesis-driven comparisons.

    Interpretive mapping (use in writeup):
    - Heavy tails in `total_volume`/`total_trades` -> heterogeneous attention/information;
      small agents can dominate order flow (speculation / informed trading).
    - Concentration (Gini/HHI/top shares) -> breadth of participation; broader participation
      tends to improve belief aggregation and reduce manipulation risk.
    - `yes_trade_rate` differences across groups -> directional crowding/herding and belief skew.
    - Daily volume concentration -> episodic attention shocks; tail-end markets may show bursts
      as resolution approaches, consistent with speculation and information arrival.
    """

    if trader_by_group.empty:
        raise ValueError("trader_by_group is empty")

    df = trader_by_group.copy()
    df["total_volume"] = pd.to_numeric(df["total_volume"], errors="coerce")
    df["total_trades"] = pd.to_numeric(df["total_trades"], errors="coerce")
    df["yes_trade_rate"] = pd.to_numeric(df.get("yes_trade_rate"), errors="coerce")

    if heavy_tail_log:
        df["log1p_volume"] = np.log1p(df["total_volume"].clip(lower=0))
        df["log1p_trades"] = np.log1p(df["total_trades"].clip(lower=0))

    summaries: list[pd.DataFrame] = []
    for group, gdf in df.groupby("group", sort=True):
        summaries.append(_summarize_distribution(gdf["total_volume"], name=f"{group}: total_volume"))
        summaries.append(_summarize_distribution(gdf["total_trades"], name=f"{group}: total_trades"))
        summaries.append(_summarize_distribution(gdf["yes_trade_rate"], name=f"{group}: yes_trade_rate"))
        if heavy_tail_log:
            summaries.append(_summarize_distribution(gdf["log1p_volume"], name=f"{group}: log1p_volume"))
            summaries.append(_summarize_distribution(gdf["log1p_trades"], name=f"{group}: log1p_trades"))
    dist_summary = pd.concat(summaries, ignore_index=True)

    conc_rows: list[dict] = []
    for group, gdf in df.groupby("group", sort=True):
        vols = pd.to_numeric(gdf["total_volume"], errors="coerce").fillna(0).to_numpy(float)
        total = float(np.sum(vols))
        shares = (vols / total) if total > 0 else np.zeros_like(vols)
        shares_sorted = np.sort(shares)[::-1]
        conc_rows.append(
            {
                "group": group,
                "n_traders": int(gdf.shape[0]),
                "volume_gini": _gini(vols),
                "volume_hhi": _hhi_from_shares(shares),
                "top1_share": float(shares_sorted[0]) if shares_sorted.size else np.nan,
                "top5_share": float(shares_sorted[:5].sum()) if shares_sorted.size >= 5 else float(shares_sorted.sum()),
                "top10_share": float(shares_sorted[:10].sum())
                if shares_sorted.size >= 10
                else float(shares_sorted.sum()),
            }
        )
    concentration = pd.DataFrame(conc_rows).sort_values("group").reset_index(drop=True)

    groups = sorted(df["group"].dropna().astype(str).unique().tolist())
    feature = "log1p_volume" if heavy_tail_log else "total_volume"
    test_rows: list[dict] = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            g1, g2 = groups[i], groups[j]
            x = df.loc[df["group"] == g1, feature].to_numpy()
            y = df.loc[df["group"] == g2, feature].to_numpy()
            t = _welch_ttest(x, y)
            ks = _ks_statistic(x, y)
            ks_p = _permutation_pvalue(x, y, stat_fn=_ks_statistic, n_perm=1000, seed=0)
            mean_stat = lambda a, b: abs(float(np.mean(a) - np.mean(b)))
            perm_p = _permutation_pvalue(x, y, stat_fn=mean_stat, n_perm=2000, seed=1)
            test_rows.append(
                {
                    "group_a": g1,
                    "group_b": g2,
                    "feature": feature,
                    "welch_t": t.get("t"),
                    "welch_df": t.get("df"),
                    "perm_p_mean_diff": perm_p,
                    "ks_D": ks,
                    "ks_perm_p": ks_p,
                    "n_a": t.get("nx"),
                    "n_b": t.get("ny"),
                }
            )
    hypothesis_tests = pd.DataFrame(test_rows).sort_values(["group_a", "group_b"]).reset_index(drop=True)

    time_concentration = pd.DataFrame()
    if group_daily_volume is not None and not group_daily_volume.empty:
        gdf = group_daily_volume.copy()
        gdf["daily_volume"] = pd.to_numeric(gdf["daily_volume"], errors="coerce").fillna(0.0)
        rows = []
        for group, gg in gdf.groupby("group", sort=True):
            daily = gg["daily_volume"].to_numpy(float)
            total = float(daily.sum())
            shares = (daily / total) if total > 0 else np.zeros_like(daily)
            rows.append(
                {
                    "group": group,
                    "n_days": int(gg.shape[0]),
                    "daily_volume_gini": _gini(daily),
                    "daily_volume_hhi": _hhi_from_shares(shares),
                }
            )
        time_concentration = pd.DataFrame(rows).sort_values("group").reset_index(drop=True)

    return {
        "distribution_summary": dist_summary,
        "concentration": concentration,
        "hypothesis_tests": hypothesis_tests,
        "time_concentration": time_concentration,
    }

def main(argv: Sequence[str] | None = None) -> None:
    """
    Backwards-compatible entrypoint:
    - No args: runs the existing volatility workflow.
    - `aggregate-trades ...`: runs memory-safe CSV aggregation.
    """
    import argparse

    parser = argparse.ArgumentParser(prog="python -m fetch.trades")
    sub = parser.add_subparsers(dest="cmd")

    agg = sub.add_parser("aggregate-trades", help="Stream market_trades.csv and compute aggregates")
    agg.add_argument("--csv", required=True, help="Path to market_trades.csv")
    agg.add_argument(
        "--condition-id",
        action="append",
        required=True,
        dest="condition_ids",
        help="ConditionId to include (repeatable)",
    )
    agg.add_argument("--chunksize", type=int, default=1_000_000)
    agg.add_argument("--volume-mode", choices=["size", "notional"], default="size")
    agg.add_argument("--out-market", default="market_aggregates.csv")
    agg.add_argument("--out-trader", default="trader_aggregates.csv")

    ana = sub.add_parser("analyze-traders", help="Compute thesis-oriented trader behavior statistics")
    ana.add_argument("--csv", required=True, help="Path to market_trades.csv")
    ana.add_argument(
        "--tailend-id",
        action="append",
        default=[],
        dest="tailend_ids",
        help="Tail-end conditionId (repeatable)",
    )
    ana.add_argument(
        "--control-id",
        action="append",
        default=[],
        dest="control_ids",
        help="Control/non-tail-end conditionId (repeatable)",
    )
    ana.add_argument("--chunksize", type=int, default=1_000_000)
    ana.add_argument("--volume-mode", choices=["size", "notional"], default="size")
    ana.add_argument("--out-dir", default="trader_analysis_out", help="Directory to write analysis CSVs")

    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.cmd == "aggregate-trades":
        market_df, trader_df = aggregate_market_trades_csv(
            args.csv,
            condition_ids=args.condition_ids,
            chunksize=args.chunksize,
            volume_mode=args.volume_mode,
        )
        market_df.to_csv(args.out_market, index=False)
        trader_df.to_csv(args.out_trader, index=False)
        print(f"Wrote {len(market_df)} markets to {args.out_market}")
        print(f"Wrote {len(trader_df)} traders to {args.out_trader}")
        return

    if args.cmd == "analyze-traders":
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        mapping: dict[str, str] = {}
        for cid in args.tailend_ids:
            mapping[str(cid)] = "tailend"
        for cid in args.control_ids:
            mapping[str(cid)] = "control"
        if not mapping:
            raise SystemExit("Provide at least one --tailend-id and/or --control-id")

        market_df, trader_df, group_daily_volume = aggregate_market_trades_by_group_csv(
            args.csv,
            condition_id_to_group=mapping,
            chunksize=args.chunksize,
            volume_mode=args.volume_mode,
        )
        market_df.to_csv(out_dir / "market_by_group.csv", index=False)
        trader_df.to_csv(out_dir / "trader_by_group.csv", index=False)

        if isinstance(group_daily_volume, pd.DataFrame) and not group_daily_volume.empty:
            group_daily_volume.to_csv(out_dir / "group_daily_volume.csv", index=False)

        results = analyze_trader_behavior(
            trader_df,
            group_daily_volume=group_daily_volume if not group_daily_volume.empty else None,
        )
        for name, df in results.items():
            df.to_csv(out_dir / f"{name}.csv", index=False)

        print(f"Wrote outputs to {out_dir}")
        return

    volatility_main()


if __name__ == "__main__":
    main()
