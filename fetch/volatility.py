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
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from imports import *
try:
    import yfinance as yf  # type: ignore
except ImportError:  # pragma: no cover
    yf = None


@dataclass
class VolatilityResult:
    """Container for price/returns/volatility series of a single market."""

    market_id: int
    prices: pd.DataFrame
    returns: pd.DataFrame
    rolling_volatility: pd.DataFrame


@dataclass
class AssetVolatilityResult:
    """Container for external asset volatility series (e.g., SP500, BTC)."""

    symbol: str
    label: str
    prices: pd.DataFrame
    returns: pd.DataFrame
    rolling_volatility: pd.DataFrame


def asset_volatility_from_rolling_series(
    rolling_volatility: pd.DataFrame,
    *,
    symbol: str,
    label: Optional[str] = None,
    ts_col: str = "t",
    vol_col: str = "rolling_vol",
) -> AssetVolatilityResult:
    """
    Wrap a precomputed rolling-volatility time series as an AssetVolatilityResult.
    This is useful for plotting custom/reference lines alongside other assets.
    """
    df = rolling_volatility[[ts_col, vol_col]].copy()
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df[vol_col] = pd.to_numeric(df[vol_col], errors="coerce")
    df = df.dropna(subset=[ts_col, vol_col]).sort_values(ts_col).reset_index(drop=True)
    df["market_id"] = pd.Series(symbol, index=df.index, dtype="string")
    df = df.rename(columns={ts_col: "t", vol_col: "rolling_vol"})

    empty_prices = pd.DataFrame(columns=["t", "price", "market_id"])
    empty_returns = pd.DataFrame(columns=["t", "price", "log_return", "market_id"])
    return AssetVolatilityResult(
        symbol=symbol,
        label=label or symbol,
        prices=empty_prices,
        returns=empty_returns,
        rolling_volatility=df[["t", "rolling_vol", "market_id"]],
    )


def _normalize_price_frame(
    prices: pd.DataFrame,
    *,
    market_col: str = "market_id",
    ts_col: str = "t",
    price_col: str = "p",
) -> pd.DataFrame:
    df = prices[[market_col, ts_col, price_col]].copy()
    df[market_col] = pd.to_numeric(df[market_col], errors="coerce").astype("Int64")
    df[ts_col] = pd.to_datetime(df[ts_col], unit="s", utc=True, errors="coerce")
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=[market_col, ts_col, price_col])
    return df.sort_values([market_col, ts_col]).reset_index(drop=True)


def compute_log_returns(
    prices: pd.DataFrame,
    *,
    market_col: str = "market_id",
    ts_col: str = "t",
    price_col: str = "p",
) -> pd.DataFrame:
    """
    Compute log returns per market. Returns DataFrame with columns:
    [market_id, t, price, log_return].
    """
    df = _normalize_price_frame(prices, market_col=market_col, ts_col=ts_col, price_col=price_col)
    df[price_col] = df[price_col].clip(1e-9, 1 - 1e-9)
    df["log_price"] = np.log(df[price_col])
    df["log_return"] = (
        df.groupby(market_col)["log_price"]
        .diff()
        .fillna(0.0)
    )
    return df.drop(columns=["log_price"])


def compute_rolling_volatility(
    returns: pd.DataFrame,
    *,
    market_col: str = "market_id",
    ts_col: str = "t",
    return_col: str = "log_return",
    window: int = 24,
    annualize: bool = False,
    freq_per_year: int = 365,
) -> pd.DataFrame:
    """
    Compute rolling standard deviation of returns for each market.
    If annualize is True, multiply rolling std by sqrt(freq_per_year).
    """
    df = returns.copy()
    df = df.sort_values([market_col, ts_col])
    df["rolling_vol"] = (
        df.groupby(market_col)[return_col]
        .rolling(window=window, min_periods=max(2, window // 2))
        .std()
        .reset_index(level=0, drop=True)
    )
    if annualize:
        df["rolling_vol"] = df["rolling_vol"] * np.sqrt(freq_per_year)
    return df.dropna(subset=["rolling_vol"])


def analyze_market_volatility(
    prices: pd.DataFrame,
    market_id: int,
    *,
    window: int = 24,
    market_col: str = "market_id",
    ts_col: str = "t",
    price_col: str = "p",
    annualize: bool = False,
    freq_per_year: int = 365,
) -> VolatilityResult:
    """Return price, return, and rolling-volatility series for one market."""
    prices_norm = _normalize_price_frame(prices, market_col=market_col, ts_col=ts_col, price_col=price_col)
    market_prices = prices_norm[prices_norm[market_col] == int(market_id)].reset_index(drop=True)
    if market_prices.empty:
        raise ValueError(f"No prices for market_id={market_id}")

    returns = compute_log_returns(market_prices, market_col=market_col, ts_col=ts_col, price_col=price_col)
    rolling_vol = compute_rolling_volatility(
        returns,
        market_col=market_col,
        ts_col=ts_col,
        return_col="log_return",
        window=window,
        annualize=annualize,
        freq_per_year=freq_per_year,
    )
    return VolatilityResult(
        market_id=int(market_id),
        prices=market_prices,
        returns=returns,
        rolling_volatility=rolling_vol,
    )


def plot_price_and_volatility(
    result: VolatilityResult,
    *,
    price_label: str = "Price / probability",
    vol_label: str = "Rolling volatility",
    title: Optional[str] = None,
) -> plt.Figure:
    """Plot price series and rolling volatility on twin axes."""
    fig, ax_price = plt.subplots(figsize=(10, 5))
    ax_vol = ax_price.twinx()

    ax_price.plot(result.prices["t"], result.prices["p"], color="#1f77b4", label=price_label)
    ax_price.set_ylabel(price_label)
    ax_price.set_xlabel("Timestamp (UTC)")

    ax_vol.plot(
        result.rolling_volatility["t"],
        result.rolling_volatility["rolling_vol"],
        color="#d62728",
        label=vol_label,
        alpha=0.7,
    )
    ax_vol.set_ylabel(vol_label)

    if title is None:
        title = f"Market {result.market_id}: Price & Rolling Volatility"
    ax_price.set_title(title)

    ax_price.grid(True, axis="both", alpha=0.3, linestyle="--")

    lines, labels = ax_price.get_legend_handles_labels()
    lines2, labels2 = ax_vol.get_legend_handles_labels()
    ax_price.legend(lines + lines2, labels + labels2, loc="upper left")
    fig.tight_layout()
    return fig


def plot_volatility_distribution(
    rolling_vol: pd.DataFrame,
    *,
    bins: int = 50,
    column: str = "rolling_vol",
    title: str = "Rolling volatility distribution",
) -> plt.Figure:
    """Plot histogram of rolling volatility values across markets."""
    fig, ax = plt.subplots(figsize=(8, 4))
    vals = pd.to_numeric(rolling_vol[column], errors="coerce").dropna()
    ax.hist(vals, bins=bins, alpha=0.7, color="#9467bd", edgecolor="white")
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig


def _require_yfinance() -> None:
    if yf is None:  # pragma: no cover - runtime check
        raise ImportError(
            "yfinance is required for external volatility comparisons. "
            "Install it via `pip install yfinance`."
        )


def fetch_asset_volatility(
    symbol: str,
    *,
    label: Optional[str] = None,
    period: str = "6mo",
    interval: str = "1d",
    window: int = 30,
    annualize: bool = True,
    trading_days: int = 252,
) -> AssetVolatilityResult:
    """
    Download historical prices for a macro asset (e.g., 'SPY', 'ETH-USD') and compute
    rolling volatility over the requested window.
    """
    _require_yfinance()

    data = yf.download(  # type: ignore[arg-type]
        symbol,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )
    if data.empty:
        raise ValueError(f"No price data returned for symbol '{symbol}'")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    price_column = "Close"
    if "Adj Close" in data.columns:
        price_column = "Adj Close"
    if price_column not in data.columns:
        raise ValueError(f"Expected '{price_column}' column in yfinance data for {symbol}")

    label = label or symbol
    prices = (
        data.reset_index()
        .rename(columns={"Date": "t", price_column: "price"})
        .loc[:, ["t", "price"]]
    )
    prices["t"] = pd.to_datetime(prices["t"], utc=True, errors="coerce")
    prices["price"] = pd.to_numeric(prices["price"], errors="coerce")
    prices = prices.dropna(subset=["t", "price"]).reset_index(drop=True)
    prices["market_id"] = pd.Series(symbol, index=prices.index, dtype="string")

    returns = prices.copy()
    returns["log_return"] = np.log(returns["price"]).diff().fillna(0.0)

    rolling = returns.copy()
    rolling["rolling_vol"] = (
        rolling["log_return"]
        .rolling(window=window, min_periods=max(2, window // 2))
        .std()
    )
    if annualize:
        rolling["rolling_vol"] = rolling["rolling_vol"] * np.sqrt(trading_days)
    rolling = rolling.dropna(subset=["rolling_vol"]).reset_index(drop=True)

    return AssetVolatilityResult(
        symbol=symbol,
        label=label,
        prices=prices,
        returns=returns,
        rolling_volatility=rolling,
    )


def fetch_reference_volatilities(
    assets: Sequence[str] | Mapping[str, str],
    **kwargs,
) -> List[AssetVolatilityResult]:
    items: Iterable[tuple[str, str]]
    if isinstance(assets, Mapping):
        items = list(assets.items())
    else:
        if isinstance(assets, str):
            symbols = [assets]
        else:
            symbols = list(assets)
        items = [(symbol, symbol) for symbol in symbols]

    results: List[AssetVolatilityResult] = []
    for symbol, label in items:
        results.append(fetch_asset_volatility(symbol, label=label, **kwargs))
    return results


def plot_reference_volatility_bars(
    references: Sequence[AssetVolatilityResult],
    *,
    top_n: Optional[int] = None,
    title: str = "Reference asset volatility (latest rolling values)",
) -> plt.Figure:
    """
    Plot horizontal bars of the latest rolling volatility for each reference asset.
    """
    summary: List[tuple[str, float]] = []
    for res in references:
        series = pd.to_numeric(res.rolling_volatility["rolling_vol"], errors="coerce").dropna()
        if series.empty:
            continue
        summary.append((res.label, float(series.iloc[-1])))

    if not summary:
        raise ValueError("No reference volatilities available to plot.")

    summary.sort(key=lambda x: x[1], reverse=True)
    if top_n is not None:
        summary = summary[:top_n]

    labels, values = zip(*summary)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(labels))))
    ax.barh(labels, values, color="#6baed6")
    ax.set_xlabel("Rolling volatility")
    ax.set_ylabel("Asset")
    ax.set_title(title)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3, linestyle="--")
    fig.tight_layout()
    return fig


def plot_market_and_reference_volatility(
    market_result: VolatilityResult,
    references: Sequence[AssetVolatilityResult],
    *,
    title: Optional[str] = None,
    price_label: str = "Market price / probability",
    market_vol_label: str = "Market rolling volatility",
) -> plt.Figure:
    """
    Plot a market's price/volatility along with external asset volatilities on the same axes.
    """
    fig, ax_price = plt.subplots(figsize=(11, 6))
    ax_vol = ax_price.twinx()

    ax_price.plot(market_result.prices["t"], market_result.prices["p"], color="#1f77b4", label=price_label)
    ax_price.set_ylabel(price_label)
    ax_price.set_xlabel("Timestamp (UTC)")

    ax_vol.plot(
        market_result.rolling_volatility["t"],
        market_result.rolling_volatility["rolling_vol"],
        color="#d62728",
        label=market_vol_label,
        linewidth=2.0,
    )

    palette = plt.colormaps["tab10"]
    for idx, asset in enumerate(references):
        color = palette(idx % palette.N)
        ax_vol.plot(
            asset.rolling_volatility["t"],
            asset.rolling_volatility["rolling_vol"],
            linestyle="--",
            linewidth=1.5,
            color=color,
            label=f"{asset.label} vol",
            alpha=0.9,
        )

    if title is None:
        title = f"Market {market_result.market_id} vs macro volatility"
    ax_price.set_title(title)
    ax_price.grid(True, axis="both", alpha=0.3, linestyle="--")

    lines, labels = ax_price.get_legend_handles_labels()
    lines2, labels2 = ax_vol.get_legend_handles_labels()
    ax_price.legend(lines + lines2, labels + labels2, loc="upper left")
    fig.tight_layout()
    return fig


def plot_reference_volatility_lines(
    *,
    references: Optional[Sequence[AssetVolatilityResult]] = None,
    tailend_result: Optional[VolatilityResult] = None,
    tailend_label: Optional[str] = None,
    alpha: int = 0.3,
    title: str = "Reference asset rolling volatility comparison",
) -> plt.Figure:
    references = list(references) if references is not None else []
    fig, ax = plt.subplots(figsize=(11, 6))
    palette = plt.colormaps["tab10"]

    for idx, asset in enumerate(references):
        color = palette(idx % palette.N)
        ax.plot(
            asset.rolling_volatility["t"],
            asset.rolling_volatility["rolling_vol"],
            label=asset.label,
            linewidth=2.0,
            color=color,
            alpha=alpha,
        )
    if tailend_result is not None:
        label = tailend_label or f"Tail-end market {tailend_result.market_id}"
        ax.plot(
            tailend_result.rolling_volatility["t"],
            tailend_result.rolling_volatility["rolling_vol"],
            label=label,
            linewidth=2.5,
            color="#41b9c3"
        )

    ax.set_title(title)
    ax.set_xlabel("Timestamp (UTC)")
    ax.set_ylabel("Rolling volatility")
    ax.grid(True, axis="both", alpha=0.3, linestyle="--")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper left")
    else:
        ax.text(
            0.5,
            0.5,
            "No reference series provided",
            transform=ax.transAxes,
            ha="center",
            va="center",
            alpha=0.7,
        )
    fig.tight_layout()
    return fig


def compute_average_tailend_volatility(
    prices: pd.DataFrame,
    tailend_market_ids: Sequence[int],
    *,
    market_col: str = "market_id",
    ts_col: str = "t",
    price_col: str = "p",
    window: int = 30,
    smoothing: int = 7,
    annualize: bool = True,
    freq_per_year: int = 365,
) -> pd.DataFrame:
    """
    Compute the time-aggregated average rolling volatility across the provided tail-end markets.
    """
    if not tailend_market_ids:
        raise ValueError("No tail-end market ids provided.")

    prices = _normalize_price_frame(prices, market_col=market_col, ts_col=ts_col, price_col=price_col)
    ids = pd.Series(tailend_market_ids, dtype="Int64").dropna().unique()
    prices = prices[prices[market_col].isin(ids)]
    if prices.empty:
        raise ValueError("No prices available for the specified tail-end markets.")
    #prices.to_csv("a.csv", index=False)
    returns = compute_log_returns(prices, market_col=market_col, ts_col=ts_col, price_col=price_col)
    rolling = compute_rolling_volatility(
        returns,
        market_col=market_col,
        ts_col=ts_col,
        return_col="log_return",
        window=window,
        annualize=annualize,
        freq_per_year=freq_per_year,
    )
    if rolling.empty:
        raise ValueError("Rolling volatility computation returned no data.")
    #rolling.to_csv("rolling.csv", index=False)

    agg = (
        rolling.assign(day=pd.to_datetime(rolling[ts_col]).dt.floor("d"))
        .groupby("day", sort=True)["rolling_vol"]
        .mean()
        .reset_index()
        .rename(columns={"day": ts_col})
    )
    agg["rolling_vol"] = agg["rolling_vol"].rolling(window=smoothing, min_periods=1).mean()
    agg["market_id"] = -1
    return agg[[market_col, ts_col, "rolling_vol"]]


def main() -> None:

    mode = 4
    annualize = False

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
    from fetch.tail_end_func import find_tailend_markets_by_merged_prices;
    markets_orig = filter_by_duration(markets_orig, DAYS)
    markets = find_tailend_markets_by_merged_prices(markets_orig, prices, 0.9, 0.5)
   # markets.to_csv("filtered_markets.csv", index=False)
    if mode == 4:
        threshold = 0.9
        required_pct = 0.5
        tailendprices = prices[prices["market_id"].isin(markets["id"])]
        tailendprices = find_tailend_prices(markets=markets, prices=prices)
        returns_tailend = compute_log_returns(tailendprices)
        rolling_t = compute_rolling_volatility(returns_tailend, window=WINDOW, freq_per_year=365, annualize=True)
        avg_vol_tailend = (
            rolling_t.groupby("market_id")["rolling_vol"]
            .mean()
            .reset_index()
            .rename(columns={"rolling_vol": "avg_volatility"})
        )
        
        #not_tailend_prices = 
        markets_orig['classicWinner'] = np.where(
                markets_orig['prob_yes'] == 1.0,
                'yes',
                'no'
            )
        nt_p = prices.merge(
                markets_orig[["id", "classicWinner"]],
                left_on=["market_id", "token"],
                right_on=["id", "classicWinner"],
                how="inner",
            )

        #nt_p = prices[prices['market_id']["token"] == markets['id']['winnerToken']]
        nt_p = nt_p[~nt_p['market_id'].isin(markets['id'])]
        returns = compute_log_returns(nt_p)
        rolling_nt = compute_rolling_volatility(returns, window=WINDOW, freq_per_year=365, annualize=True)
        avg_vol_nt = (
            rolling_nt.groupby("market_id")["rolling_vol"]
            .mean()
            .reset_index()
            .rename(columns={"rolling_vol": "avg_volatility"})
        )
        
        
        tailend_mean = avg_vol_tailend["avg_volatility"].mean()
        non_tailend_mean = avg_vol_nt["avg_volatility"].mean()
        print(
            f"Tailend markets: {len(avg_vol_tailend)} avg volatility={tailend_mean:.2f}; "
            f"Non-tailend markets: {len(avg_vol_nt)} avg volatility={non_tailend_mean:.2f}"
        )

        # Time-series averages (daily) + plot, similar to fetch/liquidation.py
        tailend_ts = (
            rolling_t.assign(day=pd.to_datetime(rolling_t["t"]).dt.floor("d"))
            .groupby("day", sort=True)["rolling_vol"]
            .mean()
            .reset_index()
            .rename(columns={"day": "t"})
        )
        tailend_ts["rolling_vol"] = tailend_ts["rolling_vol"].rolling(window=7, min_periods=1).mean()
        tailend_ts_mean = float(pd.to_numeric(tailend_ts["rolling_vol"], errors="coerce").dropna().mean())

        non_tailend_ts = (
            rolling_nt.assign(day=pd.to_datetime(rolling_nt["t"]).dt.floor("d"))
            .groupby("day", sort=True)["rolling_vol"]
            .mean()
            .reset_index()
            .rename(columns={"day": "t"})
        )
        non_tailend_ts["rolling_vol"] = non_tailend_ts["rolling_vol"].rolling(window=7, min_periods=1).mean()
        non_tailend_ts_mean = float(pd.to_numeric(non_tailend_ts["rolling_vol"], errors="coerce").dropna().mean())

        # Post-tailend (after each market enters tail-end stage), same idea as liquidation.py
        from fetch.trades import _tailend_start_time

        tailendprices_norm = tailendprices.copy()
        tailendprices_norm["t"] = pd.to_datetime(tailendprices_norm["t"], unit="s", utc=True, errors="coerce")
        tailendprices_norm["p"] = pd.to_numeric(tailendprices_norm["p"], errors="coerce")
        tailendprices_norm["market_id"] = pd.to_numeric(tailendprices_norm["market_id"], errors="coerce").astype("Int64")
        tailendprices_norm = tailendprices_norm.dropna(subset=["market_id", "t", "p"]).sort_values(
            ["market_id", "t"]
        )

        start_times = (
            tailendprices_norm.groupby("market_id", sort=False)[["t", "p"]]
            .apply(lambda df: _tailend_start_time(df, threshold=threshold, required_pct=required_pct))
            .dropna()
            .rename("tailend_start")
            .reset_index()
        )
        prices_post = tailendprices_norm.merge(start_times, on="market_id", how="inner")
        prices_post = prices_post[prices_post["t"] >= prices_post["tailend_start"]]

        returns_post = compute_log_returns(prices_post[["market_id", "t", "p"]])
        rolling_post = compute_rolling_volatility(
            returns_post,
            window=WINDOW,
            freq_per_year=365,
            annualize=True,
        )
        avg_post_ts = (
            rolling_post.assign(day=pd.to_datetime(rolling_post["t"]).dt.floor("d"))
            .groupby("day", sort=True)["rolling_vol"]
            .mean()
            .reset_index()
            .rename(columns={"day": "t"})
        )
        avg_post_ts["rolling_vol"] = avg_post_ts["rolling_vol"].rolling(window=7, min_periods=1).mean()
        avg_post_ts_mean = float(pd.to_numeric(avg_post_ts["rolling_vol"], errors="coerce").dropna().mean())

        def _label_with_daily_stats(name: str, mean_annual_vol: float, *, p0: float = 0.9) -> str:
            sigma_daily = mean_annual_vol / float(np.sqrt(365))
            expected_abs_daily_log_return = float(np.sqrt(2 / np.pi) * sigma_daily)
            expected_abs_daily_dp_at_p0 = float(p0 * expected_abs_daily_log_return)
            return (
                f"{name} \n  Mean={mean_annual_vol:.3f}, "
              #  f"\nσ_day={sigma_daily:.4f}, "
                f"\n  E|r_day|={expected_abs_daily_log_return:.4f} - expected absolute daily return (%), "
                f"\n  E|Δp|@p={p0:.1f}≈{expected_abs_daily_dp_at_p0:.4f} - expected daily price movement ($)"
            )

        refs = [
            
            asset_volatility_from_rolling_series(
                non_tailend_ts,
                symbol="NONTAILEND",
                label=_label_with_daily_stats("Non-tailend markets", non_tailend_ts_mean, p0=0.9),
            ),
            asset_volatility_from_rolling_series(
                tailend_ts,
                symbol="TAILEND",
                label=_label_with_daily_stats("Original tailend markets", tailend_ts_mean, p0=0.9),
            ),
            asset_volatility_from_rolling_series(
                avg_post_ts,
                symbol="POST",
                label=_label_with_daily_stats("After entering tail-end stage", avg_post_ts_mean, p0=0.9),
            ),
        ]

        try:
            gold = fetch_asset_volatility("GLD", label="Gold", window=WINDOW, period="3y", annualize=True)
            gold_mean_vol = float(pd.to_numeric(gold.rolling_volatility["rolling_vol"], errors="coerce").dropna().mean())
            gold.label = _label_with_daily_stats("Gold (GLD)", gold_mean_vol, p0=4365)
            refs.append(gold)
        except (ImportError, ValueError) as exc:
            print(f"Skipping Gold reference: {exc}")

        plot_reference_volatility_lines(
            references=refs,
            tailend_result=None,
            title=f"Comparison between Non-tailend, Tailend and Post-tailend avg rolling volatility (window={WINDOW})",
            alpha=0.9
        )
        plt.show()
    if mode == 1:
        reference_symbols = {
            "SPY": "S&P 500",
            "BTC-USD": "BTC",
            "ETH-USD": "ETH",
            "GLD": "Gold",
        }
        tailend_avg_result: Optional[VolatilityResult] = None
        try:
            #yes_prices = prices[prices["token"] == "yes"]
            tailend_prices = find_tailend_prices(markets, prices)
            tailend_ids = pd.to_numeric(markets["id"], errors="coerce").dropna().astype(int).tolist()
            avg_vol = compute_average_tailend_volatility(
                tailend_prices,
                tailend_ids,
                window=WINDOW,
                annualize=annualize,
            )
            #avg_vol.to_csv("volatility_summary.csv", index=False)
            tailend_avg_result = VolatilityResult(
                market_id=-1,
                prices=pd.DataFrame(columns=["market_id", "t", "p"]),
                returns=pd.DataFrame(columns=["market_id", "t", "log_return"]),
                rolling_volatility=avg_vol,
            )
            references = fetch_reference_volatilities(
                reference_symbols,
                window=WINDOW,
                period="3y",
                annualize=annualize,
            )
            
            plot_reference_volatility_lines(
                references=references,
                tailend_result=tailend_avg_result,
                tailend_label="7d Avg tail-end market volatility" if tailend_avg_result is not None else None,
                title=f"Annual Reference rolling volatility comparison rolling window={WINDOW}, min markets duration={DAYS} days, total markets={len(markets)}",
            )
            plt.show()

        except ImportError as exc:
            print(f"Skipping reference volatility plots (missing dependency): {exc}")
        except ValueError as exc:
            print(f"Skipping reference volatility plots: {exc}")
    if mode == 2:
        from fetch.tail_end_func import filter_markets_with_prices
        duration_buckets = [7, 14, 30, 60, 90]
        #yes_prices = prices[prices["token"] == "yes"]
        tailend_prices = find_tailend_prices(markets=markets, prices=prices)
        fig, ax = plt.subplots(figsize=(11, 6))
        palette = plt.colormaps["tab10"]
        for idx, min_days in enumerate(duration_buckets):
            subset = filter_by_duration(markets, min_days)
            ids = pd.to_numeric(subset["id"], errors="coerce").dropna().astype(int).tolist()
            avg_vol = compute_average_tailend_volatility(
                tailend_prices,
                ids,
                window=WINDOW,
                smoothing=50,
                annualize=annualize,
            )
            color = palette(idx % palette.N)
            ax.plot(
                pd.to_datetime(avg_vol["t"]),
                avg_vol["rolling_vol"],
                label=f"Duration ≥ {min_days}d",
                color=color,
                linewidth=2.0,
            )
        ax.set_title("Average tail-end volatility by minimum market duration")
        ax.set_xlabel("Timestamp (UTC)")
        ax.set_ylabel("Average rolling volatility")
        ax.grid(True, axis="both", alpha=0.3, linestyle="--")
        ax.legend(loc="upper left")
        plt.show()
    if mode == 3:
        #prices = prices[prices['market_id'].isin(markets['id'])]
        #yes_prices = prices[prices["token"] == "yes"]
        tailend_prices = find_tailend_prices(markets, prices)
        returns = compute_log_returns(tailend_prices)
        rolling = compute_rolling_volatility(returns, window=WINDOW, freq_per_year=365, annualize=True)
        avg_vol = (
            rolling.groupby("market_id")["rolling_vol"]
            .mean()
            .reset_index()
            .rename(columns={"rolling_vol": "avg_volatility"})
        )
        merged = avg_vol.merge(
            markets[["id", "question"]],
            left_on="market_id",
            right_on="id",
            how="left",
        )
        more5 = merged[merged['avg_volatility'] >= 5]
      #  merged = merged[merged['avg_volatility'] < 5]
        print("You deleted ", more5.size, " markets volatility above 5")
        merged = merged.sort_values("avg_volatility")
        #merged = merged[merged["avg_volatility"] > 0.5]
        merged.to_csv("volatility_summary.csv", index=False)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(merged["avg_volatility"], bins=999, color="#1f77b4", alpha=0.4)
        ax.set_xlabel(f"Average volatility tailend (rolling window={WINDOW}) for {len(merged)} markets which duration is at least 30 days. (1 == 100%)")
        ax.set_ylabel("Markets")
        ax.set_title("Distribution of average volatility")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        reference_symbols = {
            "SPY": "S&P 500",
            "BTC-USD": "BTC",
            "ETH-USD": "ETH",
            "GLD": "Gold",
        }
        stats: List[tuple[str, float]] = []
        try:
            references = fetch_reference_volatilities(
                reference_symbols,
                period="1y",
                interval="1d",
                window=WINDOW,
                annualize=True,
            )
            stats = [
                (
                    res.label,
                    float(pd.to_numeric(res.rolling_volatility["rolling_vol"], errors="coerce").dropna().iloc[-1]),
                )
                for res in references
                if not res.rolling_volatility.empty
            ]
            if stats:
                y_text = 0.95
                ax.text(
                    0.78,
                    y_text,
                    "Reference vols (last annualized):",
                    transform=ax.transAxes,
                    fontsize=10,
                    ha="left",
                )
                box_width = 0.015
                box_height = 0.02
                palette = plt.colormaps["tab10"]
                y_step = 0.06
                for idx, (label, value) in enumerate(stats):
                    y_text -= y_step
                    color = palette(idx % palette.N)
                    box = mpatches.Rectangle(
                        (0.78, y_text - box_height / 2),
                        box_width,
                        box_height,
                        transform=ax.transAxes,
                        facecolor=color,
                        edgecolor=color,
                        alpha=0.9,
                        clip_on=False,
                    )
                    ax.add_patch(box)
                    ax.text(
                        0.78 + box_width + 0.01,
                        y_text,
                        f"{label}: {value:.2%}",
                        transform=ax.transAxes,
                        fontsize=9,
                        ha="left",
                        va="center",
                    )
                    ax.axvline(
                        value,
                        color=color,
                        linestyle="--",
                        linewidth=1.8,
                        alpha=0.9,
                        label=f"{label} ref",
                    )
                    y_min, y_max = ax.get_ylim()
                    span = y_max - y_min
                    ax.text(
                        value,
                        y_min - 0.03 * span,
                        f"{value:.2f}",
                        color=color,
                        fontsize=8,
                        ha="center",
                        va="top",
                        clip_on=False,
                    )
        except (ImportError, ValueError) as exc:
            ax.text(
                0.65,
                0.95,
                f"Reference data unavailable: {exc}",
                transform=ax.transAxes,
                fontsize=9,
                ha="left",
            )
        data_max = float(merged["avg_volatility"].max()) if not merged.empty else None
        right_limit = data_max
        ax.set_xlim(left=0.0, right=5)
        plt.tight_layout()
        plt.show()

   
   
   

if __name__ == "__main__":
    main()
