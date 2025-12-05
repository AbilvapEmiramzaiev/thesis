from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class VolatilityResult:
    """Container for price/returns/volatility series of a single market."""

    market_id: int
    prices: pd.DataFrame
    returns: pd.DataFrame
    rolling_volatility: pd.DataFrame


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


def main() -> None:
    from config import PROJECT_ROOT
    from utils import read_markets_csv, read_prices_csv
    WINDOW = 10
    markets_b = read_markets_csv(PROJECT_ROOT / "data/binary_markets.csv")
    markets_c = read_markets_csv(PROJECT_ROOT / "data/categorical_markets_all.csv")
    prices_b = read_prices_csv(PROJECT_ROOT / "data/prices_binary_all.csv")
    prices_c = read_prices_csv(PROJECT_ROOT / "data/prices_categorical_all.csv")

    markets = pd.concat([markets_b, markets_c], ignore_index=True)
    prices = pd.concat([prices_b, prices_c], ignore_index=True)
    from fetch.tail_end_func import find_tailend_markets_by_merged_prices;
    markets = find_tailend_markets_by_merged_prices(markets, prices)
    markets.to_csv("tailend_markets.csv", index=False)
    returns = compute_log_returns(prices)
    rolling = compute_rolling_volatility(returns, window=WINDOW, annualize=True, freq_per_year=365)

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
    merged = merged.sort_values("avg_volatility")
    merged = merged[merged["avg_volatility"] > 0]
    merged.to_csv("volatility_summary.csv", index=False)
    print(f"Computed average rolling volatility for {len(merged)} markets.")
    print(merged.head())

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(merged["avg_volatility"], bins=60, color="#1f77b4", alpha=0.75)
    ax.set_xlabel(f"Average volatility (rolling window={WINDOW})")
    ax.set_ylabel("Markets")
    ax.set_title("Distribution of average volatility")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
