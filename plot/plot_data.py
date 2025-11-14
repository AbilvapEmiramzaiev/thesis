from imports import *

def plot_market_history(prices: pd.DataFrame):
    x_utc = pd.to_datetime(prices['t'], unit='s', utc=True)

    plt.figure(figsize=(11,5))
    plt.plot(x_utc, prices['p'], label="price", color="blue")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%M%d', tz=timezone.utc))
    plt.gcf().autofmt_xdate()
    plt.ylabel("probability")
    plt.xlabel("time")
    plt.ylim(0,1)
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.yticks(np.arange(0, 1.05, 0.05)) 
    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1))
    plt.legend()
    plt.tight_layout()
    plt.show()

    
def add_stats_panel(ax, rows, loc="upper left"):
    """
    rows: list of (label, value) tuples -> displayed in two columns.
    """
    # Make a two-column monospace block
    left_w = max(len(lbl) for lbl, _ in rows)
    text = "\n".join(f"{lbl:<{left_w}}  {val}" for lbl, val in rows)

    ta = TextArea(text, textprops=dict(family="monospace"))
    box = AnchoredOffsetbox(
        loc=loc, child=ta, pad=0.4, borderpad=0.6, frameon=True
    )
    box.patch.set_boxstyle("round,pad=0.4,rounding_size=0.2")
    box.patch.set_alpha(0.9)
    box.set_zorder(10)   
    ax.add_artist(box)
    return box


def add_market_apy_line(
    ax: plt.Axes,
    prices: pd.DataFrame,
    *,
    resolution_time,
    price_col: str = "p",
    ts_col: str = "t",
    label: str = "APY",
    linewidth: int = 2,
    alpha: float = 0.9,
    ax2: Optional[plt.Axes] = None,
):
    """Overlay APY line for a single market's price series on an existing plot.

    APY = ((1 - p) / p) * (365 / days_to_resolution)

    Pass the same `ax` across calls to stack multiple APY lines on one figure.
    Optionally pass an existing right axis `ax2` to reuse the same secondary y-axis.
    Returns a tuple (ax2, apy_values) where `apy_values` is a 1D numpy array
    of computed APY values after masking (e.g., excluding last 3 days).
    """
    t = pd.to_datetime(prices[ts_col], unit="s", utc=True)
    p = prices[price_col].clip(1e-9, 1 - 1e-9)
    res_ts = pd.to_datetime(resolution_time, utc=True)
    dt_days = (res_ts - t).dt.total_seconds() / 86400.0
    m = dt_days > 3.0 #how much days before resolution to ignore
    t, p, dt_days = t[m], p[m], dt_days[m]
    apy = ((1.0 - p) / p) * (365.0 / dt_days)
    #t.to_csv('debug_times.csv', index=False)
    #apy.to_csv('debug_apy.csv', index=False)
    ax2 = ax2 or ax.twinx()
    ax2.plot(t, apy, linewidth=linewidth, label=label, alpha=alpha)
    ax2.set_ylabel("annualized yield (*100%)")
    ax2.grid(False)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="best", frameon=False)
    return ax2, apy.values if hasattr(apy, 'values') else np.asarray(apy)





def add_apy_axis_from_path(
    ax: plt.Axes,
    path: pd.DataFrame,
    *,
    resolution_time,                 # e.g. "2025-01-15T00:00:00Z" or pd.Timestamp
    price_col="p_top10_smooth",
    label="annualized yield (APY)",
    linewidth=2,
    alpha=0.9,
):
    """
    Compute and plot APY from a path with columns ['ts_bin', price_col] where price_col is the
    smoothed top-10% daily average probability (the red line).
    APY = ((1 - p) / p) * (365 / days_to_resolution)
    """
    if path.empty or price_col not in path:
        return None

    r = path[["ts_bin", price_col]].dropna().copy()
    r["ts_bin"] = pd.to_datetime(r["ts_bin"], utc=True)
    res_ts = pd.to_datetime(resolution_time, utc=True)

    # days to resolution; ignore non-positive horizons
    dt_days = (res_ts - r["ts_bin"]).dt.total_seconds() / 86400.0
    r = r.assign(days_to_res=np.where(dt_days > 0, dt_days, np.nan)).dropna(subset=["days_to_res"])

    p = r[price_col].clip(1e-9, 1 - 1e-9)
    r["apy"] = ((1.0 - p) / p) * (365.0 / r["days_to_res"])

    ax2 = ax.twinx()
    ax2.plot(r["ts_bin"], r["apy"], linewidth=linewidth, alpha=alpha, label=label)
    ax2.set_ylabel("annualized yield")
    ax2.grid(False)

    # merge legends
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="best", frameon=False)
    return ax2

def compute_global_top_decile_path(
    df: pd.DataFrame,
    *,
    time_col="t",
    price_col="p",
    time_bin="1D",
    smooth_window=3,
) -> pd.DataFrame:
    """Per day across ALL markets: take the highest 10% p and average; then centered smooth."""
    d = df.copy()
    d["time"] = pd.to_datetime(d[time_col], unit="s", utc=True)
    d["ts_bin"] = d["time"].dt.floor(time_bin)

    rows = []
    for ts_bin, g in d.groupby("ts_bin", sort=False):
        n = len(g)
        if n == 0:
            continue
        k = max(1, math.ceil(0.10 * n))
        top = g.nlargest(k, price_col)[price_col]
        rows.append({"ts_bin": ts_bin, "p_top10": float(top.mean())})

    path = pd.DataFrame(rows).sort_values("ts_bin").reset_index(drop=True)
    if path.empty:
        return path.assign(p_top10_smooth=pd.Series(dtype=float))

    path["p_top10_smooth"] = (
        path["p_top10"].rolling(window=smooth_window, center=True, min_periods=1).mean()
    )
    return path


def plot_prices(
    df: pd.DataFrame,
    *,
    show: bool = True,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    time_col = "t"
    market_col = "market_id"
    price_col = "p"
    ax = ax or plt.gca()
    df = df.copy()
    df["time"] = pd.to_datetime(df[time_col], unit="s", utc=True)

    # For each market, aggregate overlapping dots and plot in its own color
    for i, (_, g) in enumerate(df.groupby(market_col)):
        g = g.copy()
        g["ts_rounded"] = g["time"].dt.floor("1D")   # merge close timestamps (try "12h" or "1D")
        g["p_rounded"]  = g[price_col].round(3)      # merge nearby probabilities

        # count density within bins
        counts = g.groupby(["ts_rounded", "p_rounded"]).size().reset_index(name="n")

        # scale point size & transparency by density
        ax.scatter(
            counts["ts_rounded"],
            counts["p_rounded"],
            s=counts["n"] ** 1.5 * 10,
            alpha=0.5 + (counts["n"] / counts["n"].max()) * 0.5,
            edgecolors="none",
            #color=(0.55, 0.75, 0.95),  # light pastel blue (RGB tuple 0–1 range)
        )

    ax.set_xlabel("time (UTC)")
    ax.set_ylabel("probability")
    ax.set_ylim(0.8, 1)
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    ax.figure.autofmt_xdate()
    path=False
    if show:
        path = compute_global_top_decile_path(
            df, time_col=time_col, price_col=price_col, time_bin="5D", smooth_window=5
        )
        ax.plot(
                path["ts_bin"], path["p_top10_smooth"],
                lw=2.5, alpha=0.95, color="tab:red", label="top 10% daily avg price"
            )
    if path is not False and not path.empty:
        add_apy_axis_from_path(ax, path, resolution_time="2025-01-01T00:00:00Z")
     # --- Per-market top-10% mean (smoothed) + APY on right axis
    """ path_all = compute_top_decile_paths(
        df, time_col="time", market_col=market_col, price_col='p',
        time_bin='1D', smooth_window=3
    ) """

    # plot the (smoothed) red path per market
    """ for mid, g in path_all.groupby(market_col, sort=False):
        ax.plot(
            g["ts_bin"], g["p_top10_smooth"],
            linewidth=2, alpha=0.9, label=f"{mid} top 10% avg (3D)"
        ) """

    # secondary y-axis for APY; one line per market (labels include market id)
    """ ax2 = ax.twinx()
    for mid, g in path_all.groupby(market_col, sort=False):
        # APY only where days_to_res > 0
        gg = g.dropna(subset=["apy"])
        if gg.empty:
            continue
        ax2.plot(
            gg["ts_bin"], gg["apy"],
            linewidth=2, alpha=0.9, label=f"{mid} APY"
        ) """
    #add_mode_path(ax, df, time_col="t", price_col="p", time_bin="1D", price_round=3, color="black", label="most popular")
    #add_median_path(ax, df, time_col="t", price_col="p", time_bin="1D", color="tab:red", label="median")
    if show:
        plt.show()
    return ax

def add_top_decile_mean_path(
    ax,
    df,
    *,
    time_col="t",
    price_col="p",
    time_bin="1D",
    smooth_window: int = 3,
    color="tab:red",
    linewidth=2,
    alpha=0.9,
    label="top 10% avg",
) -> pd.DataFrame:
    """Plot the mean of the highest 10% prices per time_bin, then apply centered smoothing."""
    d = df.copy()
    d["ts"] = pd.to_datetime(d[time_col], unit="s", utc=True)
    d["ts_bin"] = d["ts"].dt.floor(time_bin)

    # Compute top-decile mean per bin
    pieces = []
    for ts_bin, g in d.groupby("ts_bin"):
        n = len(g)
        if n == 0:
            continue
        k = max(1, math.ceil(0.10 * n))
        top = g.nlargest(k, price_col)[price_col]
        pieces.append({"ts_bin": ts_bin, "p_top10": float(top.mean())})

    if not pieces:
        return pd.DataFrame(columns=["ts_bin", "p_top10", "p_top10_smooth"])

    path = pd.DataFrame(pieces).sort_values("ts_bin").reset_index(drop=True)

    # Smooth with centered rolling window
    path["p_top10_smooth"] = (
        path["p_top10"]
        .rolling(window=smooth_window, center=True, min_periods=1)
        .mean()
    )

    # Plot (smoothed) line
    ax.plot(
        path["ts_bin"], path["p_top10_smooth"],
        color=color, linewidth=linewidth, alpha=alpha, label=label
    )
    return path


def compute_top_decile_paths(
    df: pd.DataFrame,
    *,
    time_col: str = "time",        # already to_datetime
    market_col: str = "market_id",
    price_col: str = "p",
    time_bin: str = "1D",
    smooth_window: int = 3,
) -> pd.DataFrame:
    """
    Returns a tidy DataFrame with columns:
      market_id, ts_bin, p_top10, p_top10_smooth, first_ts, last_ts (resolution_time),
      days_to_res, apy
    """
    d = df.copy()
    d["ts_bin"] = d[time_col].dt.floor(time_bin)

    frames = []
    for mid, g in d.groupby(market_col, sort=False):
        g = g.sort_values(time_col)
        if g.empty:
            continue

        # infer resolution_time (last timestamp); keep first for reference
        first_ts = g[time_col].iloc[0]
        last_ts  = g[time_col].iloc[-1]  # <-- resolution_time by your rule

        # per-day top-10% mean
        rows = []
        for ts_bin, gg in g.groupby("ts_bin", sort=False):
            n = len(gg)
            k = max(1, math.ceil(0.10 * n))
            top = gg.nlargest(k, price_col)[price_col]
            rows.append({"ts_bin": ts_bin, "p_top10": float(top.mean())})

        if not rows:
            continue

        path = pd.DataFrame(rows).sort_values("ts_bin").reset_index(drop=True)

        # centered smoothing (3-day default)
        path["p_top10_smooth"] = (
            path["p_top10"].rolling(window=smooth_window, center=True, min_periods=1).mean()
        )

        # APY: ((1 - p) / p) * (365 / days_to_resolution)
        # days_to_resolution measured from each ts_bin to this market's last_ts
        dt_days = (last_ts - path["ts_bin"]).dt.total_seconds() / 86400.0
        # ignore non-positive horizons
        dt_days = np.where(dt_days > 0, dt_days, np.nan)
        p = path["p_top10_smooth"].clip(1e-9, 1 - 1e-9)
        apy = ((1.0 - p) / p) * (365.0 / dt_days)

        out = path.assign(
            **{
                market_col: mid,
                "first_ts": first_ts,
                "last_ts": last_ts,
                "days_to_res": dt_days,
                "apy": apy,
            }
        )
        frames.append(out)

    if not frames:
        return pd.DataFrame(columns=[market_col, "ts_bin", "p_top10", "p_top10_smooth", "first_ts", "last_ts", "days_to_res", "apy"])

    return pd.concat(frames, ignore_index=True)

def add_apy_secondary_axis(
    ax: plt.Axes,
    path_df: pd.DataFrame,
    *,
    resolution_time: pd.Timestamp,
    price_col: str = "p_top10_smooth",
    label: str = "annualized yield",
    linewidth: int = 2,
    alpha: float = 0.9,
):
    """
    Plot APY on a secondary y-axis:
      APY = ((1 - p) / p) * (365 / days_to_resolution)
    where days_to_resolution = (resolution_time - ts_bin) in days.
    """
    r = path_df.copy()
    r = r[["ts_bin", price_col]].dropna()
    if r.empty:
        return

    # Ensure timezone-aware timestamp
    resolution_time = pd.to_datetime(resolution_time, utc=True)

    # Days to resolution (clip negatives to NaN to avoid div-by-zero/negative periods)
    dt_days = (resolution_time - r["ts_bin"]).dt.total_seconds() / 86400.0
    r["days_to_res"] = np.where(dt_days > 0, dt_days, np.nan)

    # Clip p into (0, 1) to avoid division singularities
    p = r[price_col].clip(1e-9, 1 - 1e-9)

    # APY formula as requested
    r["apy"] = ((1.0 - p) / p) * (365.0 / r["days_to_res"])

    # Secondary axis
    ax2 = ax.twinx()
    ax2.plot(r["ts_bin"], r["apy"], linewidth=linewidth, alpha=alpha, label=label)
    ax2.set_ylabel("annualized yield")
    ax2.grid(False)

    # Keep legends readable: merge handles from both axes
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="best", frameon=False)


def add_median_path(
    ax, df, *, time_col="t", price_col="p", time_bin="1D",
    color="tab:red", linewidth=2, alpha=0.9, label="median"
):
    d = df.copy()
    d["ts"] = pd.to_datetime(d[time_col], unit="s", utc=True)
    d["ts_bin"] = d["ts"].dt.floor(time_bin)

    med = (d.groupby("ts_bin", as_index=False)[price_col]
             .median()
             .rename(columns={price_col: "p_med"}))

    ax.plot(med["ts_bin"], med["p_med"], color=color, linewidth=linewidth, alpha=alpha, label=label)
    return ax

def add_mode_path(
    ax, df, *, time_col="t", price_col="p", time_bin="1D", price_round=3,
    color="black", linewidth=2, alpha=0.9, label="most popular"
):
    d = df.copy()
    d["ts"] = pd.to_datetime(d[time_col], unit="s", utc=True)
    d["ts_bin"] = d["ts"].dt.floor(time_bin)
    d["p_bin"]  = d[price_col].round(price_round)

    # count points in each (time bin, price bin)
    cnt = (d.groupby(["ts_bin", "p_bin"])
             .size()
             .rename("n")
             .reset_index())

    # for each time bin, take the price with max count
    top = (cnt.sort_values(["ts_bin", "n"], ascending=[True, False])
               .groupby("ts_bin", as_index=False)
               .first())

    ax.plot(top["ts_bin"], top["p_bin"], color=color, linewidth=linewidth, alpha=alpha, label=label)
    return ax

def plot_trades(trades: pd.DataFrame):
    yes_line = trades[trades["outcome"] == "Yes"]
    no_line  = trades[trades["outcome"] == "No"]
    plt.figure(figsize=(11,5))
    plt.plot(yes_line.index, yes_line["price"], label="YES", color="blue")
    plt.plot(no_line.index, no_line["price"],  label="NO",  color="red")
    plt.ylabel("probability")
    plt.xlabel("Index")
    plt.ylim(0,1)
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)
    #plt.title(fetch_market(TEST_MARKET_ID)['title'])
    plt.yticks(np.arange(0, 1.05, 0.05)) 
    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1))
    plt.legend()
    plt.tight_layout()
    plt.show()   



def main(argv: Iterable[str] | None = None) -> int:
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
