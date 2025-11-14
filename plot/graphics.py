from imports import *
from plot.plot_data import *
import mplcursors

mind = 1
maxd = 365
amount = 200
start = pd.Timestamp('2024-01-01T00:00:00Z')
end = pd.Timestamp('2025-12-01T00:00:00Z')

#quants aggregated APY
q1_mark = 0.00
q2_mark = 0.025
q3_mark = 0.3

graphic_descrioption = f"Average annualized return (APY) of tail-end markets (p > {int(TAILEND_PERCENT*100)}%). Reflects expected gain from holding near-certain outcome tokens until resolution."

def prepare_apy_graphics(markets:pd.DataFrame, prices:pd.DataFrame):
    markets = filter_by_duration(markets, mind, maxd)
    s = pd.to_datetime(markets["startDate"], utc=True, unit='s', errors="coerce")
    e = pd.to_datetime(markets["endDate"], utc=True, unit='s', errors="coerce")
    # Select markets that INSIDE calendar year 2024
    mask = (s >= start) & (e <= end)
    yearMarkets = markets.loc[mask].copy()

    prices = prices.copy()
    prices['market_id'] = prices['market_id'].astype(int)
    yearMarkets['id'] = yearMarkets['id'].astype(int)
    in_markets = prices['market_id'].isin(yearMarkets['id'])
    #t_utc = pd.to_datetime(prices['t'], unit='s', utc=True)
    #in_2024 = (t_utc >= start) & (t_utc < end)
    filtered_prices = prices[in_markets]
    yearMarkets = yearMarkets.head(amount) if amount and amount > 0 else yearMarkets.copy()
    return yearMarkets, filtered_prices
   
def finish_apy_graphics(yearMarkets:pd.DataFrame,
                        per_market_mean_apy:List[float] = [],
                        medians:List[float] = [],
                        ax:plt.Axes = None,
                        hoverEffect: bool = True):
    liquidity_avg = round(yearMarkets['liquidityNum'].mean(), 2)
    # average time to resolution (in days) for the displayed markets
    avg_res_days = pd.to_numeric(yearMarkets.get('duration_days', pd.Series(dtype=float))).mean()
    avg_res_days_str = f"{avg_res_days:.1f} days" if pd.notna(avg_res_days) else "n/a"

    avg_apy = float(np.mean(per_market_mean_apy)) if per_market_mean_apy else float('nan')
    media_apy = float(np.median(medians)) if per_market_mean_apy else float('nan')
    avg_apy_str = f"{avg_apy * 100:.2f} %" if np.isfinite(avg_apy) else "n/a"
    rows = [
        ("Markets", amount if amount > 0 else len(yearMarkets)),
        ("Duration (min)", f"{mind} days"),
        ("Duration (max)", f"{maxd} days"),
        ("Liquidity (avg)", f"{liquidity_avg if liquidity_avg else 'nan'} $"),
        ("APY (avg)", avg_apy_str),
        ("APY (median)", f"{media_apy * 100:.2f} %"),
        ("Resolution (avg)", avg_res_days_str),
        ("Tailendness ", f"{TAILEND_PERCENT * 100}%"),
        ("Tailend rate ", f"{TAILEND_RATE * 100}%"),

    ]
    ax.text(
        0.5, 1.02, graphic_descrioption,
        transform=ax.transAxes,
        ha="center", va="bottom",
        family="monospace", fontsize=9,
    )
    #ax.set_ylim(0, 0.1)
    add_stats_panel(ax, rows, loc="upper left")   # or "upper right", etc.
    if hoverEffect:
        cursor = mplcursors.cursor(hover=True)
        @cursor.connect("add")
        def on_hover(sel):
            # sel.artist is the line you hovered on
            line = sel.artist
            label = line.get_label()  # your "market_id APY" label
            sel.annotation.set_text(label)
            sel.annotation.get_bbox_patch().set(fc="white", alpha=0.8)
        

#plot graphic with multiple APY lines for each market (markets are tailended)
# and filtered by year or half-year
def graphic_apy_per_market(markets:pd.DataFrame, prices: pd.DataFrame):
    # you send already tail-ended markets!
    yearMarkets, filtered_prices = prepare_apy_graphics(markets, prices)
    #plot = plot_prices(filtered_prices, show=False)
    #ax = plt.gca()
    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)  # вместо plt.gca()
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    ax2 = None
    per_market_mean_apy = []
    medians = []
    subset = yearMarkets.head(amount) if amount > 0 else yearMarkets

    for _, m in subset.iterrows():
        mp = filtered_prices[filtered_prices['market_id'] == m['id']]
        res = m['closedTime']
        label = f"M {m['id']}"
        ax2, apy_vals = add_market_apy_line(ax, mp, resolution_time=res, label=label, ax2=ax2)
        if apy_vals is not None and len(apy_vals) > 0:
            per_market_mean_apy.append(float(np.mean(apy_vals)))
            medians.append(float(np.median(apy_vals)))
    #plt.tight_layout()
    #plt.ylim(0, 0.3) # zoom

    finish_apy_graphics(yearMarkets, per_market_mean_apy, medians, ax)
    ax.set_title(f"APY per market {start.strftime('%d/%m/%y')} - {end.strftime('%d/%m/%y')}", fontsize=12, loc="center", pad=8)  # loc: 'left'|'center'|'right'
    plt.show()
    plt.close(fig) 



# setup for kick off graphic with prices and top-10 prices and APY
def graphic_kickoff(markets:pd.DataFrame, prices:pd.DataFrame):
    filtered = filter_by_timeframe(markets, end_ts=pd.Timestamp('2024-12-31T12:59:59Z'))
    print(len(filtered), 'markets with duration > 70 days')
    tailended = markets.copy()#find_tailend_markets(filtered, prices, 0.90, 0.60)
    print(len(tailended), 'tailend markets found')
    print(len(prices))
    prices['market_id'] = prices['market_id'].astype(int)
    tailended['id'] = tailended['id'].astype(int)
    tailenedPrices = prices[prices['market_id'].isin(tailended['id'])]
    print(len(tailenedPrices), 'price points for tailend markets')
    print(tailenedPrices['t'].min(), 'to', tailenedPrices['t'].max())
    #plot_prices(tailenedPrices)

def graphic_min_apy_line(markets: pd.DataFrame, prices: pd.DataFrame):
    markets = filter_by_duration(markets, mind)
    s = pd.to_datetime(markets["startDate"], utc=True, errors="coerce")
    e = pd.to_datetime(markets["endDate"], utc=True, errors="coerce")
    start = pd.Timestamp('2024-01-01T00:00:00Z')
    end = pd.Timestamp('2025-01-01T00:00:00Z')
    mask = (s >= start) & (e <= end)
    yearMarkets = markets.loc[mask].copy()

    prices = prices.copy()
    prices['market_id'] = prices['market_id'].astype(int)
    yearMarkets['id'] = yearMarkets['id'].astype(int)
    fp = prices[prices['market_id'].isin(yearMarkets['id'])].copy()

    ym = yearMarkets[['id','closedTime','endDate','startDate']].copy()
    ym['res_ts'] = pd.to_datetime(ym['closedTime'], utc=True, errors='coerce')
    mna = ym['res_ts'].isna()
    ym.loc[mna, 'res_ts'] = pd.to_datetime(ym.loc[mna, 'endDate'], utc=True, errors='coerce')
    ym['start_ts'] = pd.to_datetime(ym['startDate'], utc=True, errors='coerce')

    fp = fp.merge(ym[['id','res_ts','start_ts']], left_on='market_id', right_on='id', how='left')
    fp.drop(columns=['id'], inplace=True)

    last_dt_map = pd.to_datetime(fp.groupby('market_id')['t'].transform('max'), unit='s', utc=True)
    fp['res_ts'] = fp['res_ts'].fillna(last_dt_map)

    t = pd.to_datetime(fp['t'], unit='s', utc=True)
    p = fp['p'].clip(1e-9, 1 - 1e-9)
    dt_days = (fp['res_ts'] - t).dt.total_seconds() / 86400.0
    since_days = (t - fp['start_ts']).dt.total_seconds() / 86400.0
    since_days_exclude = 3.0
    last_days_exclude = 3.0
    m = (dt_days > last_days_exclude) & (since_days > since_days_exclude)
    fp = fp[m].copy()
    t = t[m]
    p = p[m]
    dt_days = dt_days[m]
    apy = ((1.0 - p) / p) * (365.0 / dt_days)

    df = pd.DataFrame({'ts': t, 'apy': apy})
    df['ts_bin'] = df['ts'].dt.floor('1D')
    series = df.groupby('ts_bin', as_index=True)['apy'].min()

    ax = plt.gca()
    ax.plot(series.index, series.values, label='min APY')
    ax.set_xlabel('time (UTC)')
    ax.set_ylabel('annualized yield')
    ax.grid(True, axis='x', linestyle='--', alpha=0.3)
    rows = [
        ("Duration (min)", f"{mind} days"),
       # ("Duration (max)", f"{maxd} days"),
        ("Markets  ", f"{len(yearMarkets)}"),
        ("Tailend ", f"{TAILEND_PERCENT * 100}%"),
        ("Tailend rate ", f"{TAILEND_RATE * 100}%"),
        ("Excluded days: first", f"{since_days_exclude}, last {last_days_exclude}"),


    ]
    ax.set_title("Aggregated min APY", fontsize=12, loc="center", pad=8)  # loc: 'left'|'center'|'right'
    add_stats_panel(ax, rows, loc="upper left")   # or "upper right", etc.

    plt.tight_layout()
    plt.show()


def graphic_apy_aggregated(
    markets: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    price_col: str = "p",
    ts_col: str = "t",
    resample_rule: str = "1D",  # aggregate time grid (daily)
    min_days_before_res: float = 3.0,
    title_prefix: str = "Aggregated APY (10/50/90)",
    show_count_markers: bool = True,
    marker_rule: str = "1W",     # e.g. "1W", "2W", "1M"
):
    # You send already tail-ended markets!
    yearMarkets, filtered_prices = prepare_apy_graphics(markets, prices)
    apy_frames = []
    per_market_mean_apy = []
    medians = []
    for _, m in yearMarkets.iterrows():
        mid = m["id"]
        mp = filtered_prices[filtered_prices["market_id"] == mid]
        s = compute_market_apy_series(
            mp,
            resolution_time=m["closedTime"],
            price_col=price_col,
            ts_col=ts_col,
            min_days_before_res=min_days_before_res,
        )
        # Resample to a common grid for clean aggregation (median/quantiles)
        sr = s.resample(resample_rule).median()
        if s is not None and len(s) > 0:
            per_market_mean_apy.append(float(np.mean(sr)))
            medians.append(float(np.median(sr)))
        sr.name = f"M{mid}"
        apy_frames.append(sr)

    apy_mat = pd.concat(apy_frames, axis=1)

    # Optionally forward-fill short gaps (keeps band continuous without over-smoothing)
    # Limit ffill to a small window to avoid bleeding too far:
    apy_mat = apy_mat.ffill(limit=3)

    # Compute timewise percentiles
    
    title_prefix = f"Aggregated APY ({int(q1_mark*100)}%/{int(q2_mark*100)}%/{int(q3_mark*100)}%)"
    q1 = apy_mat.quantile(q1_mark, axis=1)
    q2 = apy_mat.quantile(q2_mark, axis=1)
    q3 = apy_mat.quantile(q3_mark, axis=1)

    # Overall quantiles (across markets × time), useful for legend title
    flat_vals = apy_mat.stack().dropna().values
    overall_q1 = float(np.percentile(flat_vals, q1_mark*100)) if flat_vals.size else np.nan
    overall_q2 = float(np.percentile(flat_vals, q2_mark*100)) if flat_vals.size else np.nan
    overall_q3 = float(np.percentile(flat_vals, q3_mark*100)) if flat_vals.size else np.nan

    # Time range for title
    start_ts = pd.to_datetime(filtered_prices[ts_col], unit="s", utc=True).min()
    end_ts = pd.to_datetime(filtered_prices[ts_col], unit="s", utc=True).max()

    # Plot: band + median
    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)

    ax.fill_between(q1.index, q1.values, q3.values, alpha=0.2, label=f"P{q1_mark*100}%–P{q3_mark*100}%")
    ax.plot(q2.index, q2.values, linewidth=2, label=f"Median (P{q2_mark*100}%)")

    # Niceties
    ax.set_ylabel("annualized yield (*100%)")
    n_markets = apy_mat.shape[1]
    title = (
        f"{title_prefix} across {n_markets} markets\n"
        f"{start_ts.strftime('%d/%m/%y')} – {end_ts.strftime('%d/%m/%y')}"
    )
    ax.set_ylim(0, 0.5)
    ax.set_title(title, fontsize=12, loc="center", pad=8, y=1.04)

    # Minimal weekly boxes with contributing market counts on the median line
    if show_count_markers and not apy_mat.empty:
        counts = apy_mat.count(axis=1).resample(marker_rule).mean().round()
        med = q2.resample(marker_rule).median()
        for t, n in counts.dropna().items():
            y = med.get(t)
            if pd.isna(y):
                continue
            ax.text(
                t, float(y), f"{int(n)}",
                ha="center", va="center", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="tab:blue", lw=0.8),
                zorder=5,
            )
    finish_apy_graphics(yearMarkets, [],[],ax=ax, hoverEffect=False)
    leg_title = f"Overall P{q1_mark*100}/P{q2_mark*100}/P{q3_mark*100}:\n{overall_q1:.2f} / {overall_q2:.2f} / {overall_q3:.2f}"
    leg = ax.legend(frameon=False, title=leg_title, loc="upper right")
    if leg and leg.get_title():
        leg.get_title().set_fontsize(9)

    plt.show()
    plt.close(fig)


def graphic_apy_aggregated_many_years(
    markets: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    price_col: str = "p",
    ts_col: str = "t",
    resample_rule: str = "1D",
    min_days_before_res: float = 3.0,
    q_lo_mark: float = q1_mark,     # 5th pct
    q_md_mark: float = q2_mark,     # median
    q_hi_mark: float = q3_mark,     # 60th pct (keeps spikes out)
    shade_years: bool = True,
    show_count_markers: bool = True,
    marker_rule: str = "1W",
    title_prefix: str = "Aggregated APY by Year",
):
    # ---------- 1) Same prep as your function ----------
    yearMarkets, filtered_prices = prepare_apy_graphics(markets, prices)

    frames = []
    for _, m in yearMarkets.iterrows():
        mid = m["id"]
        mp = filtered_prices[filtered_prices["market_id"] == mid]
        s = compute_market_apy_series(
            mp,
            resolution_time=m["closedTime"],
            price_col=price_col,
            ts_col=ts_col,
            min_days_before_res=min_days_before_res,
        )
        sr = s.resample(resample_rule).median()  # align to daily, robust
        sr.name = f"M{mid}"
        frames.append(sr)

    apy_mat = pd.concat(frames, axis=1).ffill(limit=3)
    flat_vals = apy_mat.stack().dropna().values
    overall_lo = float(np.percentile(flat_vals, q_lo_mark * 100))
    overall_md = float(np.percentile(flat_vals, q_md_mark * 100))
    overall_hi = float(np.percentile(flat_vals, q_hi_mark * 100))

    # These are timewise (row-wise) quantiles across markets for *all* timestamps;
    # we’ll slice them by year when plotting.
    q_lo_all = apy_mat.quantile(q_lo_mark, axis=1)
    q_md_all = apy_mat.quantile(q_md_mark, axis=1)
    q_hi_all = apy_mat.quantile(q_hi_mark, axis=1)

    # ---------- 2) Figure ----------
    fig, ax = plt.subplots(figsize=(16, 6), constrained_layout=True)
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    ax.set_ylabel("annualized yield (*100%)")

    # Year partitions present in the time index
    years = sorted(np.unique(apy_mat.index.year))

    # Nice repeating color cycle (same hue for band+line per year)
    # You can use your own palette if you want.
    base_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0","C1","C2","C3","C4","C5"])

    # ---------- 3) Plot per-year with background shade ----------
    for i, yr in enumerate(years):
        mask = (apy_mat.index.year == yr)
        if not mask.any():
            continue

        # background shade
        if shade_years:
            ax.axvspan(
                apy_mat.index[mask][0],
                apy_mat.index[mask][-1],
                facecolor="0.92" if i % 2 == 0 else "0.97",
                alpha=1.0,
                zorder=0,
            )

        # slice quantile series to that year
        q_lo = q_lo_all[mask]
        q_md = q_md_all[mask]
        q_hi = q_hi_all[mask]

        c = base_colors[i % len(base_colors)]
        ax.fill_between(q_lo.index, q_lo.values, q_hi.values, alpha=0.25, label=f"{yr}: P{int(q_lo_mark*100)}–P{int(q_hi_mark*100)}", color=c)
        ax.plot(q_md.index, q_md.values, linewidth=2, label=f"{yr}: Median (P{int(q_md_mark*100)})", color=c)
       # ax.set_ylim(0, 0.8)
        # optional per-year count markers along the median
        if show_count_markers:
            counts = apy_mat.loc[mask].count(axis=1).resample(marker_rule).mean().round()
            med = q_md.resample(marker_rule).median()
            for t, n in counts.dropna().items():
                y = med.get(t)
                if pd.isna(y):
                    continue
                ax.text(
                    t, float(y), f"{int(n)}",
                    ha="center", va="center", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=c, lw=0.8),
                    zorder=5,
                )

    # ---------- 4) Title, legend ----------
    start_ts = apy_mat.index.min()
    end_ts   = apy_mat.index.max()
    title = (
        f"{title_prefix}  (P{int(q_lo_mark*100)}/P{int(q_md_mark*100)}/P{int(q_hi_mark*100)})\n"
        f"{start_ts.strftime('%d/%m/%y')} – {end_ts.strftime('%d/%m/%y')}"
    )
    ax.set_title(title, fontsize=12, loc="center", pad=8, y=1.04)
    finish_apy_graphics(yearMarkets, [], [], ax=ax, hoverEffect=False)
    leg_title = (
        f"Overall P{int(q_lo_mark*100)}/P{int(q_md_mark*100)}/P{int(q_hi_mark*100)}:\n"
        f"{overall_lo:.2f} / {overall_md:.2f} / {overall_hi:.2f}"
    )
    #leg = ax.legend(frameon=False, title=leg_title, loc="upper right", ncol=1)
    #if leg and leg.get_title():
    #    leg.get_title().set_fontsize(9)

    plt.show()
    plt.close(fig)
