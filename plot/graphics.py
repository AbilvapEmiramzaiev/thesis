from imports import *
from plot.plot_data import *
from fetch.tail_end_func import find_tailend_markets
import mplcursors

mind = 14
maxd = 30
amount = 0
start = pd.Timestamp('2023-01-01T00:00:00Z')
end = pd.Timestamp('2024-01-01T00:00:00Z')
#plot graphic with multiple APY lines for each market (markets are tailended)
# and filtered by year or half-year
def graphic_apy_per_market(markets:pd.DataFrame, prices: pd.DataFrame):
    # you send already tail-ended markets!
    markets = filter_by_duration(markets, mind, maxd)
    s = pd.to_datetime(markets["startDate"], utc=True, errors="coerce")
    e = pd.to_datetime(markets["endDate"], utc=True, errors="coerce")

    # Select markets that overlap calendar year 2024
    mask = (s >= start) & (e <= end)
    yearMarkets = markets.loc[mask].copy()

    prices = prices.copy()
    prices['market_id'] = prices['market_id'].astype(int)
    yearMarkets['id'] = yearMarkets['id'].astype(int)
    in_markets = prices['market_id'].isin(yearMarkets['id'])
    #t_utc = pd.to_datetime(prices['t'], unit='s', utc=True)
    #in_2024 = (t_utc >= start) & (t_utc < end)
    filtered_prices = prices[in_markets]
    
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
    #plt.ylim(0, 10) # zoom


    liquidity_avg = yearMarkets['liquidityNum'].mean().round(2)
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
    add_stats_panel(ax, rows, loc="upper left")   # or "upper right", etc.

    
    cursor = mplcursors.cursor(hover=True)
    if ax.get_legend():
            ax.get_legend().set_visible(False)
    @cursor.connect("add")
    def on_hover(sel):
        # sel.artist is the line you hovered on
        line = sel.artist
        label = line.get_label()  # your "market_id APY" label
        sel.annotation.set_text(label)
        sel.annotation.get_bbox_patch().set(fc="white", alpha=0.8)
    ax.set_title(f"APY per market {start.strftime('%d/%m/%y')} - {end.strftime('%d/%m/%y')}", fontsize=12, loc="center", pad=8)  # loc: 'left'|'center'|'right'

    plt.show()
    plt.close(fig) 



# setup for kick off graphic with prices and top-10 prices and APY
def graphic_kickoff(markets:pd.DataFrame, prices:pd.DataFrame):
    filtered = filter_by_timeframe(markets, end_ts=pd.Timestamp('2024-12-31T12:59:59Z'))
    print(len(filtered), 'markets with duration > 70 days')
    tailended = find_tailend_markets(filtered, prices, 0.90, 0.60)
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
