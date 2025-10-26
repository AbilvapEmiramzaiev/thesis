from imports import *
from plot.plot_data import *
from fetch.tail_end_func import find_tailend_markets
import mplcursors

#plot graphic with multiple APY lines for each market (markets are tailended)
# and filtered by year or half-year
def graphic_apy_per_market(markets:pd.DataFrame, prices: pd.DataFrame):
    # you send already tail-ended markets!
    s = pd.to_datetime(markets["startDate"], utc=True, errors="coerce")
    e = pd.to_datetime(markets["endDate"], utc=True, errors="coerce")

    # Select markets that overlap calendar year 2024
    start = pd.Timestamp('2024-01-01T00:00:00Z')
    end = pd.Timestamp('2025-01-01T00:00:00Z')
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
    ax = plt.gca()
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    ax2 = None

    for _, m in yearMarkets.head(500).iterrows():
        mp = filtered_prices[filtered_prices['market_id'] == m['id']]
        res = m['closedTime']
        label = f"M {m['id']}"
        if m['id'] == 254097:
             print(1)
        ax2 = add_market_apy_line(ax, mp, resolution_time=res, label=label, ax2=ax2)
    plt.tight_layout()
    
    
    
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
    
    plt.show()



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
