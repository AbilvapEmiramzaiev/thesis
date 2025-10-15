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
    

def plot_prices(
    df: pd.DataFrame,
    *,
    show: bool = True,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    ax = ax or plt.gca()

    # group by market_id and plot each market with day counter
    for _, group in df.groupby("market_id"):
        day_index = range(len(group))
        ax.scatter(
            day_index,
            group["p"],
            s=3,
            alpha=0.7,
            edgecolors="none",
        )
    ax.set_xlabel("day index (0 = market start)")
    ax.set_ylabel("probability")
    ax.set_ylim(0.8, 1)
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)

    legend = ax.get_legend()
    if legend:
        legend.remove()

    if show and ax.figure:
        plt.show()

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
    plt.title(fetch_market(TEST_MARKET_ID)['title'])
    plt.yticks(np.arange(0, 1.05, 0.05)) 
    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1))
    plt.legend()
    plt.tight_layout()
    plt.show()   



def main(argv: Iterable[str] | None = None) -> int:
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
