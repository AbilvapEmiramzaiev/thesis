from imports import *
from fetch.tail_end_func import fetch_market, fetch_market_prices_history, fetch_markets

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
    plt.title(fetch_market(TEST_MARKET_ID)['question'])
    plt.yticks(np.arange(0, 1.05, 0.05)) 
    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1))
    plt.legend()
    plt.tight_layout()
    plt.show()
    

def plot_market_from_csv(
    csv_path: Path,
    *,
    show: bool = True,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot every market contained in a cached CSV file."""
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"No rows found in {csv_path}")

    df["ts"] = pd.to_datetime(df["t"], unit="s", utc=True)
    ax = ax or plt.gca()

    for _, group in df.groupby("market_id"):
        ax.scatter(
            group["ts"],
            group["p"],
            s=14,
            alpha=0.5,
            edgecolors="none",
        )

    ax.set_ylabel("probability")
    ax.set_xlabel("time")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    legend = ax.get_legend()
    if legend:
        legend.remove()
    if show and ax.figure:
        ax.figure.autofmt_xdate()
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
    csv_path = Path(__file__).resolve().parents[1] / CSV_OUTPUT_PATH
    plot_market_from_csv(csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
