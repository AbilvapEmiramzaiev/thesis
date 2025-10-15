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
            color=(0.55, 0.75, 0.95),  # light pastel blue (RGB tuple 0–1 range)
        )

    ax.set_xlabel("time (UTC)")
    ax.set_ylabel("probability")
    ax.set_ylim(0.8, 1)
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    ax.figure.autofmt_xdate()
    #add_mode_path(ax, df, time_col="t", price_col="p", time_bin="1D", price_round=3, color="black", label="most popular")
    add_median_path(ax, df, time_col="t", price_col="p", time_bin="1D", color="tab:red", label="median")

    leg = ax.legend(loc="best", frameon=False)
    plt.show()
    return ax

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
