from imports import *


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


def main(argv: Iterable[str] | None = None) -> int:
    csv_path = Path(__file__).resolve().parents[1] / CSV_OUTPUT_PATH
    plot_market_from_csv(csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
