"""Legacy pipeline preserved from the original implementation."""

import pandas as pd

from fetch.tail_end_func import add_market_bucket, fetch_all_market_prices, fetch_markets


def main() -> None:
    markets = fetch_markets(500)
    print(markets.head())
    print(f"Fetched {len(markets)} markets.")

    df_labeled = pd.DataFrame()
    for idx, (_, market) in enumerate(markets.iterrows()):
        try:
            prices = fetch_all_market_prices(market["id"])
            if not prices.empty:
                market_labeled = add_market_bucket(market, prices)
                df_labeled = pd.concat([df_labeled, pd.DataFrame([market_labeled])], ignore_index=True)
                print(
                    f"[idx={idx}] Price history found for market {market['id']}, "
                    f"market_bucket: {market_labeled['market_bucket']}"
                )
            else:
                print(f"[idx={idx}] No price history for market {market['id']}, skipping (old market).")
        except Exception as exc:
            print(f"[idx={idx}] Error processing market {market['id']}: {exc}")

    if df_labeled.empty:
        print("No labeled markets found.")
    else:
        print(df_labeled[["id", "title", "market_bucket"]])


if __name__ == "__main__":
    main()
