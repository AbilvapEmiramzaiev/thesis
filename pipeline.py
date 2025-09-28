import pandas as pd
from tail_end_func import *

if __name__ == "__main__":
	markets = fetch_markets(500, 25800)
	print(f"Fetched {len(markets)} markets.")

	df_labeled = pd.DataFrame()
	for idx, (i, market) in enumerate(markets.iterrows()):
		try:
			prices = fetch_all_market_prices(market['id'])
			if not prices.empty:
				market_labeled = add_market_bucket(market, prices)
				df_labeled = pd.concat([df_labeled, pd.DataFrame([market_labeled])], ignore_index=True)
				print(f"[idx={idx}] Price history found for market {market['id']}, market_bucket: {market_labeled['market_bucket']}")
			else:
				print(f"[idx={idx}] No price history for market {market['id']}, skipping (old market).")
		except Exception as e:
			print(f"[idx={idx}] Error processing market {market['id']}: {e}")

	# Output results
	if not df_labeled.empty:
		print(df_labeled[["id", "title", "market_bucket"]])
	else:
		print("No labeled markets found.")
