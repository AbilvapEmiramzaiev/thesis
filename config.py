from enum import Enum

class MarketBucket(Enum):
    # High-side buckets (pick the tightest/highest satisfied)
    ALWAYS_HIGH_90 = (0,  ">=90%")
    ALWAYS_HIGH_92 = (1,  ">=92%")
    ALWAYS_HIGH_95 = (2,  ">=95%")
    ALWAYS_HIGH_97 = (3,  ">=97%")
    ALWAYS_HIGH_99 = (4,  ">=99%")

    # Low-side buckets (pick the tightest/smallest satisfied)
    ALWAYS_LOW_10  = (5,  "<=10%")
    ALWAYS_LOW_8   = (6,  "<=8%")
    ALWAYS_LOW_5   = (7,  "<=5%")
    ALWAYS_LOW_3   = (8,  "<=3%")
    ALWAYS_LOW_1   = (9,  "<=1%")

    NOT_TAILEND    = (10, "NOT_TAILEND")
    EMPTY          = (11, "EMPTY")

    def __init__(self, idx: int, label: str):
        self.idx = idx
        self.label = label

HIGH_TO_ENUM = {
    0.90: MarketBucket.ALWAYS_HIGH_90,
    0.92: MarketBucket.ALWAYS_HIGH_92,
    0.95: MarketBucket.ALWAYS_HIGH_95,
    0.97: MarketBucket.ALWAYS_HIGH_97,
    0.99: MarketBucket.ALWAYS_HIGH_99,
}
LOW_TO_ENUM = {
    0.10: MarketBucket.ALWAYS_LOW_10,
    0.08: MarketBucket.ALWAYS_LOW_8,
    0.05: MarketBucket.ALWAYS_LOW_5,
    0.03: MarketBucket.ALWAYS_LOW_3,
    0.01: MarketBucket.ALWAYS_LOW_1,
}

# Constants
YES_INDEX = 0
NO_INDEX  = 1
TOKEN_NAMES = ["YES","NO"]
DATA_API = "https://data-api.polymarket.com"
TEST_MARKET_ID = "563398"
TEST_TAILEND_MARKET_ID = "592998"
CLOB_API  = "https://clob.polymarket.com"
GAMMA_API = "https://gamma-api.polymarket.com"
UNIX_STANDARD = 's'
CLOB_MAX_INTERVAL='max'
CLOB_FIDELITY_TIME='m'
GAMMA_API_DEAD_MARKETS_OFFSET=1997
CSV_OUTPUT_PATH = "data/market_prices.csv"