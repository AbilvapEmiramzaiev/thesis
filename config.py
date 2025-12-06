from enum import Enum
from pathlib import Path
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
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

def make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=5,                  # total retries
        connect=5, read=5,
        backoff_factor=1,       # 0.5, 1.0, 2.0, 4.0 ...
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],  # retry only safe methods
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=100, pool_maxsize=100)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({"User-Agent": "thesis-fetcher/1.0"})
    return s

SESSION = make_session()
DEFAULT_TIMEOUT = (7, 40)

# Constants
PROJECT_ROOT = Path(__file__).resolve().parent
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
GAMMA_API_OLD_MARKETS_OFFSET=4750
GAMMA_API_LAST_EVENTS_OFFSET=9760
GAMMA_API_LAST_PIPELINE_OFFSET=73920#98900
CSV_OUTPUT_PATH = "data/market_prices.csv"
TIME_COLS = ["endDate", "startDate", "closedTime"]
TAILEND_PERCENT=0.7
TAILEND_RATE=0.4
TAILEND_LOSER_PRICE=0.2