import json
from typing import Optional

def identify_market_outcome_winner_index(market: json) -> Optional[str]:
    """Identify the winning outcome of a market using the clobid field."""
    tol = 1e-3
    outcomes = market.get("outcomePrices", [])
    if not outcomes:
        return None

    # Find the outcome with the highest clobid
    for i, p_str in enumerate(outcomes):
            try:
                p = float(p_str)
            except ValueError:
                continue
            # Check if p is “close enough” to 1.0
            if abs(p - 1.0) <= tol:
                return i
