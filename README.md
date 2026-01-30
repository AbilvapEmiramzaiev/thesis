# Leverage Mechanisms and Market Efficiency in Blockchain-Based Prediction Markets

Prediction markets aggregate dispersed information by letting participants trade tokenized claims on future outcomes; prices map to implied probabilities (e.g., 0.90 = 90%).  
In many markets a single outcome becomes the perceived frontrunner (p ≥ 0.90) well before resolution.  
We call these **tail-end markets**. They tend to attract few counterparties, exhibit thin liquidity, and often display persistent mispricing (over- or underestimation of probabilities).

This thesis explores whether **leverage** can counter those frictions. The premise is that carefully designed leverage can:

- Draw in informed but capital-constrained traders,  
- Deepen order books and liquidity pools,  
- Improve calibration of probabilities,  

all **without introducing excessive liquidation risk or bad debt**.

---

## Research Objectives

We evaluate tail-end markets on Polymarket by measuring mispricing and participation patterns, then compare how leverage mechanisms (perps/futures, self-collateralized looping, and USD-backed reverse positions) affect capital efficiency, liquidity, and bad-debt risk, and finally test system stability under shocks by examining liquidation feasibility given latency and historical volatility

## Methodology

The empirical work combines live-market data with counterfactual simulation:

- **Gamma Markets API** – market metadata, prices, liquidity, timestamps  
- **CLOB / Data API** – trades, order books, spreads, tail-end probability series  
- **dYdX documentation** – leverage mechanics, liquidation design  
- **Flipr** – social and leverage layer on Polymarket  
- **Research Reference:** *Improved Liquidity for Prediction Markets* by Lukas Kapp-Schwoerer:contentReference[oaicite:7]{index=7}

Note: Python files with `liquidation` in the name contain liquidation-focused analysis and are being expanded.

---

## Code Map

- `pipelines/pipeline.py` – end-to-end data collection pipeline (markets, prices, trades) with CLI flags.
- `fetch/tail_end_func.py` – API wrappers + tail-end market identification utilities.
- `fetch/trades.py` – trade aggregation, trader behavior analysis, and plotting helpers.
- `fetch/volatility.py` – volatility calculations, tail-end vs non-tail-end comparisons.
- `fetch/liquidation.py` – liquidation analysis.

---
