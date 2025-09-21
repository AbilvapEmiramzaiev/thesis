# Leverage Mechanisms and Market Efficiency in Blockchain-Based Prediction Markets

Prediction markets aggregate dispersed information by letting participants trade tokenized claims on future outcomes; prices map to implied probabilities (e.g., 0.90 = 90%).  
In many markets a single outcome becomes the perceived frontrunner (p ≥ 0.90) well before resolution.  
We call these **tail-end markets**. They tend to attract few counterparties, exhibit thin liquidity, and often display persistent mispricing (over- or underestimation of probabilities).

This thesis explores whether **leverage** can counter those frictions. The premise is that carefully designed leverage can:

- Draw in informed but capital-constrained traders,  
- Deepen order books and liquidity pools,  
- Improve calibration of probabilities,  

all **without introducing excessive liquidation risk or bad debt**:contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}.

---

## Research Objectives

We evaluate tail-end markets, diagnose their mispricing and participation patterns, and assess the introduction of leverage across multiple mechanism designs.

### RQ1: Pricing Inefficiency in Tail-End Events
- **Question:** To what extent do tail-end markets in prediction platforms like Polymarket exhibit pricing inefficiencies or constant volume declines?  
- **Approach:**  
  - Quantify mispricing by comparing pre-resolution implied probabilities to actual outcomes.  
  - Use calibration curves and Brier scores (usual, relative, ordinal).  
  - Analyze profitability and liquidity patterns.  
  - Show correlations between market volume and probability brackets:contentReference[oaicite:2]{index=2}.

### RQ2: Comparative Analysis of Leverage Mechanisms
- **Question:** How do different leverage mechanisms differ in efficiency, risk, and impact on tail-end pricing?  
- **Mechanisms Studied:**  
  A) **Perpetual/futures contracts** with market probability as spot  
  B) **Self-collateralized lending** (looping against position tokens)  
  C) **USD-backed reverse positions**  
- **Focus:**  
  - Capital efficiency  
  - Liquidation dynamics  
  - Bad debt risks  
  - Practical impact on liquidity and mispricing:contentReference[oaicite:3]{index=3}:contentReference[oaicite:4]{index=4}  

### RQ3: Stability, Cascades & Bad Debt
- **Question:** How do prediction markets handle extreme price movements?  
- **Focus:**  
  - Liquidity and shock absorption (can large trades be absorbed without cascades?)  
  - Liquidation feasibility given oracle latency, block time, gas constraints  
  - Historical analysis of volatility spikes and whether liquidation would have been possible before debt accrual:contentReference[oaicite:5]{index=5}:contentReference[oaicite:6]{index=6}  

---

## Methodology

The empirical work combines live-market data with counterfactual simulation:

- **Gamma Markets API** – market metadata, prices, liquidity, timestamps  
- **CLOB / Data API** – trades, order books, spreads, tail-end probability series  
- **dYdX documentation** – leverage mechanics, liquidation design  
- **Flipr** – social and leverage layer on Polymarket  
- **Research Reference:** *Improved Liquidity for Prediction Markets* by Lukas Kapp-Schwoerer:contentReference[oaicite:7]{index=7}

---

## Repository Structure

