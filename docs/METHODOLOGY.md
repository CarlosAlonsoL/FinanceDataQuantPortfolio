# Prediction Methodology & Portfolio Construction

## 1. Prediction Methodology

### 1.1 Problem Formulation

We formulate S&P 500 membership prediction as two binary classification problems:
- **Join prediction**: P(stock joins S&P 500 within next 63 trading days | features at time t)
- **Leave prediction**: P(stock leaves S&P 500 within next 63 trading days | features at time t)

The 63-day horizon (approximately 3 calendar months) balances prediction lead time with label accuracy.

### 1.2 Feature Engineering

Features are computed from CRSP daily stock data with strict no-lookahead constraints (all features lagged by 1 trading day):

| Feature Group | Variables | Rationale |
|--------------|-----------|-----------|
| **Momentum** | 21d, 63d, 126d, 252d cumulative returns; 12m skip-month momentum | Price momentum predicts future membership changes; skip-month avoids short-term reversal |
| **Volatility** | 21d, 63d, 126d rolling standard deviation | Lower volatility stocks are more likely to be added |
| **Size** | Market capitalization, market cap rank, size percentile | Primary eligibility criterion for S&P 500 inclusion |
| **Liquidity** | Turnover ratio, rolling average volume | Minimum liquidity required for inclusion |
| **Abnormal performance** | Excess returns vs market-cap-weighted benchmark | Outperformance signals potential candidacy |
| **Quality proxy** | Return / volatility ratio (1-year) | Proxy for financial viability without accounting data |

### 1.3 Model

**Algorithm:** XGBoost (gradient-boosted decision trees)
- `n_estimators=100`, `max_depth=4`, `learning_rate=0.1`
- Class imbalance handled via `scale_pos_weight = n_negative / n_positive`
- GPU acceleration when available (CUDA or Apple MPS)

**Validation:** Rolling time-series cross-validation
- Train window: 5 years, Test window: 1 year
- 20 folds covering the full sample period
- No data leakage: strict temporal ordering, features lagged by 1 day

### 1.4 Model Performance

| Metric | Join Model | Leave Model |
|--------|-----------|-------------|
| **ROC-AUC** | 0.94 (mean across folds) | 0.92 |
| **Brier Score** | 0.026 | 0.011 |
| **OOS Accuracy** | 96.4% | 98.2% |
| **IC (21-day)** | 0.017 | — |
| **ICIR** | 0.17 | — |

The model discriminates well between future joiners/leavers and non-events (high AUC), but the extreme class imbalance (~2-4% positive rate) means precision at any threshold is inherently low.

## 2. Portfolio Construction Rules

### 2.1 Strategy Variants

We implement and compare two portfolio construction approaches:

**Quantile-based (baseline):** Select the top decile (10%) of stocks by predicted probability for each leg. This yields ~743 stocks per side — a diversified but diluted portfolio.

**Top-N (concentrated):** Select only the top N stocks by predicted probability. This concentrates capital in the highest-conviction names, at the cost of higher idiosyncratic risk.

In both cases:
1. **Long leg:** top stocks ranked by P(join) — highest probability of entering S&P 500
2. **Short leg:** top stocks ranked by P(leave) — highest probability of exiting S&P 500
3. **Weighting:** equal-weight or probability-weighted within each leg

### 2.2 Position Sizing

| Parameter | Quantile Strategy | Top-N Strategy |
|-----------|-------------------|----------------|
| **Positions per side** | ~743 (top 10%) | 5, 10, 20, 30, 50 (swept) |
| **Gross exposure** | ~1.1x | ~1.3x |
| **Net exposure** | ~0.2x | ~0x |
| **Weighting** | Equal | Equal or probability-weighted |

### 2.3 Rebalancing

- **Frequency:** Monthly (first trading day of each month)
- **Between rebalances:** Weights are held constant (no drift adjustment)
- **At rebalance:** Previous positions are fully closed, new positions opened based on current model predictions

### 2.4 Transaction Costs

- **Cost model:** Proportional to turnover at 10 bps per unit of turnover
- **Turnover:** Measured as sum of absolute weight changes at each rebalance

## 3. Benchmark: Perfect Foresight (Omniscient) Strategy

The omniscient benchmark uses realized future S&P 500 membership to construct the portfolio:
- **Long:** Stocks that will actually join the index within 63 trading days
- **Short:** Stocks that will actually leave the index within 63 trading days
- **Weighting:** Equal-weight within each leg

This provides an upper bound on performance — the maximum alpha achievable if the investor had perfect knowledge of future index changes. The gap between predictive and omniscient strategies quantifies the cost of prediction uncertainty.

## 4. Results

### 4.1 Strategy Comparison

| Strategy | Annual Return | Volatility | Sharpe | Sortino | Max Drawdown | VaR (5%) |
|----------|--------------|------------|--------|---------|--------------|----------|
| **Omniscient** | 15.6% | 16.3% | 0.97 | 1.19 | 54.6% | -1.49% |
| **Quantile (top 10%)** | 3.8% | 6.5% | **0.61** | 0.63 | **29.9%** | -0.63% |
| **Top-5 (prob-weighted)** | **9.6%** | 28.3% | 0.47 | 0.51 | 70.1% | -2.71% |
| **Top-5 (equal)** | 9.5% | 28.3% | 0.46 | 0.50 | 71.0% | -2.69% |
| **Top-10 (prob-weighted)** | 6.2% | 22.8% | 0.38 | 0.40 | 65.2% | -2.18% |
| **Top-20 (equal)** | 2.3% | 19.0% | 0.22 | 0.22 | 62.0% | -1.84% |
| **Top-30 (equal)** | 2.1% | 17.1% | 0.21 | 0.21 | 58.1% | -1.66% |
| **Top-50 (equal)** | 1.0% | 14.8% | 0.14 | 0.14 | 50.6% | -1.44% |

### 4.2 Key Findings

1. **The predictive model generates positive risk-adjusted returns.** The best risk-adjusted strategy (quantile, top 10%) achieves a Sharpe ratio of 0.61 with only 29.9% maximum drawdown — roughly 63% of the omniscient benchmark's Sharpe.

2. **Concentration increases returns but at the cost of higher risk.** The top-5 strategy delivers the highest annual return (9.6%) but with extreme volatility (28.3%) and 70% drawdown. The quantile strategy offers a better risk-return tradeoff.

3. **Probability weighting marginally improves performance.** Across all top-N variants, probability-weighted portfolios slightly outperform equal-weighted ones, suggesting the model's relative ranking carries useful information about conviction strength.

4. **The predictive strategy captures roughly 25-60% of the omniscient benchmark's return,** depending on concentration. This quantifies the cost of replacing perfect foresight with noisy model predictions.

### 4.3 Economic Interpretation

The positive performance of the predictive strategy is consistent with the academic literature on the S&P 500 index effect:

- **Addition effect:** Stocks predicted to join the index tend to exhibit positive abnormal returns, driven by anticipated demand from index-tracking funds and the signaling value of inclusion.
- **Deletion effect:** Stocks predicted to leave tend to underperform, reflecting both the mechanical selling pressure from index funds and the fundamental deterioration that often triggers deletion.
- **Signal decay with portfolio size:** As N increases from 5 to 50, Sharpe ratios decline monotonically (0.47 to 0.14). This is expected: the model's signal is strongest for the highest-ranked stocks, and diluting capital across lower-conviction names reduces the portfolio's information ratio.
- **Quantile strategy outperforms on risk-adjusted basis:** Despite lower returns, the diversification benefit of holding ~743 stocks per side reduces volatility sufficiently to produce the highest Sharpe ratio. This reflects the well-known diversification-concentration tradeoff in portfolio management.

## 5. Robustness Framework

The strategy is tested across a grid of parameter variations:

| Dimension | Values Tested |
|-----------|---------------|
| **Holding period** | 1, 3, 6, 12 months |
| **Number of positions (N)** | 5, 10, 20, 30, 50 |
| **Weighting scheme** | Equal, probability-weighted |

This produces a heatmap of Sharpe ratios across the (holding period x N) grid, revealing which parameter combinations are robust and which are sensitive to specification. Results are saved in `results/tables/robustness_holding_periods.csv` and visualized in `results/figures/robustness_heatmap_*.png`.

## 6. Performance Metrics

| Metric | Description |
|--------|-------------|
| **Annual return** | Geometric annualized return |
| **Annual volatility** | Annualized standard deviation of daily returns |
| **Sharpe ratio** | Excess return per unit of risk (annualized) |
| **Sortino ratio** | Excess return per unit of downside risk |
| **Maximum drawdown** | Largest peak-to-trough decline |
| **Calmar ratio** | Annual return / maximum drawdown |
| **VaR (5%)** | 5th percentile of daily return distribution |
| **Skewness** | Asymmetry of return distribution |
| **Turnover** | Average daily absolute weight change |
| **Factor exposure** | Betas to Market, SMB, HML, MOM factors |
