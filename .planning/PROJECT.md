# S&P 500 Joiners & Leavers — Academic Project (HEC Paris)

## What This Is

A quantitative research project for HEC Paris (Augustin Landier, Intro to Hedge Funds & Asset Management) that builds and evaluates trading strategies that anticipate S&P 500 index additions and deletions. The deliverable is a professional Jupyter notebook that demonstrates a full research pipeline: event study, predictive modeling, portfolio construction, and performance evaluation — with emphasis on model quality and economic interpretation.

## Core Value

The strategy must demonstrably predict which firms will join/leave the S&P 500 *before* the announcement, converting that signal into a profitable long-short portfolio with rigorous out-of-sample evidence.

## Requirements

### Validated (existing codebase)

- ✓ Data loading pipeline (CRSP daily panel + S&P 500 events Excel) — existing
- ✓ Event study engine (CAR, volume, volatility around announcement) — existing
- ✓ Feature engineering (momentum skip-month, quality proxy, excess returns, market cap rank, size percentile) — existing
- ✓ ML models with rolling OOS evaluation (Logistic, RF, GradientBoosting, XGBoost) — existing
- ✓ Portfolio construction (equal/probability/rank weighting, long-short dollar-neutral) — existing
- ✓ Perfect foresight benchmark — existing
- ✓ Backtester with transaction costs (10bps) — existing
- ✓ Performance metrics (Sharpe, Sortino, Calmar, VaR, skewness, subperiod) — existing
- ✓ 4-factor Fama-French attribution — existing

### Active

- [ ] Information Coefficient (IC) analysis — rank correlation between model score and realized return, with decay curve by horizon
- [ ] Master deliverable notebook (03_master_analysis.ipynb) — end-to-end story from raw data to economic conclusions
- [ ] S&P 500 institutional rules documentation section in notebook
- [ ] Perfect foresight benchmark vs. realistic strategies performance comparison table
- [ ] Holding horizon robustness analysis (1d, 5d, 21d, 63d windows) — explicitly required by rubric
- [ ] Feature importance with SHAP values — interpretable signal attribution
- [ ] Model comparison: AUC/Brier score + portfolio Sharpe by model type
- [ ] Regime analysis (pre/post 2008, bull/bear subperiods)
- [ ] Economic interpretation section (why does this work? what limits it?)
- [ ] Bootstrap confidence intervals on Sharpe ratio (is alpha statistically significant?)

### Out of Scope

- Real-time trading infrastructure — academic project only
- Accounting data (Compustat) features — not in provided dataset
- PEAD/earnings surprise features — requires IBES, not available
- Mobile/web deliverable — Jupyter notebook only

## Context

- **Course**: Intro to Hedge Funds & Asset Management, HEC Paris, W1 2026
- **Professor**: Augustin Landier
- **Deadline**: End of current week (March 2026)
- **Data**: CRSP daily panel + S&P 500 joiners/leavers 1995–Feb 2026
- **Key rubric points**: prediction methodology clarity, portfolio construction rules, robustness (varying holding horizon), economic interpretation
- **Differentiator**: IC/ICIR analysis is standard in professional quant research but unlikely to be done by most students; SHAP values make feature attribution academically rigorous

## Constraints

- **Timeline**: Must be done this week — prioritize highest-impact improvements
- **Tech Stack**: Python, scikit-learn, XGBoost, pandas, matplotlib/seaborn — no new dependencies unless trivial
- **Format**: Jupyter notebook as primary deliverable (data must run end-to-end)
- **Academic integrity**: Must reflect genuine analysis, not fabricated results

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| IC/ICIR as primary model quality metric | Standard in quant finance; shows predictive value directly; professor unlikely to expect it | — Pending |
| SHAP for feature attribution | More interpretable than permutation importance; rigorous for academic submission | — Pending |
| Master notebook structure | Single notebook tells complete story; easier to grade and evaluate | — Pending |
| Bootstrap Sharpe CI | Addresses overfitting concern from syllabus explicitly | — Pending |

---
*Last updated: 2026-03-18 after initialization*
