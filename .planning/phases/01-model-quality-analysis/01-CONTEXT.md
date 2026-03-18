# Phase 1: Model Quality Analysis - Context

**Gathered:** 2026-03-18
**Status:** Ready for planning
**Source:** User-provided session context

<domain>
## Phase Boundary

Phase 1 delivers IC/ICIR analysis and SHAP feature attribution for the best predictive model. The existing ML pipeline (join_prediction.py, backtester.py, portfolio_construction.py) is the input — this phase adds the model quality layer on top. Outputs are importable functions or saved artifacts consumed by Phase 3.

</domain>

<decisions>
## Implementation Decisions

### Model Quality Metrics
- IC (Information Coefficient) = rank correlation between predicted probability and realized 21-day forward return — industry-standard, professor unlikely to expect it (differentiator)
- ICIR = IC / std(IC) — signal quality metric showing consistency
- IC decay across 1d, 5d, 21d, 63d horizons — shows how quickly the signal fades
- SHAP chosen over permutation importance — more rigorous for academic submission

### Model Comparison Table
- Must include: AUC, Brier score, OOS accuracy, portfolio Sharpe
- All four model types must be present

### Known Bugs in Existing Codebase (from previous session — relevant for correctness)
- **Sharpe ratio bug**: `performance_metrics.py` uses total returns instead of excess returns in denominator — must use excess returns over risk-free rate
- **Class imbalance bug**: ~0.3% base rate for S&P 500 events is unaddressed in `join_prediction.py` — requires `class_weight='balanced'` and `scale_pos_weight` parameters
- **Perfect foresight leakage**: join/leave prediction labels in `join_prediction.py` may use future information — label logic must use announcement date, not effective date

### Novel Academic Contributions (from previous session — may surface in SHAP Phase)
- Amihud illiquidity ratio predicting price impact
- 52-week high proximity for psychological anchoring
- Idiosyncratic volatility connecting to inattention theory
- Short-term reversal signal
- Buffer zone proximity to rank 500 threshold
- λ-inattention model from Bouchaud applied to event study CARs

### Claude's Discretion
- File structure: whether to implement as a new module (src/evaluation/model_quality.py) or extend existing files
- Whether to run model quality analysis in a new notebook cell or as standalone script
- Output format: saved CSV/pickle artifacts vs. in-memory results

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Existing Implementation
- `src/models/join_prediction.py` — Ensemble modeling with rolling window validation (the model to evaluate)
- `src/evaluation/performance_metrics.py` — Existing performance metrics (Sharpe bug lives here)
- `src/evaluation/factor_analysis.py` — Four-factor alpha attribution
- `src/portfolio/portfolio_construction.py` — Long-short portfolio construction
- `src/backtesting/backtester.py` — Backtesting with transaction costs

### Notebooks
- `notebooks/01_event_study.ipynb` — Event study methodology
- `notebooks/02_feature_exploration.ipynb` — Feature engineering context

### Planning Artifacts
- `.planning/REQUIREMENTS.md` — MODEL-01 through MODEL-05 definitions
- `.planning/ROADMAP.md` — Phase 1 goal and success criteria

</canonical_refs>

<specifics>
## Specific Ideas

### IC Computation
- Use scipy.stats.spearmanr or pandas rank correlation
- Compute per model type (logistic, random forest, gradient boosting, etc.)
- Realized return = 21-day forward return from announcement date (not effective date — avoid look-ahead)

### IC Decay
- Compute IC at 1d, 5d, 21d, 63d forward returns
- Plot as line chart: x = horizon, y = IC value
- Apply to best model only (highest ICIR)

### SHAP
- Use `shap` library: TreeExplainer for tree models, LinearExplainer for logistic
- Bar chart: mean(|SHAP|) per feature, sorted descending
- Save SHAP values for Phase 3 notebook import

### Deadline Constraint
- Deadline end of current week (March 2026) — prioritize MODEL-01 through MODEL-05 as atomic deliverables

</specifics>

<deferred>
## Deferred Ideas

- λ-inattention model fitting to event study CARs (complex, may be Phase 3 academic enrichment)
- PEAD features (requires IBES data, out of scope)
- Novel features (Amihud, 52-week high, idiovol, reversal) — Phase 2 scope if time permits

</deferred>

---

*Phase: 01-model-quality-analysis*
*Context gathered: 2026-03-18*
