# Phase 1: Model Quality Analysis - Research

**Researched:** 2026-03-18
**Domain:** Quantitative finance model evaluation — IC/ICIR, SHAP, OOS model comparison
**Confidence:** HIGH (codebase fully read; all libraries verified against installed versions)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- IC (Information Coefficient) = rank correlation between predicted probability and realized 21-day forward return
- ICIR = IC / std(IC) — signal quality metric showing consistency
- IC decay across 1d, 5d, 21d, 63d horizons — shows how quickly the signal fades
- SHAP chosen over permutation importance — more rigorous for academic submission
- Model comparison table must include: AUC, Brier score, OOS accuracy, portfolio Sharpe
- All four model types must be present in the comparison table
- Known bugs must be fixed:
  - Sharpe ratio bug: `performance_metrics.py` uses total returns instead of excess returns in denominator
  - Class imbalance bug: ~0.3% base rate unaddressed — requires `class_weight='balanced'` and `scale_pos_weight`
  - Perfect foresight leakage: label logic must use announcement date, not effective date

### Claude's Discretion

- File structure: whether to implement as a new module (`src/evaluation/model_quality.py`) or extend existing files
- Whether to run model quality analysis in a new notebook cell or as standalone script
- Output format: saved CSV/pickle artifacts vs. in-memory results

### Deferred Ideas (OUT OF SCOPE)

- lambda-inattention model fitting to event study CARs
- PEAD features (requires IBES data, out of scope)
- Novel features (Amihud, 52-week high, idiovol, reversal) — Phase 2 scope if time permits
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| MODEL-01 | IC computed per model: rank correlation between predicted join/leave probability and realized 21-day forward return | `scipy.stats.spearmanr` or `pd.Series.rank()` + Pearson correlation; realized 21-day return already exists as `ret_21d` in features parquet |
| MODEL-02 | IC decay curve by holding horizon (1d, 5d, 21d, 63d) for best model | `ret_21d`, `ret_63d` already in features parquet; need `ret_1d` and `ret_5d` to be computed or joined from raw returns; IC at each horizon via Spearman rank correlation |
| MODEL-03 | ICIR (IC / std(IC)) per model — industry-standard signal quality metric | Computed from per-period IC series produced during MODEL-01; std(IC) over rolling OOS folds |
| MODEL-04 | SHAP feature importance for best model | `shap` library not currently installed (confirmed); must install; TreeExplainer for RF/GBM/XGB, LinearExplainer for logistic |
| MODEL-05 | Model comparison table: AUC, Brier score, OOS accuracy, portfolio Sharpe per model type | AUC already computed in `model_utils.train_and_evaluate`; Brier score and OOS accuracy missing; portfolio Sharpe requires backtester integration; Sharpe bug fix required |
</phase_requirements>

---

## Summary

Phase 1 adds a model quality analysis layer on top of the existing ML pipeline. The existing infrastructure in `src/models/join_prediction.py` produces predicted probabilities via rolling-window cross-validation, and `data/processed/features_join.parquet` / `features_leave.parquet` hold the full feature panel. However, **`label_join` is entirely zeros** in the current parquet — the pipeline ran but produced no positive labels, indicating the raw feature panel was built from a price file that was not joined with the events Excel. This is the single largest blocker: before IC/SHAP can be computed, the models must be trained on data with non-zero labels.

The `shap` library is not installed. All other required libraries (scipy, pandas, scikit-learn, matplotlib, statsmodels, numpy) are present. The Sharpe computation in `performance_metrics.py` has a confirmed denominator bug: `excess.mean() / ret.std()` divides excess returns by total-return std rather than excess-return std. The class imbalance fix (`class_weight='balanced'`) is absent from `join_prediction.py`.

**Primary recommendation:** Phase 1 must (1) regenerate feature parquets with real positive labels, (2) fix the three known bugs, (3) train models and produce OOS scores, (4) implement `src/evaluation/model_quality.py` with IC/ICIR/decay/SHAP functions, and (5) save artifacts for Phase 3 consumption.

---

## Codebase Reality Check

### Critical Finding: label_join is All Zeros

```
features_join.parquet shape: (19,776,423, 20)
label_join unique values: [0]   <-- ALL zeros, no positive examples
```

`features_leave.parquet` correctly has 202 positive labels out of 19,776,423 rows (0.001% base rate). The join labels being entirely zero means either:
- `build_feature_panel` was run without an `is_sp500` column populated from the events file
- The ticker/PERMNO bridge join between events Excel and CRSP daily.csv produced no matches

**Consequence:** `run_join_prediction` currently produces no meaningful model (every fold skips due to `y_train.nunique() < 2`), and `join_scores.parquet` is empty. The pipeline code is structurally sound but has not been successfully end-to-end executed with real data.

### Confirmed Sharpe Bug

In `performance_metrics.py` line 30:
```python
sharpe = (excess.mean() / ret.std() * np.sqrt(periods_per_year))
```
`ret.std()` should be `excess.std()` — this overstates Sharpe when the risk-free rate is non-trivial.

### Confirmed Missing: Brier Score and OOS Accuracy

`train_and_evaluate` in `model_utils.py` computes: `roc_auc`, `precision`, `recall`, `f1`. It does NOT compute:
- Brier score (required by MODEL-05)
- OOS accuracy (required by MODEL-05)

### Confirmed Missing: class_weight

`_get_model` in `join_prediction.py` does not pass `class_weight='balanced'` to LogisticRegression or RandomForestClassifier, and does not pass `scale_pos_weight` to XGBoost.

### XGBoost Broken

XGBoost is installed but non-functional: `libomp.dylib` missing. Fix: `brew install libomp`. This is a runtime dependency issue, not a code issue.

### shap Not Installed

`shap` is not installed in the current Python environment. Must install before MODEL-04 can run.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| scipy | 1.17.1 (installed) | `spearmanr` for IC computation | Standard rank correlation; no dependencies beyond numpy |
| pandas | 2.3.3 (installed) | Data manipulation, IC per fold | Already project dependency |
| scikit-learn | 1.8.0 (installed) | `brier_score_loss`, `accuracy_score` — missing metrics | Already project dependency |
| shap | latest (NOT installed) | SHAP values for tree + linear models | Locked decision; most rigorous feature attribution |
| matplotlib | 3.10.8 (installed) | IC decay line chart, SHAP bar chart | Already project dependency |
| numpy | 2.4.3 (installed) | Numerical computation | Already project dependency |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| statsmodels | 0.14.6 (installed) | Sharpe t-test if needed | Factor regression already uses it |

### Installation Required
```bash
pip install shap
brew install libomp  # fix xgboost on macOS
```

---

## Architecture Patterns

### Recommended Module Structure

```
src/evaluation/
├── performance_metrics.py    # exists — fix Sharpe bug here
├── factor_analysis.py        # exists — no changes needed in Phase 1
└── model_quality.py          # NEW — IC, ICIR, decay, SHAP, comparison table
```

The discretion decision is: **create `src/evaluation/model_quality.py`**. Extending existing files risks breaking existing interfaces. A new module keeps concerns separated and is cleanly importable by Phase 3.

### Pattern 1: IC Computation Per Model Per Fold

**What:** For each OOS fold, compute Spearman rank correlation between predicted probability and realized forward return. Aggregate IC time series across folds, then compute mean IC and ICIR.

**When to use:** After `run_join_prediction` produces `join_scores.parquet` with OOS probabilities.

**Example:**
```python
# Source: scipy.stats.spearmanr documentation
from scipy.stats import spearmanr
import pandas as pd

def compute_ic(scores: pd.DataFrame, returns: pd.DataFrame,
               prob_col: str, ret_col: str = "ret_21d",
               date_col: str = "date") -> pd.Series:
    """IC per period: Spearman rank corr(predicted_prob, realized_return)."""
    merged = scores.merge(returns[["date", "permno", ret_col]],
                          on=["date", "permno"], how="inner")
    ic_by_date = (
        merged.groupby(date_col)
        .apply(lambda g: spearmanr(g[prob_col], g[ret_col])[0]
               if len(g) >= 5 else float("nan"))
    )
    return ic_by_date.dropna()
```

### Pattern 2: ICIR Computation

**What:** ICIR = mean(IC) / std(IC) across the IC time series. Higher ICIR means more consistent signal quality.

```python
def compute_icir(ic_series: pd.Series) -> float:
    """ICIR = mean(IC) / std(IC). Returns nan if std is 0."""
    return float(ic_series.mean() / ic_series.std()) if ic_series.std() > 0 else float("nan")
```

### Pattern 3: IC Decay Across Horizons

**What:** For the best model (highest ICIR), compute IC at 1d, 5d, 21d, 63d forward return horizons.

**Key issue:** `features_join.parquet` currently has `ret_21d`, `ret_63d`, `ret_126d`, `ret_252d` but NOT `ret_1d` or `ret_5d`. These need to be added when regenerating the feature panel, or computed from the raw panel separately.

```python
horizons = [1, 5, 21, 63]
ret_cols = {1: "ret_1d", 5: "ret_5d", 21: "ret_21d", 63: "ret_63d"}
ic_decay = {h: compute_ic(scores, returns, prob_col, ret_col=ret_cols[h]).mean()
            for h in horizons}
```

### Pattern 4: SHAP for Tree vs Linear Models

**What:** Use `shap.TreeExplainer` for RandomForest, GradientBoosting, XGBoost. Use `shap.LinearExplainer` for LogisticRegression. Mean absolute SHAP value per feature = feature importance.

```python
import shap

def compute_shap(model, X_sample: pd.DataFrame) -> pd.Series:
    """Mean |SHAP| per feature, sorted descending."""
    model_type = type(model).__name__
    if model_type in ("RandomForestClassifier", "GradientBoostingClassifier", "XGBClassifier"):
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.LinearExplainer(model, X_sample)
    shap_values = explainer.shap_values(X_sample)
    # For binary classifiers, shap_values may be list[array]; take class-1
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    importance = pd.Series(
        abs(shap_values).mean(axis=0),
        index=X_sample.columns
    ).sort_values(ascending=False)
    return importance
```

**SHAP sample size:** TreeExplainer is slow on large datasets. Subsample to 1000-5000 rows from the OOS test set.

### Pattern 5: Model Comparison Table

**What:** Aggregate per-fold metrics across all models. Must include: AUC, Brier score, OOS accuracy, portfolio Sharpe.

**Missing metrics to add to `model_utils.train_and_evaluate`:**
```python
from sklearn.metrics import brier_score_loss, accuracy_score

res["brier_score"] = brier_score_loss(y_test, proba)
res["oos_accuracy"] = accuracy_score(y_test, pred)
```

**Portfolio Sharpe** requires running the full backtester per model and extracting the Sharpe ratio — this is the most expensive step. The fixed Sharpe formula is:
```python
# Fix: use excess.std() not ret.std()
sharpe = (excess.mean() / excess.std() * np.sqrt(periods_per_year))
```

### Anti-Patterns to Avoid

- **Computing IC on the full in-sample set:** IC must be computed only on OOS predictions. The `join_scores.parquet` produced by the rolling-window pipeline is already OOS — do not re-merge with training data.
- **Using `ret_21d` from the features file as the realized return without verification:** The `ret_21d` feature is a lagged 21-day rolling return (look-back), not a forward return. IC requires forward returns. Forward returns must be computed separately or confirmed to be truly forward-looking.
- **Running SHAP on all 19M rows:** Always subsample to a representative OOS test partition.
- **Calling `shap_values[0]` for binary classifier:** TreeExplainer returns a list of 2 arrays for binary classification. Use index `[1]` for the positive class.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Rank correlation | Custom Spearman loop | `scipy.stats.spearmanr` | Handles ties correctly, tested, fast |
| SHAP values | Manual feature attribution | `shap.TreeExplainer` / `LinearExplainer` | Accounts for feature interactions; TreeExplainer is exact for tree models |
| Brier score | `mean((p - y)^2)` by hand | `sklearn.metrics.brier_score_loss` | Edge cases handled |
| Feature importance bar chart | Custom matplotlib loop | `shap.summary_plot` or `pd.Series.plot(kind='barh')` | Both are fine; `pd.Series.plot` is simpler for sorted bar chart |

---

## Common Pitfalls

### Pitfall 1: Forward Return vs Look-Back Return Confusion

**What goes wrong:** `ret_21d` in the features parquet is computed as a look-back rolling window (past 21 days of cumulative return). Using this as the "realized return" for IC gives a spurious correlation between past momentum and current prediction — not the IC we want.

**Why it happens:** The feature engineering function uses `shift(1)` on rolling returns for no-lookahead features, but the column name `ret_21d` does not signal direction.

**How to avoid:** Compute forward returns explicitly:
```python
# forward_ret_21d = return from date t to t+21
panel["fwd_ret_21d"] = panel.groupby("permno")["ret"].transform(
    lambda x: x.rolling(21).apply(lambda y: (1+y).prod()-1).shift(-21)
)
```
Or verify by checking sign correlation with the event date — stocks joining S&P 500 should have positive forward returns if the model is working.

**Warning signs:** Mean IC very close to 0 or negative for all models.

### Pitfall 2: IC Computed on Empty or Near-Empty Groups

**What goes wrong:** If a date has fewer than 5 stocks with OOS predictions, `spearmanr` returns NaN or undefined correlation.

**How to avoid:** Apply minimum group size filter (`if len(g) >= 5`) before computing per-date IC.

### Pitfall 3: SHAP with Stale/Refitted Model

**What goes wrong:** SHAP must be applied to the actual fitted model object, not a re-initialized one. The rolling-window loop in `join_prediction.py` overwrites `model` in each fold — the last fold's model is what remains.

**How to avoid:** Save the best-fold model object explicitly, or re-fit on the full OOS partition using the best model configuration. Best practice: save model after last fold or identify the best fold by highest AUC and save that model.

### Pitfall 4: Class Imbalance Skewing Metrics

**What goes wrong:** With 202 positives out of 19.7M rows (0.001% base rate), OOS accuracy will be ~99.999% even for a trivially all-zero classifier. Brier score will also be near zero.

**How to avoid:** Add `class_weight='balanced'` to sklearn models. Report metrics that are meaningful under imbalance: AUC-ROC, Brier score calibrated against no-skill baseline, precision@k. The IC metric is inherently rank-based so is less sensitive to class imbalance.

### Pitfall 5: label_join All-Zero Requires Pipeline Regeneration

**What goes wrong:** The current `features_join.parquet` has `label_join = 0` for all rows. This means models cannot train (fold loop exits with `y_train.nunique() < 2`).

**How to avoid:** The data pipeline must be re-run with an `is_sp500` column that is populated from the events Excel file. The `load_events` function and `build_ticker_permno_bridge` exist — the preprocess step must join them to the price panel before calling `build_feature_panel`.

**Root cause hypothesis:** `preprocess_data.py` likely builds the price panel without executing the events join, or the ticker normalization produces no matches between the Excel file's ticker format and CRSP's PERMNO-Ticker mapping.

### Pitfall 6: XGBoost Requires libomp on macOS

**What goes wrong:** `import xgboost` raises `XGBoostError: Library not loaded: @rpath/libomp.dylib`.

**How to avoid:** `brew install libomp` before running models. This is a one-line fix but must happen before any XGBoost-dependent code runs.

---

## Code Examples

### IC and ICIR (Complete)

```python
# Source: scipy documentation + quant finance convention
from scipy.stats import spearmanr
import pandas as pd
import numpy as np

def compute_ic_series(scores_df: pd.DataFrame,
                      fwd_returns_df: pd.DataFrame,
                      prob_col: str,
                      fwd_ret_col: str = "fwd_ret_21d",
                      date_col: str = "date") -> pd.Series:
    """Per-date IC: Spearman(predicted_prob, forward_return). OOS only."""
    merged = scores_df.merge(
        fwd_returns_df[["date", "permno", fwd_ret_col]],
        on=["date", "permno"], how="inner"
    )
    ic = (
        merged.dropna(subset=[prob_col, fwd_ret_col])
        .groupby(date_col)
        .apply(lambda g: spearmanr(g[prob_col], g[fwd_ret_col])[0]
               if len(g) >= 5 else np.nan)
    )
    return ic.dropna().rename("ic")

def compute_icir(ic_series: pd.Series) -> float:
    return float(ic_series.mean() / ic_series.std()) if ic_series.std() > 0 else np.nan
```

### Sharpe Bug Fix

```python
# Fix in performance_metrics.py line 30:
# BEFORE (bug): sharpe = (excess.mean() / ret.std() * np.sqrt(periods_per_year))
# AFTER (fix):
sharpe = (excess.mean() / excess.std() * np.sqrt(periods_per_year)) if excess.std() > 0 else np.nan
```

### Adding Missing Metrics to model_utils.py

```python
# Add to train_and_evaluate after existing metrics:
from sklearn.metrics import brier_score_loss, accuracy_score

res["brier_score"] = brier_score_loss(y_test, proba)
res["oos_accuracy"] = float(accuracy_score(y_test, pred))
```

### Brier Score No-Skill Baseline

The no-skill Brier score baseline is `base_rate * (1 - base_rate)`. For 0.001% base rate, this equals ~0.001. Any model Brier score near this is essentially uninformative.

### IC Decay Plot

```python
import matplotlib.pyplot as plt

def plot_ic_decay(ic_decay: dict[int, float], model_name: str) -> plt.Figure:
    """ic_decay = {1: 0.02, 5: 0.03, 21: 0.05, 63: 0.04}"""
    horizons = sorted(ic_decay)
    values = [ic_decay[h] for h in horizons]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(horizons, values, marker="o", linewidth=2)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Holding Horizon (trading days)")
    ax.set_ylabel("IC (Spearman rank correlation)")
    ax.set_title(f"IC Decay — {model_name}")
    ax.set_xticks(horizons)
    ax.set_xticklabels(["1d", "5d", "21d", "63d"])
    return fig
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Permutation importance | SHAP TreeExplainer | ~2018 (shap paper) | Accounts for feature interactions; consistent sign |
| Accuracy / F1 | AUC-ROC + Brier score + IC | Academic standard | Better for imbalanced classification and ranking |
| In-sample evaluation | Rolling-window OOS | Standard in quant research | Prevents look-ahead bias |

---

## Open Questions

1. **Forward return availability**
   - What we know: `ret_21d` in features parquet is a look-back return; `ret_63d`, `ret_126d`, `ret_252d` same.
   - What's unclear: Whether forward returns need to be recomputed from scratch or if any existing pipeline step computes them.
   - Recommendation: Add `fwd_ret_1d`, `fwd_ret_5d`, `fwd_ret_21d`, `fwd_ret_63d` columns when regenerating the feature panel. Use `.shift(-N)` on cumulative product.

2. **Pipeline regeneration scope**
   - What we know: `label_join` is all zeros; `label_leave` has 202 positives.
   - What's unclear: Whether the preprocess step needs a full re-run from raw daily.csv (slow, ~20M rows) or whether a targeted join of events to the existing panel can be done.
   - Recommendation: Read `src/data/preprocess_data.py` before planning to understand whether it populates `is_sp500` from events data.

3. **Portfolio Sharpe source for MODEL-05**
   - What we know: MODEL-05 requires "portfolio Sharpe per model type" in the comparison table.
   - What's unclear: Whether this means running the full backtester for each model (expensive) or a simplified Sharpe from the top-decile predictions.
   - Recommendation: Use simplified approach: long top-decile predicted probability stocks, compute mean next-period return, annualize Sharpe. Full backtester is Phase 2 scope.

4. **Best model selection criterion**
   - What we know: CONTEXT.md says "best model = highest ICIR."
   - What's unclear: ICIR will be undefined if IC series has all NaN (from zero labels problem).
   - Recommendation: Fall back to AUC if ICIR undefined; document selection criterion explicitly.

---

## Data Flow for This Phase

```
data/raw/daily.csv                     [exists, ~20M rows CRSP]
data/raw/SPX_index leavers & joiners_17-Feb-2026.xlsx   [exists]
         |
         v
src/data/preprocess_data.py            [MUST populate is_sp500 from events]
         |
         v
src/features/feature_engineering.py   [add fwd_ret_1d, fwd_ret_5d columns]
         |
         v
data/processed/features_join.parquet  [regenerate with real label_join]
         |
         v
src/models/join_prediction.py         [fix class_weight; add Brier/accuracy]
         |
         v
data/processed/join_scores.parquet    [OOS probabilities per date/permno/model]
         |
         v
src/evaluation/model_quality.py       [NEW: IC, ICIR, decay, SHAP, comparison]
         |
         v
results/tables/model_comparison.csv   [MODEL-05 artifact]
results/figures/ic_decay.png          [MODEL-02 artifact]
results/figures/shap_importance.png   [MODEL-04 artifact]
```

---

## Sources

### Primary (HIGH confidence)
- Source code inspection: `src/models/join_prediction.py`, `src/models/model_utils.py`, `src/evaluation/performance_metrics.py`, `src/features/feature_engineering.py`, `src/data/load_data.py`
- Runtime verification: `python3` version checks for scipy 1.17.1, pandas 2.3.3, sklearn 1.8.0, matplotlib 3.10.8, numpy 2.4.3
- Data inspection: `features_join.parquet` (19.7M rows, label_join all zeros), `features_leave.parquet` (202 positives)

### Secondary (MEDIUM confidence)
- SHAP library API patterns: based on shap 0.4x documentation conventions (TreeExplainer/LinearExplainer interface)
- IC/ICIR formulas: standard quant finance definition; Grinold & Kahn "Active Portfolio Management" convention

### Tertiary (LOW confidence)
- XGBoost libomp fix via `brew install libomp`: based on common macOS XGBoost troubleshooting pattern

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — installed versions verified directly
- Architecture: HIGH — existing module interfaces read directly; new module placement is idiomatic
- Pitfalls: HIGH — all-zero label_join confirmed by direct data inspection; Sharpe bug confirmed by reading line 30 of performance_metrics.py; missing metrics confirmed by reading model_utils.py
- IC/ICIR patterns: HIGH — scipy.stats.spearmanr is the canonical approach, no exotic API
- SHAP patterns: MEDIUM — shap not installed, so API verified from documentation patterns only

**Research date:** 2026-03-18
**Valid until:** 2026-04-18 (stable libraries)
