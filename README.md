# Forecasting S&P 500 Stock Volatility

**INSY 674 · Enterprise Data Science · McGill Desautels MMA · Fall 2026**

Predicting how much an individual S&P 500 stock's price may fluctuate over the next five trading days, and examining whether those forecasts help size positions more consistently.

## Project objective and framing

The business objective is to support investors and risk managers who size positions against a fixed risk budget. A higher predicted volatility leads to a smaller position. The research question is whether machine-learning forecasts improve on historical forecasts and the HAR benchmark.

This is a **supervised regression**: each stock-date has historical features and a numeric future-variance label. Models are trained offline in batches and evaluated chronologically. The target concerns individual stocks that are index members on the forecast date, rather than the S&P 500 index itself. 

The five-day target is:

```text
y_var = r²(t+1) + r²(t+2) + ... + r²(t+5)
r(t) = log(1 + daily return)
Annualized volatility (%) = sqrt((252 / 5) × y_var) × 100
```

All five future returns must be available. Learned models fit `z = log(y_var + 1e-8)`. Forecasts assume access to information available at the close of day t.

## Files and execution

The repository screenshot identifies `INSY_modeling.ipynb` as the updated modeling notebook. The acquisition and feature-preparation notebooks supply its upstream data.

To reproduce the modeling workflow:

1. Open the main modeling notebook in Google Colab and mount Google Drive.
2. Provide `sp500_panel_features.parquet`, produced by the upstream acquisition and preparation workflow. The default Drive location is `/content/drive/MyDrive/sp500_vol/sp500_panel_features.parquet`; the notebook also checks its existing alternative paths.
3. Use an environment containing the existing dependencies: NumPy, pandas, Matplotlib, SciPy, scikit-learn, XGBoost, and a pandas-compatible parquet engine. Google Colab supplies the Drive integration. Exact package versions are not pinned in the supplied notebook.
4. Run cells from top to bottom. Reloading VIX requires access to the existing FRED source or fallback CSV. Exploratory figures are saved to the configured Drive figure folder.

The results below are from the supplied notebook's saved outputs. They were not independently rerun for this README because the underlying parquet file was not supplied. WRDS/CRSP source access and upstream extraction are separate prerequisites for rebuilding the panel from scratch.

## Notebook walkthrough

The following sections follow all eight numbered sections of the main notebook. The setup and code-block subheadings identify the steps within each section; wording varies slightly between the original and annotated copies.

### Setup: Connect to Google Drive

Connect Colab to the Drive location containing the project data.

### Setup: Load libraries and set project constants

Load the existing imports, locate the parquet file, and set the five-day horizon, annualization factor, split years, and random seed (`69`).

## 1. Data audit

### Audit the input panel

The panel contains **1,039,122 stock-date rows**, **637 stock identifiers**, and **1,760 trading dates**, covering 2019–2025. The presentation identifies WRDS/CRSP as the stock-data source and FRED as the VIX source. Histories include observations outside a stock's index-membership period so earlier returns can support feature construction.

| Attribute | Type / meaning | Role and audit finding |
|---|---|---|
| `permno` | Integer stock identifier | Groups each stock's history |
| `date` | Trading date | Orders observations and defines time splits |
| `return` | Numeric daily simple return | Target inputs and market aggregation; 306 missing values, about 0.029% of panel rows |
| `price` | Numeric stock price | Included in extreme-return inspection |
| `volumn` | Numeric trading volume; source spelling retained | Builds log and relative-volume features |
| `log_return`, `r2` | Numeric log return and its square | Builds the target and historical variance features |
| `future_r2_sum`, `full_r2` | Upstream target and completeness flag | Checks consistency of the recomputed target |
| `isin_sp500`, `in_index` | Membership flags | Restricts forecast-date eligibility |
| `VIXCLS` | Upstream VIX values | Only 2,975 non-missing rows and one distinct value; replaced for modeling |

There are **zero duplicate stock-date rows**, **zero internal trading-date gaps**, and **zero detected five-zero-return windows**. These checks do not prove that every price or extreme return is correct. Missing percentages for every input attribute are not reported by the saved audit.

### Expand the membership flag

The existing code extends each flagged membership run by one stock observation on each side. It adds **739 rows** and produces daily membership counts of **501 / 503 / 507** (minimum / median / maximum). This approximates inclusive membership boundaries; original membership intervals were not available for independent verification.

### Reload and align VIX

VIX is reloaded, indexed by date, aligned to the trading calendar, and forward-filled for at most three dates. The saved FRED run has **zero missing aligned dates** and a changing VIX series, replacing the unusable constant/missing column.

## 2. Features

### Build the target and 18 features

The recomputed target matches the upstream target on the complete, non-missing rows compared. Rolling features are calculated separately for each stock using information through day t, including available history before index entry.

| Group | Features | Purpose |
|---|---|---|
| Historical variance | `rv_d`, `rv_w`, `rv_m`, `rv_q` | Capture persistence over 1, 5, 22, and 66 days |
| Downside variance | `rvneg_w`, `rvneg_m` | Summarize squared negative returns over 5 and 22 days |
| Returns | `ret_d`, `ret_lag1`, `ret_w`, `ret_m` | Capture current, lagged, weekly, and monthly returns |
| Smoothed variance | `ewma` | Weight recent squared returns more heavily; decay factor 0.94 |
| Volume | `log_vol`, `rel_vol` | Describe activity level and activity relative to the 22-day mean |
| Market | `mkt_ret`, `mkt_rv_w`, `mkt_rv_m`, `vix`, `vix_chg_w` | Describe member-market returns, variance, VIX, and its change |

Variance features use logs with a small offset. Volume uses `log1p` and a relative-volume ratio. Ridge standardization is fitted within its training pipeline. Complete-window requirements and the existing EWMA warm-up determine feature availability.

## 3. Sample, split, and purge & EDA

### Filter rows and define time splits

Keep eligible member rows from 2020 onward with complete features and targets. **3,763 member rows are excluded** for incomplete features or targets.

| Sample | Eligible dates | Rows | Stocks |
|---|---|---:|---:|
| Train | 2020-01-02 to 2023-12-21 | 503,898 | 571 |
| Validation | 2024-01-02 to 2024-12-31 | 126,476 | 519 |
| Refit | 2020-01-02 to 2024-12-23 | 630,369 | 588 |
| Test | 2025-01-02 to 2025-12-23 | 123,059 | 521 |

Refit overlaps the earlier train and validation periods. Training and refit drop their last five trading dates to prevent their targets extending into the next period. **Validation is not similarly purged:** its final targets can contain early-2025 returns, affecting model selection and calibration. The reported test results therefore have this boundary limitation.

### Prepare the exploratory charts

Construct a plotting table and express realized volatility as annualized percent. EDA uses the full eligible sample, including 2025; it is descriptive rather than training-only analysis.

### Compare three target scales

Raw variance has skewness **227.3**, annualized volatility **5.0**, and log variance **0.2**. The log scale compresses the long tail. Histogram values are capped at the 99.5th percentile for display only; model-training observations are not clipped by this chart.

### Compare past volatility with future variance

Density plots and binned averages show that higher past variance is associated with higher future variance. The relationship is easier to see on logs, while the spread shows substantial uncertainty about individual outcomes.

### Inspect feature correlations

Spearman correlations with the log target are strongest for **EWMA (0.52)**, **66-day variance (0.51)**, and **22-day variance (0.50)**. Return-feature correlations are smaller. EWMA and 22-day variance correlate **0.95**, indicating overlapping information. These are associations, not causal effects.

## 4. Metrics and calibration

### Define metrics, calibration, and baselines

| Metric | Interpretation |
|---|---|
| QLIKE | Variance forecasting loss; lower is better |
| MAE | Average absolute error in annualized volatility percentage points |
| RMSE | Volatility error measure giving extra weight to large misses |
| Bias | Mean forecast minus realized volatility; negative means average underprediction |
| Under % | Percentage of forecasts below realized volatility |

The notebook estimates separate multiplicative adjustments for variance and volatility from 2024 predictions, then keeps them fixed for 2025. These adjustments do not guarantee unbiased forecasts in every subgroup or year. Because the two forecasts are calibrated separately, the volatility forecast need not equal the square root of the variance forecast.

## 5. Stage 1 — fit on 2020–2023, tune on 2024

### Tune the existing models on 2024

| Model | Role / reason for comparison | Existing search |
|---|---|---|
| Naive 5D, Hist 22D, EWMA | Simple historical benchmarks | No hyperparameter search |
| HAR | Three-horizon linear benchmark | Ordinary least squares on log-variance features |
| Ridge | Penalized linear model for correlated predictors | Six penalties from 0.01 to 1,000,000 |
| Random Forest | Nonlinear tree ensemble | Depth 10 or 14; 100 trees; 20% bootstrap samples |
| XGBoost | Sequential tree ensemble | Depth 4, 6, or 8; learning rate 0.05; validation early stopping |

The search compares a small, explicit set of settings using validation log-MSE. The stock-only XGBoost comparison and raw-variance diagnostic reuse full-feature XGBoost's chosen settings. They are not independently tuned.

### Calibrate and select by validation QLIKE

After calibration, **XGBoost without market features wins validation QLIKE at 0.5901**. Random Forest, Ridge, and full-feature XGBoost score 0.5934, 0.5944, and 0.5955. The difference among these candidates is small. HAR scores 0.6658.

The raw-variance XGBoost diagnostic has QLIKE **12,981.1707** and positive volatility bias. Near-zero variance forecasts can generate very large QLIKE when outcomes are positive, even if average forecasts are too high.

**Validation design:** the notebook uses a chronological holdout, not N-fold cross-validation. It does not report fold means or standard deviations. Random Forest and XGBoost are ensembles; no additional cross-model ensemble is implemented. Selection and calibration share the same validation year.

## 6. Stage 2 — refit on 2020–2024, evaluate once on 2025

### Refit and read the 2025 test results

Models are refitted with the selected settings on the purged refit sample. Calibration remains fixed from 2024. The table reports the saved 2025 comparison; the validation-selected model is retained regardless of test ranking.

| Model | Test QLIKE ↓ | Volatility RMSE, pp ↓ |
|---|---:|---:|
| Ridge | 0.635 | 18.605 |
| XGBoost | 0.645 | 19.354 |
| Random Forest | 0.646 | 19.024 |
| **XGBoost without market features — selected** | **0.648** | **18.965** |
| HAR | 0.696 | 19.476 |
| EWMA | 0.771 | 21.245 |
| Hist 22D | 0.851 | 21.988 |
| Naive 5D | 1.760 | 24.641 |

The selected model improves on HAR by **6.891% in QLIKE** and **2.624% in RMSE**. Ridge has the best displayed test scores, but was not the validation winner. These comparisons retain the validation-boundary limitation noted above.

### Compare paired daily losses

Diebold–Mariano-style comparisons average stock-level loss differences by date and use a Newey–West adjustment with lag 5. Selected XGBoost versus HAR has p-values **0.0001 for QLIKE** and **0.0046 for squared volatility error**. The selected-model comparison with Ridge does not detect a difference. Only the listed model pairs are tested; non-significance does not establish equivalence.

### Read errors for high realized volatility

The training 90th-percentile threshold is **55.3% annualized volatility**; **7.5%** of test rows exceed it. Across the listed models, roughly **87%–98%** of these high outcomes are underestimated. The selected model's overall bias is **-0.831 pp**, versus **-45.040 pp** in the high-outcome group. This split uses realized future outcomes and is not known when forecasting.

### Read the four diagnostic charts

- **Monthly QLIKE:** compares loss over time; the selected model generally improves on HAR.
- **Daily median volatility:** shows the large realized spike around April exceeding model forecasts.
- **Forecast-decile calibration:** contrasts forecast and realized averages against an equality line.
- **Feature importance:** EWMA dominates the selected XGBoost, followed by 66-day variance; importance is model-specific rather than causal.

## 7. Risk-budget application

### Translate forecasts into position exposure

The exercise evaluates **24,610 stock decisions across 49 dates**, using every fifth eligible date. Exposure follows:

```text
Exposure = min(1, 0.02 / max(forecast five-day volatility, 0.005))
Realized risk proxy = Exposure × sqrt(y_var)
```

A separate scale factor fitted on 2024 makes average risk-taking more comparable. It does not make 2025 risk levels identical.

| Calibrated method | Mean realized risk | Budget RMSE, pp | Risk above 3% |
|---|---:|---:|---:|
| Random Forest | 2.06% | 1.33 | 13.12% |
| Selected XGBoost | 2.09% | 1.34 | 14.37% |
| EWMA | 2.02% | 1.39 | 14.04% |
| HAR | 2.14% | 1.45 | 16.04% |
| Naive 5D | 2.06% | 1.62 | 16.33% |
| Fully invested | 3.96% | 3.52 | 54.87% |

ML forecasts track the 2% budget somewhat more closely than HAR. The advantage is modest. This is individual-stock risk sizing, not a portfolio-return backtest or a guaranteed risk cap.

## 8. Summary / Presentation takeaways

The validation-selected stock-only XGBoost improves on HAR in the saved 2025 comparison, while Ridge leads the test ranking. Differences among ML models are small. Average calibration improves the overall forecast level, but large shocks remain difficult to predict. Risk-budget results show a modest practical benefit.

The evidence is limited by one validation year, late-validation targets overlapping early test returns, full-sample EDA, approximate membership repair, correlated observations, a noisy five-return target, and historical-data requirements. New index membership does not necessarily imply missing features when earlier stock history is available.

## Appendix — fixes for the upstream notebooks / Scope and provenance

The original notebook discusses inclusive membership dates, setting the VIX date index before alignment, and complete five-day targets. The separate upstream notebooks were not supplied for independent verification or edited as part of this documentation.

The presentation's proposed launch process includes tracking forecasting loss and bias against HAR and updating calibration and model fits as data arrive. These are proposed operations, not an implemented deployment.

## Rubric coverage and reporting limits

| Rubric area | Evidence in the project | Reporting limit |
|---|---|---|
| Frame the problem | Business user, five-day numeric target, supervised regression, offline training, risk-sizing use | Business value is assessed through a risk proxy, not investment returns |
| Explore the data | Panel audit, target distributions, feature-target plots, rank correlations, chart takeaways | Complete attribute-level missingness and noise profiles are not reported |
| Prepare and compare | Feature engineering, log transforms, Ridge scaling, multiple model families, historical and HAR benchmarks | Chronological holdout replaces N-fold CV; validation has the disclosed boundary overlap |
| Fine-tune and evaluate | Explicit parameter searches, tree ensembles, fixed validation selection, refit and test comparison | No CV mean/std or additional blended ensemble is reported |



## Team

Sarah Liu · Shaoze Ying · Pengxin Wu · Francois-Xavier Charbonneau · Renkai Zhu
