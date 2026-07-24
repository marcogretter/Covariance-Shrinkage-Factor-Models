# Covariance Shrinkage and Statistical Factor Models in Python

This repository contains a Python implementation of covariance-matrix
estimation, portfolio optimization, and statistical factor analysis for the
Euro Stoxx 50 equity universe.

The project studies two related problems:

1. reducing estimation noise in covariance matrices;
2. identifying the main common factors driving stock returns.

The analysis includes:

- sample covariance estimation;
- constant-correlation shrinkage;
- single-factor covariance shrinkage;
- minimum-variance portfolio construction;
- mean-variance portfolio construction;
- rolling out-of-sample backtesting;
- covariance-matrix conditioning diagnostics;
- Principal Component Analysis;
- factor turnover and stability analysis;
- sector-based interpretation of statistical factors;
- comparison between shrinkage and PCA-based denoising.

## Project Overview

The case study is framed from the perspective of a junior analyst working in an
Equity Statistical Arbitrage team.

The investment universe consists of Euro Stoxx 50 constituents.

The analysis uses:

- total return indices denominated in EUR;
- simple daily returns;
- monthly rebalancing at the end of each month;
- a rolling two-year estimation window;
- no risk-free rate.

At every rebalance date, the most recent two years of data are used to estimate
expected returns, covariance matrices, portfolio weights, and statistical
factors.

## Objectives

The project has two main objectives.

### Covariance Estimation

The first objective is to reduce the noise and instability of the sample
covariance matrix.

The following estimators are compared:

- sample covariance matrix;
- constant-correlation shrinkage estimator;
- single-factor shrinkage estimator.

Each covariance estimate is used to construct:

- a minimum-variance portfolio;
- a mean-variance portfolio.

### Statistical Factor Analysis

The second objective is to identify the main pervasive sources of equity risk.

Principal Component Analysis is applied to the covariance matrix in order to
study:

- the percentage of variance explained by each component;
- the stability of the components through time;
- the relationship between components and market conditions;
- the economic interpretation of the first factors.

## Data

The project uses two course-provided datasets:

```text
sx5e_underlyings.csv
ticker_details.csv
```

The first dataset contains Euro Stoxx 50 total return indices.

The second dataset contains descriptive information such as:

- ticker;
- company name;
- sector;
- industry classification.

The sector information is used to interpret the first principal components.

## Return Calculation

Simple daily returns are computed from total return indices.

```text
return_t =
    price_t / price_t_minus_1
    - 1
```

The risk-free rate is ignored.

Missing data should be managed consistently before calculating covariance
matrices and portfolio weights.

Possible treatments include:

- retaining only securities with sufficient history;
- aligning all series on common trading dates;
- removing incomplete observations within each estimation window.

The exact procedure should be documented in the code.

## Rolling Estimation Framework

The project uses a rolling two-year window.

At every end-of-month rebalance date:

1. select the previous two years of daily returns;
2. estimate expected returns;
3. estimate the covariance matrices;
4. construct portfolio weights;
5. hold the portfolio until the next rebalance;
6. record out-of-sample returns;
7. move the estimation window forward.

A typical approximation is:

```text
estimation_window = 504 trading days
rebalancing_frequency = monthly
```

The exact number of observations may vary because of holidays and missing data.

## Sample Covariance Matrix

The sample covariance matrix is estimated directly from historical returns.

```text
sample_covariance =
    covariance_matrix(returns_window)
```

The sample covariance matrix is unbiased under standard assumptions, but it may
be noisy and unstable when:

- the number of assets is large relative to the number of observations;
- returns are highly correlated;
- the estimation window is short;
- the true covariance structure changes over time.

Noise in the covariance matrix can generate unstable portfolio weights and
large out-of-sample errors.

## Why Covariance Estimation Matters

Portfolio optimization depends heavily on the inverse of the covariance matrix.

Small estimation errors in the covariance matrix can therefore produce large
changes in portfolio weights.

This is especially relevant for the mean-variance portfolio because the
optimization also depends on estimated expected returns, which are themselves
very noisy.

A poorly conditioned covariance matrix may produce:

- extreme long and short positions;
- high gross exposure;
- unstable weights;
- high turnover;
- poor out-of-sample performance;
- excessive sensitivity to small data changes.

## Condition Number

The condition number measures how close a matrix is to being numerically
singular.

```text
condition_number =
    largest_eigenvalue
    / smallest_eigenvalue
```

A large condition number indicates that the covariance matrix is difficult to
invert reliably.

The project compares the condition numbers of:

- the sample covariance matrix;
- the constant-correlation shrunk matrix;
- the single-factor shrunk matrix.

A successful shrinkage estimator should generally improve numerical
conditioning.

## Shrinkage Estimation

Shrinkage combines the sample covariance matrix with a more structured target.

The general form is:

```text
shrunk_covariance =
    shrinkage_intensity * target_matrix
    + (1 - shrinkage_intensity) * sample_covariance
```

The shrinkage intensity lies between zero and one.

```text
0 <= shrinkage_intensity <= 1
```

Interpretation:

```text
shrinkage_intensity = 0
```

uses only the sample covariance matrix.

```text
shrinkage_intensity = 1
```

uses only the structured target.

Intermediate values combine the flexibility of the sample estimate with the
stability of the target.

## Constant-Correlation Shrinkage

The constant-correlation target assumes that all pairwise correlations are equal
to the average sample correlation.

Individual asset volatilities are preserved.

Let:

```text
average_correlation =
    average of all off-diagonal sample correlations
```

The target covariance between assets `i` and `j` is:

```text
target_covariance_ij =
    average_correlation
    * volatility_i
    * volatility_j
```

For diagonal elements:

```text
target_covariance_ii =
    variance_i
```

The target simplifies the dependence structure while retaining heterogeneous
asset volatilities.

## Rationale of the Constant-Correlation Target

The constant-correlation model assumes that the cross-sectional dependence
structure is broadly homogeneous.

Its main advantages are:

- simple interpretation;
- improved numerical stability;
- preservation of individual volatilities;
- reduction of noise in pairwise correlations.

Its main limitation is that it ignores:

- sector clusters;
- heterogeneous correlation structures;
- different exposures to market factors.

## Single-Factor Shrinkage

The single-factor target assumes that stock returns are driven by one common
market factor plus idiosyncratic noise.

A generic model is:

```text
asset_return_i =
    alpha_i
    + beta_i * market_return
    + residual_i
```

The market proxy is constructed from the equity universe, for example as an
equally weighted portfolio.

The target covariance is:

```text
target_covariance_ij =
    beta_i
    * beta_j
    * market_variance
```

For diagonal elements, idiosyncratic variance is also included:

```text
target_variance_i =
    beta_i^2 * market_variance
    + residual_variance_i
```

The single-factor target preserves heterogeneous market exposures while
eliminating noisy residual cross-correlations.

## Rationale of the Single-Factor Target

The single-factor target reflects the idea that a large part of equity
co-movement is driven by market-wide risk.

Its main advantages are:

- economically interpretable dependence structure;
- heterogeneous market betas;
- reduced estimation noise;
- improved covariance-matrix conditioning.

Its limitations include:

- reliance on a single market factor;
- inability to represent sector-specific co-movement;
- sensitivity to the selected market proxy;
- omission of additional systematic factors.

## Shrinkage Intensity

The relative shrinkage intensity is estimated for each rolling window.

The intensity reflects the trade-off between:

- sampling error in the covariance matrix;
- misspecification error in the shrinkage target.

A high shrinkage intensity indicates that the sample covariance matrix is
estimated with substantial noise relative to the target.

A low shrinkage intensity indicates that the data contain enough information to
rely more heavily on the sample estimate.

The project tracks the evolution of the shrinkage intensity over time for both
targets.

## Minimum-Variance Portfolio

The minimum-variance portfolio minimizes total portfolio variance subject to a
full-investment constraint.

The optimization problem is:

```text
minimize:
    weights_transpose
    * covariance_matrix
    * weights

subject to:
    sum(weights) = 1
```

Depending on the implementation, short positions may be allowed.

If short selling is allowed, weights may be positive or negative.

The analytical solution is:

```text
minimum_variance_weights =
    inverse_covariance * ones
    / (
        ones_transpose
        * inverse_covariance
        * ones
    )
```

For numerical stability, solving a linear system is preferable to explicitly
calculating the matrix inverse.

## Mean-Variance Portfolio

The mean-variance portfolio balances expected return and variance.

A representative objective is:

```text
maximize:
    expected_return_transpose * weights
    - risk_aversion / 2
      * weights_transpose
      * covariance_matrix
      * weights
```

The exact formulation should match the implementation provided for the
assignment.

When no full-investment constraint is imposed, a common solution is:

```text
mean_variance_weights =
    inverse_covariance
    * expected_returns
    / risk_aversion
```

When constraints are imposed, the weights are obtained through numerical
optimization.

## Risk-Aversion Calibration

The risk-aversion parameter is calibrated on the first complete estimation
window.

The target is a gross exposure of approximately three.

Gross exposure is defined as:

```text
gross_exposure =
    sum(abs(weights))
```

The risk-aversion parameter is selected so that:

```text
gross_exposure approximately equals 3
```

Once calibrated, the same risk-aversion parameter is used for all subsequent
rebalancing dates and covariance estimators.

This ensures that the rolling comparison is not distorted by repeatedly
recalibrating the investor's risk preferences.

## Why Gross Exposure Matters

A portfolio can satisfy:

```text
sum(weights) = 1
```

while still having substantial leverage through offsetting long and short
positions.

For example:

```text
long exposure = 2
short exposure = -1
net exposure = 1
gross exposure = 3
```

Gross exposure is therefore a useful measure of:

- leverage;
- concentration in long-short positions;
- sensitivity to estimation error;
- trading and funding requirements.

## Portfolio Estimators

For each rebalance date, six portfolios are considered.

### Minimum-Variance Portfolios

- sample covariance minimum variance;
- constant-correlation shrinkage minimum variance;
- single-factor shrinkage minimum variance.

### Mean-Variance Portfolios

- sample covariance mean variance;
- constant-correlation shrinkage mean variance;
- single-factor shrinkage mean variance.

The portfolios are evaluated using the same estimation windows and rebalance
dates.

## Out-of-Sample Backtest

Portfolio weights are estimated using information available at the rebalance
date.

The weights are then applied to returns observed after the rebalance.

This avoids look-ahead bias.

A representative holding-period return is:

```text
portfolio_return_t =
    sum(
        weight_i_at_last_rebalance
        * asset_return_i_t
    )
```

Weights remain fixed until the next monthly rebalance unless the implementation
explicitly accounts for weight drift.

## Performance Metrics

The project compares the portfolios using:

- cumulative performance;
- realized volatility;
- turnover;
- gross exposure;
- condition number of the covariance estimator.

### Cumulative Performance

```text
cumulative_wealth_t =
    product(
        1 + portfolio_return_s
        for all s up to t
    )
```

### Annualized Volatility

```text
annualized_volatility =
    standard_deviation(daily_returns)
    * sqrt(252)
```

### Turnover

A simple turnover measure is:

```text
turnover_t =
    sum(
        abs(
            new_weight_i
            - previous_weight_i
        )
    )
```

A more precise implementation may compare new target weights with drifted
pre-trade weights.

The selected definition should be documented.

### Gross Exposure

```text
gross_exposure_t =
    sum(abs(weight_i_t))
```

## Expected Effects of Shrinkage

Shrinkage is expected to:

- reduce the condition number;
- stabilize portfolio weights;
- reduce extreme positions;
- reduce turnover;
- improve out-of-sample risk estimates;
- make optimization less sensitive to small data changes.

The effect may differ between minimum-variance and mean-variance portfolios.

Mean-variance portfolios may remain unstable because expected-return estimation
error is not corrected by covariance shrinkage.

## Comparing the Two Shrinkage Targets

The constant-correlation target and the single-factor target have different
economic rationales.

### Constant-Correlation Target

Assumes that all asset pairs share the same average correlation.

Emphasizes:

- statistical simplicity;
- homogeneous dependence;
- robust covariance regularization.

### Single-Factor Target

Assumes that co-movement is generated by a common market factor.

Emphasizes:

- market-beta exposure;
- systematic versus idiosyncratic risk;
- economically interpretable covariance structure.

The comparison focuses on:

- shrinkage intensity;
- condition number;
- weight stability;
- realized volatility;
- cumulative performance;
- turnover;
- gross exposure.

## Principal Component Analysis

Principal Component Analysis is applied to the covariance matrix.

The covariance matrix is decomposed into eigenvalues and eigenvectors.

```text
covariance_matrix
    * eigenvector_j
    =
eigenvalue_j
    * eigenvector_j
```

The eigenvectors represent statistical factors.

The eigenvalues measure the amount of variance associated with each factor.

The components are ordered from the largest eigenvalue to the smallest.

## Explained Variance

The percentage of variance explained by component `j` is:

```text
explained_variance_ratio_j =
    eigenvalue_j
    / sum(all_eigenvalues)
```

The cumulative explained variance of the first `k` components is:

```text
cumulative_explained_variance_k =
    sum(
        explained_variance_ratio_j
        for j from 1 to k
    )
```

The project tracks the explained variance through time.

## Interpretation of the First Component

The first principal component is often associated with a broad market factor.

A market-like component typically has:

- loadings with the same sign across most stocks;
- relatively uniform exposure across sectors;
- a large share of total variance explained.

Its importance often increases during periods of market stress because
cross-sectional correlations rise.

## Interpretation of Later Components

The second and third components may capture relative movements such as:

- cyclical versus defensive sectors;
- financials versus non-financials;
- industrial versus consumer companies;
- country or regional effects;
- high-volatility versus low-volatility stocks.

The sign of a principal component is arbitrary.

If an eigenvector is multiplied by minus one, it represents the same statistical
factor.

Economic interpretation should therefore focus on relative signs and magnitudes,
not on the absolute orientation of the vector.

## Factor Interpretation on 31 January 2019

The first three factors are interpreted using the PCA loadings estimated on
31 January 2019.

For each component:

1. rank stocks by loading;
2. identify the largest positive loadings;
3. identify the largest negative loadings;
4. map companies to sectors;
5. evaluate whether the component represents a market or relative-sector factor.

A useful output table is:

| Ticker | Company | Sector | PC1 loading | PC2 loading | PC3 loading |
|---|---|---|---:|---:|---:|

Possible interpretations should be derived from the actual loadings rather than
imposed in advance.

## Factor Turnover

The stability of principal components across consecutive rebalance dates is
measured using cosine similarity.

For two consecutive loading vectors:

```text
cosine_similarity =
    absolute_value(
        dot(previous_loading, current_loading)
        / (
            norm(previous_loading)
            * norm(current_loading)
        )
    )
```

The absolute value is used because eigenvector signs are arbitrary.

Interpretation:

```text
cosine_similarity close to 1
```

indicates a stable factor.

```text
cosine_similarity close to 0
```

indicates a substantial change in the factor direction.

## Matching Components Through Time

Principal components may change order across rebalance dates.

Comparing only components with the same numerical index may therefore be
misleading.

A more robust procedure is:

1. compute cosine similarities between all previous and current components;
2. match components according to the highest absolute similarity;
3. align the signs;
4. record the matched similarity.

A Hungarian assignment algorithm can be used when several components are tracked
simultaneously.

## Estimation Window and Factor Turnover

The length of the estimation window affects factor stability.

A longer window generally:

- averages out short-term noise;
- produces more stable covariance estimates;
- reduces factor turnover;
- reacts more slowly to structural changes.

A shorter window generally:

- adapts more quickly to market changes;
- produces noisier loadings;
- increases factor turnover;
- may better capture temporary market regimes.

The two-year window used in this project represents a compromise between
stability and responsiveness.

## Market Conditions and Explained Variance

The variance explained by the first component often changes with market
conditions.

During stressed markets:

- correlations tend to increase;
- the first eigenvalue tends to rise;
- a larger share of total variance becomes market-driven.

During calmer markets:

- sector-specific and idiosyncratic effects become more relevant;
- variance may be distributed across more components;
- the first component may explain a smaller proportion of total variance.

The project compares the rolling factor structure with major market conditions
observed in the sample.

## Covariance PCA vs Correlation PCA

PCA can be applied to either the covariance matrix or the correlation matrix.

### Covariance-Matrix PCA

High-volatility stocks receive greater influence.

The factors reflect both:

- correlations;
- individual volatility levels.

### Correlation-Matrix PCA

Each asset is standardized to unit variance.

The factors reflect the dependence structure independently of individual
volatility levels.

The first components may therefore differ materially between the two methods.

Covariance PCA is more directly connected to portfolio variance.

Correlation PCA may be more appropriate for identifying common standardized
return patterns.

## Optional — PCA-Based Covariance Denoising

PCA can also be used to construct a lower-rank approximation of the covariance
matrix.

If only the first `k` components are retained:

```text
denoised_covariance =
    sum(
        eigenvalue_j
        * outer_product(
            eigenvector_j,
            eigenvector_j
        )
        for j from 1 to k
    )
```

A diagonal or residual-noise component may be added to preserve total variance
or ensure positive definiteness.

The assignment considers:

```text
k = 1
k = 3
k = 5
k = 10
```

## Shrinkage vs PCA Denoising

Shrinkage and PCA denoising both regularize covariance matrices, but they operate
differently.

### Shrinkage

```text
shrunk_covariance =
    weighted average of
    sample covariance
    and structured target
```

Shrinkage modifies the complete eigenvalue spectrum indirectly.

It generally preserves full rank.

### PCA Denoising

```text
denoised_covariance =
    reconstruction from selected components
```

PCA denoising directly retains dominant eigenvectors and suppresses smaller
components.

A truncated PCA matrix may be singular unless a residual variance correction is
introduced.

## Eigenvalue-Spectrum Comparison

The optional analysis compares the eigenvalues of:

- the sample covariance matrix;
- the constant-correlation shrunk matrix;
- the single-factor shrunk matrix;
- PCA-denoised covariance matrices.

The reference date is 31 January 2019.

Expected effects include:

- compression of extreme sample eigenvalues under shrinkage;
- stabilization of small eigenvalues;
- lower condition numbers;
- removal of small eigenvalues under PCA truncation;
- preservation of dominant factors.

A useful figure plots the ordered eigenvalues for all estimators.

## Limited Cross-Sectional Dimension

The Euro Stoxx 50 universe has a relatively limited number of assets compared
with the number of observations in a two-year daily window.

Approximately:

```text
number_of_assets around 50
number_of_observations around 500
```

The ratio is therefore relatively favorable.

```text
number_of_observations
    much greater than
number_of_assets
```

In this setting, the sample covariance matrix is less noisy than it would be in a
high-dimensional problem where the number of assets is close to or larger than
the number of observations.

Consequently, the benefits of shrinkage may be less dramatic than in larger
equity universes or shorter estimation windows.

## Numerical Validation

The implementation should include the following checks.

### Covariance Symmetry

```text
covariance_matrix
    approximately equals
covariance_matrix_transpose
```

### Positive Semidefiniteness

All covariance-matrix eigenvalues should be non-negative within numerical
tolerance.

### Shrinkage Bounds

```text
0 <= shrinkage_intensity <= 1
```

### Portfolio Budget

When a full-investment constraint is imposed:

```text
sum(weights) approximately equals 1
```

### Gross Exposure Target

For the first mean-variance estimation window:

```text
gross_exposure approximately equals 3
```

### Out-of-Sample Timing

Weights estimated at a rebalance date must only be applied to subsequent
returns.

### Eigenvector Normalization

```text
norm(eigenvector_j) approximately equals 1
```

### Explained Variance

```text
sum(explained_variance_ratios)
    approximately equals 1
```

### Factor Sign Alignment

Consecutive eigenvectors should be sign-aligned before calculating factor
turnover.

### Portfolio Variance

The minimum-variance portfolio should have no higher estimated variance than
other feasible fully invested portfolios under the same covariance estimator.

### Condition Number

Shrunk covariance matrices should generally have a lower condition number than
the sample covariance matrix.

## Suggested Repository Structure

```text
covariance-shrinkage-factor-models/
|
|-- README.md
|-- requirements.txt
|
|-- src/
|   |-- data_loader.py
|   |-- returns.py
|   |-- rolling_windows.py
|   |-- sample_covariance.py
|   |-- constant_correlation_shrinkage.py
|   |-- single_factor_shrinkage.py
|   |-- portfolio_optimization.py
|   |-- risk_aversion.py
|   |-- backtest.py
|   |-- performance_metrics.py
|   |-- pca_analysis.py
|   |-- factor_matching.py
|   |-- covariance_denoising.py
|   `-- validation.py
|
|-- notebooks/
|   `-- covariance_and_factor_analysis.ipynb
|
|-- scripts/
|   `-- run_analysis.py
|
|-- data/
|   `-- README.md
|
|-- results/
|   |-- shrinkage_intensities.csv
|   |-- portfolio_weights.csv
|   |-- portfolio_performance.csv
|   |-- pca_explained_variance.csv
|   |-- factor_turnover.csv
|   `-- figures/
|       |-- shrinkage_intensity.png
|       |-- condition_numbers.png
|       |-- cumulative_performance.png
|       |-- realized_volatility.png
|       |-- portfolio_turnover.png
|       |-- gross_exposure.png
|       |-- explained_variance.png
|       |-- factor_cosine_similarity.png
|       |-- factor_loadings_2019_01_31.png
|       `-- eigenvalue_spectrum.png
|
`-- report/
    `-- assignment_report.pdf
```

The file and folder names can be adapted to the structure of the actual Python
implementation.

## Requirements

A representative Python environment may include:

```text
numpy
pandas
scipy
matplotlib
scikit-learn
```

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Running the Project

A possible execution command is:

```bash
python scripts/run_analysis.py
```

Alternatively, the complete workflow can be executed from:

```text
notebooks/covariance_and_factor_analysis.ipynb
```

## Main Outputs

The project reports:

- calibrated risk-aversion parameter;
- rolling sample covariance matrices;
- rolling constant-correlation shrinkage intensities;
- rolling single-factor shrinkage intensities;
- covariance-matrix condition numbers;
- minimum-variance portfolio weights;
- mean-variance portfolio weights;
- portfolio gross exposure;
- portfolio turnover;
- cumulative out-of-sample performance;
- annualized realized volatility;
- PCA eigenvalues and eigenvectors;
- explained-variance ratios;
- cosine similarity between consecutive factors;
- sector interpretation of the first three factors;
- covariance-PCA and correlation-PCA comparison;
- optional eigenvalue-spectrum comparison;
- optional PCA-denoised covariance matrices.

## Technologies

- Python
- NumPy
- pandas
- SciPy
- scikit-learn
- Matplotlib
- Portfolio optimization
- Covariance shrinkage
- Principal Component Analysis
- Statistical factor models
- Quantitative asset management

## Data

The project uses course-provided Euro Stoxx 50 market data.

Expected input files include:

```text
sx5e_underlyings.csv
ticker_details.csv
```

Course-provided or proprietary datasets should not be included in a public
repository unless redistribution is explicitly permitted.

When the original data cannot be published, the `data` folder should contain a
description of:

- expected files;
- required columns;
- date format;
- return-index units;
- ticker identifiers;
- sector classifications;
- missing-data conventions.

## Academic Context

This project was developed as part of the Buy Side section of the Financial
Engineering course at Politecnico di Milano.

The repository presents the Python implementation, rolling portfolio analysis,
covariance regularization methods, and statistical factor interpretation
developed for the assignment.
