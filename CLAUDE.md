# CLAUDE.md — Pairs Trading with Machine Learning

## Project Overview

This is an academic research project (CS7641 Machine Learning) implementing a **market-neutral pairs trading strategy** using unsupervised and supervised machine learning. The strategy identifies cointegrated stock pairs from the Russell 3000 universe, then uses statistical signals to generate long/short trades.

Two trading strategy implementations are provided:
1. **Linear Regression** (polynomial LASSO, z-score based) — implemented in Jupyter notebooks
2. **Kalman Filter** (online linear regression) — implemented as a qstrader strategy in Python scripts

## Repository Structure

```
pair-trading/
├── dataFiltering.ipynb          # Step 1: Filter Russell 3000 stocks from raw WRDS data
├── impute_reduce-prices.ipynb   # Step 2a: Impute missing prices + PCA (→15 components)
├── impute_reduce-ratios.ipynb   # Step 2b: Impute missing ratios + PCA (→5 components)
├── KMeans Clustering.ipynb      # Step 3a: K-Means clustering (31 clusters, with t-SNE viz)
├── Clustering.ipynb             # Step 3b: Additional clustering analysis/visualizations
├── DBSCAN and Pair Selection.ipynb  # Step 3c: DBSCAN (eps=1.8, minPts=3) + pair selection
├── pairSelection.ipynb          # Step 4: ADF cointegration tests, final pair selection
├── tradingStrategy.ipynb        # Step 5: Z-score computation, LASSO polynomial regression
├── Backtesting.ipynb            # Step 6: Linear regression backtesting (moving windows)
├── performanceMetrics.ipynb     # Step 7: Sharpe, alpha, drawdown, etc.
│
├── kalman_qstrader_strategy.py  # Kalman filter trading strategy (qstrader AbstractStrategy)
├── kalman_qstrader_backtest.py  # Runs qstrader backtest with Kalman filter strategy
├── parse_data.py                # Parses raw CRSP CSV into per-ticker CSV files for qstrader
│
├── params.json                  # Project metadata / GitHub Pages config
├── _config.yml                  # Jekyll config for GitHub Pages
├── index.html                   # GitHub Pages landing page
├── pictures/                    # Output charts (cluster plots, t-SNE, equity curves)
│   ├── Kmeans_plots/
│   └── DBSCAN_plots/
├── README.md                    # Full project description with results and methodology
└── .gitignore                   # Ignores .venv/ and .vscode/
```

**Note**: Raw data files (`data/`, `*.csv` from WRDS) are not committed to the repository and must be obtained separately (see Data section below).

## Analysis Pipeline

The pipeline runs sequentially through Jupyter notebooks in this order:

### 1. Data Filtering (`dataFiltering.ipynb`)
- Loads daily stock prices (Russell 3000, ~3000 stocks) and monthly financial ratios
- Removes delisted, negative-price, low-volume, and data-sparse stocks
- Outputs: ~2060 eligible stocks
- Financial ratios used: 20 metrics (Asset Turnover, BM, P/E, Sharpe-related ratios, etc.)

### 2. Data Imputation + PCA
- `impute_reduce-prices.ipynb`: Imputes missing prices with per-stock mean; PCA → 15 components (>99% variance)
- `impute_reduce-ratios.ipynb`: Imputes missing ratios with dataset mean; PCA → 5 components (>99% variance)
- Final combined feature matrix: 20 dimensions (15 price + 5 ratio PCs)

### 3. Clustering
- `KMeans Clustering.ipynb`: Elbow method (distortion, silhouette, Calinski-Harabasz scores) → 31 clusters; t-SNE visualization
- `DBSCAN and Pair Selection.ipynb`: DBSCAN with eps=1.8, minPts=3 → 11 clusters; includes ADF-based pair selection
- `Clustering.ipynb`: Supplementary cluster visualizations

### 4. Pair Selection (`pairSelection.ipynb`, also in `DBSCAN and Pair Selection.ipynb`)
- ADF test (Augmented Dickey-Fuller) on spread of each stock pair within each cluster
- Pairs with p-value < 0.05 are cointegrated (spread is stationary → mean-reverting)
- At least one pair selected per cluster for portfolio diversification

### 5. Trading Strategy (`tradingStrategy.ipynb`)
- Spread: `spread = lasso_poly_model.predict(log(A)) - log(B)`
- Z-score: `z = spread / std_dev(spread_train)`
- Model: polynomial regression (degree=4) with LassoCV regularization
- Rolling window: 700 trading days for training
- Trade signals: enter when |z| > threshold, exit on mean reversion

### 6. Backtesting (`Backtesting.ipynb`)
- Moving window backtest over 2016–2019 testing period
- Initial investment: $1,000,000; allocation per pair = total_assets / num_pairs
- No commission model; daily short price mark-to-market

### 7. Performance Metrics (`performanceMetrics.ipynb`)
- Uses `empyrical` library: max drawdown, alpha/beta, annual volatility, Sharpe ratio, Sortino ratio
- Results (testing period 2016–2019):
  | Metric | Linear Regression | Kalman Filter |
  |---|---|---|
  | Max Drawdown | -16.2% | -3.7% |
  | Alpha | 20.0% | 6.4% |
  | Sharpe Ratio | 1.20 | 3.15 |
  | Sortino Ratio | 1.72 | 5.24 |

## Kalman Filter Implementation (`kalman_qstrader_strategy.py`)

The `KalmanPairsTradingStrategy` class extends qstrader's `AbstractStrategy`:

- Maintains per-pair Kalman state: `theta` (hedge ratio), `P` (covariance), `R`, `C`, `delta=1e-4`, `vt=1e-3`
- Entry threshold: `|et| > factorA * sqrt(Qt)` where `factorA = 1.5`
- Exit threshold: mean reversion within `factorC * sqrt(Qt)` where `factorC = 1.1`
- Tickers list must be ordered as pairs: `[stock_A1, stock_B1, stock_A2, stock_B2, ...]`
- Stock identifiers are CRSP PERMNO numbers (e.g., `'43350'`, `'82651'`)

To run the Kalman backtest:
```bash
python kalman_qstrader_backtest.py
```
Outputs cumulative returns to `6pair.csv` and a tearsheet to `output`.

## Data

### Sources (not included in repo)
- **CRSP daily stock files**: `russel3000PriceDaily.csv` — download from WRDS
- **Compustat quarterly financial ratios**: financial ratios CSV — download from WRDS
- **Testing data**: `tetsing_data.csv` (note: intentional typo in filename) — raw CRSP format

### Data Format for qstrader
The `parse_data.py` script converts raw CRSP data into qstrader-compatible per-ticker CSVs:
```
data/<PERMNO>.csv
```
Format: `Date,Open,High,Low,Close,Adj Close,Volume` (all price columns set to same value from CRSP)

### Key Pairs Used (CRSP PERMNO identifiers)
| Pair | Stock A | Stock B |
|---|---|---|
| 1 | 43350 | 82651 |
| 2 | 44644 | 90458 |
| 3 | 24969 | 24985 |
| 4 | 42585 | 83621 |
| 5 | 60186 | 81095 |
| 6 | 16548 | 81577 |

## Dependencies

### Jupyter Notebooks
```
pandas
numpy
matplotlib
seaborn
scikit-learn       # KMeans, DBSCAN, PCA, LassoCV, PolynomialFeatures, StandardScaler
statsmodels        # ADF test (tsa.stattools), OLS (api.OLS)
yellowbrick        # KElbowVisualizer for cluster evaluation
empyrical          # Financial performance metrics
missingno          # Missing data visualization
scipy              # cdist for distance computations
```

### Python Scripts (qstrader)
```
qstrader           # Backtesting framework (event-driven)
numpy
click              # CLI interface (imported but not actively used in main())
```

Install in a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels yellowbrick empyrical missingno scipy qstrader
```

## Key Conventions

### Stock Identifiers
- Stocks are referenced by **CRSP PERMNO** (numeric strings), not ticker symbols
- Ticker symbols (WDFC, HSIC, etc.) appear only in documentation/examples

### Notebook Execution Order
Notebooks must be run **in the pipeline order** listed above. Each notebook produces outputs (processed DataFrames, cluster assignments, selected pairs) that downstream notebooks consume. Intermediate results are typically saved to CSV files in the working directory.

### Spread and Z-score Convention
- Spread is always defined as `predict(log(A)) - log(B)` where A is the independent variable
- Entry long: `z < -threshold` (stock B relatively underpriced)
- Entry short: `z > +threshold` (stock B relatively overpriced)

### Training vs Testing Split
- **Training**: 2010-01-01 to 2015-12-31 (used for clustering, pair selection, model fitting)
- **Testing**: 2016-01-01 to 2019-12-31 (backtesting only)

### Kalman Filter Pair Order
In `kalman_qstrader_backtest.py`, the tickers list must strictly alternate: `[A, B, A, B, ...]`. The strategy uses integer division to assign tickers to pairs (`pair_idx = ticker_index // 2`).

## Development Branch

Active development branch: `claude/add-claude-documentation-JpV2t`

## References

- [Pairs Trading Basics (QuantInsti)](https://blog.quantinsti.com/pairs-trading-basics/)
- [Kalman Filter Pairs Trading in qstrader (QuantStart)](https://www.quantstart.com/articles/kalman-filter-based-pairs-trading-strategy-in-qstrader/)
- [Ornstein-Uhlenbeck process (Wikipedia)](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process)
- [ADF Test (Wikipedia)](https://en.wikipedia.org/wiki/Augmented_Dickey%E2%80%93Fuller_test)
