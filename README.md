# Time Series Forecasting for Retail Sales

## Overview

This project tackles the challenge of predicting weekly sales for Walmart stores using historical data. Accurate sales forecasting is crucial for retail operations - it helps with inventory management, staffing decisions, and resource allocation. We explored three different approaches to see which method works best: classical statistical models (ARIMA/SARIMA), traditional machine learning (regression models), and modern deep learning techniques (RNN/LSTM).

The project uses real Walmart sales data spanning 143 weeks across 45 stores, with various external factors like temperature, fuel prices, CPI, and unemployment rates. Our goal was not just to build models, but to understand which forecasting methods perform best under different time horizons.

## Dataset

**Source:** Walmart Sales Forecasting Data  
**Size:** 6,435 observations  
**Stores:** 45 locations  
**Time Period:** 143 weeks of historical data  

**Features:**
- Weekly sales (target variable)
- Store identifier
- Date information
- Holiday flags
- Temperature
- Fuel prices
- Consumer Price Index (CPI)
- Unemployment rate

The data exhibits strong weekly seasonality (52-week cycle), which influenced our modeling approach significantly.

## Methodology

### 1. ARIMA/SARIMA Models (Classical Statistical Approach)

We started with classical time series models because they're interpretable and work well with seasonal data. The SARIMA models explicitly handle the 52-week seasonality we observed in the data.

**Approach:**
- Used expanding-origin cross-validation with training windows at 78, 91, 104, 117, and 130 weeks
- Tested multiple forecast horizons: 1 week, 4 weeks, 8 weeks, and 13 weeks ahead
- Compared against baseline models (Naive and Seasonal-Naive)
- Systematically searched through different ARIMA and SARIMA configurations

**Key Features:**
- Feature engineering included calendar variables (year, month, week-of-year)
- Created lag features (1, 4, 13, 52 weeks) for exogenous variables
- Rolling averages to capture trends
- Strict no-data-leakage policy - only used past information

**Results:**

The SARIMA models significantly outperformed basic ARIMA models across all horizons:

| Horizon | Best Model | Average RMSE | Configuration |
|---------|-----------|--------------|---------------|
| 1 week | SARIMA | 34,029 | (3,0,0)×(1,1,1)[52] |
| 4 weeks | SARIMA | 48,181 | (0,0,2)×(1,0,1)[52] |
| 8 weeks | SARIMA | 56,216 | (0,0,2)×(1,0,1)[52] |
| 13 weeks | SARIMA | 63,133 | (0,0,2)×(1,0,1)[52] |

**Insights:**
- Seasonal-Naive baseline achieved RMSE of ~54k for 1-week and ~72k for 13-week forecasts
- SARIMA models beat the seasonal baseline by about 37% for short-term (1-week) forecasts
- The seasonal component (52-week period) was critical - models without it performed much worse
- Performance degrades predictably as forecast horizon increases

### 2. Self-Regression Models (Machine Learning Approach)

Next, we explored traditional machine learning regression models to see if they could capture non-linear patterns in the data.

**Approach:**
- Adjusted dates to align with actual sales week start (subtracted 7 days)
- Created comprehensive feature set including temporal features and store identifiers
- Tested multiple regression models with 5-fold cross-validation
- Used Recursive Feature Elimination (RFE) to identify optimal feature subset

**Models Tested:**
- Linear Regression (baseline)
- Polynomial Regression (degree 2 and 3)
- Ridge Regression (L2 regularization)
- Lasso Regression (L1 regularization)
- ElasticNet (combined L1/L2)

**Results:**

| Model | Test RMSE | Train RMSE | Test R² | Train R² |
|-------|-----------|------------|---------|----------|
| Linear Regression | 145,762 | 143,756 | 0.9345 | 0.9364 |
| **Polynomial (degree 2)** | **127,896** | 71,722 | **0.9481** | 0.9842 |
| Polynomial (degree 3) | 359,599 | 0 | 0.5945 | 1.0000 |
| Ridge | 145,732 | 143,763 | 0.9346 | 0.9364 |
| Lasso | 145,730 | 143,768 | 0.9346 | 0.9363 |
| ElasticNet | 145,876 | 143,937 | 0.9344 | 0.9362 |

**Feature Selection Results:**
- RFE identified 59 optimal features (out of many candidates)
- Selected features included: Temperature, CPI, Unemployment, Holiday flags, yearly and monthly indicators, store identifiers
- Optimal feature set achieved Test RMSE of 145,663

**Insights:**
- Polynomial regression (degree 2) performed best, showing non-linear relationships exist
- Degree 3 polynomial severely overfitted (train RMSE = 0, but worst test performance)
- Regularization methods (Ridge, Lasso, ElasticNet) showed similar performance to linear regression
- Store-specific features and temporal indicators were consistently selected as important
- The model achieved strong R² scores (>0.93), indicating good explanatory power

### 3. RNN & LSTM Models (Deep Learning Approach)

Finally, we explored recurrent neural networks to leverage their ability to learn sequential patterns.

**Approach:**
- Built upon the same feature engineering pipeline (84 leak-safe features)
- Used expanding-origin cross-validation matching the ARIMA setup
- Implemented multiple architectures: Simple RNN, GRU, LSTM variants
- Included uncertainty quantification through quantile regression
- Optimized for M1/MPS acceleration

**Architecture Details:**
- Lookback window: 26 weeks of historical data
- Feature set: Lags, rolling statistics, cyclical encodings, store dummy variables
- Training with proper fold-aware scaling
- Multi-horizon predictions (1, 4, 8, 13 weeks)

**Results:**

| Model | Average RMSE | Parameters | Status |
|-------|--------------|------------|---------|
| **Simple RNN** | **196,351** | 156K | Best performer |
| GRU | ~205,000 | 234K | Competitive |
| LSTM (various configs) | 210,000+ | 300K+ | Good but slower |

**Insights:**
- Surprisingly, the simpler RNN architecture outperformed complex LSTM/GRU variants
- This suggests that for this dataset, the sequential patterns aren't as complex as expected
- Deep learning models were computationally more expensive without significant accuracy gains
- The strong seasonality might be better captured by explicit seasonal terms (as in SARIMA)

## Model Comparison Summary

When we put all approaches together, here's what we found:

**Best Overall Performance by Horizon:**

| Forecast Horizon | Winner | RMSE | Method Type |
|------------------|--------|------|-------------|
| 1 week ahead | SARIMA | 34,029 | Statistical |
| 4 weeks ahead | SARIMA | 48,181 | Statistical |
| 8 weeks ahead | SARIMA | 56,216 | Statistical |
| 13 weeks ahead | SARIMA | 63,133 | Statistical |

**Cross-Method Comparison:**
- **SARIMA models:** Best for all horizons, particularly strong at short-term forecasts
- **Polynomial Regression:** Solid performance (RMSE ~128k) with simpler implementation
- **RNN/LSTM:** Competitive but computationally expensive (RMSE ~196k average)

**Why SARIMA Won:**
1. The data has strong weekly seasonality (52-week cycle) that SARIMA explicitly models
2. SARIMA models are specifically designed for time series forecasting
3. Less prone to overfitting on this structured temporal data
4. More interpretable - we can see the seasonal and trend components

**When to Use Each Approach:**
- **SARIMA:** When you have clear seasonality and need interpretable, accurate forecasts
- **Polynomial Regression:** When you need quick results and can accept slightly lower accuracy
- **RNN/LSTM:** When you have very large datasets, complex patterns, or multivariate dependencies that simpler models can't capture

## Technical Implementation

### Feature Engineering Highlights

We were very careful about data leakage throughout the project:

1. **Date Alignment:** Subtracted 7 days from dates since they represent week-end, not week-start
2. **Lag Features:** Created lags at 1, 4, 13, and 52 weeks for all exogenous variables
3. **Rolling Statistics:** Computed rolling means over 4, 8, and 13 week windows
4. **Cyclical Encoding:** For deep learning models, used sine/cosine transformations of temporal features
5. **Strict Train/Test Split:** Never used future information - only past data for predictions

### Cross-Validation Strategy

We used expanding-origin cross-validation, which is the gold standard for time series:
- Training windows: 78, 91, 104, 117, 130 weeks
- Multiple forecast horizons: 1, 4, 8, 13 weeks
- This mimics real-world deployment where you retrain as more data arrives

## Requirements

```python
# Core libraries
numpy
pandas
matplotlib
seaborn

# Statistical models
statsmodels
scipy

# Machine learning
scikit-learn

# Deep learning (for RNN/LSTM notebooks)
torch
```

## How to Use

1. **Clone the repository:**
```bash
git clone https://github.com/ashwini0008/TIME-SERIES-FORECASTING-FOR-RETAIL-SALES.git
cd TIME-SERIES-FORECASTING-FOR-RETAIL-SALES
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the notebooks:**
- `ARIMASarimaCompleted.ipynb` - Statistical time series models
- `selfRegressionDone.ipynb` - Machine learning regression approaches  
- `RNN&Lstm.ipynb` - Deep learning models

4. **Data:**
- The Walmart.csv dataset should be in the root directory
- Format: Store, Date, Weekly_Sales, Holiday_Flag, Temperature, Fuel_Price, CPI, Unemployment

## Key Learnings

1. **Seasonality Matters:** The 52-week seasonal pattern was the most important signal in the data. Models that explicitly handled this performed best.

2. **Simple Often Wins:** SARIMA, despite being a "classical" method, outperformed complex deep learning models. This reminds us to always establish strong baselines before jumping to complex solutions.

3. **Feature Engineering is Critical:** Careful construction of lag features and rolling statistics, combined with strict no-leakage policies, was more important than model choice.

4. **Forecast Horizon Trade-off:** All models degraded gracefully as forecast horizon increased. This is expected - predicting further into the future is inherently harder.

5. **Computational Efficiency:** For this dataset size, SARIMA and regression models train in seconds/minutes, while deep learning takes significantly longer without accuracy benefits.

## Future Work

Some directions for improvement:

- **Hierarchical Forecasting:** Build separate models per store or store cluster
- **External Data:** Incorporate additional features like promotions, competitive pricing, local events
- **Ensemble Methods:** Combine predictions from multiple models
- **Prophet:** Try Facebook's Prophet library which handles seasonality well
- **Transformer Models:** Explore attention-based architectures designed for time series
- **Probabilistic Forecasting:** Enhance uncertainty quantification with better prediction intervals

## Conclusion

This project demonstrates that for retail sales forecasting with strong seasonal patterns, classical statistical methods (SARIMA) still offer excellent performance. While machine learning and deep learning have their place, they didn't provide significant advantages for this particular problem - a good reminder that understanding your data and choosing the right tool matters more than using the newest technique.

The best model achieved RMSE of 34,029 for 1-week forecasts and 63,133 for 13-week forecasts on Walmart sales data, providing actionable predictions for inventory and resource planning.

---

**Author:** Ashwini  
**Project Type:** Time Series Analysis & Forecasting  
**Domain:** Retail Analytics