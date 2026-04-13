# Complete Feature Extraction - UK Financial Early Warning System

## 📋 Summary

Successfully created a comprehensive ML-ready dataset combining:
- **Financial market indicators** (FTSE 100, Gold, Silver)
- **News sentiment features** (keyword analysis, crisis detection)
- **Advanced engineered features** (interactions, lags, forward labels)

---

## 📊 Dataset Overview

### Output Files Created

1. **`ml_ready_dataset.csv`** (6,199 records × 110 columns)
   - Complete dataset ready for machine learning
   - Date range: December 6, 2000 to February 6, 2026
   - Includes all features + target variables

2. **`feature_summary.txt`**
   - Detailed list of all 103 features
   - Statistics for each feature (mean, std, range)
   - Target variable distributions

3. **`complete_feature_extraction.py`**
   - Reusable feature extraction pipeline
   - Can be run on updated data

---

## 🎯 Feature Breakdown

### Total Features: **103**

| Category | Count | Examples |
|----------|-------|----------|
| **Financial Features** | 55 | FTSE_Close, Gold_Return, FTSE_Vol_20d, RSI_14 |
| **News Features** | 37 | news_sentiment_mean, news_crisis_keywords, news_fear_mentions |
| **Lagged Features** | 28 | FTSE_Return_lag1, news_sentiment_mean_lag7 |
| **Interaction Features** | 4 | vol_sentiment_interaction, momentum_sentiment_divergence |

### Target Variables

1. **`Stress_Label`** - Current stress indicator (1 = stress, 0 = normal)
   - 1,651 stress instances (26.6%)

2. **`Stress_Future_5d`** - Stress occurs within next 5 days
   - 1,651 instances (26.6%)
   - Use this for early warning prediction

3. **`Stress_Future_10d`** - Stress occurs within next 10 days
   - 1,651 instances (26.6%)

4. **`Stress_Rolling_Future_5d`** - Maximum stress in next 5 days

---

## 📰 News Sentiment Features Extracted

### Keyword-Based Features (Without TextBlob)

Since TextBlob wasn't installed, we used **keyword detection** for crisis signals:

- **Crisis Keywords**: crash, plunge, tumble, collapse, meltdown, sell-off
- **Volatility Keywords**: volatile, volatility, turbulent, uncertainty
- **Fear Keywords**: fear, panic, anxiety, worried, concerns
- **Recession Keywords**: recession, downturn, contraction, slowdown, crisis
- **Risk Keywords**: risk, danger, threat, warning, alert
- **Positive Keywords**: rally, surge, jump, soar, gain, optimism, recovery

### News Features Created

- `news_crisis_keywords_sum` - Total crisis keywords per day
- `news_fear_mentions` - Number of fear-related mentions
- `news_volatility_mentions` - Volatility keyword count
- `news_article_count` - Number of articles per day
- `news_sentiment_ma_7d` - 7-day moving average of sentiment
- `news_crisis_change_7d` - Change in crisis keywords over 7 days
- `news_volume_change_7d` - Change in news volume

**News Coverage**: 290 unique days with news data (2018-2020)
**Average articles per day**: 4.4

---

## 🔧 Engineered Features

### 1. Financial Indicators

**Volatility Features:**
- `FTSE_Vol_5d`, `FTSE_Vol_10d`, `FTSE_Vol_20d`, `FTSE_Vol_60d` - Rolling volatility
- `FTSE_HL_Range` - High-Low range (intraday volatility)

**Momentum Features:**
- `FTSE_Momentum_5d`, `FTSE_Momentum_10d`, `FTSE_Momentum_20d`, `FTSE_Momentum_60d`
- `FTSE_MA_Cross` - Moving average crossover signal
- `FTSE_RSI_14` - Relative Strength Index

**Safe-Haven Indicators:**
- `Gold_Silver_Ratio` - Flight to safety indicator
- `Gold_FTSE_Corr_30d` - 30-day correlation between Gold and FTSE
- `Safe_Haven_Strength` - 20-day rolling safe-haven demand
- `Gold_FTSE_Mom_Spread` - Momentum spread between Gold and FTSE

### 2. Interaction Features

Created features combining financial and news data:
- `vol_sentiment_interaction` - Volatility × Negative sentiment
- `crisis_vol_interaction` - Crisis keywords × Volatility
- `return_news_volume` - Returns × News volume
- `safe_haven_fear_signal` - Safe haven strength × Fear mentions
- `momentum_sentiment_divergence` - Momentum vs Sentiment trend

### 3. Lagged Features

Time-series features for prediction (lag 1, 3, 5, 7 days):
- `FTSE_Return_lag1`, `FTSE_Return_lag3`, etc.
- `FTSE_Vol_20d_lag1`, `FTSE_Vol_20d_lag3`, etc.
- `news_sentiment_mean_lag1`, `news_sentiment_mean_lag7`, etc.
- `Safe_Haven_Strength_lag1`, etc.

These ensure we only use **past information** for predictions (no data leakage).

---

## ⚠️ Known Limitations

### 1. News Data Coverage
- News data only covers **2018-2020** (290 days)
- Older periods (2000-2017) and newer periods (2021-2026) filled with forward-fill
- **Impact**: News features less reliable outside 2018-2020 range

### 2. Missing Sentiment Scores
- TextBlob not installed → no polarity sentiment scores
- Only keyword-based features available
- **Solution**: Install TextBlob for richer sentiment analysis:
  ```bash
  pip install textblob
  python -m textblob.download_corpora
  ```

### 3. Class Imbalance
- 26.6% stress periods vs 73.4% normal periods
- **Solution**: Use class weights or SMOTE when training models

---

## 🚀 Next Steps - Machine Learning

### Step 1: Install Dependencies (Optional but Recommended)

```bash
pip install textblob scikit-learn xgboost shap
python -m textblob.download_corpora
```

Then re-run feature extraction for sentiment scores:
```bash
python complete_feature_extraction.py
```

### Step 2: Load the Dataset

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Load data
df = pd.read_csv('ml_ready_dataset.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Define features and target
exclude_cols = ['Date', 'Stress_Label', 'Stress_Future_5d', 
                'Stress_Future_10d', 'Stress_Future_20d', 
                'Stress_Rolling_Future_5d', 'Stress_Rolling_Future_10d']
feature_cols = [c for c in df.columns if c not in exclude_cols]

X = df[feature_cols]
y = df['Stress_Future_5d']  # Predict stress 5 days ahead
dates = df['Date']
```

### Step 3: Time-Series Cross-Validation

```python
# Use TimeSeriesSplit to avoid data leakage
tscv = TimeSeriesSplit(n_splits=5)

for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Train your model here
```

### Step 4: Train Models

```python
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Random Forest with class weights
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced',  # Handle imbalance
    random_state=42
)

# XGBoost
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    scale_pos_weight=2.77,  # Ratio of negative to positive (73.4/26.6)
    random_state=42
)

# Train
rf_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)
```

### Step 5: Evaluate

```python
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# Predictions
y_pred_rf = rf_model.predict(X_test)
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]

# Metrics
print("ROC-AUC:", roc_auc_score(y_test, y_pred_proba_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Features:")
print(feature_importance.head(10))
```

### Step 6: SHAP Interpretability

```python
import shap

# Create SHAP explainer
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test, feature_names=feature_cols)

# Individual prediction explanation
shap.force_plot(explainer.expected_value[1], 
                shap_values[1][0], 
                X_test.iloc[0])
```

---

## 📈 Evaluation Metrics to Track

1. **ROC-AUC Score** - Overall predictive power
2. **F1-Score** - Balance between precision and recall
3. **Recall** - Important for early warning (catching true stress periods)
4. **Precision** - Avoiding false alarms
5. **Lead Time** - How many days before stress does the model predict?

### Lead Time Analysis

```python
# Calculate lead time for correct predictions
test_data = df.iloc[test_idx].copy()
test_data['y_pred'] = y_pred_rf
test_data['y_pred_proba'] = y_pred_proba_rf

# Find correct early warnings
correct_warnings = test_data[(test_data['y_pred'] == 1) & 
                              (test_data['Stress_Future_5d'] == 1)]

# Calculate how many days before actual stress
# (This requires more complex analysis of when stress actually occurs)
```

---

## 📚 Feature Importance Analysis

After training, analyze which features are most important:

1. **Financial volatility** (FTSE_Vol_20d, FTSE_Vol_60d)
2. **Drawdown** (current market decline)
3. **Safe-haven indicators** (Gold_FTSE_Corr_30d, Gold_Silver_Ratio)
4. **News crisis signals** (news_crisis_keywords, news_fear_mentions)
5. **Momentum indicators** (FTSE_Momentum_10d, FTSE_MA_Cross)
6. **Lagged features** (past volatility, past sentiment)

---

## 🔍 Model Recommendations

### For Interpretability:
- **Logistic Regression** (baseline)
- **Decision Trees**
- **Random Forest**

### For Performance:
- **XGBoost**
- **LightGBM**
- **Neural Networks** (LSTM for time-series)

### For Handling Imbalance:
- Use `class_weight='balanced'`
- Apply SMOTE (Synthetic Minority Over-sampling)
- Use stratified sampling

---

## 🎯 Project Objectives Completion

✅ **Objective 1**: Unified dataset created (6,199 records)  
✅ **Objective 2**: Stress periods labeled using 10% drawdown threshold  
✅ **Objective 3**: 103 features engineered (financial + news + interactions)  
⚠️ **Objective 4**: News sentiment partially completed (keyword-based only)  
▶️ **Next**: Build ML models and validate performance

---

## 📝 Files Structure

```
c:\Final year Project\
├── ml_ready_dataset.csv           # ← Main ML dataset
├── feature_summary.txt             # ← Feature documentation
├── complete_feature_extraction.py # ← Feature extraction script
├── ews_processed_data.csv         # Financial features only
├── cnbc_headlines.csv             # Raw news data
├── ftse_cleaned.csv               # Cleaned FTSE data
├── gold_cleaned.csv               # Cleaned Gold data
├── silver_cleaned.csv             # Cleaned Silver data
└── financial_ews_analysis.py      # Initial analysis script
```

---

## 💡 Tips for Success

1. **Start Simple**: Begin with Logistic Regression baseline
2. **Use TimeSeriesSplit**: Avoid data leakage
3. **Handle Imbalance**: Use class weights or SMOTE
4. **Track Lead Time**: Measure how early your model predicts stress
5. **Interpret Results**: Use SHAP to explain predictions
6. **Test on Real Events**: Brexit (2016), COVID-19 (2020), etc.

---

## 🤝 Need Help?

If you encounter issues:
1. Check `feature_summary.txt` for feature details
2. Verify no NaN values in your features
3. Ensure proper time-series split (no future data leakage)
4. Monitor class imbalance in train/test splits

---

**Good luck with your Early Warning System! 🎯**
