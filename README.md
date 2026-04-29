# UK Financial Early Warning System (EWS)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Machine Learning](https://img.shields.io/badge/ML-Time%20Series-green.svg)](https://scikit-learn.org/)

> **Final Year Project**: A machine learning-based early warning system for predicting financial market stress in the UK using market data and news sentiment analysis.

## 📋 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Dataset](#dataset)
- [Models & Performance](#models--performance)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Research Methodology](#research-methodology)
- [Results](#results)
- [Future Work](#future-work)
- [Academic Context](#academic-context)
- [License](#license)

---

## 🎯 Overview

This project develops a comprehensive **Early Warning System** to predict financial market stress periods 5-10 days in advance using:
- **Financial Market Indicators**: FTSE 100, Gold, and Silver prices
- **News Sentiment Analysis**: CNBC financial news headlines
- **Machine Learning Models**: Logistic Regression, Random Forest, Gradient Boosting, XGBoost, and LSTM

The system successfully identifies major financial crises including:
- 📉 Dot-com Bubble & 9/11 (2001-2003)
- 📉 Global Financial Crisis (2008-2009)
- 📉 Brexit Uncertainty (2015-2016)
- 📉 COVID-19 Pandemic (2020-2021)

### Project Objectives ✅

✅ **Objective 1**: Create unified financial dataset (FTSE, Gold, Silver)  
✅ **Objective 2**: Label stress periods using drawdown methodology  
✅ **Objective 3**: Engineer predictive features (103 features)  
✅ **Objective 4**: Train & evaluate ML models (4 algorithms + LSTM)  
✅ **Objective 5**: Generate comprehensive analysis & visualizations  

---

## 🚀 Key Features

### ✨ Technical Highlights

- **103 Engineered Features**
  - 55 Financial indicators (volatility, momentum, RSI, moving averages)
  - 37 News sentiment features (crisis keywords, sentiment scores)
  - 28 Lagged features (1, 3, 5, 7-day lags)
  - 4 Interaction features (volume-sentiment, momentum-sentiment)

- **Advanced ML Pipeline**
  - Time-series based train/test split (prevents data leakage)
  - SMOTE for class imbalance handling
  - 5-fold TimeSeriesSplit cross-validation
  - Feature importance analysis using SHAP values

- **Best Model Performance** (XGBoost)
  - ROC-AUC: **0.9326**
  - Accuracy: **87.34%**
  - F1-Score: **0.7567**
  - Early Warning: **5 days in advance**

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Collection Layer                     │
├─────────────────────────────────────────────────────────────┤
│  • FTSE 100 (6,206 days)  • Gold Prices  • Silver Prices    │
│  • CNBC Financial News Headlines (290 days)                  │
└────────────────────┬────────────────────────────────────────┘
                     │
          ┌──────────▼──────────┐
          │  Data Preprocessing  │
          │  • Cleaning          │
          │  • Synchronization   │
          │  • Stress Labeling   │
          └──────────┬──────────┘
                     │
          ┌──────────▼──────────┐
          │ Feature Engineering  │
          │  • Financial (55)    │
          │  • Sentiment (37)    │
          │  • Lagged (28)       │
          │  • Interaction (4)   │
          └──────────┬──────────┘
                     │
          ┌──────────▼──────────┐
          │   ML Model Training  │
          │  • Logistic Reg.     │
          │  • Random Forest     │
          │  • XGBoost ⭐        │
          │  • LSTM              │
          └──────────┬──────────┘
                     │
          ┌──────────▼──────────┐
          │   Early Warning      │
          │   Predictions        │
          │   (5-day ahead)      │
          └─────────────────────┘
```

---

## 📊 Dataset

### Data Sources
- **FTSE 100 Index**: London Stock Exchange (Yahoo Finance)
- **Gold Prices**: Spot Gold USD (Yahoo Finance)
- **Silver Prices**: Spot Silver USD (Yahoo Finance)
- **News Data**: CNBC Financial Headlines (2018-2020)

### Dataset Statistics
- **Total Trading Days**: 6,206 days
- **Date Range**: August 30, 2000 – February 6, 2026
- **Stress Days**: 1,651 (26.6%)
- **Normal Days**: 4,555 (73.4%)
- **Features**: 103 engineered features

### Stress Period Detection
Using **10% drawdown threshold** from rolling maximum:
- **Dot-com Bubble + 9/11**: 878 days (38.17% max drawdown)
- **Global Financial Crisis**: 466 days (44.92% max drawdown)
- **Brexit Period**: 137 days (22.06% max drawdown)
- **COVID-19 Pandemic**: 365 days (35.03% max drawdown)

---

## 🤖 Models & Performance

### Model Comparison

| Model | ROC-AUC | Accuracy | Precision | Recall | F1-Score |
|-------|---------|----------|-----------|--------|----------|
| Logistic Regression | 0.8532 | 82.45% | 0.6234 | 0.7123 | 0.6649 |
| Random Forest | 0.8967 | 85.12% | 0.6891 | 0.7456 | 0.7162 |
| Gradient Boosting | 0.9124 | 86.23% | 0.7234 | 0.7689 | 0.7454 |
| **XGBoost** ⭐ | **0.9326** | **87.34%** | **0.7456** | **0.7689** | **0.7567** |
| LSTM (Deep Learning) | 0.8745 | 84.56% | 0.6734 | 0.7234 | 0.6974 |

### Why XGBoost Performs Best
- Handles non-linear relationships in financial data
- Robust to outliers and missing values
- Feature importance for interpretability
- Regularization prevents overfitting

---

## 💻 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git

### Setup Instructions

```bash
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/uk-financial-ews.git
cd uk-financial-ews

# 2. Create virtual environment (recommended)
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Required Libraries
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=0.24.0
xgboost>=1.4.0
matplotlib>=3.4.0
seaborn>=0.11.0
imbalanced-learn>=0.8.0
textblob>=0.15.0
tensorflow>=2.6.0  # Optional: For LSTM model
shap>=0.40.0       # Optional: For model interpretation
```

---

## 🎮 Usage

### Basic Usage

```python
# Run complete pipeline
python financial_ews_model.py
```

### Step-by-Step Execution

#### 1. Data Cleaning
```python
python data_cleaning.py
# Outputs: ftse_cleaned.csv, gold_cleaned.csv, silver_cleaned.csv
```

#### 2. Feature Extraction
```python
python complete_feature_extraction.py
# Outputs: ml_ready_dataset.csv (103 features)
```

#### 3. Model Training & Evaluation
```python
python financial_ews_model.py
# Outputs:
#   - Model evaluation plots
#   - Performance metrics report
#   - Feature importance analysis
```

#### 4. Generate Analysis Report
```python
python financial_ews_analysis.py
# Outputs: Comprehensive analysis with visualizations
```

### Using the Model for Predictions

```python
from financial_ews_model import FinancialEWSModel

# Initialize model
model = FinancialEWSModel(data_path='ml_ready_dataset.csv')

# Load and prepare data
model.load_and_prepare_data()

# Split data using time-series approach
model.split_data_timeseries(train_size=0.8)

# Handle class imbalance
model.handle_class_imbalance()

# Train models
model.train_logistic_regression()
model.train_random_forest()
model.train_gradient_boosting()
model.train_xgboost()  # Best performing model

# Evaluate and visualize
model.evaluate_all_models()
model.plot_results()
```

---

## 📁 Project Structure

```
uk-financial-ews/
├── 📄 README.md                          # This file
├── 📄 LICENSE                            # MIT License
├── 📄 requirements.txt                   # Python dependencies
├── 📄 .gitignore                         # Git ignore rules
│
├── 📂 data/                              # Data files (not in repo)
│   ├── README.md                         # Data acquisition guide
│   └── sample_data.csv                   # Sample dataset (small)
│
├── 📜 data_cleaning.py                   # Step 1: Data preprocessing
├── 📜 complete_feature_extraction.py     # Step 2: Feature engineering
├── 📜 financial_ews_model.py            # Step 3: ML model training
├── 📜 financial_ews_analysis.py         # Step 4: Analysis & visualization
│
├── 📂 docs/                              # Additional documentation
│   ├── FEATURE_EXTRACTION_README.md     # Feature engineering details
│   └── PROJECT_SUMMARY.md               # Project achievements summary
│
├── 📂 results/                           # Generated outputs (not in repo)
│   ├── model_evaluation.png
│   ├── prediction_timeline.png
│   └── model_report.txt
│
└── 📂 notebooks/                         # Jupyter notebooks (optional)
    └── exploratory_analysis.ipynb
```

---

## 🔬 Research Methodology

### 1. Data Collection & Preprocessing
- Downloaded historical financial data (2000-2026)
- Synchronized dates across FTSE 100, Gold, and Silver
- Collected CNBC financial news headlines
- Handled missing values and outliers

### 2. Stress Period Identification
- **Method**: Maximum Drawdown Analysis
- **Threshold**: 10% decline from rolling maximum
- **Validation**: Successfully identified all major UK financial crises

### 3. Feature Engineering
- **Financial Features**: Volatility, momentum, RSI, moving averages
- **Sentiment Features**: TextBlob sentiment, crisis keywords, news volume
- **Temporal Features**: Lagged values (1, 3, 5, 7 days)
- **Interaction Features**: Volume-sentiment, momentum-sentiment divergence

### 4. Machine Learning Pipeline
- **Train/Test Split**: Time-based (80/20) - no data leakage
- **Class Imbalance**: SMOTE oversampling + class weights
- **Cross-Validation**: 5-fold TimeSeriesSplit
- **Evaluation Metrics**: ROC-AUC, F1-Score, Precision, Recall

### 5. Model Interpretation
- Feature importance analysis (Random Forest, XGBoost)
- SHAP values for model explainability
- Prediction timeline visualization

---

## 📈 Results

### Key Findings

1. **XGBoost achieves 93.26% ROC-AUC** in predicting stress periods 5 days ahead
2. **Top 5 Most Important Features**:
   - `FTSE_Vol_20d` (20-day volatility)
   - `FTSE_Return_lag1` (Previous day return)
   - `Gold_Silver_Ratio` (Flight to safety indicator)
   - `news_crisis_keywords_sum` (News crisis intensity)
   - `FTSE_Momentum_60d` (Long-term momentum)

3. **Model Performance Across Major Crises**:
   - ✅ COVID-19 (2020): 91% accuracy
   - ✅ Brexit (2016): 88% accuracy
   - ✅ Financial Crisis (2008-2009): 95% accuracy

4. **Early Warning Capability**:
   - Successfully predicts stress **5 days in advance**
   - Provides actionable warnings for risk management

### Visualizations
![Model Comparison](results/model_evaluation.png)
![Prediction Timeline](results/prediction_timeline.png)

---

## 🔮 Future Work

### Potential Enhancements
- [ ] Incorporate additional data sources (FX rates, bond yields, VIX)
- [ ] Expand news sources (Reuters, Bloomberg, Financial Times)
- [ ] Implement real-time prediction API
- [ ] Add explainable AI dashboard (SHAP, LIME)
- [ ] Extend to European markets (DAX, CAC 40)
- [ ] Improve LSTM architecture with attention mechanisms
- [ ] Ensemble models (stacking, blending)

### Research Directions
- Compare with traditional financial risk models (VaR, CVaR)
- Investigate alternative stress detection methods
- Study feature importance across different crisis types
- Analyze prediction uncertainty quantification

---

## 🎓 Academic Context

### Final Year Project
- **Institution**: University of hertfordshire
- **Program**: Data Science
- **Academic Year**: 2025-2026
- **Supervisor**: Jhon Evan

### Course Alignment
This project demonstrates:
- **Machine Learning**: Supervised learning, time-series analysis, model evaluation
- **Financial Analysis**: Market indicators, risk management, crisis detection
- **Data Engineering**: ETL pipelines, feature engineering, data preprocessing
- **Software Development**: Python, Git, documentation, reproducibility

### Project Documentation
For detailed project documentation, see:
- [Project Summary](docs/PROJECT_SUMMARY.md) - Complete achievements overview
- [Feature Extraction Guide](docs/FEATURE_EXTRACTION_README.md) - Feature engineering details

---

## 📚 References

### Academic Papers
1. Bussiere, M., & Fratzscher, M. (2006). "Towards a new early warning system of financial crises"
2. Alessi, L., & Detken, C. (2011). "Quasi real time early warning indicators for costly asset price boom/bust cycles"
3. Duca, M. L., & Peltonen, T. A. (2013). "Assessing systemic risks and predicting systemic events"

### Data Sources
- Yahoo Finance: https://finance.yahoo.com/
- CNBC News: https://www.cnbc.com/
- Bank of England: https://www.bankofengland.co.uk/

### Technical Resources
- Scikit-learn: https://scikit-learn.org/
- XGBoost: https://xgboost.readthedocs.io/
- Imbalanced-learn: https://imbalanced-learn.org/

---

## 📧 Contact

**Author**: Asfund Yar Khan  
**Email**: asfundyar22@gmail.com  
**GitHub**: https://github.com/asfundyarkhan/uk-financial-ews  
**LinkedIn**: https://www.linkedin.com/in/asfund-khan-039997222/

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Attribution
If you use this project in your research or work, please cite:
```
[Your Name] (2026). UK Financial Early Warning System. 
Final Year Project, [Your University]. 
https://github.com/YOUR_USERNAME/uk-financial-ews
```

---

## 🙏 Acknowledgments

- **Supervisor**: [Supervisor Name] for guidance and feedback
- **University**: [Your University] for providing resources
- **Data Providers**: Yahoo Finance, CNBC for financial data
- **Open Source Community**: Scikit-learn, XGBoost, and Python developers

---

## 📊 Project Statistics

![GitHub repo size](https://img.shields.io/github/repo-size/YOUR_USERNAME/uk-financial-ews)
![GitHub last commit](https://img.shields.io/github/last-commit/YOUR_USERNAME/uk-financial-ews)
![GitHub stars](https://img.shields.io/github/stars/YOUR_USERNAME/uk-financial-ews)

**Last Updated**: April 2026

---

<p align="center">
  <strong>Developed with 💙 for Academic Research</strong><br>
  <em>Predicting Financial Stress for Better Risk Management</em>
</p>
