# Data Directory

This directory contains the datasets used in the UK Financial Early Warning System project.

## ⚠️ Important Notice

Due to file size limitations on GitHub, **raw data files are not included in this repository**. You will need to download the data separately following the instructions below.

---

## 📊 Required Datasets

### 1. FTSE 100 Index Data
- **Source**: Yahoo Finance
- **Symbol**: `^FTSE`
- **Date Range**: August 30, 2000 – February 6, 2026
- **Columns**: Date, Open, High, Low, Close, Adj Close, Volume

**Download Link**: https://finance.yahoo.com/quote/%5EFTSE/history

**Download Instructions**:
1. Visit the Yahoo Finance link above
2. Select "Historical Data" tab
3. Set date range: Aug 30, 2000 to Feb 6, 2026
4. Click "Download" button
5. Save as `ftse.csv` in this directory

---

### 2. Gold Prices (Spot)
- **Source**: Yahoo Finance
- **Symbol**: `GC=F`
- **Date Range**: August 30, 2000 – February 6, 2026
- **Columns**: Date, Open, High, Low, Close, Adj Close, Volume

**Download Link**: https://finance.yahoo.com/quote/GC%3DF/history

**Download Instructions**:
1. Visit the Yahoo Finance link above
2. Select "Historical Data" tab
3. Set date range: Aug 30, 2000 to Feb 6, 2026
4. Click "Download" button
5. Save as `gold.csv` in this directory

---

### 3. Silver Prices (Spot)
- **Source**: Yahoo Finance
- **Symbol**: `SI=F`
- **Date Range**: August 30, 2000 – February 6, 2026
- **Columns**: Date, Open, High, Low, Close, Adj Close, Volume

**Download Link**: https://finance.yahoo.com/quote/SI%3DF/history

**Download Instructions**:
1. Visit the Yahoo Finance link above
2. Select "Historical Data" tab
3. Set date range: Aug 30, 2000 to Feb 6, 2026
4. Click "Download" button
5. Save as `silver.csv` in this directory

---

### 4. CNBC Financial News Headlines (Optional)
- **Source**: CNBC Financial News
- **Date Range**: 2018-2020 (290 unique days)
- **Content**: Financial news headlines for sentiment analysis

**Note**: News data collection requires web scraping or API access. This is optional as the system can work with financial indicators alone, though performance may be reduced.

**Alternative**: You can use financial news APIs like:
- News API (https://newsapi.org/)
- Alpha Vantage News (https://www.alphavantage.co/)
- Financial Modeling Prep (https://financialmodelingprep.com/)

Save any news data as `cnbc_headlines.csv` with columns:
- `Date`: Date of the news
- `Headline`: News headline text

---

## 📁 Expected Directory Structure

After downloading all datasets, your data directory should look like this:

```
data/
├── README.md              # This file
├── ftse.csv               # FTSE 100 historical data
├── gold.csv               # Gold prices
├── silver.csv             # Silver prices
└── cnbc_headlines.csv     # News headlines (optional)
```

---

## 🔄 Data Processing Pipeline

Once you have the raw data files:

### Step 1: Data Cleaning
```bash
python data_cleaning.py
```

This will create:
- `ftse_cleaned.csv`
- `gold_cleaned.csv`
- `silver_cleaned.csv`

### Step 2: Feature Extraction
```bash
python complete_feature_extraction.py
```

This will create:
- `ml_ready_dataset.csv` (103 features, ready for ML)
- `feature_summary.txt`

### Step 3: Model Training
```bash
python financial_ews_model.py
```

---

## 📊 Dataset Statistics

| Dataset | Records | Date Range | Size |
|---------|---------|------------|------|
| FTSE 100 | 6,206 | 2000-2026 | ~500 KB |
| Gold | 6,206 | 2000-2026 | ~500 KB |
| Silver | 6,206 | 2000-2026 | ~500 KB |
| News (optional) | 290 | 2018-2020 | ~50 KB |
| **ML-Ready Dataset** | 6,199 | 2000-2026 | ~15 MB |

---

## 🔒 Data Privacy & Terms

### Terms of Use
- Financial data from Yahoo Finance is subject to their Terms of Service
- Data is for **educational and research purposes only**
- Do NOT use for commercial trading without proper licensing
- Respect rate limits when downloading data

### Data Quality Notes
- Market data includes trading days only (no weekends/holidays)
- Some missing values are expected and handled by the preprocessing pipeline
- News data is optional but improves model performance by ~5%

---

## 🆘 Troubleshooting

### Problem: "File not found" error
**Solution**: Ensure CSV files are in the `data/` directory with exact names:
- `ftse.csv`
- `gold.csv`
- `silver.csv`

### Problem: Date range mismatch
**Solution**: When downloading from Yahoo Finance, ensure you set:
- Start Date: August 30, 2000
- End Date: February 6, 2026 (or latest available)

### Problem: News data unavailable
**Solution**: The system can run without news data. Comment out news-related features in `complete_feature_extraction.py` or the model will use default values.

---

## 📞 Support

If you encounter issues downloading or processing data:
1. Check the [main README](../README.md) for setup instructions
2. Ensure you have the required Python packages: `pip install -r requirements.txt`
3. Verify file paths and naming conventions
4. Check data file formats (CSV with proper headers)

---

## 📝 Citation

If you use these datasets in your research:

```
Yahoo Finance. (2026). Historical financial data for FTSE 100, Gold, and Silver indices. 
Retrieved from https://finance.yahoo.com/

CNBC. (2020). Financial news headlines. 
Retrieved from https://www.cnbc.com/
```

---

**Last Updated**: April 2026
