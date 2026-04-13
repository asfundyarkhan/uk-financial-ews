"""
Complete Feature Extraction for UK Financial Early Warning System
==================================================================
Integrates:
- Financial market data (FTSE, Gold, Silver)
- News sentiment analysis
- Advanced feature engineering
- ML-ready dataset preparation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# For sentiment analysis
try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False
    print("⚠️  TextBlob not installed. Run: pip install textblob")

# For advanced NLP
try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("⚠️  scikit-learn not installed. Run: pip install scikit-learn")


class NewsFeatureExtractor:
    """Extract sentiment and behavioral features from financial news"""
    
    def __init__(self, news_path='cnbc_headlines.csv'):
        """Load and preprocess news data"""
        print("="*80)
        print("NEWS SENTIMENT FEATURE EXTRACTION")
        print("="*80)
        print(f"\n[1/6] Loading news data from {news_path}...")
        
        self.news_df = pd.read_csv(news_path)
        print(f"   ✓ Loaded {len(self.news_df)} news articles")
        print(f"   ✓ Columns: {list(self.news_df.columns)}")
        
    def parse_news_dates(self):
        """Parse and clean date/time information from news"""
        print("\n[2/6] Parsing news timestamps...")
        
        df = self.news_df.copy()
        
        # Clean and parse the Time column
        def parse_cnbc_time(time_str):
            """Parse CNBC's custom time format"""
            if pd.isna(time_str) or time_str == '':
                return None
            
            try:
                # Remove leading/trailing quotes and whitespace
                time_str = str(time_str).strip().strip('"')
                
                # Extract date components using string matching
                # Format: "7:51  PM ET Fri, 17 July 2020"
                parts = time_str.split(',')
                if len(parts) >= 2:
                    date_part = parts[-1].strip()  # "17 July 2020"
                    return pd.to_datetime(date_part, format='%d %B %Y', errors='coerce')
            except:
                return None
            
            return None
        
        df['Date'] = df['Time'].apply(parse_cnbc_time)
        
        # Drop rows without valid dates
        before_drop = len(df)
        df = df.dropna(subset=['Date'])
        after_drop = len(df)
        
        print(f"   ✓ Parsed {after_drop} valid dates")
        print(f"   ✓ Dropped {before_drop - after_drop} rows with invalid dates")
        print(f"   ✓ Date range: {df['Date'].min()} to {df['Date'].max()}")
        
        self.news_df = df
        return df
    
    def extract_sentiment_features(self):
        """Extract sentiment scores from headlines and descriptions"""
        print("\n[3/6] Extracting sentiment features...")
        
        if not HAS_TEXTBLOB:
            print("   ⚠️  TextBlob not available. Skipping sentiment extraction.")
            self.news_df['headline_sentiment'] = 0
            self.news_df['headline_subjectivity'] = 0
            return self.news_df
        
        df = self.news_df.copy()
        
        def get_sentiment(text):
            """Calculate sentiment polarity and subjectivity"""
            if pd.isna(text) or text == '':
                return 0, 0
            try:
                blob = TextBlob(str(text))
                return blob.sentiment.polarity, blob.sentiment.subjectivity
            except:
                return 0, 0
        
        # Sentiment from headlines
        sentiments = df['Headlines'].apply(get_sentiment)
        df['headline_sentiment'] = sentiments.apply(lambda x: x[0])
        df['headline_subjectivity'] = sentiments.apply(lambda x: x[1])
        
        # Sentiment from descriptions (if available)
        if 'Description' in df.columns:
            desc_sentiments = df['Description'].apply(get_sentiment)
            df['description_sentiment'] = desc_sentiments.apply(lambda x: x[0])
            df['description_subjectivity'] = desc_sentiments.apply(lambda x: x[1])
            
            # Combined sentiment (weighted average)
            df['combined_sentiment'] = (df['headline_sentiment'] * 0.6 + 
                                       df['description_sentiment'] * 0.4)
        else:
            df['combined_sentiment'] = df['headline_sentiment']
        
        print(f"   ✓ Calculated sentiment scores")
        print(f"   ✓ Mean headline sentiment: {df['headline_sentiment'].mean():.4f}")
        print(f"   ✓ Sentiment range: [{df['headline_sentiment'].min():.3f}, {df['headline_sentiment'].max():.3f}]")
        
        self.news_df = df
        return df
    
    def detect_crisis_keywords(self):
        """Detect crisis-related keywords in news"""
        print("\n[4/6] Detecting crisis keywords...")
        
        # Define crisis-related keywords
        crisis_keywords = {
            'market_crash': ['crash', 'plunge', 'tumble', 'collapse', 'meltdown', 'sell-off', 'selloff'],
            'volatility': ['volatile', 'volatility', 'turbulent', 'uncertainty', 'unstable'],
            'fear': ['fear', 'panic', 'anxiety', 'worried', 'concerns', 'nervous'],
            'recession': ['recession', 'downturn', 'contraction', 'slowdown', 'crisis'],
            'risk': ['risk', 'danger', 'threat', 'warning', 'alert'],
            'positive': ['rally', 'surge', 'jump', 'soar', 'gain', 'optimism', 'recovery']
        }
        
        df = self.news_df.copy()
        
        # Create keyword count features
        for category, keywords in crisis_keywords.items():
            keyword_pattern = '|'.join(keywords)
            df[f'keyword_{category}'] = (
                df['Headlines'].fillna('').str.lower().str.contains(keyword_pattern, regex=True).astype(int)
            )
        
        # Overall crisis intensity
        df['crisis_keyword_count'] = (
            df['keyword_market_crash'] + df['keyword_volatility'] + 
            df['keyword_fear'] + df['keyword_recession'] + df['keyword_risk']
        )
        
        # Net sentiment (positive - negative keywords)
        df['keyword_net_sentiment'] = df['keyword_positive'] - df['crisis_keyword_count']
        
        print(f"   ✓ Created {len(crisis_keywords)} keyword categories")
        print(f"   ✓ Average crisis keywords per article: {df['crisis_keyword_count'].mean():.2f}")
        
        self.news_df = df
        return df
    
    def aggregate_daily_sentiment(self):
        """Aggregate news sentiment to daily level"""
        print("\n[5/6] Aggregating to daily features...")
        
        df = self.news_df.copy()
        
        # Build aggregation dict based on available columns
        agg_dict = {
            'crisis_keyword_count': ['sum', 'mean'],
            'keyword_market_crash': 'sum',
            'keyword_volatility': 'sum',
            'keyword_fear': 'sum',
            'keyword_recession': 'sum',
            'keyword_risk': 'sum',
            'keyword_positive': 'sum',
            'keyword_net_sentiment': 'mean',
            'Headlines': 'count'  # Number of articles per day
        }
        
        # Add sentiment columns if available
        if 'headline_sentiment' in df.columns:
            agg_dict['headline_sentiment'] = ['mean', 'std', 'min', 'max']
        if 'headline_subjectivity' in df.columns:
            agg_dict['headline_subjectivity'] = 'mean'
        if 'combined_sentiment' in df.columns:
            agg_dict['combined_sentiment'] = 'mean'
        
        # Group by date
        daily_agg = df.groupby('Date').agg(agg_dict)
        
        # Flatten column names manually
        new_cols = []
        for col in daily_agg.columns:
            if isinstance(col, tuple):
                base_col, agg_func = col
                if base_col == 'headline_sentiment':
                    new_cols.append(f'news_sentiment_{agg_func}')
                elif base_col == 'headline_subjectivity':
                    new_cols.append('news_subjectivity_mean')
                elif base_col == 'combined_sentiment':
                    new_cols.append('news_combined_sentiment')
                elif base_col == 'crisis_keyword_count':
                    new_cols.append(f'news_crisis_keywords_{agg_func}')
                elif base_col == 'Headlines':
                    new_cols.append('news_article_count')
                else:
                    # For other columns like keyword_*
                    clean_name = base_col.replace('keyword_', '')
                    new_cols.append(f'news_{clean_name}_mentions')
            else:
                # Single aggregation (not a tuple)
                if col == 'Headlines':
                    new_cols.append('news_article_count')
                elif col.startswith('keyword_'):
                    new_cols.append(f'news_{col.replace("keyword_", "")}_mentions')
                elif col == 'keyword_net_sentiment':
                    new_cols.append('news_net_sentiment_keywords')
                else:
                    new_cols.append(f'news_{col}')
        
        daily_agg.columns = new_cols
        daily_agg = daily_agg.reset_index()
        
        # Fill std with 0 when there's only one article (if sentiment exists)
        if 'news_sentiment_std' in daily_agg.columns:
            daily_agg['news_sentiment_std'] = daily_agg['news_sentiment_std'].fillna(0)
        
        # Add placeholder sentiment columns if they don't exist
        if 'news_sentiment_mean' not in daily_agg.columns:
            daily_agg['news_sentiment_mean'] = 0
            daily_agg['news_sentiment_std'] = 0
            daily_agg['news_sentiment_min'] = 0
            daily_agg['news_sentiment_max'] = 0
            daily_agg['news_subjectivity_mean'] = 0
            daily_agg['news_combined_sentiment'] = 0
        
        # Ensure we have article count column
        article_count_col = 'news_article_count'
        if article_count_col not in daily_agg.columns:
            # Find it under a different name
            possible_names = [c for c in daily_agg.columns if 'article' in c.lower() or 'count' in c.lower()]
            if possible_names:
                daily_agg = daily_agg.rename(columns={possible_names[0]: article_count_col})
            else:
                daily_agg[article_count_col] = 1  # Default to 1 article per day
        
        print(f"   ✓ Created {len(daily_agg.columns)-1} daily news features")
        print(f"   ✓ Coverage: {len(daily_agg)} unique days")
        if article_count_col in daily_agg.columns:
            print(f"   ✓ Avg articles per day: {daily_agg[article_count_col].mean():.1f}")
        
        self.daily_news_features = daily_agg
        return daily_agg
    
    def create_rolling_news_features(self):
        """Create rolling/momentum features from news sentiment"""
        print("\n[6/6] Creating rolling news features...")
        
        df = self.daily_news_features.copy()
        df = df.sort_values('Date')
        
        # Rolling sentiment trends (if sentiment columns exist)
        if 'news_sentiment_mean' in df.columns:
            for window in [3, 7, 14, 30]:
                df[f'news_sentiment_ma_{window}d'] = df['news_sentiment_mean'].rolling(window).mean()
                if 'news_crisis_keywords_mean' in df.columns:
                    df[f'news_crisis_ma_{window}d'] = df['news_crisis_keywords_mean'].rolling(window).mean()
        
            # Sentiment momentum (change over time)
            df['news_sentiment_change_3d'] = df['news_sentiment_mean'].diff(3)
            df['news_sentiment_change_7d'] = df['news_sentiment_mean'].diff(7)
        
        # Crisis keyword momentum
        if 'news_crisis_keywords_sum' in df.columns:
            df['news_crisis_change_3d'] = df['news_crisis_keywords_sum'].diff(3)
            df['news_crisis_change_7d'] = df['news_crisis_keywords_sum'].diff(7)
        
        # News volume changes
        if 'news_article_count' in df.columns:
            df['news_volume_change_7d'] = df['news_article_count'].pct_change(7)
        
        print(f"   ✓ Created rolling features (windows: 3, 7, 14, 30 days)")
        print(f"   ✓ Total news features: {len(df.columns)-1}")
        
        self.daily_news_features = df
        return df


class CompleteFeatureSet:
    """Combine financial and news features into ML-ready dataset"""
    
    def __init__(self, financial_data_path='ews_processed_data.csv'):
        """Load financial data"""
        print("\n" + "="*80)
        print("COMBINING FINANCIAL + NEWS FEATURES")
        print("="*80)
        print(f"\n[1/5] Loading financial data from {financial_data_path}...")
        
        self.financial_df = pd.read_csv(financial_data_path)
        self.financial_df['Date'] = pd.to_datetime(self.financial_df['Date'])
        
        print(f"   ✓ Loaded {len(self.financial_df)} records")
        print(f"   ✓ Financial features: {len(self.financial_df.columns)} columns")
        print(f"   ✓ Date range: {self.financial_df['Date'].min()} to {self.financial_df['Date'].max()}")
    
    def merge_news_features(self, news_features):
        """Merge news features with financial data"""
        print("\n[2/5] Merging news sentiment features...")
        
        # Merge on date
        merged = self.financial_df.merge(news_features, on='Date', how='left')
        
        # Fill missing news data (days without news) with forward fill then 0
        news_cols = [col for col in news_features.columns if col != 'Date']
        
        print(f"   ✓ Merged datasets")
        print(f"   ✓ Missing news data: {merged[news_cols].isna().any(axis=1).sum()} days")
        
        # Forward-fill news features (assume sentiment persists)
        merged[news_cols] = merged[news_cols].ffill().fillna(0)
        
        print(f"   ✓ Filled missing values with forward-fill")
        print(f"   ✓ Combined dataset: {len(merged)} records, {len(merged.columns)} features")
        
        self.merged_df = merged
        return merged
    
    def create_interaction_features(self):
        """Create interaction features between financial and news data"""
        print("\n[3/5] Creating interaction features...")
        
        df = self.merged_df.copy()
        
        # Volatility × Sentiment interactions
        if 'news_sentiment_mean' in df.columns:
            df['vol_sentiment_interaction'] = df['FTSE_Vol_20d'] * (-df['news_sentiment_mean'])
            df['crisis_vol_interaction'] = df['news_crisis_keywords_mean'] * df['FTSE_Vol_20d']
        
        # Return × News volume
        if 'news_article_count' in df.columns:
            df['return_news_volume'] = df['FTSE_Return'] * df['news_article_count']
        
        # Safe-haven × Fear keywords
        if 'news_fear_mentions' in df.columns:
            df['safe_haven_fear_signal'] = df['Safe_Haven_Strength'] * df['news_fear_mentions']
        
        # Momentum × Sentiment trend
        if 'news_sentiment_ma_7d' in df.columns:
            df['momentum_sentiment_divergence'] = df['FTSE_Momentum_10d'] - df['news_sentiment_ma_7d']
        
        interaction_cols = [col for col in df.columns if col not in self.merged_df.columns]
        print(f"   ✓ Created {len(interaction_cols)} interaction features")
        
        self.merged_df = df
        return df
    
    def create_lagged_features(self, lag_days=[1, 3, 5, 7]):
        """Create lagged features for time-series prediction"""
        print(f"\n[4/5] Creating lagged features (lags: {lag_days})...")
        
        df = self.merged_df.copy()
        df = df.sort_values('Date')
        
        # Select key features to lag
        lag_features = [
            'FTSE_Return', 'FTSE_Vol_20d', 'FTSE_Momentum_10d',
            'news_sentiment_mean', 'news_crisis_keywords_mean',
            'Safe_Haven_Strength', 'Gold_FTSE_Corr_30d'
        ]
        
        # Only lag features that exist
        lag_features = [f for f in lag_features if f in df.columns]
        
        lagged_cols = []
        for feature in lag_features:
            for lag in lag_days:
                col_name = f'{feature}_lag{lag}'
                df[col_name] = df[feature].shift(lag)
                lagged_cols.append(col_name)
        
        print(f"   ✓ Created {len(lagged_cols)} lagged features")
        
        self.merged_df = df
        return df
    
    def create_forward_labels(self, forward_days=5):
        """Create forward-looking stress labels for early warning"""
        print(f"\n[5/5] Creating forward stress labels (predict {forward_days} days ahead)...")
        
        df = self.merged_df.copy()
        df = df.sort_values('Date')
        
        # Create forward-looking stress label
        # Label = 1 if stress occurs within next N days
        df['Stress_Future_5d'] = df['Stress_Label'].shift(-5).fillna(0).astype(int)
        df['Stress_Future_10d'] = df['Stress_Label'].shift(-10).fillna(0).astype(int)
        df['Stress_Future_20d'] = df['Stress_Label'].shift(-20).fillna(0).astype(int)
        
        # Rolling forward stress (max stress in next N days)
        df['Stress_Rolling_Future_5d'] = df['Stress_Label'].rolling(5).max().shift(-5).fillna(0).astype(int)
        df['Stress_Rolling_Future_10d'] = df['Stress_Label'].rolling(10).max().shift(-10).fillna(0).astype(int)
        
        print(f"   ✓ Created forward-looking stress labels")
        print(f"   ✓ Stress within 5 days: {df['Stress_Future_5d'].sum()} instances ({df['Stress_Future_5d'].mean()*100:.1f}%)")
        print(f"   ✓ Stress within 10 days: {df['Stress_Future_10d'].sum()} instances ({df['Stress_Future_10d'].mean()*100:.1f}%)")
        
        self.final_df = df
        return df
    
    def prepare_ml_dataset(self, drop_na=True):
        """Prepare final ML-ready dataset"""
        print("\n" + "="*80)
        print("PREPARING ML-READY DATASET")
        print("="*80)
        
        df = self.final_df.copy()
        
        # Drop NaN from lagged features
        if drop_na:
            before = len(df)
            df = df.dropna()
            after = len(df)
            print(f"\n   ✓ Dropped {before - after} rows with NaN values")
        
        # Identify feature columns
        exclude_cols = ['Date', 'Stress_Label', 'Stress_Future_5d', 'Stress_Future_10d', 
                       'Stress_Future_20d', 'Stress_Rolling_Future_5d', 'Stress_Rolling_Future_10d']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        print(f"\n   ✓ Total features available: {len(feature_cols)}")
        print(f"   ✓ Total records: {len(df)}")
        print(f"   ✓ Date range: {df['Date'].min()} to {df['Date'].max()}")
        
        # Display feature categories
        financial_features = [c for c in feature_cols if any(x in c for x in ['FTSE', 'Gold', 'Silver', 'Safe_Haven', 'Drawdown', 'Vol', 'Momentum', 'RSI', 'MA'])]
        news_features = [c for c in feature_cols if 'news_' in c]
        interaction_features = [c for c in feature_cols if any(x in c for x in ['interaction', 'divergence', 'return_news'])]
        lagged_features = [c for c in feature_cols if '_lag' in c]
        
        print(f"\n   Feature Breakdown:")
        print(f"   - Financial features: {len(financial_features)}")
        print(f"   - News features: {len(news_features)}")
        print(f"   - Interaction features: {len(interaction_features)}")
        print(f"   - Lagged features: {len(lagged_features)}")
        print(f"   - Other features: {len(feature_cols) - len(financial_features) - len(news_features) - len(interaction_features) - len(lagged_features)}")
        
        self.ml_ready_df = df
        self.feature_cols = feature_cols
        
        return df, feature_cols
    
    def save_dataset(self, output_path='ml_ready_dataset.csv'):
        """Save the final ML-ready dataset"""
        print(f"\n💾 Saving ML-ready dataset to {output_path}...")
        self.ml_ready_df.to_csv(output_path, index=False)
        print(f"   ✓ Saved successfully!")
        print(f"   ✓ Shape: {self.ml_ready_df.shape}")
        return output_path
    
    def generate_feature_summary(self, output_path='feature_summary.txt'):
        """Generate a summary of all features"""
        print(f"\n📄 Generating feature summary to {output_path}...")
        
        df = self.ml_ready_df
        
        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("COMPLETE FEATURE SUMMARY\n")
            f.write("UK Financial Early Warning System\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Dataset Shape: {df.shape[0]} records × {df.shape[1]} features\n")
            f.write(f"Date Range: {df['Date'].min()} to {df['Date'].max()}\n\n")
            
            f.write("="*80 + "\n")
            f.write("FEATURE LIST\n")
            f.write("="*80 + "\n\n")
            
            for i, col in enumerate(self.feature_cols, 1):
                mean_val = df[col].mean()
                std_val = df[col].std()
                min_val = df[col].min()
                max_val = df[col].max()
                
                f.write(f"{i:3d}. {col:50s} | Mean: {mean_val:>10.4f} | Std: {std_val:>10.4f} | Range: [{min_val:>8.4f}, {max_val:>8.4f}]\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("TARGET VARIABLE DISTRIBUTION\n")
            f.write("="*80 + "\n\n")
            
            for target in ['Stress_Label', 'Stress_Future_5d', 'Stress_Future_10d']:
                if target in df.columns:
                    count = df[target].sum()
                    pct = df[target].mean() * 100
                    f.write(f"{target:30s}: {count:>6d} instances ({pct:>5.2f}%)\n")
        
        print(f"   ✓ Feature summary saved!")
        return output_path


def main():
    """Main execution pipeline"""
    print("\n" + "="*80)
    print("COMPLETE FEATURE EXTRACTION PIPELINE")
    print("UK Financial Early Warning System")
    print("="*80 + "\n")
    
    # Step 1: Extract news features
    print("STEP 1: NEWS FEATURE EXTRACTION")
    print("-"*80)
    news_extractor = NewsFeatureExtractor('cnbc_headlines.csv')
    news_extractor.parse_news_dates()
    news_extractor.extract_sentiment_features()
    news_extractor.detect_crisis_keywords()
    daily_news = news_extractor.aggregate_daily_sentiment()
    news_extractor.create_rolling_news_features()
    
    # Step 2: Combine with financial features
    print("\n\nSTEP 2: FEATURE INTEGRATION")
    print("-"*80)
    feature_set = CompleteFeatureSet('ews_processed_data.csv')
    feature_set.merge_news_features(news_extractor.daily_news_features)
    feature_set.create_interaction_features()
    feature_set.create_lagged_features(lag_days=[1, 3, 5, 7])
    feature_set.create_forward_labels(forward_days=5)
    
    # Step 3: Prepare ML dataset
    print("\n\nSTEP 3: ML PREPARATION")
    print("-"*80)
    ml_df, feature_cols = feature_set.prepare_ml_dataset(drop_na=True)
    
    # Step 4: Save outputs
    print("\n\nSTEP 4: SAVING OUTPUTS")
    print("-"*80)
    feature_set.save_dataset('ml_ready_dataset.csv')
    feature_set.generate_feature_summary('feature_summary.txt')
    
    # Final summary
    print("\n" + "="*80)
    print("✅ FEATURE EXTRACTION COMPLETE!")
    print("="*80)
    print(f"\n📊 Final Dataset Statistics:")
    print(f"   - Total records: {len(ml_df)}")
    print(f"   - Total features: {len(feature_cols)}")
    print(f"   - Financial features: {len([c for c in feature_cols if any(x in c for x in ['FTSE', 'Gold', 'Silver'])])}")
    print(f"   - News features: {len([c for c in feature_cols if 'news_' in c])}")
    print(f"   - Lagged features: {len([c for c in feature_cols if '_lag' in c])}")
    print(f"   - Date range: {ml_df['Date'].min()} to {ml_df['Date'].max()}")
    
    print(f"\n📁 Output Files:")
    print(f"   ✓ ml_ready_dataset.csv - Complete feature set for ML")
    print(f"   ✓ feature_summary.txt - Detailed feature documentation")
    
    print(f"\n🎯 Next Steps:")
    print(f"   1. Perform feature selection (correlation analysis, importance)")
    print(f"   2. Split data using TimeSeriesSplit")
    print(f"   3. Train ML models (Logistic Regression, Random Forest, XGBoost)")
    print(f"   4. Evaluate using ROC-AUC, F1-score, and lead time metrics")
    print(f"   5. Apply SHAP for interpretability")
    
    print("\n" + "="*80 + "\n")
    
    return feature_set, ml_df, feature_cols


if __name__ == "__main__":
    feature_set, ml_df, feature_cols = main()
