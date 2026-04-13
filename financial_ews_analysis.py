import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

class FinancialEWS:
    """
    Early Warning System for UK Financial Market Stress
    Based on FTSE100, Gold, and Silver safe-haven indicators
    """
    
    def __init__(self, ftse_path='ftse_cleaned.csv', gold_path='gold_cleaned.csv', 
                 silver_path='silver_cleaned.csv'):
        """Load cleaned datasets"""
        print("="*70)
        print("UK FINANCIAL EARLY WARNING SYSTEM")
        print("="*70)
        print("\n[1/7] Loading cleaned datasets...")
        
        self.ftse = pd.read_csv(ftse_path)
        self.gold = pd.read_csv(gold_path)
        self.silver = pd.read_csv(silver_path)
        
        # Convert dates
        self.ftse['Date'] = pd.to_datetime(self.ftse['Date'])
        self.gold['Date'] = pd.to_datetime(self.gold['Date'])
        self.silver['Date'] = pd.to_datetime(self.silver['Date'])
        
        print(f"   ✓ FTSE100: {len(self.ftse)} records")
        print(f"   ✓ Gold: {len(self.gold)} records")
        print(f"   ✓ Silver: {len(self.silver)} records")
        print(f"   ✓ Date range: {self.ftse['Date'].min()} to {self.ftse['Date'].max()}")
        
    def create_unified_dataset(self):
        """Merge all datasets into unified daily dataset (Objective 1)"""
        print("\n[2/7] Creating unified dataset...")
        
        # Merge datasets
        df = self.ftse.copy()
        df = df.rename(columns={col: f'FTSE_{col}' for col in df.columns if col != 'Date'})
        
        gold_renamed = self.gold.rename(columns={col: f'Gold_{col}' for col in self.gold.columns if col != 'Date'})
        silver_renamed = self.silver.rename(columns={col: f'Silver_{col}' for col in self.silver.columns if col != 'Date'})
        
        df = df.merge(gold_renamed, on='Date', how='inner')
        df = df.merge(silver_renamed, on='Date', how='inner')
        
        df = df.sort_values('Date').reset_index(drop=True)
        
        print(f"   ✓ Unified dataset created: {len(df)} records, {len(df.columns)} features")
        self.unified_df = df
        return df
    
    def label_stress_periods(self, drawdown_threshold=0.10, window=252):
        """
        Define crash/stress periods using drawdown methodology (Objective 2)
        
        Drawdown = (Peak - Current) / Peak
        Stress detected when drawdown exceeds threshold (default 10%)
        """
        print(f"\n[3/7] Labeling stress periods (drawdown threshold: {drawdown_threshold*100}%)...")
        
        df = self.unified_df.copy()
        
        # Calculate rolling peak (252 trading days = 1 year)
        df['FTSE_Peak'] = df['FTSE_Close'].rolling(window=window, min_periods=1).max()
        
        # Calculate drawdown
        df['Drawdown'] = (df['FTSE_Peak'] - df['FTSE_Close']) / df['FTSE_Peak']
        
        # Label stress periods
        df['Stress_Label'] = (df['Drawdown'] >= drawdown_threshold).astype(int)
        
        # Calculate statistics
        stress_periods = df[df['Stress_Label'] == 1]
        stress_count = stress_periods['Stress_Label'].sum()
        stress_pct = (stress_count / len(df)) * 100
        
        print(f"   ✓ Stress periods identified: {stress_count} days ({stress_pct:.2f}%)")
        print(f"   ✓ Normal periods: {len(df) - stress_count} days ({100-stress_pct:.2f}%)")
        print(f"   ✓ Max drawdown: {df['Drawdown'].max()*100:.2f}%")
        
        # Identify major crisis periods
        print("\n   Major Crisis Periods Detected:")
        stress_streaks = []
        in_stress = False
        start_date = None
        
        for idx, row in df.iterrows():
            if row['Stress_Label'] == 1 and not in_stress:
                in_stress = True
                start_date = row['Date']
            elif row['Stress_Label'] == 0 and in_stress:
                in_stress = False
                if start_date:
                    stress_streaks.append((start_date, df.loc[idx-1, 'Date'], 
                                          df.loc[(df['Date'] >= start_date) & (df['Date'] <= df.loc[idx-1, 'Date']), 'Drawdown'].max()))
        
        # Show top 5 longest stress periods
        stress_streaks_sorted = sorted(stress_streaks, key=lambda x: (x[1] - x[0]).days, reverse=True)[:5]
        for i, (start, end, max_dd) in enumerate(stress_streaks_sorted, 1):
            duration = (end - start).days
            print(f"   {i}. {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')} "
                  f"({duration} days, max drawdown: {max_dd*100:.2f}%)")
        
        self.unified_df = df
        return df
    
    def engineer_features(self):
        """
        Engineer predictive features (Objective 3):
        - Volatility indicators
        - Momentum indicators  
        - Flight-to-safety indicators
        """
        print("\n[4/7] Engineering predictive features...")
        
        df = self.unified_df.copy()
        
        # === VOLATILITY FEATURES ===
        print("   Creating volatility features...")
        
        # Calculate returns
        df['FTSE_Return'] = df['FTSE_Close'].pct_change()
        df['Gold_Return'] = df['Gold_Close'].pct_change()
        df['Silver_Return'] = df['Silver_Close'].pct_change()
        
        # Realized volatility (rolling standard deviation)
        for window in [5, 10, 20, 60]:
            df[f'FTSE_Vol_{window}d'] = df['FTSE_Return'].rolling(window).std() * np.sqrt(252)
        
        # Intraday volatility (High-Low range)
        df['FTSE_HL_Range'] = (df['FTSE_High'] - df['FTSE_Low']) / df['FTSE_Close']
        
        # === MOMENTUM FEATURES ===
        print("   Creating momentum features...")
        
        # Price momentum (rate of change)
        for window in [5, 10, 20, 60]:
            df[f'FTSE_Momentum_{window}d'] = df['FTSE_Close'].pct_change(window)
        
        # Moving average crossovers
        df['FTSE_MA_10'] = df['FTSE_Close'].rolling(10).mean()
        df['FTSE_MA_50'] = df['FTSE_Close'].rolling(50).mean()
        df['FTSE_MA_Cross'] = (df['FTSE_MA_10'] - df['FTSE_MA_50']) / df['FTSE_Close']
        
        # RSI (Relative Strength Index)
        def calculate_rsi(data, window=14):
            delta = data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        df['FTSE_RSI_14'] = calculate_rsi(df['FTSE_Close'])
        
        # === FLIGHT-TO-SAFETY INDICATORS ===
        print("   Creating flight-to-safety indicators...")
        
        # Gold/Silver ratio (higher = more flight to safety)
        df['Gold_Silver_Ratio'] = df['Gold_Close'] / df['Silver_Close']
        
        # Gold vs FTSE correlation (rolling)
        df['Gold_FTSE_Corr_30d'] = df['Gold_Return'].rolling(30).corr(df['FTSE_Return'])
        
        # Gold momentum vs FTSE momentum
        df['Gold_Momentum_10d'] = df['Gold_Close'].pct_change(10)
        df['Silver_Momentum_10d'] = df['Silver_Close'].pct_change(10)
        df['Gold_FTSE_Mom_Spread'] = df['Gold_Momentum_10d'] - df['FTSE_Momentum_10d']
        
        # Safe-haven demand signal (gold outperforming FTSE)
        df['Safe_Haven_Signal'] = (df['Gold_Return'] > df['FTSE_Return']).astype(int)
        df['Safe_Haven_Strength'] = df['Safe_Haven_Signal'].rolling(20).mean()
        
        # Volume surge detection
        df['FTSE_Volume_MA_20'] = df['FTSE_Volume'].rolling(20).mean()
        df['FTSE_Volume_Surge'] = df['FTSE_Volume'] / df['FTSE_Volume_MA_20']
        
        # Drop NaN values from feature engineering
        initial_len = len(df)
        df = df.dropna()
        
        print(f"   ✓ Created {len([col for col in df.columns if col not in self.unified_df.columns])} new features")
        print(f"   ✓ Dropped {initial_len - len(df)} rows with NaN (from rolling calculations)")
        print(f"   ✓ Final dataset: {len(df)} records, {len(df.columns)} features")
        
        self.feature_df = df
        return df
    
    def display_summary_statistics(self):
        """Display comprehensive summary statistics"""
        print("\n[5/7] Summary Statistics and Feature Overview...")
        print("\n" + "="*70)
        print("FEATURE SUMMARY")
        print("="*70)
        
        df = self.feature_df
        
        # Key features to display
        key_features = [
            'FTSE_Return', 'FTSE_Vol_20d', 'FTSE_Momentum_20d', 'FTSE_RSI_14',
            'Gold_FTSE_Corr_30d', 'Gold_Silver_Ratio', 'Safe_Haven_Strength',
            'Drawdown', 'Stress_Label'
        ]
        
        print("\nKey Features Statistics:")
        print(df[key_features].describe().round(4))
        
        # Correlation with stress label
        print("\n" + "="*70)
        print("FEATURE CORRELATION WITH STRESS PERIODS")
        print("="*70)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in ['Date', 'Stress_Label']]
        
        correlations = df[feature_cols].corrwith(df['Stress_Label']).abs().sort_values(ascending=False)
        
        print("\nTop 15 Features Most Correlated with Market Stress:")
        for i, (feature, corr) in enumerate(correlations.head(15).items(), 1):
            print(f"   {i:2d}. {feature:30s} : {corr:.4f}")
        
        # Stress vs Normal comparison
        print("\n" + "="*70)
        print("STRESS vs NORMAL PERIOD COMPARISON")
        print("="*70)
        
        stress_df = df[df['Stress_Label'] == 1]
        normal_df = df[df['Stress_Label'] == 0]
        
        comparison_features = ['FTSE_Vol_20d', 'FTSE_Return', 'Safe_Haven_Strength', 
                              'Gold_FTSE_Corr_30d', 'FTSE_Volume_Surge']
        
        print(f"\n{'Feature':<25} {'Normal (Mean)':<15} {'Stress (Mean)':<15} {'Difference %':<15}")
        print("-"*70)
        for feat in comparison_features:
            normal_mean = normal_df[feat].mean()
            stress_mean = stress_df[feat].mean()
            diff_pct = ((stress_mean - normal_mean) / abs(normal_mean) * 100) if normal_mean != 0 else 0
            print(f"{feat:<25} {normal_mean:>14.4f} {stress_mean:>14.4f} {diff_pct:>14.2f}%")
    
    def identify_complications(self):
        """Identify potential complications and data quality issues"""
        print("\n[6/7] Identifying Potential Complications...")
        print("\n" + "="*70)
        print("COMPLICATION ANALYSIS")
        print("="*70)
        
        df = self.feature_df
        complications = []
        
        # 1. Class imbalance
        stress_ratio = df['Stress_Label'].sum() / len(df)
        print(f"\n1. CLASS IMBALANCE:")
        print(f"   Stress periods: {stress_ratio*100:.2f}%")
        print(f"   Normal periods: {(1-stress_ratio)*100:.2f}%")
        if stress_ratio < 0.15:
            print("   ⚠ WARNING: Severe class imbalance detected!")
            print("   IMPACT: Models may be biased toward predicting 'normal'")
            print("   SOLUTION: Use SMOTE, class weights, or stratified sampling")
            complications.append("Class Imbalance")
        
        # 2. Missing sentiment data
        print(f"\n2. MISSING NEWS SENTIMENT DATA:")
        print("   ⚠ WARNING: Financial news sentiment data not yet integrated")
        print("   IMPACT: Missing a key behavioral indicator from objectives")
        print("   SOLUTION: Integrate news API or sentiment dataset (e.g., FinBERT)")
        complications.append("Missing News Sentiment")
        
        # 3. Temporal dependency
        print(f"\n3. TEMPORAL DEPENDENCY:")
        print("   ⚠ CAUTION: Time-series data requires special ML considerations")
        print("   IMPACT: Standard cross-validation can leak future information")
        print("   SOLUTION: Use TimeSeriesSplit or walk-forward validation")
        complications.append("Temporal Structure")
        
        # 4. Feature correlation
        print(f"\n4. FEATURE MULTICOLLINEARITY:")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in ['Date', 'Stress_Label']]
        
        corr_matrix = df[feature_cols].corr().abs()
        high_corr = (corr_matrix > 0.9) & (corr_matrix < 1.0)
        high_corr_pairs = high_corr.sum().sum() / 2
        
        print(f"   High correlation pairs (>0.9): {int(high_corr_pairs)}")
        if high_corr_pairs > 10:
            print("   ⚠ WARNING: Many highly correlated features detected")
            print("   IMPACT: May cause model instability, especially for linear models")
            print("   SOLUTION: Apply PCA, feature selection, or regularization")
            complications.append("Multicollinearity")
        
        # 5. Lead time requirement
        print(f"\n5. EARLY WARNING LEAD TIME:")
        print("   ⚠ CONSIDERATION: Need to predict stress BEFORE it happens")
        print("   IMPACT: Features should use lagged values only")
        print("   SOLUTION: Shift stress labels forward (e.g., predict 5-10 days ahead)")
        complications.append("Lead Time Design")
        
        # 6. Regime changes
        years_span = (df['Date'].max() - df['Date'].min()).days / 365.25
        print(f"\n6. MARKET REGIME CHANGES:")
        print(f"   Dataset spans {years_span:.1f} years (2000-2026)")
        print("   ⚠ WARNING: Market dynamics change over time (regime shifts)")
        print("   IMPACT: Models trained on old crises may not generalize")
        print("   SOLUTION: Use recent data for training, test on holdout period")
        complications.append("Non-Stationarity")
        
        # 7. Data quality checks
        print(f"\n7. DATA QUALITY:")
        zero_volume_days = (df['FTSE_Volume'] == 0).sum()
        print(f"   Zero volume days: {zero_volume_days}")
        if zero_volume_days > 0:
            print("   ⚠ CAUTION: Some days have zero volume (may be holidays/errors)")
            complications.append("Zero Volume Days")
        
        return complications
    
    def generate_recommendations(self):
        """Generate next steps and recommendations"""
        print("\n[7/7] Next Steps and Recommendations...")
        print("\n" + "="*70)
        print("RECOMMENDED NEXT STEPS")
        print("="*70)
        
        print("\n📊 DATA ENHANCEMENT:")
        print("   1. Integrate financial news sentiment data (scrape or API)")
        print("   2. Add VIX (volatility index) or UK equivalent data")
        print("   3. Include economic indicators (interest rates, GDP, etc.)")
        
        print("\n🔧 FEATURE ENGINEERING:")
        print("   4. Create lagged features (1-day, 5-day, 10-day lags)")
        print("   5. Apply feature selection (recursive elimination, importance)")
        print("   6. Engineer interaction terms (e.g., vol × sentiment)")
        
        print("\n🤖 MODEL DEVELOPMENT:")
        print("   7. Implement baseline Logistic Regression")
        print("   8. Build Random Forest with class weighting")
        print("   9. Train XGBoost with optimized hyperparameters")
        print("   10. Use TimeSeriesSplit for cross-validation")
        
        print("\n📈 EVALUATION:")
        print("   11. Calculate ROC-AUC, F1, Recall, Precision")
        print("   12. Measure early warning lead time (days before stress)")
        print("   13. Test on Brexit (2016) and COVID-19 (2020) periods")
        
        print("\n🔍 INTERPRETABILITY:")
        print("   14. Apply SHAP values for feature importance")
        print("   15. Visualize decision boundaries and predictions")
        print("   16. Create explainable reports for stakeholders")
    
    def save_processed_data(self, output_path='ews_processed_data.csv'):
        """Save the processed dataset"""
        print(f"\n💾 Saving processed dataset to {output_path}...")
        self.feature_df.to_csv(output_path, index=False)
        print(f"   ✓ Saved successfully! ({len(self.feature_df)} records, {len(self.feature_df.columns)} columns)")
        return output_path


def main():
    """Main execution function"""
    # Initialize EWS
    ews = FinancialEWS()
    
    # Execute pipeline
    ews.create_unified_dataset()
    ews.label_stress_periods(drawdown_threshold=0.10)
    ews.engineer_features()
    ews.display_summary_statistics()
    complications = ews.identify_complications()
    ews.generate_recommendations()
    ews.save_processed_data()
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\n✅ Successfully processed {len(ews.feature_df)} days of market data")
    print(f"✅ Engineered {len(ews.feature_df.columns)} features")
    print(f"⚠️  Identified {len(complications)} complications to address")
    print("\n🎯 Ready for machine learning model development!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
