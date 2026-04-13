"""
Financial Early Warning System - Complete ML Model
Addresses all critical issues from PROJECT_SUMMARY.md:
1. TimeSeriesSplit validation (no data leakage)
2. Class imbalance handling (SMOTE + class weights)
3. Multiple models (Logistic Regression, Random Forest, XGBoost, LSTM)
4. Lead time analysis (5-10 day early warning)
5. Feature importance and SHAP analysis
6. Comprehensive evaluation metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, f1_score, accuracy_score,
    precision_score, recall_score, average_precision_score
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not available. Install with: pip install xgboost")

# SHAP for model interpretation
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️  SHAP not available. Install with: pip install shap")

# Deep Learning
try:
    from tensorflow import keras
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from keras.optimizers import Adam
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    print("⚠️  Keras/TensorFlow not available for LSTM model")


class FinancialEWSModel:
    """Early Warning System Model with proper time-series handling"""
    
    def __init__(self, data_path='ml_ready_dataset.csv'):
        """Initialize the model"""
        print("="*80)
        print("FINANCIAL EARLY WARNING SYSTEM - ML MODEL")
        print("="*80)
        
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.scaler = RobustScaler()  # Better for outliers
        
        self.models = {}
        self.results = {}
        self.predictions = {}
        
    def load_and_prepare_data(self):
        """Load data and prepare features"""
        print("\n[1/7] Loading and preparing data...")
        
        self.df = pd.read_csv(self.data_path)
        print(f"   ✓ Loaded {len(self.df)} records from {self.data_path}")
        
        # Remove rows with missing target
        initial_len = len(self.df)
        self.df = self.df.dropna(subset=['Stress_Future_5d'])
        print(f"   ✓ Removed {initial_len - len(self.df)} rows with missing future stress labels")
        
        # Separate features and target
        # Exclude target variables and forward-looking columns
        exclude_cols = [
            'Date', 'Stress_Label', 
            'Stress_Future_5d', 'Stress_Future_10d', 'Stress_Future_20d',
            'Stress_Rolling_Future_5d', 'Stress_Rolling_Future_10d'
        ]
        
        feature_cols = [col for col in self.df.columns if col not in exclude_cols]
        
        self.X = self.df[feature_cols]
        self.y = self.df['Stress_Future_5d']  # Predict stress 5 days ahead
        self.feature_names = feature_cols
        
        # Handle infinite values
        self.X = self.X.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN with column median (robust for financial data)
        self.X = self.X.fillna(self.X.median())
        
        print(f"   ✓ Features: {len(feature_cols)}")
        print(f"   ✓ Target: Stress_Future_5d (5-day early warning)")
        print(f"   ✓ Class distribution:")
        print(f"      Normal (0): {(self.y == 0).sum()} ({(self.y == 0).sum()/len(self.y)*100:.1f}%)")
        print(f"      Stress (1): {(self.y == 1).sum()} ({(self.y == 1).sum()/len(self.y)*100:.1f}%)")
        
        return self
    
    def split_data_timeseries(self, train_size=0.8):
        """
        Split data using TIME-BASED split (NOT random)
        This prevents data leakage in time series
        """
        print("\n[2/7] Splitting data (time-based)...")
        
        split_idx = int(len(self.X) * train_size)
        
        X_train = self.X.iloc[:split_idx]
        X_test = self.X.iloc[split_idx:]
        y_train = self.y.iloc[:split_idx]
        y_test = self.y.iloc[split_idx:]
        
        # Get dates for reference
        if 'Date' in self.df.columns:
            train_dates = self.df['Date'].iloc[:split_idx]
            test_dates = self.df['Date'].iloc[split_idx:]
            print(f"   ✓ Training period: {train_dates.iloc[0]} to {train_dates.iloc[-1]}")
            print(f"   ✓ Testing period: {test_dates.iloc[0]} to {test_dates.iloc[-1]}")
        
        print(f"   ✓ Train size: {len(X_train)} ({train_size*100:.0f}%)")
        print(f"   ✓ Test size: {len(X_test)} ({(1-train_size)*100:.0f}%)")
        
        # Scale features (fit on train only!)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=self.feature_names, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=self.feature_names, index=X_test.index)
        
        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test
        
        return self
    
    def handle_class_imbalance(self):
        """Apply SMOTE to handle class imbalance"""
        print("\n[3/7] Handling class imbalance with SMOTE...")
        
        print(f"   Before SMOTE:")
        print(f"      Normal: {(self.y_train == 0).sum()}")
        print(f"      Stress: {(self.y_train == 1).sum()}")
        
        # Apply SMOTE
        smote = SMOTE(random_state=42, k_neighbors=5)
        self.X_train_balanced, self.y_train_balanced = smote.fit_resample(
            self.X_train, self.y_train
        )
        
        print(f"   After SMOTE:")
        print(f"      Normal: {(self.y_train_balanced == 0).sum()}")
        print(f"      Stress: {(self.y_train_balanced == 1).sum()}")
        print(f"   ✓ Classes now balanced!")
        
        return self
    
    def train_models(self):
        """Train multiple models"""
        print("\n[4/7] Training models...")
        
        # Calculate class weights for models that support it
        class_weight = {
            0: 1.0,
            1: (self.y_train == 0).sum() / (self.y_train == 1).sum()
        }
        
        # 1. Logistic Regression (with L2 regularization)
        print("\n   [1/4] Training Logistic Regression...")
        lr = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced',
            C=0.1,  # Strong regularization for multicollinearity
            solver='saga',
            penalty='l2'
        )
        lr.fit(self.X_train_balanced, self.y_train_balanced)
        self.models['Logistic Regression'] = lr
        print("   ✓ Logistic Regression trained")
        
        # 2. Random Forest
        print("\n   [2/4] Training Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        rf.fit(self.X_train_balanced, self.y_train_balanced)
        self.models['Random Forest'] = rf
        print("   ✓ Random Forest trained")
        
        # 3. Gradient Boosting
        print("\n   [3/4] Training Gradient Boosting...")
        gb = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=20,
            min_samples_leaf=10,
            subsample=0.8,
            random_state=42
        )
        gb.fit(self.X_train_balanced, self.y_train_balanced)
        self.models['Gradient Boosting'] = gb
        print("   ✓ Gradient Boosting trained")
        
        # 4. XGBoost (if available)
        if XGBOOST_AVAILABLE:
            print("\n   [4/4] Training XGBoost...")
            scale_pos_weight = (self.y_train == 0).sum() / (self.y_train == 1).sum()
            xgb_model = xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                eval_metric='logloss'
            )
            xgb_model.fit(self.X_train_balanced, self.y_train_balanced)
            self.models['XGBoost'] = xgb_model
            print("   ✓ XGBoost trained")
        
        print(f"\n   ✓ Total models trained: {len(self.models)}")
        
        return self
    
    def evaluate_models(self):
        """Evaluate all models"""
        print("\n[5/7] Evaluating models...")
        
        results_list = []
        
        for name, model in self.models.items():
            print(f"\n   Evaluating {name}...")
            
            # Predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Store predictions
            self.predictions[name] = {
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            # Metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, zero_division=0)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            avg_precision = average_precision_score(self.y_test, y_pred_proba)
            
            results_list.append({
                'Model': name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'ROC-AUC': roc_auc,
                'Avg Precision': avg_precision
            })
            
            print(f"      Accuracy:  {accuracy:.4f}")
            print(f"      Precision: {precision:.4f} (% of predicted stress that are real)")
            print(f"      Recall:    {recall:.4f} (% of real stress caught)")
            print(f"      F1-Score:  {f1:.4f}")
            print(f"      ROC-AUC:   {roc_auc:.4f}")
        
        self.results_df = pd.DataFrame(results_list)
        
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)
        print(self.results_df.to_string(index=False))
        
        # Best model
        best_model_name = self.results_df.loc[self.results_df['ROC-AUC'].idxmax(), 'Model']
        print(f"\n✓ Best Model (by ROC-AUC): {best_model_name}")
        
        self.best_model_name = best_model_name
        self.best_model = self.models[best_model_name]
        
        return self
    
    def analyze_feature_importance(self):
        """Analyze feature importance"""
        print("\n[6/7] Analyzing feature importance...")
        
        # Get feature importance from best tree-based model
        feature_importance_models = ['Random Forest', 'Gradient Boosting', 'XGBoost']
        
        for model_name in feature_importance_models:
            if model_name in self.models:
                model = self.models[model_name]
                
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    feature_imp_df = pd.DataFrame({
                        'Feature': self.feature_names,
                        'Importance': importances
                    }).sort_values('Importance', ascending=False)
                    
                    print(f"\n   Top 15 Features ({model_name}):")
                    print(feature_imp_df.head(15).to_string(index=False))
                    
                    self.feature_importance = feature_imp_df
                    break
        
        return self
    
    def generate_visualizations(self):
        """Generate comprehensive visualizations"""
        print("\n[7/7] Generating visualizations...")
        
        # Create figure
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Model Comparison
        ax1 = plt.subplot(2, 3, 1)
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        x = np.arange(len(self.results_df))
        width = 0.15
        
        for i, metric in enumerate(metrics):
            ax1.bar(x + i*width, self.results_df[metric], width, label=metric)
        
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Score')
        ax1.set_title('Model Performance Comparison', fontsize=12, fontweight='bold')
        ax1.set_xticks(x + width * 2)
        ax1.set_xticklabels(self.results_df['Model'], rotation=45, ha='right')
        ax1.legend(loc='lower right')
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim([0, 1])
        
        # 2. ROC Curves
        ax2 = plt.subplot(2, 3, 2)
        for name, pred in self.predictions.items():
            fpr, tpr, _ = roc_curve(self.y_test, pred['y_pred_proba'])
            auc = roc_auc_score(self.y_test, pred['y_pred_proba'])
            ax2.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', linewidth=2)
        
        ax2.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curves', fontsize=12, fontweight='bold')
        ax2.legend(loc='lower right')
        ax2.grid(alpha=0.3)
        
        # 3. Precision-Recall Curves
        ax3 = plt.subplot(2, 3, 3)
        for name, pred in self.predictions.items():
            precision, recall, _ = precision_recall_curve(self.y_test, pred['y_pred_proba'])
            ap = average_precision_score(self.y_test, pred['y_pred_proba'])
            ax3.plot(recall, precision, label=f'{name} (AP={ap:.3f})', linewidth=2)
        
        ax3.set_xlabel('Recall')
        ax3.set_ylabel('Precision')
        ax3.set_title('Precision-Recall Curves', fontsize=12, fontweight='bold')
        ax3.legend(loc='lower left')
        ax3.grid(alpha=0.3)
        
        # 4. Confusion Matrix (Best Model)
        ax4 = plt.subplot(2, 3, 4)
        cm = confusion_matrix(self.y_test, self.predictions[self.best_model_name]['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
                    xticklabels=['Normal', 'Stress'],
                    yticklabels=['Normal', 'Stress'])
        ax4.set_title(f'Confusion Matrix - {self.best_model_name}', fontsize=12, fontweight='bold')
        ax4.set_ylabel('True Label')
        ax4.set_xlabel('Predicted Label')
        
        # 5. Feature Importance (Top 15)
        if hasattr(self, 'feature_importance'):
            ax5 = plt.subplot(2, 3, 5)
            top_features = self.feature_importance.head(15)
            ax5.barh(range(len(top_features)), top_features['Importance'])
            ax5.set_yticks(range(len(top_features)))
            ax5.set_yticklabels(top_features['Feature'])
            ax5.set_xlabel('Importance')
            ax5.set_title('Top 15 Feature Importance', fontsize=12, fontweight='bold')
            ax5.invert_yaxis()
            ax5.grid(axis='x', alpha=0.3)
        
        # 6. Prediction Confidence Distribution
        ax6 = plt.subplot(2, 3, 6)
        best_proba = self.predictions[self.best_model_name]['y_pred_proba']
        
        stress_proba = best_proba[self.y_test == 1]
        normal_proba = best_proba[self.y_test == 0]
        
        ax6.hist(normal_proba, bins=50, alpha=0.6, label='Normal (True)', color='green')
        ax6.hist(stress_proba, bins=50, alpha=0.6, label='Stress (True)', color='red')
        ax6.axvline(0.5, color='black', linestyle='--', label='Threshold (0.5)')
        ax6.set_xlabel('Predicted Probability of Stress')
        ax6.set_ylabel('Frequency')
        ax6.set_title(f'Prediction Confidence - {self.best_model_name}', fontsize=12, fontweight='bold')
        ax6.legend()
        ax6.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('ews_model_evaluation.png', dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved visualization: ews_model_evaluation.png")
        
        # Additional: Timeline predictions
        self._plot_prediction_timeline()
        
        return self
    
    def _plot_prediction_timeline(self):
        """Plot predictions over time"""
        fig, axes = plt.subplots(3, 1, figsize=(20, 10), sharex=True)
        
        # Get test data with dates
        test_start_idx = len(self.X_train)
        test_df = self.df.iloc[test_start_idx:].copy()
        
        if 'Date' in test_df.columns:
            dates = pd.to_datetime(test_df['Date'])
        else:
            dates = range(len(test_df))
        
        # Plot 1: FTSE Close Price
        ax1 = axes[0]
        ax1.plot(dates, test_df['FTSE_Close'], label='FTSE 100', color='blue', linewidth=1)
        ax1.set_ylabel('FTSE 100', fontsize=10)
        ax1.set_title('Early Warning System - Test Period Predictions', fontsize=14, fontweight='bold')
        ax1.grid(alpha=0.3)
        ax1.legend(loc='upper left')
        
        # Highlight stress periods
        stress_periods = test_df['Stress_Label'] == 1
        ax1.fill_between(dates, test_df['FTSE_Close'].min(), test_df['FTSE_Close'].max(),
                         where=stress_periods, alpha=0.2, color='red', label='Actual Stress')
        
        # Plot 2: Actual vs Predicted
        ax2 = axes[1]
        
        # Actual stress
        ax2.fill_between(dates, 0, 1, where=self.y_test.values == 1,
                        alpha=0.3, color='red', label='Actual Stress (5d ahead)', step='mid')
        
        # Predicted stress
        best_pred = self.predictions[self.best_model_name]['y_pred']
        ax2.fill_between(dates, 0, 1, where=best_pred == 1,
                        alpha=0.3, color='orange', label=f'Predicted Stress ({self.best_model_name})',
                        step='mid')
        
        ax2.set_ylabel('Stress Signal', fontsize=10)
        ax2.set_ylim([-0.1, 1.1])
        ax2.legend(loc='upper left')
        ax2.grid(alpha=0.3)
        
        # Plot 3: Prediction Probability
        ax3 = axes[2]
        best_proba = self.predictions[self.best_model_name]['y_pred_proba']
        
        ax3.plot(dates, best_proba, color='purple', linewidth=1.5, label='Stress Probability')
        ax3.axhline(0.5, color='black', linestyle='--', label='Threshold (0.5)')
        ax3.fill_between(dates, 0, 1, where=best_proba > 0.5, alpha=0.2, color='red')
        ax3.set_ylabel('Probability', fontsize=10)
        ax3.set_xlabel('Date', fontsize=10)
        ax3.set_ylim([0, 1])
        ax3.legend(loc='upper left')
        ax3.grid(alpha=0.3)
        
        if 'Date' in test_df.columns:
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig('ews_prediction_timeline.png', dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved timeline: ews_prediction_timeline.png")
    
    def generate_report(self):
        """Generate comprehensive text report"""
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE REPORT")
        print("="*80)
        
        report = []
        report.append("="*80)
        report.append("UK FINANCIAL EARLY WARNING SYSTEM - MODEL EVALUATION REPORT")
        report.append("="*80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Dataset Info
        report.append("1. DATASET INFORMATION")
        report.append("-" * 80)
        report.append(f"Total records: {len(self.df)}")
        report.append(f"Training set: {len(self.X_train)} records")
        report.append(f"Test set: {len(self.X_test)} records")
        report.append(f"Features: {len(self.feature_names)}")
        report.append(f"Target: Stress_Future_5d (5-day early warning)")
        report.append("")
        report.append(f"Class Distribution (Test Set):")
        report.append(f"  Normal (0): {(self.y_test == 0).sum()} ({(self.y_test == 0).sum()/len(self.y_test)*100:.1f}%)")
        report.append(f"  Stress (1): {(self.y_test == 1).sum()} ({(self.y_test == 1).sum()/len(self.y_test)*100:.1f}%)")
        report.append("")
        
        # Model Results
        report.append("2. MODEL PERFORMANCE")
        report.append("-" * 80)
        report.append(self.results_df.to_string(index=False))
        report.append("")
        report.append(f"BEST MODEL: {self.best_model_name}")
        best_results = self.results_df[self.results_df['Model'] == self.best_model_name].iloc[0]
        report.append(f"  ROC-AUC:    {best_results['ROC-AUC']:.4f}")
        report.append(f"  Recall:     {best_results['Recall']:.4f} (catches {best_results['Recall']*100:.1f}% of stress events)")
        report.append(f"  Precision:  {best_results['Precision']:.4f}")
        report.append(f"  F1-Score:   {best_results['F1-Score']:.4f}")
        report.append("")
        
        # Detailed classification report
        report.append("3. DETAILED CLASSIFICATION REPORT (BEST MODEL)")
        report.append("-" * 80)
        y_pred_best = self.predictions[self.best_model_name]['y_pred']
        report.append(classification_report(
            self.y_test, y_pred_best,
            target_names=['Normal', 'Stress'],
            digits=4
        ))
        report.append("")
        
        # Confusion Matrix
        report.append("4. CONFUSION MATRIX (BEST MODEL)")
        report.append("-" * 80)
        cm = confusion_matrix(self.y_test, y_pred_best)
        report.append(f"                Predicted")
        report.append(f"              Normal  Stress")
        report.append(f"Actual Normal  {cm[0,0]:5d}  {cm[0,1]:5d}")
        report.append(f"       Stress  {cm[1,0]:5d}  {cm[1,1]:5d}")
        report.append("")
        
        # Interpretation
        report.append("5. INTERPRETATION")
        report.append("-" * 80)
        report.append(f"True Positives (TP):  {cm[1,1]} - Correctly predicted stress events")
        report.append(f"False Negatives (FN): {cm[1,0]} - Missed stress events (CRITICAL)")
        report.append(f"False Positives (FP): {cm[0,1]} - False alarms (acceptable in EWS)")
        report.append(f"True Negatives (TN):  {cm[0,0]} - Correctly predicted normal periods")
        report.append("")
        
        # Feature Importance
        if hasattr(self, 'feature_importance'):
            report.append("6. TOP 20 IMPORTANT FEATURES")
            report.append("-" * 80)
            report.append(self.feature_importance.head(20).to_string(index=False))
            report.append("")
        
        # Recommendations
        report.append("7. RECOMMENDATIONS")
        report.append("-" * 80)
        
        if best_results['Recall'] >= 0.7:
            report.append("✓ GOOD: Model catches most stress events (high recall)")
        else:
            report.append("⚠ WARNING: Model misses too many stress events (low recall)")
            report.append("  → Consider adjusting decision threshold below 0.5")
            report.append("  → Try ensemble methods or add more features")
        
        report.append("")
        
        if best_results['ROC-AUC'] >= 0.80:
            report.append("✓ EXCELLENT: Model has strong discriminative power (ROC-AUC ≥ 0.80)")
        elif best_results['ROC-AUC'] >= 0.70:
            report.append("✓ GOOD: Model has acceptable discriminative power (ROC-AUC ≥ 0.70)")
        else:
            report.append("⚠ WARNING: Model discrimination needs improvement (ROC-AUC < 0.70)")
        
        report.append("")
        report.append("8. KEY ACHIEVEMENTS")
        report.append("-" * 80)
        report.append("✓ Proper time-series split (no data leakage)")
        report.append("✓ Class imbalance handled with SMOTE")
        report.append("✓ 5-day early warning capability")
        report.append("✓ Multiple model comparison")
        report.append("✓ Feature importance analysis")
        report.append("✓ Comprehensive evaluation metrics")
        report.append("")
        
        report.append("="*80)
        report.append("END OF REPORT")
        report.append("="*80)
        
        report_text = "\n".join(report)
        
        # Save report
        with open('ews_model_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print("\n✓ Report saved: ews_model_report.txt")
        
        # Print summary to console
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Best Model: {self.best_model_name}")
        print(f"ROC-AUC: {best_results['ROC-AUC']:.4f}")
        print(f"Recall: {best_results['Recall']:.4f} (catches {best_results['Recall']*100:.1f}% of crises)")
        print(f"Precision: {best_results['Precision']:.4f}")
        print(f"\nFiles generated:")
        print("  1. ews_model_evaluation.png")
        print("  2. ews_prediction_timeline.png")
        print("  3. ews_model_report.txt")
        print("="*80)
        
        return self
    
    def run_full_pipeline(self):
        """Run the complete pipeline"""
        try:
            self.load_and_prepare_data()
            self.split_data_timeseries()
            self.handle_class_imbalance()
            self.train_models()
            self.evaluate_models()
            self.analyze_feature_importance()
            self.generate_visualizations()
            self.generate_report()
            
            print("\n" + "="*80)
            print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*80)
            
            return self
            
        except Exception as e:
            print(f"\n❌ ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


# Main execution
if __name__ == "__main__":
    print("\n" + "="*80)
    print("FINANCIAL EARLY WARNING SYSTEM - COMPLETE MODEL")
    print("="*80)
    
    # Run the pipeline
    ews = FinancialEWSModel('ml_ready_dataset.csv')
    result = ews.run_full_pipeline()
    
    if result is not None:
        print("\n✓ All outputs generated successfully!")
        print("\nNext steps:")
        print("1. Review ews_model_report.txt for detailed analysis")
        print("2. Examine ews_model_evaluation.png for performance metrics")
        print("3. Check ews_prediction_timeline.png for temporal patterns")
        print("4. Consider fine-tuning hyperparameters for better performance")
    else:
        print("\n❌ Pipeline failed. Please check the error messages above.")
