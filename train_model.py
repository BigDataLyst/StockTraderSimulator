import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

# Suppress minor future integration warnings
warnings.filterwarnings('ignore')

# Establish path references matching your data architecture
BASE_DIR = Path(__file__).resolve().parent
ML_DATA_FILE = BASE_DIR / "ml_training_features.parquet"

def load_and_prepare_ml_data(file_path):
    """Loads feature matrix, separates inputs, and prepares arrays for modeling."""
    if not file_path.exists():
        raise FileNotFoundError(f"Feature matrix missing at {file_path.name}. Run data collection script first.")
        
    print(f"Reading target data matrix from local storage: {file_path.name}...")
    df = pd.read_parquet(file_path)
    
    feature_cols = [col for col in df.columns if "_Lag_" in col or "_RollVol_" in col]
    df = df.sort_index()
    X = df[feature_cols]
    
    # Map classes: -1 -> 0 (Down), 0 -> 1 (Flat), 1 -> 2 (Up)
    y = df["ML_Target_Class"].map({-1: 0, 0: 1, 1: 2})
    contexts = df["Asset_Pair_Context"]
    
    return X, y, contexts, feature_cols

def train_high_confidence_xgboost(confidence_threshold):
    try:
        X, y, contexts, feature_names = load_and_prepare_ml_data(ML_DATA_FILE)
    except FileNotFoundError as e:
        print(e)
        return

    print(f"Dataset compiled successfully. Shapes: Matrix={X.shape}, Unique Targets={dict(y.value_counts())}")
    print(f"Operational Target Set to Strict High-Confidence Filter: >= {confidence_threshold*100}% Certainty Required.")

    # 1. Initialize Walk-Forward Framework
    tscv = TimeSeriesSplit(n_splits=5)
    
    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        objective="multi:softprob",
        num_class=3,
        random_state=42,
        eval_metric="mlogloss",
        tree_method="hist"
    )

    print("\nInitiating Walk-Forward Cross Validation with Probability Filtering...")
    
    filtered_fold_accuracies = []
    total_trades_executed = 0
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
        model.fit(X_train, y_train, sample_weight=sample_weights)
        
        # EXTRACT PROBABILITIES instead of raw guesses
        # probas shape: [n_samples, 3] -> Column 0=Down, 1=Flat, 2=Up
        probas = model.predict_proba(X_test)
        max_probas = np.max(probas, axis=1)
        raw_preds = np.argmax(probas, axis=1)
        
        # Apply the strict 75% filter
        high_conf_mask = max_probas >= confidence_threshold
        
        if np.sum(high_conf_mask) > 0:
            filtered_preds = raw_preds[high_conf_mask]
            filtered_actuals = y_test.iloc[high_conf_mask]
            
            fold_acc = accuracy_score(filtered_actuals, filtered_preds)
            filtered_fold_accuracies.append(fold_acc)
            total_trades_executed += np.sum(high_conf_mask)
            print(f" -> Fold {fold} Complete. Executed {np.sum(high_conf_mask)} high-confidence signals. Filtered Accuracy: {fold_acc:.4f}")
        else:
            print(f" -> Fold {fold} Complete. Zero signals crossed the {confidence_threshold*100}% threshold.")

    if filtered_fold_accuracies:
        print(f"\nMean Walk-Forward Filtered Accuracy Score: {np.mean(filtered_fold_accuracies):.4f}")
    print(f"Total trading opportunities filtered and accepted across backtest splits: {total_trades_executed}")

    # 2. Final Full-Sample Fit & Filtered Scorecard Generation
    print("\n" + "="*75)
    final_weights = compute_sample_weight(class_weight="balanced", y=y)
    model.fit(X, y, sample_weight=final_weights)
    
    final_probas = model.predict_proba(X)
    final_max_probas = np.max(final_probas, axis=1)
    final_raw_preds = np.argmax(final_probas, axis=1)
    
    final_mask = final_max_probas >= confidence_threshold
    # Save the trained model to disk so the live scanner can load it instantly
    
    model.save_model(BASE_DIR / "trained_sniper_model.json")
    print("💾 Trained model successfully saved locally as 'trained_sniper_model.json'")
    print("                HIGH-CONFIDENCE SNIPER MODEL PERFORMANCE SCORECARD           ")
    print("="*75)
    
    if np.sum(final_mask) > 0:
        y_filtered = y.iloc[final_mask]
        preds_filtered = final_raw_preds[final_mask]
        
        # Calculate percentage of total bars that actually get traded
        trade_frequency = (np.sum(final_mask) / len(y)) * 100
        print(f"🎯 Execution Traffic: Model triggered trades on {np.sum(final_mask)} out of {len(y)} total bars ({trade_frequency:.2f}% activity rate).")
        print(f"   The other {100 - trade_frequency:.2f}% of bars were successfully skipped as low-probability noise.\n")
        
        print("📊 FILTERED HIGH-CONFIDENCE PERFORMANCE METRICS:")
        print(classification_report(y_filtered, preds_filtered, labels=[0, 1, 2], target_names=["Down (-1)", "Flat (0)", "Up (+1)"], zero_division=0))
        
        print("🧱 FILTERED CONFUSION MATRIX DENSITIES:")
        print(confusion_matrix(y_filtered, preds_filtered))
    else:
        print(f"❌ Structural Warning: No vectors crossed the {confidence_threshold*100}% filter on the full dataset.")
    print("="*75)

if __name__ == "__main__":
    # Run script with your target 75% certainty threshold
    train_high_confidence_xgboost(confidence_threshold=0.55)