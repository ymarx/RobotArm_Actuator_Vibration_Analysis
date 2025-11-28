"""
ECMiner CSV Performance Test
Test XGBoost with ECMiner's 4 parameters only:
1. 18 features (Phase2 Step 3-4 equivalent)
2. 12 features (Phase3-1 equivalent)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import xgboost as xgb

# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).parent
CSV_PATH = PROJECT_ROOT / "ecminer_stage1_output.csv"
OUTPUT_DIR = PROJECT_ROOT / "docs" / "ecminer_test_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ECMiner에서 조정 가능한 4개 파라미터만 사용
ECMINER_PARAMS = {
    'n_estimators': 100,
    'max_depth': 3,
    'subsample': 0.8,
    'learning_rate': 0.1,
    # 나머지는 XGBoost 기본값 사용
    'random_state': 42,
    'eval_metric': 'logloss'
}

# 18 features (Phase2 Step 3-4)
FEATURES_18 = [
    'acc_Y_rms', 'acc_Y_peak', 'acc_Y_crest',
    'acc_Sum_rms', 'acc_Sum_peak', 'acc_Sum_crest',
    'Gyro_Y_rms', 'Gyro_Y_peak', 'Gyro_Y_crest',
    'acc_Y_rms_low', 'acc_Y_rms_mid', 'acc_Y_rms_high',
    'acc_Sum_rms_low', 'acc_Sum_rms_mid', 'acc_Sum_rms_high',
    'Gyro_Y_rms_low', 'Gyro_Y_rms_mid', 'Gyro_Y_rms_high'
]

# 12 features (Phase3-1 core)
FEATURES_12 = [
    'acc_Y_rms', 'acc_Y_peak', 'acc_Y_crest',
    'acc_Sum_rms', 'acc_Sum_peak', 'acc_Sum_crest',
    'Gyro_Y_rms', 'Gyro_Y_peak', 'Gyro_Y_crest',
    'acc_Y_rms_high', 'Gyro_Y_rms_high', 'Gyro_Y_rms_low'
]


def load_data():
    """Load ECMiner CSV and prepare datasets"""
    df = pd.read_csv(CSV_PATH)

    # Split by dataset_type
    train_df = df[df['dataset_type'] == 'train'].copy()
    val_df = df[df['dataset_type'] == 'val'].copy()
    test_df = df[df['dataset_type'] == 'test'].copy()

    print(f"Loaded data:")
    print(f"  Train: {len(train_df)} (Normal: {sum(train_df['label_binary']==1)}, Abnormal: {sum(train_df['label_binary']==0)})")
    print(f"  Val:   {len(val_df)} (Normal: {sum(val_df['label_binary']==1)}, Abnormal: {sum(val_df['label_binary']==0)})")
    print(f"  Test:  {len(test_df)} (Normal: {sum(test_df['label_binary']==1)}, Abnormal: {sum(test_df['label_binary']==0)})")

    return train_df, val_df, test_df


def train_and_evaluate(train_df, val_df, test_df, features, feature_set_name):
    """Train XGBoost with ECMiner parameters and evaluate"""

    print("\n" + "=" * 80)
    print(f"Testing: {feature_set_name} ({len(features)} features)")
    print("=" * 80)

    # Prepare data
    X_train = train_df[features].values
    y_train = train_df['label_binary'].values

    X_val = val_df[features].values
    y_val = val_df['label_binary'].values

    X_test = test_df[features].values
    y_test = test_df['label_binary'].values

    # Calculate scale_pos_weight (class imbalance handling)
    n_abnormal = sum(y_train == 0)
    n_normal = sum(y_train == 1)
    scale_pos_weight = n_abnormal / n_normal if n_normal > 0 else 1.0

    print(f"\n[1] Training with ECMiner parameters:")
    print(f"  n_estimators: {ECMINER_PARAMS['n_estimators']}")
    print(f"  max_depth: {ECMINER_PARAMS['max_depth']}")
    print(f"  subsample: {ECMINER_PARAMS['subsample']}")
    print(f"  learning_rate: {ECMINER_PARAMS['learning_rate']}")
    print(f"  scale_pos_weight: {scale_pos_weight:.2f} (auto-calculated)")

    # Train model with ECMiner parameters only
    model = xgb.XGBClassifier(
        n_estimators=ECMINER_PARAMS['n_estimators'],
        max_depth=ECMINER_PARAMS['max_depth'],
        subsample=ECMINER_PARAMS['subsample'],
        learning_rate=ECMINER_PARAMS['learning_rate'],
        scale_pos_weight=scale_pos_weight,
        random_state=ECMINER_PARAMS['random_state'],
        eval_metric=ECMINER_PARAMS['eval_metric']
    )

    model.fit(X_train, y_train)

    # ========================================================================
    # Cross-Validation (5-Fold Stratified)
    # ========================================================================
    print(f"\n[2] 5-Fold Cross-Validation (Train set)...")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        X_fold_train = X_train[train_idx]
        y_fold_train = y_train[train_idx]
        X_fold_val = X_train[val_idx]
        y_fold_val = y_train[val_idx]

        # Recalculate scale_pos_weight for fold
        fold_scale = sum(y_fold_train == 0) / sum(y_fold_train == 1) if sum(y_fold_train == 1) > 0 else 1.0

        fold_model = xgb.XGBClassifier(
            n_estimators=ECMINER_PARAMS['n_estimators'],
            max_depth=ECMINER_PARAMS['max_depth'],
            subsample=ECMINER_PARAMS['subsample'],
            learning_rate=ECMINER_PARAMS['learning_rate'],
            scale_pos_weight=fold_scale,
            random_state=ECMINER_PARAMS['random_state'],
            eval_metric=ECMINER_PARAMS['eval_metric']
        )

        fold_model.fit(X_fold_train, y_fold_train)
        y_pred_proba = fold_model.predict_proba(X_fold_val)[:, 1]
        fold_auc = roc_auc_score(y_fold_val, y_pred_proba)
        cv_scores.append(fold_auc)

        print(f"  Fold {fold_idx + 1}: AUC={fold_auc:.4f}")

    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    print(f"\nCV Summary:")
    print(f"  Mean AUC: {cv_mean:.4f} ± {cv_std:.4f}")

    # ========================================================================
    # Test Set Evaluation
    # ========================================================================
    print(f"\n[3] Test Set Evaluation...")

    y_test_pred_proba = model.predict_proba(X_test)[:, 1]
    y_test_pred = model.predict(X_test)

    test_auc = roc_auc_score(y_test, y_test_pred_proba)

    print(f"\nTest AUC: {test_auc:.4f}")
    print(f"CV-Test Gap: {cv_mean - test_auc:.4f}")

    print(f"\nClassification Report:")
    print(classification_report(y_test, y_test_pred,
                                target_names=['Abnormal', 'Normal'],
                                digits=3))

    print(f"Confusion Matrix:")
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)
    print(f"\n              Predicted")
    print(f"              Abn   Norm")
    print(f"Actual  Abn   {cm[0,0]:3d}   {cm[0,1]:3d}   (Recall: {cm[0,0]/(cm[0,0]+cm[0,1]):.3f})")
    print(f"        Norm  {cm[1,0]:3d}   {cm[1,1]:3d}   (Recall: {cm[1,1]/(cm[1,0]+cm[1,1]):.3f})")

    # Feature importance
    print(f"\n[4] Feature Importance (Top 10)...")
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(feature_importance.head(10).to_string(index=False))

    # Save results
    results = {
        'feature_set': feature_set_name,
        'n_features': len(features),
        'cv_auc_mean': cv_mean,
        'cv_auc_std': cv_std,
        'test_auc': test_auc,
        'cv_test_gap': cv_mean - test_auc,
        'confusion_matrix': cm.tolist(),
        'feature_importance': feature_importance.to_dict('records')
    }

    return results


def main():
    """Main test pipeline"""

    print("=" * 80)
    print("ECMiner CSV Performance Test")
    print("Testing with ECMiner's 4 parameters only")
    print("=" * 80)

    # Load data
    print("\n[Loading Data]")
    train_df, val_df, test_df = load_data()

    # Test 18 features
    results_18 = train_and_evaluate(
        train_df, val_df, test_df,
        FEATURES_18,
        "Phase2 Step 3-4 (18 features)"
    )

    # Test 12 features
    results_12 = train_and_evaluate(
        train_df, val_df, test_df,
        FEATURES_12,
        "Phase3-1 Core (12 features)"
    )

    # ========================================================================
    # Comparison Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    comparison_df = pd.DataFrame([
        {
            'Model': 'Phase2 Step 3-4 (18 features)',
            'Features': results_18['n_features'],
            'CV AUC': f"{results_18['cv_auc_mean']:.4f} ± {results_18['cv_auc_std']:.4f}",
            'Test AUC': f"{results_18['test_auc']:.4f}",
            'CV-Test Gap': f"{results_18['cv_test_gap']:.4f}",
            'Test Recall (Abn)': f"{results_18['confusion_matrix'][0][0]/(results_18['confusion_matrix'][0][0]+results_18['confusion_matrix'][0][1]):.3f}",
            'Test Recall (Norm)': f"{results_18['confusion_matrix'][1][1]/(results_18['confusion_matrix'][1][0]+results_18['confusion_matrix'][1][1]):.3f}"
        },
        {
            'Model': 'Phase3-1 Core (12 features)',
            'Features': results_12['n_features'],
            'CV AUC': f"{results_12['cv_auc_mean']:.4f} ± {results_12['cv_auc_std']:.4f}",
            'Test AUC': f"{results_12['test_auc']:.4f}",
            'CV-Test Gap': f"{results_12['cv_test_gap']:.4f}",
            'Test Recall (Abn)': f"{results_12['confusion_matrix'][0][0]/(results_12['confusion_matrix'][0][0]+results_12['confusion_matrix'][0][1]):.3f}",
            'Test Recall (Norm)': f"{results_12['confusion_matrix'][1][1]/(results_12['confusion_matrix'][1][0]+results_12['confusion_matrix'][1][1]):.3f}"
        }
    ])

    print("\n")
    print(comparison_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("Key Findings:")
    print("=" * 80)

    diff_auc = results_12['test_auc'] - results_18['test_auc']
    print(f"1. Test AUC Difference (12 vs 18): {diff_auc:+.4f}")

    if abs(diff_auc) < 0.01:
        print("   → Similar performance, 12 features preferred (simpler)")
    elif diff_auc > 0:
        print("   → 12 features performs BETTER (less overfitting)")
    else:
        print("   → 18 features performs better (more information)")

    gap_18 = results_18['cv_test_gap']
    gap_12 = results_12['cv_test_gap']
    print(f"\n2. Overfitting Analysis:")
    print(f"   18 features CV-Test Gap: {gap_18:.4f}")
    print(f"   12 features CV-Test Gap: {gap_12:.4f}")

    if abs(gap_12) < abs(gap_18):
        print("   → 12 features shows LESS overfitting")
    else:
        print("   → 18 features shows less overfitting")

    print(f"\n3. ECMiner 4-Parameter Limitation:")
    print(f"   Using only: n_estimators, max_depth, subsample, learning_rate")
    print(f"   Missing: min_child_weight, reg_lambda, reg_alpha, colsample_bytree")
    print(f"   Impact: May not fully replicate Phase3-1 performance (AUC 0.797)")

    print("\n" + "=" * 80)
    print("Recommendation for ECMiner:")
    print("=" * 80)

    if results_12['test_auc'] >= 0.75:
        print("✅ 12 features with 4 ECMiner parameters achieves good performance")
        print(f"   Test AUC: {results_12['test_auc']:.4f}")
        print("   Proceed with ECMiner implementation")
    else:
        print("⚠️  Performance below target (0.75)")
        print("   Consider adding more XGBoost parameters to ECMiner")

    # Save summary
    summary_path = OUTPUT_DIR / "ecminer_test_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("ECMiner CSV Performance Test Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(comparison_df.to_string(index=False))
        f.write(f"\n\n18 features Test AUC: {results_18['test_auc']:.4f}\n")
        f.write(f"12 features Test AUC: {results_12['test_auc']:.4f}\n")
        f.write(f"Difference: {diff_auc:+.4f}\n")

    print(f"\n✓ Summary saved to: {summary_path}")

    return results_18, results_12


if __name__ == '__main__':
    results_18, results_12 = main()
