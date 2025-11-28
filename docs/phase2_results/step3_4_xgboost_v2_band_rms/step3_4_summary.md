# Step 3-4: XGBoost v2 with Band RMS Features - Summary

## Goal
Re-train XGBoost with expanded feature set (18 features = 9 original + 9 Band RMS)
to improve Recall while maintaining Precision at threshold=0.5.

## Feature Set v2 (18 features)

### Original Features (9)
- RMS core: acc_Y_rms, acc_X_rms, Gyro_Y_rms, Gyro_X_rms
- Peak: acc_Y_peak, acc_Sum_peak
- Crest: acc_Y_crest
- Kurtosis: acc_Y_kurtosis, acc_Sum_kurtosis

### New Band RMS Features (9)
- acc_Y: low/mid/high (1-10Hz, 10-50Hz, 50-150Hz)
- acc_Sum: low/mid/high
- Gyro_Y: low/mid/high

## Training Configuration
- **Algorithm**: XGBoost with binary:logistic
- **Cross-Validation**: 5-Fold StratifiedKFold
- **scale_pos_weight**: 2.61 (Abnormal/Normal ratio)
- **Hyperparameters**: max_depth=4, learning_rate=0.01, subsample=0.8
- **Early Stopping**: 50 rounds on validation AUC

## Results

### Cross-Validation (Train Set)
- **Mean AUC**: 0.9398 ± 0.0289
- **Mean Recall (Abnormal)**: 0.875 ± 0.081
- **Mean Precision (Abnormal)**: 0.949 ± 0.022

### Test Set Performance (threshold=0.5)
- **AUC**: 0.8159
- **Abnormal Class**:
  - Precision: 0.982
  - Recall: 0.567
  - F1: 0.719
- **Normal Class**:
  - Precision: 0.236
  - Recall: 0.929
  - F1: 0.377

## Comparison with Baseline Models

| Model | Features | AUC | Precision (Abn) | Recall (Abn) | F1 (Abn) | Recall (Norm) |
|-------|----------|-----|-----------------|--------------|----------|---------------|
| XGBoost v1 (Step 2) | 9 | 0.708 | 0.931 | 0.691 | 0.793 | 0.643 |
| Hybrid (Step 3-2) | 9 + rule | - | 0.929 | 0.804 | 0.862 | 0.571 |
| **XGBoost v2 (Step 3-4)** | **18** | **0.816** | **0.982** | **0.567** | **0.719** | **0.929** |

### Improvements
- **vs XGBoost v1**: Recall (Abn) -0.124 (-17.9%)
- **vs Hybrid**: Recall (Abn) -0.237 (-29.5%)

## Top 5 Most Important Features
| feature          |   importance |
|:-----------------|-------------:|
| Gyro_Y_rms_low   |      41.3806 |
| acc_Sum_rms_high |      33.5357 |
| acc_Y_peak       |      31.5186 |
| acc_Y_rms        |      30.2261 |
| acc_Sum_peak     |      28.991  |

## Key Findings

1. **Band RMS Impact**: Negative - decreased Recall

2. **Feature Importance**:
   - Top feature: Gyro_Y_rms_low (41.4)
   - Band RMS features in top 10: 5

3. **Overfitting Check**:
   - CV AUC: 0.9398
   - Test AUC: 0.8159
   - Gap: 0.1239

4. **Model Choice Recommendation**:
   Hybrid still superior for Recall priority

## Files Generated
- `xgboost_v2_final_model.json`: Trained model
- `cv_results_v2.csv`: Cross-validation detailed results
- `feature_importance_v2.csv`: Feature importance rankings
- `model_comparison.csv`: Comparison with baseline models
- `step3_4_summary.md`: This summary report

## Next Steps
1. **If XGBoost v2 shows improvement**:
   - Apply Hybrid Rule v2 (XGBoost v2 + Band RMS rules)
   - Test product-specific thresholds (100W vs 200W)

2. **If improvement is modest**:
   - Investigate feature selection (focus on high-importance Band RMS)
   - Consider threshold tuning for v2 model
   - Prepare for Autoencoder/anomaly detection approach

3. **Production Deployment**:
   - Document final model choice (v2 vs Hybrid)
   - Create inference pipeline with feature extraction
   - Plan monitoring and retraining strategy
