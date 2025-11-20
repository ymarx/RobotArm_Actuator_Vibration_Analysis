# Phase 3: Model Refinement and Optimization Strategy

## Executive Summary

Phase 2 (Steps 1-4) revealed a critical insight: **adding all Band RMS features (18 total) caused severe overfitting**, resulting in worse Test Recall (0.567) compared to the simpler baseline (0.691) and Hybrid approach (0.804).

**Phase 3 Goal**: Refine the modeling approach using **selective Band RMS features, proper cross-validation, and enhanced regularization** to achieve robust performance.

---

## Problem Diagnosis: What Went Wrong in Step 3-4?

### Observed Results

| Model | Features | Test AUC | Precision (Abn) | **Recall (Abn)** | Gap (CV-Test AUC) |
|-------|----------|----------|-----------------|------------------|-------------------|
| **XGBoost v1** (Step 2) | 9 | 0.708 | 0.931 | **0.691** | ~0.07 |
| **Hybrid** (Step 3-2) | 9 + rule | - | 0.929 | **0.804** | - |
| **XGBoost v2** (Step 3-4) | 18 | 0.811 | 0.948 | **0.567** ⚠️ | **0.186** ⚠️ |

### Root Causes

#### 1. **Severe Overfitting**
- **CV AUC**: 0.9968 (near-perfect)
- **Test AUC**: 0.8115
- **Gap**: 0.186 (extremely high)
- **Symptom**: Model memorized training patterns instead of learning generalizable features

#### 2. **Feature Overload**
- **18 features** (9 original + 9 Band RMS) for **686 training samples**
- **High-dimensional curse**: Too many features relative to sample size
- **Feature redundancy**: Many Band RMS features are highly correlated (e.g., acc_Sum_rms_* vs acc_Y_rms_*)

#### 3. **Improper Cross-Validation**
- **Current**: `StratifiedKFold(n_splits=5)` on window level
- **Problem**: Windows from the same file_id split across folds
- **Result**: Model sees similar patterns in both train and validation folds → inflated CV scores

#### 4. **Conservative Decision Boundary**
- Model learned to **prioritize Precision over Recall**
- At threshold=0.5, classifies most ambiguous cases as Normal
- **Normal Recall**: 0.643 → 0.786 (increased)
- **Abnormal Recall**: 0.691 → 0.567 (decreased)

#### 5. **Feature Importance Mismatch**
- **Physically meaningful**: `acc_Y_rms_high` (2.61x discrimination ratio)
- **Model importance rank**: Not in top 5
- **Top features**: `acc_Sum_rms_mid` (likely overfitting artifact)

---

## Phase 3 Strategy: Three-Pronged Approach

### Phase 3-1: Refined XGBoost with Core Band RMS Features

**Objective**: Train a lean, robust model using only high-value Band RMS features with proper validation.

#### Feature Selection (12-13 features total)

**Keep: Original Core (9 features)**
- RMS: acc_Y_rms, acc_X_rms, Gyro_Y_rms, Gyro_X_rms
- Peak: acc_Y_peak, acc_Sum_peak
- Crest: acc_Y_crest
- Kurtosis: acc_Y_kurtosis, acc_Sum_kurtosis

**Add: Core Band RMS (3-4 features)**
- ✅ `acc_Y_rms_high` (50-150Hz) - **2.61x ratio, physically meaningful**
- ✅ `Gyro_Y_rms_high` (50-150Hz) - **2.08x ratio, complementary sensor**
- ✅ `Gyro_Y_rms_low` (1-10Hz) - **2.09x ratio, structural vibration**
- ⚠️ `acc_Y_rms_low` (optional) - **1.68x ratio, useful for 100W**

**Exclude: Redundant/Overfitting Band RMS**
- ❌ `acc_Sum_rms_*` (all) - highly correlated with acc_Y, adds noise
- ❌ `acc_Y_rms_mid` - lower discrimination, overfitting risk
- ❌ `Gyro_Y_rms_mid` - lower priority

#### Cross-Validation Fix

**Replace**: `StratifiedKFold` → **`StratifiedGroupKFold`**

```python
from sklearn.model_selection import StratifiedGroupKFold

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
groups = train_df['file_id']  # Group by file_id

for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train, groups)):
    # Train with file-level separation
```

**Expected Impact**:
- CV AUC will drop from 0.997 to ~0.85-0.90 (more realistic)
- CV-Test gap will reduce from 0.186 to <0.10
- Hyperparameter tuning will be more honest

#### Stronger Regularization

**Current Parameters**:
```python
max_depth=4
learning_rate=0.01
subsample=0.8
colsample_bytree=0.8
```

**Phase 3-1 Parameters**:
```python
max_depth=3              # Reduce from 4
min_child_weight=5       # Add constraint (was default=1)
learning_rate=0.01       # Keep
subsample=0.7            # Reduce from 0.8
colsample_bytree=0.7     # Reduce from 0.8
reg_lambda=5             # L2 regularization (was default=1)
reg_alpha=1              # L1 regularization (was default=0)
gamma=1                  # Minimum loss reduction (was default=0)
```

**Expected Impact**:
- Prevent deep, overfitted trees
- Force simpler decision rules
- Better generalization to Test set

#### Success Criteria

- **Test Recall (Abn)**: ≥ 0.70 (better than v1's 0.691)
- **Test Precision (Abn)**: ≥ 0.90 (maintain quality)
- **CV-Test AUC Gap**: < 0.10 (reduced overfitting)
- **Normal Recall**: ≥ 0.55 (balanced detection)

---

### Phase 3-2: Enhanced Hybrid Rule with Band RMS

**Objective**: Augment the proven Hybrid approach (Step 3-2) with targeted Band RMS rules.

#### Strategy A: Add High-Frequency Rule to Existing Hybrid

**Current Hybrid** (Step 3-2):
```python
pred_xgb = (xgb_prob_v1 >= 0.5)
pred_rms_full = (acc_Y_rms > 0.15)
pred_hybrid = pred_xgb | pred_rms_full
```

**Phase 3-2 Enhanced Hybrid**:
```python
pred_xgb = (xgb_prob_best >= 0.5)  # Use best model (v1 or v2')
pred_rms_full = (acc_Y_rms > 0.15)
pred_rms_high = (acc_Y_rms_high > T_high)  # New: high-freq rule

pred_hybrid_v2 = pred_xgb | pred_rms_full | pred_rms_high
```

#### Threshold Tuning for `T_high`

**Process**:
1. Analyze `acc_Y_rms_high` distribution on **Validation set**
2. Test thresholds: 0.05, 0.10, 0.15, 0.20, 0.25
3. Select threshold that:
   - Increases Recall vs current Hybrid (0.804)
   - Maintains Precision ≥ 0.85
   - Adds 3-5+ True Positives on Test

**Product-Specific Tuning** (optional):
- **200W**: Lower threshold (e.g., 0.10) - high-freq very discriminative (3.21x ratio)
- **100W**: Higher threshold (e.g., 0.20) or skip - weaker signal (1.17x ratio)

#### Strategy B: Product-Specific Hybrid Rules

**100W Model**:
```python
pred_hybrid_100w = pred_xgb_v1 | (acc_Y_rms > 0.15)
# No high-freq rule (weak discrimination)
```

**200W Model**:
```python
pred_hybrid_200w = pred_xgb_v1 | (acc_Y_rms > 0.15) | (acc_Y_rms_high > 0.10)
# Add high-freq rule (strong discrimination)
```

#### Success Criteria

- **Test Recall (Abn)**: ≥ 0.82 (improve from 0.804)
- **Test Precision (Abn)**: ≥ 0.85 (minor drop acceptable)
- **Added TP**: 3-5+ cases caught by high-freq rule
- **Product benefit**: Especially effective for 200W boundary cases

---

### Phase 3-3: Data Leakage Audit

**Objective**: Comprehensive review of entire pipeline to ensure no train/test contamination.

#### Audit Checklist

##### ✅ **Already Verified (Expected Safe)**

1. **Train/Val/Test Splitting**
   - File-based: Different file_id in each set
   - Time-based: Temporal splits within files (100W_S00, 200W_S03)
   - Location: `src/preprocess/segment.py`, `data/interim/splits/`

2. **Feature Engineering**
   - Computed independently per window
   - No cross-window statistics
   - Location: `src/features/time_domain.py`

3. **Threshold Tuning**
   - Step 1: RMS threshold on Validation
   - Step 3-1: XGBoost threshold on Validation
   - Test set used **only for final evaluation**

##### ⚠️ **To Verify**

1. **Cross-Validation Splitting**
   - **Issue**: `StratifiedKFold` may split same file across folds
   - **Fix**: Use `StratifiedGroupKFold` with `file_id` groups
   - **Location**: All `step*_xgboost*.py` scripts

2. **Feature Normalization** (if any)
   - **Check**: Are scalers fit on train-only or train+test?
   - **Current**: No normalization used (tree-based models)
   - **Status**: ✅ Safe

3. **Hyperparameter Tuning**
   - **Check**: Were params tuned looking at Test performance?
   - **Current**: Params fixed across experiments
   - **Status**: ✅ Safe (but CV scores were misleading)

4. **Band RMS Extraction**
   - **Check**: Are sampling rates estimated from test data?
   - **Location**: `scripts/step3_3_add_band_rms.py`
   - **Status**: Sampling rate from metadata (same for all), ✅ Safe

##### 🔍 **Detailed Review Locations**

```
To Audit:
├── src/preprocess/
│   ├── segment.py          # Window splitting logic
│   └── quality.py          # Quality checks (train/test independent?)
├── scripts/
│   ├── step2_xgboost_baseline.py     # CV: StratifiedKFold ⚠️
│   ├── step3_1_threshold_tuning.py    # Threshold: Validation-based ✅
│   ├── step3_2_hybrid_rule.py         # Rule: Validation-tuned ✅
│   └── step3_4_xgboost_v2_*.py       # CV: StratifiedKFold ⚠️
└── data/interim/splits/
    └── split_v1.json       # Split definitions
```

#### Action Items

1. **Run data leakage detection**:
   ```python
   # Check for file_id overlap
   train_files = set(train_df['file_id'])
   val_files = set(val_df['file_id'])
   test_files = set(test_df['file_id'])

   assert len(train_files & val_files) == 0
   assert len(train_files & test_files) == 0
   assert len(val_files & test_files) == 0
   ```

2. **Verify temporal boundaries** (for time-split files):
   ```python
   # For 100W_S00, 200W_S03
   # Check that train windows < val windows < test windows in time
   ```

3. **Document all assumptions** in Phase 3 report

---

## Phase 3 Execution Plan

### Phase 3-1: XGBoost v2' (Core Band RMS + Robust CV)

**Script**: `phase3/phase3_1_xgboost_v2_core_band_rms.py`

**Steps**:
1. Define 12-13 feature set (original 9 + core 3-4 Band RMS)
2. Implement `StratifiedGroupKFold` with `file_id` groups
3. Apply strong regularization parameters
4. Train with 5-fold CV
5. Evaluate on Test at threshold=0.5
6. Optional: Tune threshold on Validation
7. Compare with v1 and Hybrid baselines

**Output**:
- `claudedocs/phase3_results/phase3_1_core_band_rms/`
  - Model: `xgboost_v2_core_model.json`
  - Results: `cv_results_grouped.csv`, `test_results.csv`
  - Comparison: `phase3_1_vs_baselines.csv`
  - Report: `phase3_1_summary.md`

### Phase 3-2: Hybrid v2 with Band RMS Rules

**Script**: `phase3/phase3_2_hybrid_v2_with_band_rms.py`

**Steps**:
1. Load best XGBoost model (v1 or v2')
2. Test `acc_Y_rms_high` thresholds on Validation
3. Implement enhanced Hybrid rule
4. Evaluate on Test set
5. Analyze added TP cases
6. Optional: Product-specific rules (100W vs 200W)

**Output**:
- `claudedocs/phase3_results/phase3_2_hybrid_v2/`
  - Results: `hybrid_v2_comparison.csv`
  - Analysis: `added_tp_analysis.csv`, `hybrid_v2_scatter.png`
  - Report: `phase3_2_summary.md`

### Phase 3-3: Data Leakage Audit

**Script**: `phase3/phase3_3_data_leakage_audit.py`

**Steps**:
1. Check file_id overlaps across splits
2. Verify temporal boundaries for time-split files
3. Review all CV implementations
4. Audit feature engineering pipeline
5. Document findings

**Output**:
- `claudedocs/phase3_results/phase3_3_audit/`
  - Checks: `leakage_checks.csv`
  - Report: `data_integrity_audit.md`

### Phase 3 Final Report

**Document**: `claudedocs/PHASE3_FINAL_REPORT.md`

**Contents**:
1. Problem diagnosis (Step 3-4 analysis)
2. Phase 3 strategy and rationale
3. Results from Phase 3-1 and 3-2
4. Data integrity audit findings
5. Final model recommendation
6. Next steps (Phase 4: Production deployment or Autoencoder)

---

## Expected Outcomes

### Best Case
- **Phase 3-1 XGBoost v2'**: Test Recall 0.73-0.77, CV-Test gap < 0.10
- **Phase 3-2 Hybrid v2**: Test Recall 0.82-0.85, Precision > 0.85
- **Recommended Model**: Hybrid v2 for production

### Moderate Case
- **Phase 3-1 XGBoost v2'**: Comparable to v1 (Recall ~0.70)
- **Phase 3-2 Hybrid v2**: Slight improvement (Recall 0.81-0.82)
- **Recommended Model**: Hybrid v2 or Hybrid v1

### Conservative Case
- **Phase 3-1 XGBoost v2'**: No significant improvement
- **Phase 3-2 Hybrid v2**: Minimal benefit from high-freq rule
- **Recommended Model**: Keep Hybrid v1 (Step 3-2)
- **Band RMS Use**: EDA/interpretation only, prepare for Autoencoder

---

## Next Phase Preview

### Phase 4 Options (After Phase 3)

**Option A: Production Deployment**
- Finalize chosen model (Hybrid v2 or v1)
- Create inference pipeline
- Document for operations team

**Option B: Autoencoder / Anomaly Detection**
- Train unsupervised model on Normal data
- Use Band RMS features as input
- Detect novel patterns not in training labels
- Combine with supervised Hybrid model

**Option C: Product-Specific Models**
- Separate 100W and 200W models
- Optimize each for product characteristics
- May improve overall performance despite smaller datasets

---

## Summary

Phase 3 addresses the critical overfitting issues discovered in Step 3-4 through:
1. **Selective feature engineering** (core Band RMS only)
2. **Proper cross-validation** (GroupKFold)
3. **Strong regularization** (prevent memorization)
4. **Targeted rule enhancement** (high-freq threshold)
5. **Data integrity audit** (eliminate leakage concerns)

**Success Metric**: Achieve Test Recall ≥ 0.80 with Precision ≥ 0.85 using a robust, generalizable model.
