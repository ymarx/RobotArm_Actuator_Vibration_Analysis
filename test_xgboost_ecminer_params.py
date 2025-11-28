"""
ECMiner 출력 CSV로 XGBoost 성능 테스트
ECMiner 4개 파라미터만 사용하여 모델 학습 및 평가
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import xgboost as xgb

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).parent

print("=" * 80)
print("XGBoost 성능 테스트 (ECMiner 4개 파라미터)")
print("=" * 80)

# ============================================================================
# 1. 데이터 로드
# ============================================================================

print("\n[1/5] 데이터 로드")
csv_path = PROJECT_ROOT / "ecminer_output_test.csv"

if not csv_path.exists():
    print(f"❌ 출력 CSV가 없습니다: {csv_path}")
    print("먼저 test_ecminer_package.py를 실행하세요.")
    exit(1)

df = pd.read_csv(csv_path)
print(f"  - 총 윈도우: {len(df)}")
print(f"  - Train: {len(df[df['dataset_type']=='train'])}")
print(f"  - Val: {len(df[df['dataset_type']=='val'])}")
print(f"  - Test: {len(df[df['dataset_type']=='test'])}")

# ============================================================================
# 2. 특징 및 라벨 분리
# ============================================================================

print("\n[2/5] 특징 및 라벨 분리")

# 메타데이터 컬럼
meta_cols = ['window_id', 'file_id', 'dataset_type', 'label_binary', 'product', 'sample', 'direction']

# 전체 특징 (18개)
all_features = [col for col in df.columns if col not in meta_cols]
print(f"  - 전체 특징: {len(all_features)}개")

# 12개 핵심 특징
core_features = [
    'acc_Y_rms', 'acc_Y_peak', 'acc_Y_crest',
    'acc_Sum_rms', 'acc_Sum_peak', 'acc_Sum_crest',
    'Gyro_Y_rms', 'Gyro_Y_peak', 'Gyro_Y_crest',
    'acc_Y_rms_low', 'acc_Sum_rms_low', 'Gyro_Y_rms_low'
]
print(f"  - 핵심 특징: {len(core_features)}개")

# Train/Val/Test 분할
train_df = df[df['dataset_type'] == 'train'].copy()
val_df = df[df['dataset_type'] == 'val'].copy()
test_df = df[df['dataset_type'] == 'test'].copy()

# 중복 제거 (window_id 기준)
print(f"\n  중복 제거:")
print(f"  - Train: {len(train_df)} → ", end="")
train_df = train_df.drop_duplicates(subset=['window_id'], keep='first')
print(f"{len(train_df)}")

print(f"  - Val: {len(val_df)} → ", end="")
val_df = val_df.drop_duplicates(subset=['window_id'], keep='first')
print(f"{len(val_df)}")

print(f"  - Test: {len(test_df)} → ", end="")
test_df = test_df.drop_duplicates(subset=['window_id'], keep='first')
print(f"{len(test_df)}")

# ============================================================================
# 3. ECMiner 파라미터 설정
# ============================================================================

print("\n[3/5] ECMiner 4개 파라미터 설정")

ECMINER_PARAMS = {
    'n_estimators': 100,
    'max_depth': 3,
    'subsample': 0.8,
    'learning_rate': 0.1,
    # 고정 파라미터 (ECMiner에서 조정 불가)
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'random_state': 42,
    'use_label_encoder': False
}

print(f"  ECMiner 조정 가능 파라미터:")
print(f"    - n_estimators: {ECMINER_PARAMS['n_estimators']}")
print(f"    - max_depth: {ECMINER_PARAMS['max_depth']}")
print(f"    - subsample: {ECMINER_PARAMS['subsample']}")
print(f"    - learning_rate: {ECMINER_PARAMS['learning_rate']}")

# 클래스 불균형 처리
n_normal_train = (train_df['label_binary'] == 1).sum()
n_abnormal_train = (train_df['label_binary'] == 0).sum()
scale_pos_weight = n_abnormal_train / n_normal_train if n_normal_train > 0 else 1.0

ECMINER_PARAMS['scale_pos_weight'] = scale_pos_weight
print(f"    - scale_pos_weight: {scale_pos_weight:.2f} (자동 계산)")

# ============================================================================
# 4. 모델 학습 및 평가 (18개 특징)
# ============================================================================

print("\n[4/5] 모델 학습 (18개 전체 특징)")

X_train_18 = train_df[all_features]
y_train = train_df['label_binary']

X_val_18 = val_df[all_features]
y_val = val_df['label_binary']

X_test_18 = test_df[all_features]
y_test = test_df['label_binary']

# 모델 학습
model_18 = xgb.XGBClassifier(**ECMINER_PARAMS)
model_18.fit(
    X_train_18, y_train,
    eval_set=[(X_val_18, y_val)],
    verbose=False
)

# 예측
y_pred_proba_train_18 = model_18.predict_proba(X_train_18)[:, 1]
y_pred_proba_val_18 = model_18.predict_proba(X_val_18)[:, 1]
y_pred_proba_test_18 = model_18.predict_proba(X_test_18)[:, 1]

# AUC 계산
auc_train_18 = roc_auc_score(y_train, y_pred_proba_train_18)
auc_val_18 = roc_auc_score(y_val, y_pred_proba_val_18)
auc_test_18 = roc_auc_score(y_test, y_pred_proba_test_18)

print(f"  - Train AUC: {auc_train_18:.4f}")
print(f"  - Val AUC: {auc_val_18:.4f}")
print(f"  - Test AUC: {auc_test_18:.4f}")

# ============================================================================
# 5. 모델 학습 및 평가 (12개 핵심 특징)
# ============================================================================

print("\n[5/5] 모델 학습 (12개 핵심 특징)")

X_train_12 = train_df[core_features]
X_val_12 = val_df[core_features]
X_test_12 = test_df[core_features]

# 모델 학습
model_12 = xgb.XGBClassifier(**ECMINER_PARAMS)
model_12.fit(
    X_train_12, y_train,
    eval_set=[(X_val_12, y_val)],
    verbose=False
)

# 예측
y_pred_proba_train_12 = model_12.predict_proba(X_train_12)[:, 1]
y_pred_proba_val_12 = model_12.predict_proba(X_val_12)[:, 1]
y_pred_proba_test_12 = model_12.predict_proba(X_test_12)[:, 1]

# AUC 계산
auc_train_12 = roc_auc_score(y_train, y_pred_proba_train_12)
auc_val_12 = roc_auc_score(y_val, y_pred_proba_val_12)
auc_test_12 = roc_auc_score(y_test, y_pred_proba_test_12)

print(f"  - Train AUC: {auc_train_12:.4f}")
print(f"  - Val AUC: {auc_val_12:.4f}")
print(f"  - Test AUC: {auc_test_12:.4f}")

# ============================================================================
# 6. 결과 요약
# ============================================================================

print("\n" + "=" * 80)
print("성능 요약")
print("=" * 80)

print("\n데이터 구성 (중복 제거 후):")
print(f"  Train: {len(train_df)} (Normal: {n_normal_train}, Abnormal: {n_abnormal_train})")
print(f"  Val:   {len(val_df)} (Normal: {(val_df['label_binary']==1).sum()}, Abnormal: {(val_df['label_binary']==0).sum()})")
print(f"  Test:  {len(test_df)} (Normal: {(test_df['label_binary']==1).sum()}, Abnormal: {(test_df['label_binary']==0).sum()})")

print("\nECMiner 파라미터 (4개):")
print(f"  n_estimators: {ECMINER_PARAMS['n_estimators']}")
print(f"  max_depth: {ECMINER_PARAMS['max_depth']}")
print(f"  subsample: {ECMINER_PARAMS['subsample']}")
print(f"  learning_rate: {ECMINER_PARAMS['learning_rate']}")

print("\n성능 비교:")
print(f"┌────────────┬─────────┬─────────┬─────────┐")
print(f"│ 특징 세트   │  Train  │   Val   │  Test   │")
print(f"├────────────┼─────────┼─────────┼─────────┤")
print(f"│ 18개 (전체) │ {auc_train_18:.4f}  │ {auc_val_18:.4f}  │ {auc_test_18:.4f}  │")
print(f"│ 12개 (핵심) │ {auc_train_12:.4f}  │ {auc_val_12:.4f}  │ {auc_test_12:.4f}  │")
print(f"└────────────┴─────────┴─────────┴─────────┘")

# Test 성능 상세 (18개 특징)
print(f"\nTest 성능 상세 (18개 특징):")
y_pred_test_18 = model_18.predict(X_test_18)
print(classification_report(y_test, y_pred_test_18, target_names=['Abnormal', 'Normal']))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred_test_18)
print(f"  [[TN={cm[0,0]}, FP={cm[0,1]}],")
print(f"   [FN={cm[1,0]}, TP={cm[1,1]}]]")

print("\n" + "=" * 80)
print("✅ 테스트 완료")
print("=" * 80)
