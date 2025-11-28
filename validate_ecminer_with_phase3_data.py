"""
ECMiner 파라미터 검증 (Phase3 데이터 사용)
==========================================
목적:
- Phase3 원본 데이터를 사용하여 ECMiner 4가지 파라미터만으로 성능 검증
- Stage1 스크립트 문제와 무관하게 ECMiner XGBoost 성능 평가

비교:
1. Phase3 full params: n_estimators=100, max_depth=3, subsample=0.8, learning_rate=0.1,
                       min_child_weight=5, reg_lambda=5, reg_alpha=1
2. ECMiner params only: n_estimators=100, max_depth=3, subsample=0.8, learning_rate=0.1
"""

import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import json
from datetime import datetime

# 경로 설정
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

print("="*80)
print("ECMiner 파라미터 검증 (Phase3 데이터)")
print("="*80)
print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ==============================================================================
# 1. Phase3 데이터 로드
# ==============================================================================
print("\n[1] Phase3 데이터 로드 중...")

features_df = pd.read_parquet(DATA_DIR / "processed" / "features_combined_v2_with_band_rms.parquet")
features_clean = features_df.dropna()

print(f"총 샘플 수: {len(features_clean)}")
print(f"Train: {len(features_clean[features_clean['split_set']=='train'])}")
print(f"Val:   {len(features_clean[features_clean['split_set']=='val'])}")
print(f"Test:  {len(features_clean[features_clean['split_set']=='test'])}")

# ==============================================================================
# 2. Phase3 특성 선택 (동일하게)
# ==============================================================================
print("\n[2] 특성 선택 중...")

# Phase3 정확한 특성
base_features = [
    'acc_Y_rms',
    'acc_Y_peak',
    'acc_Y_crest',
    'acc_Sum_rms',
    'acc_Sum_peak',
    'acc_Sum_crest',
    'Gyro_Y_rms',
    'Gyro_Y_peak',
    'Gyro_Y_crest'
]

core_band_rms_features = [
    'acc_Y_rms_high',
    'Gyro_Y_rms_high',
    'Gyro_Y_rms_low'
]

selected_features = base_features + core_band_rms_features

print(f"선택된 특성 ({len(selected_features)}개):")
print(f"  - 기존 특성: {len(base_features)}개")
print(f"  - Band RMS: {len(core_band_rms_features)}개")

# ==============================================================================
# 3. Train/Val/Test 분할
# ==============================================================================
print("\n[3] 데이터 분할 중...")

train_df = features_clean[features_clean['split_set'] == 'train'].copy()
val_df = features_clean[features_clean['split_set'] == 'val'].copy()
test_df = features_clean[features_clean['split_set'] == 'test'].copy()

X_train = train_df[selected_features]
y_train = train_df['label_binary']

X_val = val_df[selected_features]
y_val = val_df['label_binary']

X_test = test_df[selected_features]
y_test = test_df['label_binary']

print(f"\nTrain: {X_train.shape}")
print(f"  Normal: {(y_train==1).sum()}, Abnormal: {(y_train==0).sum()}")
print(f"Val: {X_val.shape}")
print(f"  Normal: {(y_val==1).sum()}, Abnormal: {(y_val==0).sum()}")
print(f"Test: {X_test.shape}")
print(f"  Normal: {(y_test==1).sum()}, Abnormal: {(y_test==0).sum()}")

# ==============================================================================
# 4. 세 가지 XGBoost 설정 비교
# ==============================================================================
print("\n[4] 세 가지 XGBoost 설정으로 학습 및 비교...")

# 공통 파라미터
common_params = {
    'random_state': 42,
    'eval_metric': 'logloss',
    'scale_pos_weight': len(y_train[y_train==0]) / len(y_train[y_train==1])
}

# 설정 1: Phase3 Full (모든 정규화 포함)
phase3_full_params = {
    'n_estimators': 100,
    'max_depth': 3,
    'subsample': 0.8,
    'learning_rate': 0.1,
    'min_child_weight': 5,  # 정규화
    'reg_lambda': 5,         # L2 정규화
    'reg_alpha': 1,          # L1 정규화
    'colsample_bytree': 0.8,
    **common_params
}

# 설정 2: ECMiner Only (4가지 파라미터만)
ecminer_only_params = {
    'n_estimators': 100,
    'max_depth': 3,
    'subsample': 0.8,
    'learning_rate': 0.1,
    **common_params
}

# 설정 3: ECMiner + colsample (5가지)
ecminer_plus_params = {
    'n_estimators': 100,
    'max_depth': 3,
    'subsample': 0.8,
    'learning_rate': 0.1,
    'colsample_bytree': 0.8,
    **common_params
}

results = {}

for config_name, params in [
    ("Phase3 Full", phase3_full_params),
    ("ECMiner Only (4 params)", ecminer_only_params),
    ("ECMiner + colsample (5 params)", ecminer_plus_params)
]:
    print(f"\n{'='*80}")
    print(f"{config_name}")
    print(f"{'='*80}")

    # 파라미터 출력
    print(f"\n파라미터:")
    for key, value in params.items():
        if key not in ['random_state', 'eval_metric', 'scale_pos_weight']:
            print(f"  {key}: {value}")

    # 모델 학습
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # Test 예측
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # 메트릭 계산
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_proba)

    cm = confusion_matrix(y_test, y_pred)

    print(f"\nTest Set 평가:")
    print(f"  Accuracy:  {acc:.3f}")
    print(f"  Precision: {prec:.3f}")
    print(f"  Recall:    {rec:.3f}")
    print(f"  F1-Score:  {f1:.3f}")
    print(f"  AUC:       {auc:.3f}")

    print(f"\nConfusion Matrix:")
    print(cm)

    # 결과 저장
    results[config_name] = {
        'params': {k: v for k, v in params.items() if k not in ['random_state', 'eval_metric', 'scale_pos_weight']},
        'metrics': {
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1),
            'auc': float(auc)
        },
        'confusion_matrix': cm.tolist()
    }

# ==============================================================================
# 5. 비교 요약
# ==============================================================================
print("\n" + "="*80)
print("비교 요약")
print("="*80)

print(f"\n{'설정':<30} {'AUC':>8} {'Accuracy':>10} {'F1':>8}")
print("-" * 60)
for config_name, result in results.items():
    metrics = result['metrics']
    print(f"{config_name:<30} {metrics['auc']:>8.3f} {metrics['accuracy']:>10.3f} {metrics['f1']:>8.3f}")

# AUC 차이 계산
phase3_auc = results["Phase3 Full"]['metrics']['auc']
ecminer_auc = results["ECMiner Only (4 params)"]['metrics']['auc']
ecminer_plus_auc = results["ECMiner + colsample (5 params)"]['metrics']['auc']

auc_loss_ecminer = ecminer_auc - phase3_auc
auc_loss_ecminer_plus = ecminer_plus_auc - phase3_auc

print(f"\n성능 차이 분석:")
print(f"  Phase3 Full AUC: {phase3_auc:.3f}")
print(f"  ECMiner Only AUC: {ecminer_auc:.3f} (차이: {auc_loss_ecminer:+.3f})")
print(f"  ECMiner + colsample AUC: {ecminer_plus_auc:.3f} (차이: {auc_loss_ecminer_plus:+.3f})")

if abs(auc_loss_ecminer) < 0.05:
    print(f"\n✅ ECMiner 4가지 파라미터만으로도 유사한 성능 달성 (차이 < 0.05)")
elif auc_loss_ecminer < -0.05:
    print(f"\n⚠️ ECMiner 파라미터 제한으로 인한 성능 감소")
    print(f"   정규화 파라미터 부족: min_child_weight, reg_lambda, reg_alpha")
    print(f"   → 과적합 방지 능력 감소")
else:
    print(f"\n✅ ECMiner 파라미터가 더 좋은 성능!")

# ==============================================================================
# 6. 결과 저장
# ==============================================================================
print(f"\n[6] 결과 저장 중...")

output = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'data': {
        'total_samples': len(features_clean),
        'train': len(train_df),
        'val': len(val_df),
        'test': len(test_df)
    },
    'features': selected_features,
    'results': results,
    'analysis': {
        'phase3_auc': float(phase3_auc),
        'ecminer_only_auc': float(ecminer_auc),
        'ecminer_plus_auc': float(ecminer_plus_auc),
        'auc_loss_ecminer_only': float(auc_loss_ecminer),
        'auc_loss_ecminer_plus': float(auc_loss_ecminer_plus)
    }
}

results_path = PROJECT_ROOT / "ecminer_param_comparison_results.json"
with open(results_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"저장 완료: {results_path}")

print("\n" + "="*80)
print("검증 완료!")
print("="*80)
print(f"종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
