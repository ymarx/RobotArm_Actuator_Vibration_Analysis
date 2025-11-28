"""
ECMiner Stage1 전체 데이터 검증 스크립트
==========================================
목적:
1. 전체 57개 파일 처리 (~898 윈도우)
2. ECMiner 호환 XGBoost 파라미터로 성능 검증
3. Phase3 결과와 비교

ECMiner XGBoost 4가지 파라미터:
- 총 나무 수 (n_estimators)
- 최대 트리 깊이 (max_depth)
- 샘플링 비율 (subsample)
- 학습률 (learning_rate)
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
print("ECMiner Stage1 전체 데이터 검증")
print("="*80)
print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ==============================================================================
# 1. 전체 파일 리스트 생성
# ==============================================================================
print("\n[1] 전체 파일 리스트 생성 중...")

file_master = pd.read_parquet(DATA_DIR / "interim" / "file_master_v1.parquet")

# 사용 가능한 파일만 선택 (is_usable == True)
usable_files = file_master[file_master['is_usable'] == True].copy()

print(f"전체 파일 수: {len(file_master)}")
print(f"사용 가능 파일: {len(usable_files)}")

# 파일 존재 여부 확인
from pathlib import Path
existing_files = []
missing_files = []

for idx, row in usable_files.iterrows():
    fp = Path(row['file_path'])
    if fp.exists():
        existing_files.append(row)
    else:
        missing_files.append(row['file_id'])

print(f"존재하는 파일: {len(existing_files)}개")
if missing_files:
    print(f"⚠️ 누락된 파일: {len(missing_files)}개")
    print(f"  예시: {missing_files[:3]}")

# ecmData 형식으로 변환
ecmData = pd.DataFrame(existing_files)
ecmData = ecmData[['file_path', 'label_binary', 'split_set', 'file_id']].copy()
ecmData.columns = ['file_path', 'label', 'dataset_type', 'file_id']

print(f"\n파일 분포:")
print(ecmData.groupby(['dataset_type', 'label']).size())

# 저장
file_list_path = PROJECT_ROOT / "file_list_full.csv"
ecmData.to_csv(file_list_path, index=False)
print(f"\n저장 완료: {file_list_path}")

# ==============================================================================
# 2. Stage1 스크립트 실행 (전체 데이터 처리)
# ==============================================================================
print("\n[2] Stage1 스크립트 실행 중 (전체 데이터 처리)...")
print("  (이 과정은 수 분 소요될 수 있습니다)")

from ecminer_stage1_feature_extraction import build_feature_table_from_ecmdata

feature_table = build_feature_table_from_ecmdata(ecmData)

print(f"\n✅ Feature Table 생성 완료!")
print(f"  - 총 윈도우: {len(feature_table)}")
print(f"  - 컬럼 개수: {len(feature_table.columns)}")

# NaN 체크
nan_count = feature_table.isna().sum().sum()
nan_pct = 100 * nan_count / (feature_table.shape[0] * feature_table.shape[1])
print(f"  - NaN 개수: {nan_count} ({nan_pct:.2f}%)")

if nan_pct > 5:
    print("  ⚠️ Warning: NaN 비율이 5%를 초과합니다. 데이터 품질을 확인하세요.")

# 저장
feature_table_path = PROJECT_ROOT / "feature_table_full.csv"
feature_table.to_csv(feature_table_path, index=False)
print(f"\n저장 완료: {feature_table_path}")

# 데이터 분포 확인
print(f"\n데이터셋 분포:")
dist = feature_table.groupby(['dataset_type', 'label']).size()
print(dist)

# ==============================================================================
# 3. Stage1 추출 특성 기반 선택 (Phase3 개념 유지)
# ==============================================================================
print("\n[3] 특성 선택 중...")

# Stage1에서 추출된 Phase2 Step 3-4 특성 (18개)
stage1_phase2_features = [
    # 기본 9개
    'acc_Y_rms', 'acc_Y_peak', 'acc_Y_crest',
    'acc_Sum_rms', 'acc_Sum_peak', 'acc_Sum_crest',
    'Gyro_Y_rms', 'Gyro_Y_peak', 'Gyro_Y_crest',
    # Band RMS 9개
    'acc_Y_rms_low', 'acc_Y_rms_mid', 'acc_Y_rms_high',
    'acc_Sum_rms_low', 'acc_Sum_rms_mid', 'acc_Sum_rms_high',
    'Gyro_Y_rms_low', 'Gyro_Y_rms_mid', 'Gyro_Y_rms_high'
]

# Phase3-1 정확한 12개 특성 (ECMiner Filter 후)
# 기본 9개 + 핵심 Band RMS 3개
selected_features = [
    # 기본 9개 (Phase3-1과 동일)
    'acc_Y_rms', 'acc_Y_peak', 'acc_Y_crest',
    'acc_Sum_rms', 'acc_Sum_peak', 'acc_Sum_crest',
    'Gyro_Y_rms', 'Gyro_Y_peak', 'Gyro_Y_crest',
    # 핵심 Band RMS 3개 (Phase3-1 선택)
    'acc_Y_rms_high',   # 2.61x ratio (가장 높음)
    'Gyro_Y_rms_high',  # 2.08x ratio
    'Gyro_Y_rms_low'    # 2.09x ratio
]

print(f"\n[Phase2 Step 3-4 → Phase3-1 Filter]")
print(f"Stage1 추출 (Phase2): {len(stage1_phase2_features)}개")
print(f"ECMiner Filter 후 (Phase3-1): {len(selected_features)}개")

print(f"\n기본 특성 (9개) - Phase3-1과 동일:")
for f in selected_features[:9]:
    print(f"  - {f}")

print(f"\n핵심 Band RMS (3개) - Phase3-1 선택:")
for f in selected_features[9:]:
    print(f"  - {f}")

print(f"\nECMiner Filter에서 제거할 특성 (6개):")
removed_features = [f for f in stage1_phase2_features if f not in selected_features]
for f in sorted(removed_features):
    print(f"  ❌ {f}")

# 특성 존재 확인
missing_features = [f for f in selected_features if f not in feature_table.columns]
if missing_features:
    print(f"\n  ⚠️ Error: 누락된 특성 {missing_features}")
    print(f"\n실제 feature_table 컬럼:")
    feature_cols = [c for c in feature_table.columns if c.endswith(('_rms', '_peak', '_crest', '_kurtosis'))]
    for c in sorted(feature_cols):
        print(f"    - {c}")
    raise ValueError(f"필수 특성이 feature_table에 없습니다: {missing_features}")

print(f"\n✅ 모든 특성 확인 완료 - Phase3-1과 정확히 일치")

# ==============================================================================
# 4. Train/Val/Test 분할
# ==============================================================================
print("\n[4] 데이터 분할 중...")

# NaN 제거
feature_table_clean = feature_table.dropna(subset=selected_features)

train_df = feature_table_clean[feature_table_clean['dataset_type'] == 'train'].copy()
val_df = feature_table_clean[feature_table_clean['dataset_type'] == 'val'].copy()
test_df = feature_table_clean[feature_table_clean['dataset_type'] == 'test'].copy()

X_train = train_df[selected_features]
y_train = train_df['label']

X_val = val_df[selected_features]
y_val = val_df['label']

X_test = test_df[selected_features]
y_test = test_df['label']

print(f"\nTrain: {X_train.shape}")
print(f"  Normal: {(y_train==1).sum()}, Abnormal: {(y_train==0).sum()}")
print(f"Val: {X_val.shape}")
print(f"  Normal: {(y_val==1).sum()}, Abnormal: {(y_val==0).sum()}")
print(f"Test: {X_test.shape}")
print(f"  Normal: {(y_test==1).sum()}, Abnormal: {(y_test==0).sum()}")

# ==============================================================================
# 5. ECMiner 호환 XGBoost 학습 (4가지 파라미터만)
# ==============================================================================
print("\n[5] XGBoost 모델 학습 중...")

# ECMiner XGBoost 4가지 파라미터
# Phase3 값을 ECMiner 파라미터로 매핑
ecminer_params = {
    'n_estimators': 100,        # 총 나무 수 (Phase3: 100)
    'max_depth': 3,             # 최대 트리 깊이 (Phase3: 3, 과적합 방지)
    'subsample': 0.8,           # 샘플링 비율 (Phase3: 0.8)
    'learning_rate': 0.1,       # 학습률 (Phase3: 0.1)
}

# ECMiner에서 설정 불가능하지만 모델 성능에 필요한 추가 파라미터
additional_params = {
    'random_state': 42,
    'eval_metric': 'logloss',
    'scale_pos_weight': len(y_train[y_train==0]) / len(y_train[y_train==1])
}

# 전체 파라미터 (Phase3 정규화 파라미터는 ECMiner에서 조정 불가)
all_params = {**ecminer_params, **additional_params}

print(f"\n[ECMiner 조정 가능 파라미터]")
for key, value in ecminer_params.items():
    print(f"  {key}: {value}")

print(f"\n[추가 파라미터 (ECMiner 기본값)]")
for key, value in additional_params.items():
    print(f"  {key}: {value}")

# 모델 학습
model = xgb.XGBClassifier(**all_params)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

print("\n✅ 학습 완료!")

# ==============================================================================
# 6. 모델 평가 (Test Set)
# ==============================================================================
print("\n[6] 모델 평가 중...")

# 예측
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# 메트릭 계산
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
auc = roc_auc_score(y_test, y_pred_proba)

cm = confusion_matrix(y_test, y_pred)

print("\n" + "="*80)
print("Test Set 평가 결과")
print("="*80)
print(f"\nAccuracy:  {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall:    {rec:.3f}")
print(f"F1-Score:  {f1:.3f}")
print(f"AUC:       {auc:.3f}")

print(f"\nConfusion Matrix:")
print(cm)

print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Abnormal', 'Normal'], zero_division=0))

# 특성 중요도
feature_importance = pd.DataFrame({
    'feature': selected_features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n특성 중요도 (Top 5):")
print(feature_importance.head(5).to_string(index=False))

# ==============================================================================
# 7. Phase3 결과 비교
# ==============================================================================
print("\n[7] Phase3 결과와 비교 중...")

# Phase3 최종 결과 (phase3_1_xgboost_core_band_rms.py)
phase3_results = {
    'test_auc': 0.820,  # 프로젝트 종합 보고서에서 확인된 값
    'test_acc': None,   # 보고서에 없음
    'features': 12,
    'params': {
        'n_estimators': 100,
        'max_depth': 3,
        'subsample': 0.8,
        'learning_rate': 0.1,
        'min_child_weight': 5,  # ECMiner에서 설정 불가
        'reg_lambda': 5,         # ECMiner에서 설정 불가
        'reg_alpha': 1           # ECMiner에서 설정 불가
    }
}

ecminer_results = {
    'test_auc': auc,
    'test_acc': acc,
    'features': len(selected_features),
    'params': ecminer_params
}

print("\n" + "="*80)
print("Phase3 vs ECMiner Stage1 비교")
print("="*80)

print(f"\n[Phase3 최종 결과 (프로젝트 보고서)]")
print(f"  Test AUC: {phase3_results['test_auc']:.3f}")
print(f"  특성 개수: {phase3_results['features']}개")
print(f"  정규화: min_child_weight=5, reg_lambda=5, reg_alpha=1")

print(f"\n[ECMiner Stage1 결과 (현재 검증)]")
print(f"  Test AUC: {ecminer_results['test_auc']:.3f}")
print(f"  Test Acc: {ecminer_results['test_acc']:.3f}")
print(f"  특성 개수: {ecminer_results['features']}개")
print(f"  정규화: ECMiner 기본값 (조정 불가)")

auc_diff = ecminer_results['test_auc'] - phase3_results['test_auc']
print(f"\n[성능 차이]")
print(f"  AUC 차이: {auc_diff:+.3f}")

if abs(auc_diff) < 0.05:
    print(f"  ✅ 성능 유사 (차이 < 0.05)")
elif auc_diff < 0:
    print(f"  ⚠️ 성능 감소: ECMiner 정규화 파라미터 부족으로 인한 것으로 추정")
    print(f"     → Phase3는 min_child_weight=5, reg_lambda=5, reg_alpha=1 사용")
    print(f"     → ECMiner는 이 파라미터들을 조정할 수 없음")
else:
    print(f"  ✅ 성능 향상!")

# ==============================================================================
# 8. 결과 저장
# ==============================================================================
print("\n[8] 결과 저장 중...")

results = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'data': {
        'total_files': len(existing_files),
        'total_windows': len(feature_table_clean),
        'train_windows': len(train_df),
        'val_windows': len(val_df),
        'test_windows': len(test_df)
    },
    'features': selected_features,
    'ecminer_params': ecminer_params,
    'phase3_comparison': {
        'phase3_test_auc': phase3_results['test_auc'],
        'ecminer_test_auc': ecminer_results['test_auc'],
        'ecminer_test_acc': ecminer_results['test_acc'],
        'auc_difference': float(auc_diff)
    },
    'metrics': {
        'test_accuracy': float(acc),
        'test_precision': float(prec),
        'test_recall': float(rec),
        'test_f1': float(f1),
        'test_auc': float(auc)
    },
    'confusion_matrix': cm.tolist(),
    'feature_importance': feature_importance.to_dict('records')
}

results_path = PROJECT_ROOT / "ecminer_validation_results.json"
with open(results_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"저장 완료: {results_path}")

print("\n" + "="*80)
print("검증 완료!")
print("="*80)
print(f"종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
