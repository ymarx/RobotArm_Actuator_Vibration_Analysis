"""
ECMiner Stage1 스크립트 검증 테스트
===================================
목적:
1. ecminer_stage1_feature_extraction.py 스크립트 검증
2. Raw CSV → 윈도우 Feature Table 변환 확인
3. 추출된 특성 품질 검증
4. XGBoost 모델 학습/예측 테스트
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

print("=" * 80)
print("ECMiner Stage1 스크립트 검증 테스트")
print("=" * 80)

# ==============================================================================
# 1. 테스트용 파일 리스트 생성 (file_master에서 샘플 추출)
# ==============================================================================
print("\n[1] 테스트용 파일 리스트 생성 중...")

# file_master 로드
file_master = pd.read_parquet(PROJECT_ROOT / "data" / "interim" / "file_master_v1.parquet")

# Normal과 Abnormal 각각 샘플링
normal_files = file_master[file_master['is_normal'] == True].copy()
abnormal_files = file_master[file_master['is_normal'] == False].copy()

# 샘플 선택
test_files = []

# Normal: 200W_S03_CCW_R4 (time_split - will be split into train/test during window generation)
normal_file = normal_files[normal_files['file_id'] == '200W_S03_CCW_R4'].head(1)
test_files.append(normal_file)

# Abnormal Train: 1개
abnormal_train = abnormal_files[abnormal_files['split_set'] == 'train'].sample(1, random_state=42)
test_files.append(abnormal_train)

# Abnormal Test: 1개
abnormal_test = abnormal_files[abnormal_files['split_set'] == 'test'].sample(1, random_state=42)
test_files.append(abnormal_test)

# 병합
test_file_list = pd.concat(test_files, ignore_index=True)

# ecmData 형식으로 변환
# IMPORTANT: For time_split files, we pass 'time_split' to let the script handle splitting
ecmData = pd.DataFrame({
    'file_path': test_file_list['file_path'],
    'label': test_file_list['label_binary'],
    'dataset_type': test_file_list['split_set'],  # Keep original split_set
    'file_id': test_file_list['file_id']
})

print(f"\n테스트 파일 리스트 (총 {len(ecmData)}개):")
print(ecmData.to_string(index=False))

# CSV로 저장
test_file_list_path = PROJECT_ROOT / "file_list_test.csv"
ecmData.to_csv(test_file_list_path, index=False)
print(f"\n저장 완료: {test_file_list_path}")

# ==============================================================================
# 2. Stage1 스크립트 실행 (ecminer_stage1_feature_extraction.py)
# ==============================================================================
print("\n[2] Stage1 스크립트 실행 중...")

# 스크립트 import (모듈로 로드)
from ecminer_stage1_feature_extraction import build_feature_table_from_ecmdata

# 실행
try:
    feature_table = build_feature_table_from_ecmdata(ecmData)
    print(f"\n✅ Feature Table 생성 완료!")
    print(f"  - 총 윈도우: {len(feature_table)}")
    print(f"  - 컬럼 개수: {len(feature_table.columns)}")
    print(f"  - 특성 컬럼: {len([c for c in feature_table.columns if c not in ['file_id', 'window_idx', 'start_idx', 'end_idx', 'label', 'dataset_type', 'product', 'serial', 'condition', 'load']])}")

except Exception as e:
    print(f"\n❌ 오류 발생: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ==============================================================================
# 3. 추출된 특성 품질 검증
# ==============================================================================
print("\n[3] 추출된 특성 품질 검증 중...")

# 컬럼 확인
expected_features = [
    'acc_Y_rms', 'acc_X_rms', 'Gyro_Y_rms', 'Gyro_X_rms',
    'acc_Y_peak', 'acc_Sum_peak',
    'acc_Y_crest',
    'acc_Y_kurtosis', 'acc_Sum_kurtosis',
    'acc_Y_rms_low', 'acc_Y_rms_mid', 'acc_Y_rms_high',
    'acc_Sum_rms_low', 'acc_Sum_rms_mid', 'acc_Sum_rms_high',
    'Gyro_Y_rms_low', 'Gyro_Y_rms_mid', 'Gyro_Y_rms_high'
]

print("\n특성 존재 여부:")
for feat in expected_features:
    exists = feat in feature_table.columns
    symbol = "✅" if exists else "❌"
    print(f"  {symbol} {feat}")

# NaN 확인
print("\n\nNaN 개수:")
nan_counts = feature_table[expected_features].isna().sum()
for feat, count in nan_counts.items():
    if count > 0:
        print(f"  ⚠️ {feat}: {count}개")

# 기본 통계
print("\n\n기본 통계 (Train 데이터):")
train_data = feature_table[feature_table['dataset_type'] == 'train']
if len(train_data) > 0:
    stats = train_data[expected_features].describe()
    print(stats.T[['mean', 'std', 'min', 'max']].to_string())

# Label 분포
print("\n\nLabel 분포:")
label_dist = feature_table.groupby(['dataset_type', 'label']).size()
print(label_dist)

# 저장
output_path = PROJECT_ROOT / "feature_table_test.csv"
feature_table.to_csv(output_path, index=False)
print(f"\n저장 완료: {output_path}")

# ==============================================================================
# 4. XGBoost 모델 학습 및 예측 테스트
# ==============================================================================
print("\n[4] XGBoost 모델 학습 및 예측 테스트 중...")

# 필요한 라이브러리 import
try:
    import xgboost as xgb
    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
except ImportError as e:
    print(f"⚠️ 필요한 라이브러리가 설치되지 않음: {e}")
    print("  pip install xgboost scikit-learn")
    sys.exit(1)

# 특성 선택 (XGBoost v3)
features_v3 = [
    'acc_Y_rms', 'acc_X_rms', 'Gyro_Y_rms', 'Gyro_X_rms',
    'acc_Y_peak', 'acc_Sum_peak',
    'acc_Y_crest',
    'acc_Y_kurtosis', 'acc_Sum_kurtosis',
    'acc_Y_rms_high', 'Gyro_Y_rms_high', 'Gyro_Y_rms_low'
]

# Train/Test 분할
train_df = feature_table[feature_table['dataset_type'] == 'train'].copy()
test_df = feature_table[feature_table['dataset_type'] == 'test'].copy()

# NaN 제거
train_df = train_df.dropna(subset=features_v3)
test_df = test_df.dropna(subset=features_v3)

print(f"\nTrain 세트: {len(train_df)} 윈도우 (Normal: {(train_df['label']==1).sum()}, Abnormal: {(train_df['label']==0).sum()})")
print(f"Test 세트: {len(test_df)} 윈도우 (Normal: {(test_df['label']==1).sum()}, Abnormal: {(test_df['label']==0).sum()})")

if len(train_df) == 0 or len(test_df) == 0:
    print("\n⚠️ Train 또는 Test 데이터가 부족하여 모델 학습을 건너뜁니다.")
    sys.exit(0)

# 데이터 준비
X_train = train_df[features_v3]
y_train = train_df['label']

X_test = test_df[features_v3]
y_test = test_df['label']

# XGBoost 파라미터 (프로젝트 검증값)
params = {
    'max_depth': 3,
    'min_child_weight': 5,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_lambda': 5,
    'reg_alpha': 1,
    'random_state': 42,
    'eval_metric': 'logloss',
    'scale_pos_weight': len(y_train[y_train==0]) / len(y_train[y_train==1]) if len(y_train[y_train==1]) > 0 else 1.0
}

print(f"\nXGBoost 파라미터:")
for key, value in params.items():
    print(f"  {key}: {value}")

# 학습
print("\n모델 학습 중...")
model = xgb.XGBClassifier(**params)
model.fit(X_train, y_train, verbose=False)

print("✅ 학습 완료!")

# 예측
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Abnormal (class 0) 확률

# 평가
accuracy = accuracy_score(y_test, y_pred)

# AUC 계산 (클래스가 2개 이상 있어야 함)
if len(np.unique(y_test)) > 1:
    # y_test가 {0, 1}이고 y_pred_proba는 class 1의 확률
    # AUC를 위해서는 positive class (1)의 확률 사용
    auc = roc_auc_score(y_test, y_pred_proba)
else:
    auc = np.nan

print("\n" + "=" * 80)
print("모델 평가 결과")
print("=" * 80)
print(f"\nAccuracy: {accuracy:.3f}")
if not np.isnan(auc):
    print(f"AUC: {auc:.3f}")
else:
    print("AUC: N/A (Test 세트에 단일 클래스만 존재)")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nClassification Report:")
# Handle single-class case
unique_labels = np.unique(y_test)
if len(unique_labels) == 1:
    print(f"⚠️ Warning: Test set contains only class {unique_labels[0]}")
    if unique_labels[0] == 0:
        print(classification_report(y_test, y_pred, target_names=['Abnormal'], zero_division=0))
    else:
        print(classification_report(y_test, y_pred, target_names=['Normal'], zero_division=0))
else:
    print(classification_report(y_test, y_pred, target_names=['Abnormal', 'Normal'], zero_division=0))

# 특성 중요도
print("\n특성 중요도 (Top 5):")
feature_importance = pd.DataFrame({
    'feature': features_v3,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head().to_string(index=False))

# ==============================================================================
# 5. 최종 요약
# ==============================================================================
print("\n" + "=" * 80)
print("검증 요약")
print("=" * 80)

print(f"\n✅ Stage1 스크립트 검증 완료!")
print(f"  - 입력 파일: {len(ecmData)}개")
print(f"  - 출력 윈도우: {len(feature_table)}개")
print(f"  - 추출 특성: 18개")
print(f"  - NaN 비율: {feature_table[expected_features].isna().sum().sum() / (len(feature_table) * len(expected_features)) * 100:.1f}%")

print(f"\n✅ XGBoost 모델 테스트 완료!")
print(f"  - Train: {len(train_df)} 윈도우")
print(f"  - Test: {len(test_df)} 윈도우")
print(f"  - Accuracy: {accuracy:.3f}")
if not np.isnan(auc):
    print(f"  - AUC: {auc:.3f}")

print(f"\n📋 다음 단계:")
print(f"  1. 전체 데이터로 재실행 (file_list 확장)")
print(f"  2. ECMiner에 통합 (파이썬 연동 노드)")
print(f"  3. Stage2/3 노드 구성 (Filter, XGBoost)")

print("\n" + "=" * 80)
