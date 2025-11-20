"""
Phase 3-2: Enhanced Hybrid Rule (v2)
======================================
XGBoost v3 (Phase 3-1) 기반 Hybrid 규칙 개선

구조:
- Base: XGBoost v3 (threshold=0.5)
- Rule 1: acc_Y_rms > 0.15 (기존 Hybrid v1)
- Rule 2: acc_Y_rms_high > T_high (신규 Band RMS)
- Product별 threshold 분리 (100W vs 200W)

절차:
1. Validation에서 threshold 튜닝
2. 제품별 최적 threshold 선택
3. Test 최종 평가
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# 경로 설정
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "docs" / "phase3_results" / "phase3_2_hybrid_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("PHASE 3-2: Enhanced Hybrid Rule (v2)")
print("="*80)

# ==============================================================================
# 1. 데이터 로드 및 XGBoost v3 모델 로드
# ==============================================================================
print("\n[1] 데이터 및 모델 로드 중...")

# 데이터 로드
features_df = pd.read_parquet(DATA_DIR / "processed" / "features_combined_v2_with_band_rms.parquet")
features_clean = features_df.dropna()

# 분할
train_df = features_clean[features_clean['split_set'] == 'train'].copy()
val_df = features_clean[features_clean['split_set'] == 'val'].copy()
test_df = features_clean[features_clean['split_set'] == 'test'].copy()

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# XGBoost v3 모델 로드
model_path = PROJECT_ROOT / "docs" / "phase3_results" / "phase3_1_xgboost_core" / "xgboost_v3_core_band_rms.json"
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(model_path)

# Phase 3-1에서 사용한 특성
selected_features = [
    'acc_Y_rms', 'acc_Y_peak', 'acc_Y_crest',
    'acc_Sum_rms', 'acc_Sum_peak', 'acc_Sum_crest',
    'Gyro_Y_rms', 'Gyro_Y_peak', 'Gyro_Y_crest',
    'acc_Y_rms_high', 'Gyro_Y_rms_high', 'Gyro_Y_rms_low'
]

print(f"XGBoost v3 모델 로드 완료")
print(f"사용 특성 ({len(selected_features)}개): {selected_features}")

# ==============================================================================
# 2. Validation에서 acc_Y_rms_high 분포 확인
# ==============================================================================
print("\n[2] Validation acc_Y_rms_high 분포 확인...")

val_normal = val_df[val_df['label_binary'] == 1]['acc_Y_rms_high']
val_abnormal = val_df[val_df['label_binary'] == 0]['acc_Y_rms_high']

print(f"\nValidation acc_Y_rms_high 통계:")
print(f"  Normal (n={len(val_normal)}):")
print(f"    Mean: {val_normal.mean():.4f}, Std: {val_normal.std():.4f}")
print(f"    Min: {val_normal.min():.4f}, Max: {val_normal.max():.4f}")
print(f"    Median: {val_normal.median():.4f}")
print(f"  Abnormal (n={len(val_abnormal)}):")
print(f"    Mean: {val_abnormal.mean():.4f}, Std: {val_abnormal.std():.4f}")
print(f"    Min: {val_abnormal.min():.4f}, Max: {val_abnormal.max():.4f}")
print(f"    Median: {val_abnormal.median():.4f}")
print(f"  Ratio (Abnormal/Normal): {val_abnormal.mean()/val_normal.mean():.2f}x")

# 제품별 분포
print(f"\n제품별 acc_Y_rms_high 분포:")
for product in ['100W', '200W']:
    val_prod = val_df[val_df['product'] == product]
    if len(val_prod) == 0:
        continue

    prod_normal = val_prod[val_prod['label_binary'] == 1]['acc_Y_rms_high']
    prod_abnormal = val_prod[val_prod['label_binary'] == 0]['acc_Y_rms_high']

    print(f"\n  {product}:")
    print(f"    Normal (n={len(prod_normal)}): Mean={prod_normal.mean():.4f}")
    print(f"    Abnormal (n={len(prod_abnormal)}): Mean={prod_abnormal.mean():.4f}")
    if len(prod_normal) > 0:
        print(f"    Ratio: {prod_abnormal.mean()/prod_normal.mean():.2f}x")

# ==============================================================================
# 3. XGBoost v3 예측 (Validation & Test)
# ==============================================================================
print("\n[3] XGBoost v3 예측 생성 중...")

# Validation 예측
X_val = val_df[selected_features]
y_val = val_df['label_binary']
val_df['xgb_prob'] = xgb_model.predict_proba(X_val)[:, 1]
val_df['xgb_pred'] = (val_df['xgb_prob'] >= 0.5).astype(int)

# Test 예측
X_test = test_df[selected_features]
y_test = test_df['label_binary']
test_df['xgb_prob'] = xgb_model.predict_proba(X_test)[:, 1]
test_df['xgb_pred'] = (test_df['xgb_prob'] >= 0.5).astype(int)

print("XGBoost v3 예측 완료")

# ==============================================================================
# 4. Validation에서 Threshold 튜닝
# ==============================================================================
print("\n[4] Validation에서 acc_Y_rms_high threshold 튜닝 중...")

# Threshold 후보
threshold_candidates = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]

# 기존 Hybrid v1 threshold
rms_full_threshold = 0.15

# Validation 결과 저장
val_results = []

print(f"\nThreshold 후보: {threshold_candidates}")
print(f"\n{'='*80}")
print("Validation Threshold 튜닝 결과:")
print(f"{'='*80}")

for t_high in threshold_candidates:
    # Hybrid v2 예측
    # pred = (XGB v3) OR (acc_Y_rms > 0.15) OR (acc_Y_rms_high > t_high)
    pred_hybrid_v2 = (
        (val_df['xgb_pred'] == 0) |  # XGBoost v3 Abnormal 예측
        (val_df['acc_Y_rms'] > rms_full_threshold) |  # 기존 RMS 규칙
        (val_df['acc_Y_rms_high'] > t_high)  # 신규 High-band 규칙
    )

    # 예측 변환 (True → 0 (Abnormal), False → 1 (Normal))
    y_pred_hybrid_v2 = (~pred_hybrid_v2).astype(int)

    # 성능 계산
    acc = accuracy_score(y_val, y_pred_hybrid_v2)
    prec_abnormal = precision_score(y_val, y_pred_hybrid_v2, pos_label=0, zero_division=0)
    rec_abnormal = recall_score(y_val, y_pred_hybrid_v2, pos_label=0, zero_division=0)
    f1_abnormal = f1_score(y_val, y_pred_hybrid_v2, pos_label=0, zero_division=0)

    prec_normal = precision_score(y_val, y_pred_hybrid_v2, pos_label=1, zero_division=0)
    rec_normal = recall_score(y_val, y_pred_hybrid_v2, pos_label=1, zero_division=0)
    f1_normal = f1_score(y_val, y_pred_hybrid_v2, pos_label=1, zero_division=0)

    # Hybrid v1 대비 추가 탐지 계산
    pred_hybrid_v1 = (
        (val_df['xgb_pred'] == 0) |
        (val_df['acc_Y_rms'] > rms_full_threshold)
    )

    # v1에서 놓친 것을 v2가 잡은 케이스
    new_tp = (
        (~pred_hybrid_v1) &  # v1은 Normal 예측
        (pred_hybrid_v2) &   # v2는 Abnormal 예측
        (y_val == 0)          # 실제 Abnormal
    ).sum()

    # v2에서 새로 FP가 된 케이스
    new_fp = (
        (~pred_hybrid_v1) &  # v1은 Normal 예측
        (pred_hybrid_v2) &   # v2는 Abnormal 예측
        (y_val == 1)          # 실제 Normal
    ).sum()

    print(f"\nThreshold = {t_high:.2f}:")
    print(f"  Abnormal - Precision: {prec_abnormal:.3f}, Recall: {rec_abnormal:.3f}, F1: {f1_abnormal:.3f}")
    print(f"  Normal   - Precision: {prec_normal:.3f}, Recall: {rec_normal:.3f}, F1: {f1_normal:.3f}")
    print(f"  Accuracy: {acc:.3f}")
    print(f"  New TP (v1→v2): {new_tp}, New FP (v1→v2): {new_fp}")

    val_results.append({
        'threshold': t_high,
        'accuracy': acc,
        'abnormal_precision': prec_abnormal,
        'abnormal_recall': rec_abnormal,
        'abnormal_f1': f1_abnormal,
        'normal_precision': prec_normal,
        'normal_recall': rec_normal,
        'normal_f1': f1_normal,
        'new_tp': new_tp,
        'new_fp': new_fp
    })

val_results_df = pd.DataFrame(val_results)
val_results_df.to_csv(OUTPUT_DIR / "val_threshold_tuning.csv", index=False)

print(f"\n{'='*80}")
print("Validation 튜닝 결과 저장 완료")

# ==============================================================================
# 5. 제품별 Threshold 분석
# ==============================================================================
print(f"\n[5] 제품별 Threshold 분석 중...")

product_results = []

for product in ['100W', '200W']:
    print(f"\n{'='*80}")
    print(f"{product} Threshold 분석:")
    print(f"{'='*80}")

    val_prod = val_df[val_df['product'] == product].copy()

    if len(val_prod) == 0:
        print(f"  {product} 데이터 없음")
        continue

    y_val_prod = val_prod['label_binary']

    for t_high in threshold_candidates:
        pred_hybrid_v2 = (
            (val_prod['xgb_pred'] == 0) |
            (val_prod['acc_Y_rms'] > rms_full_threshold) |
            (val_prod['acc_Y_rms_high'] > t_high)
        )

        y_pred_hybrid_v2 = (~pred_hybrid_v2).astype(int)

        prec_abnormal = precision_score(y_val_prod, y_pred_hybrid_v2, pos_label=0, zero_division=0)
        rec_abnormal = recall_score(y_val_prod, y_pred_hybrid_v2, pos_label=0, zero_division=0)
        rec_normal = recall_score(y_val_prod, y_pred_hybrid_v2, pos_label=1, zero_division=0)

        print(f"  T={t_high:.2f}: Abnormal Rec={rec_abnormal:.3f}, Prec={prec_abnormal:.3f}, Normal Rec={rec_normal:.3f}")

        product_results.append({
            'product': product,
            'threshold': t_high,
            'abnormal_precision': prec_abnormal,
            'abnormal_recall': rec_abnormal,
            'normal_recall': rec_normal
        })

product_results_df = pd.DataFrame(product_results)
product_results_df.to_csv(OUTPUT_DIR / "product_threshold_analysis.csv", index=False)

# ==============================================================================
# 6. 최적 Threshold 선택
# ==============================================================================
print(f"\n[6] 최적 Threshold 선택...")

# 선택 기준:
# 1. Abnormal Recall >= 0.80 (기존 유지)
# 2. Abnormal Precision >= 0.85
# 3. Normal Recall >= 0.60

filtered_results = val_results_df[
    (val_results_df['abnormal_recall'] >= 0.80) &
    (val_results_df['abnormal_precision'] >= 0.85) &
    (val_results_df['normal_recall'] >= 0.60)
]

if len(filtered_results) > 0:
    # F1 최대화
    best_idx = filtered_results['abnormal_f1'].idxmax()
    best_threshold = val_results_df.loc[best_idx, 'threshold']
    print(f"\n✅ 최적 Threshold 선택: {best_threshold:.2f}")
    print(f"   Abnormal Recall: {val_results_df.loc[best_idx, 'abnormal_recall']:.3f}")
    print(f"   Abnormal Precision: {val_results_df.loc[best_idx, 'abnormal_precision']:.3f}")
    print(f"   Normal Recall: {val_results_df.loc[best_idx, 'normal_recall']:.3f}")
else:
    # 기준 만족 못하면 가장 보수적인 threshold (높은 값)
    best_threshold = 0.15
    print(f"\n⚠️ 기준 만족하는 threshold 없음. 보수적 선택: {best_threshold:.2f}")

# ==============================================================================
# 7. Test 세트 최종 평가
# ==============================================================================
print(f"\n[7] Test 세트 최종 평가 중...")

# 전략 1: XGBoost v3 단독 (threshold=0.5)
pred_xgb_only = (test_df['xgb_pred'] == 0)
y_pred_xgb = (~pred_xgb_only).astype(int)

# 전략 2: Hybrid v1 (XGB v3 + acc_Y_rms > 0.15)
pred_hybrid_v1 = (
    (test_df['xgb_pred'] == 0) |
    (test_df['acc_Y_rms'] > rms_full_threshold)
)
y_pred_hybrid_v1 = (~pred_hybrid_v1).astype(int)

# 전략 3: Hybrid v2 (v1 + acc_Y_rms_high > best_threshold)
pred_hybrid_v2 = (
    (test_df['xgb_pred'] == 0) |
    (test_df['acc_Y_rms'] > rms_full_threshold) |
    (test_df['acc_Y_rms_high'] > best_threshold)
)
y_pred_hybrid_v2 = (~pred_hybrid_v2).astype(int)

# 성능 계산 함수
def evaluate_strategy(y_true, y_pred, strategy_name):
    acc = accuracy_score(y_true, y_pred)
    prec_abnormal = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    rec_abnormal = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    f1_abnormal = f1_score(y_true, y_pred, pos_label=0, zero_division=0)

    prec_normal = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec_normal = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_normal = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    cm = confusion_matrix(y_true, y_pred)

    return {
        'strategy': strategy_name,
        'accuracy': acc,
        'abnormal_precision': prec_abnormal,
        'abnormal_recall': rec_abnormal,
        'abnormal_f1': f1_abnormal,
        'normal_precision': prec_normal,
        'normal_recall': rec_normal,
        'normal_f1': f1_normal,
        'confusion_matrix': cm
    }

# 세 전략 평가
results = []
results.append(evaluate_strategy(y_test, y_pred_xgb, 'XGBoost v3 (단독)'))
results.append(evaluate_strategy(y_test, y_pred_hybrid_v1, 'Hybrid v1 (v3 + RMS)'))
results.append(evaluate_strategy(y_test, y_pred_hybrid_v2, f'Hybrid v2 (v1 + High-band T={best_threshold:.2f})'))

# 결과 출력
print(f"\n{'='*80}")
print("TEST 세트 전략별 성능 비교:")
print(f"{'='*80}")

for result in results:
    print(f"\n▶ {result['strategy']}")
    print(f"  Accuracy: {result['accuracy']:.3f}")
    print(f"  Abnormal - Precision: {result['abnormal_precision']:.3f}, Recall: {result['abnormal_recall']:.3f}, F1: {result['abnormal_f1']:.3f}")
    print(f"  Normal   - Precision: {result['normal_precision']:.3f}, Recall: {result['normal_recall']:.3f}, F1: {result['normal_f1']:.3f}")
    print(f"  Confusion Matrix:")
    print(f"    {result['confusion_matrix']}")

# 결과 DataFrame
test_comparison = pd.DataFrame([
    {
        'Strategy': r['strategy'],
        'Accuracy': r['accuracy'],
        'Abnormal_Precision': r['abnormal_precision'],
        'Abnormal_Recall': r['abnormal_recall'],
        'Abnormal_F1': r['abnormal_f1'],
        'Normal_Precision': r['normal_precision'],
        'Normal_Recall': r['normal_recall'],
        'Normal_F1': r['normal_f1']
    }
    for r in results
])

test_comparison.to_csv(OUTPUT_DIR / "test_strategy_comparison.csv", index=False)

# ==============================================================================
# 8. 제품별 Test 성능
# ==============================================================================
print(f"\n[8] 제품별 Test 성능 분석...")

product_test_results = []

for product in ['100W', '200W']:
    test_prod = test_df[test_df['product'] == product].copy()

    if len(test_prod) == 0:
        continue

    y_test_prod = test_prod['label_binary']

    print(f"\n{'='*80}")
    print(f"{product} Test 성능:")
    print(f"{'='*80}")

    # 세 전략 적용
    pred_xgb = (~(test_prod['xgb_pred'] == 0)).astype(int)

    pred_v1 = (~((test_prod['xgb_pred'] == 0) | (test_prod['acc_Y_rms'] > rms_full_threshold))).astype(int)

    pred_v2 = (~(
        (test_prod['xgb_pred'] == 0) |
        (test_prod['acc_Y_rms'] > rms_full_threshold) |
        (test_prod['acc_Y_rms_high'] > best_threshold)
    )).astype(int)

    for strategy_name, y_pred in [
        ('XGBoost v3', pred_xgb),
        ('Hybrid v1', pred_v1),
        ('Hybrid v2', pred_v2)
    ]:
        rec_abnormal = recall_score(y_test_prod, y_pred, pos_label=0, zero_division=0)
        rec_normal = recall_score(y_test_prod, y_pred, pos_label=1, zero_division=0)

        print(f"  {strategy_name}: Abnormal Rec={rec_abnormal:.3f}, Normal Rec={rec_normal:.3f}")

        product_test_results.append({
            'product': product,
            'strategy': strategy_name,
            'abnormal_recall': rec_abnormal,
            'normal_recall': rec_normal
        })

product_test_df = pd.DataFrame(product_test_results)
product_test_df.to_csv(OUTPUT_DIR / "product_test_results.csv", index=False)

# ==============================================================================
# 9. 설정 및 권장사항 저장
# ==============================================================================
print(f"\n[9] 결과 저장 중...")

config = {
    'best_threshold_acc_Y_rms_high': float(best_threshold),
    'rms_full_threshold': float(rms_full_threshold),
    'base_model': 'XGBoost v3 (Phase 3-1)',
    'strategies': {
        'strategy_1_conservative': {
            'name': 'XGBoost v3 단독',
            'rule': 'xgb_prob >= 0.5',
            'use_case': '기본 운영 모드'
        },
        'strategy_2_moderate': {
            'name': 'Hybrid v1',
            'rule': '(xgb_prob >= 0.5) OR (acc_Y_rms > 0.15)',
            'use_case': '불량 탐지 민감도 중간'
        },
        'strategy_3_sensitive': {
            'name': 'Hybrid v2',
            'rule': f'(xgb_prob >= 0.5) OR (acc_Y_rms > 0.15) OR (acc_Y_rms_high > {best_threshold})',
            'use_case': '불량 탐지 최대 민감도 (Recall 우선)'
        }
    },
    'validation_results': val_results_df.to_dict('records'),
    'test_results': test_comparison.to_dict('records')
}

with open(OUTPUT_DIR / "phase3_2_config.json", 'w', encoding='utf-8') as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

print(f"\n결과 저장 완료:")
print(f"  - val_threshold_tuning.csv")
print(f"  - product_threshold_analysis.csv")
print(f"  - test_strategy_comparison.csv")
print(f"  - product_test_results.csv")
print(f"  - phase3_2_config.json")

print("\n" + "="*80)
print("PHASE 3-2 완료")
print("="*80)
