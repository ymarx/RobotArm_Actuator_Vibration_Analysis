# Phase 3-2: Enhanced Hybrid Rule (v2) 결과 보고서

## 📋 개요

**목표**: XGBoost v3 (Phase 3-1) 기반 Hybrid 규칙 개선 및 운영 모드 정의

**접근법**:
- Base: XGBoost v3 (threshold=0.5)
- Rule 1: acc_Y_rms > 0.15 (기존 Hybrid v1)
- Rule 2: acc_Y_rms_high > T_high (신규 Band RMS 규칙)
- Validation 기준 threshold 튜닝

**핵심 발견**: ⚠️ **Hybrid v2는 Test 성능 개선 효과 없음** → XGBoost v3 단독 또는 Hybrid v1 사용 권장

---

## 🔍 주요 발견사항

### 1. Validation vs Test 성능 괴리

**Validation 결과 (T=0.05)**:
- Abnormal Recall: **0.989** (매우 높음)
- Abnormal Precision: 0.978
- Normal Recall: 0.500
- **New TP (v1→v2): 7개** (추가 탐지)

**Test 결과 (T=0.15)**:
- Hybrid v1과 **완전 동일**
- **New TP: 0개** (추가 탐지 없음)

**결론**: Validation에서의 개선이 Test로 일반화되지 않음

### 2. acc_Y_rms_high 분포 분석

#### Validation 분포

```
전체:
  Normal (n=4):    Mean=0.0817, Median=0.0781, Range=[0.046, 0.125]
  Abnormal (n=88): Mean=0.1311, Median=0.0866, Range=[0.044, 0.596]
  Ratio: 1.60x

100W (Validation):
  Normal (n=2):    Mean=0.0470
  Abnormal (n=41): Mean=0.0570
  Ratio: 1.21x  ← 판별력 약함

200W (Validation):
  Normal (n=2):    Mean=0.1164
  Abnormal (n=47): Mean=0.1958
  Ratio: 1.68x  ← 판별력 있음
```

**문제점**:
- Validation Normal 샘플이 **4개뿐** → 통계적 신뢰도 매우 낮음
- 100W는 acc_Y_rms_high 비율이 1.21x로 거의 차이 없음
- 200W는 1.68x로 차이 있지만, Normal이 2개뿐이라 불안정

### 3. Test 세트 전략별 성능 비교

| 전략 | Abnormal Recall | Abnormal Precision | Normal Recall | Normal Precision | Accuracy |
|------|----------------|-------------------|---------------|-----------------|----------|
| **XGBoost v3 (단독)** | **0.804** | **0.951** | **0.714** | **0.345** | **0.793** |
| Hybrid v1 (v3 + RMS) | **0.814** | 0.940 | 0.643 | 0.333 | 0.793 |
| Hybrid v2 (v1 + High-band) | 0.814 | 0.940 | 0.643 | 0.333 | 0.793 |

**핵심 인사이트**:
1. **Hybrid v1 = Hybrid v2** (Test에서 완전 동일)
   - acc_Y_rms_high > 0.15 규칙이 Test에서 **전혀 작동하지 않음**

2. **XGBoost v3 vs Hybrid v1 비교**:
   - Abnormal Recall: 0.804 → 0.814 (+0.010, +1%)
   - Normal Recall: 0.714 → 0.643 (**-0.071, -10%**)
   - **Trade-off**: Abnormal 1% 개선 vs Normal 10% 하락

3. **권장**: XGBoost v3 단독 사용
   - Normal Recall 0.714가 **가장 높음**
   - Abnormal Precision 0.951이 **가장 높음**
   - 균형잡힌 성능

### 4. 제품별 Test 성능 (심각한 문제 발견)

#### 100W Test 성능

| 전략 | Abnormal Recall | Normal Recall |
|------|----------------|---------------|
| XGBoost v3 | 0.575 | **0.750** |
| Hybrid v1 | 0.575 | 0.750 |
| Hybrid v2 | 0.575 | 0.750 |

- 모든 전략이 **완전 동일**
- Abnormal Recall이 0.575로 **매우 낮음** (전체 0.804 대비)

#### 200W Test 성능 ⚠️ 심각

| 전략 | Abnormal Recall | Normal Recall |
|------|----------------|---------------|
| XGBoost v3 | 0.965 | **0.500** |
| Hybrid v1 | 0.982 | **0.000** ❌ |
| Hybrid v2 | 0.982 | **0.000** ❌ |

**문제점**:
- **Hybrid v1/v2는 200W Normal을 하나도 못 잡음** (Recall=0.000)
- acc_Y_rms > 0.15 규칙이 200W에서 **과도하게 민감**
- XGBoost v3만 Normal Recall 0.500 유지

---

## 📊 Threshold 튜닝 분석

### Validation 튜닝 결과

| Threshold | Abnormal Recall | Abnormal Precision | Normal Recall | New TP (v1→v2) |
|-----------|----------------|-------------------|---------------|----------------|
| **0.05** | **0.989** ✅ | 0.978 | 0.500 | **7개** |
| 0.08~0.30 | 0.909 | 0.976 | 0.500 | 0개 |

**발견**:
- T=0.05만 New TP 7개 추가 탐지
- T=0.08 이상은 모두 동일 (추가 효과 없음)
- 하지만 **Validation에서만** 효과 있고, Test에서는 효과 없음

### 최적 Threshold 선택 실패

**선택 기준**:
- Abnormal Recall >= 0.80
- Abnormal Precision >= 0.85
- Normal Recall >= 0.60

**결과**: ⚠️ 기준을 만족하는 threshold 없음
- T=0.05: Normal Recall 0.500 (< 0.60)
- T=0.08+: Abnormal Recall 0.909 (기준은 충족하나 New TP 0개)

**최종**: 보수적으로 T=0.15 선택 (하지만 Test에서 효과 없음)

---

## ⚠️ Phase 3-2의 한계점

### 1. Validation 샘플 부족의 영향

**현황**:
- Validation Normal: **4개** (전체 92개 중 4.3%)
- Validation Abnormal: 88개

**문제**:
- Normal 4개로는 통계적 유의미한 threshold 튜닝 불가능
- Validation에서 좋은 결과 → Test 일반화 실패
- 과적합 위험 (Validation 특정 샘플에 맞춰짐)

### 2. acc_Y_rms_high의 한계

**분석 결과**:
- Phase 3-3 EDA에서 **높은 판별력**(2.61x 비율) 보였으나,
- 실제 Hybrid 규칙으로는 **추가 탐지 효과 없음**

**이유**:
1. **XGBoost v3가 이미 학습함**: acc_Y_rms_high가 특성으로 포함되어 있어, 모델이 이미 활용
2. **임계값 규칙의 한계**: 고정 threshold는 XGBoost의 비선형 결정 경계보다 단순함
3. **데이터 분포**: Test와 Validation 분포 차이로 임계값이 일반화되지 않음

### 3. 제품별 규칙의 필요성 재검토

**100W**:
- acc_Y_rms_high 비율 1.21x로 판별력 매우 약함
- Hybrid 규칙 추가 효과 없음

**200W**:
- acc_Y_rms_high 비율 1.68x로 판별력 있으나,
- **acc_Y_rms > 0.15 규칙이 이미 과도하게 민감** (Normal Recall=0)
- 추가 규칙은 불필요 (오히려 악화)

---

## 🎯 최종 권장사항

### 운영 모드 정의

#### 모드 1: 균형 모드 (권장 기본 설정) ✅

```
전략: XGBoost v3 단독 (threshold=0.5)
규칙: xgb_prob >= 0.5

성능:
- Abnormal Recall: 0.804
- Abnormal Precision: 0.951
- Normal Recall: 0.714 (최고)
- Accuracy: 0.793

적용 대상: 일반 운영 환경
장점: Normal/Abnormal 균형, 높은 Precision
```

#### 모드 2: 불량 민감 모드 (선택 사항)

```
전략: Hybrid v1 (v3 + RMS 규칙)
규칙: (xgb_prob >= 0.5) OR (acc_Y_rms > 0.15)

성능:
- Abnormal Recall: 0.814 (+1%)
- Abnormal Precision: 0.940
- Normal Recall: 0.643 (-10%)
- Accuracy: 0.793

적용 대상: 불량 탐지가 매우 중요한 환경
단점: Normal을 10% 더 많이 불량으로 오판
제약: 200W에서 Normal Recall=0 (사용 불가)
```

#### 모드 3: Hybrid v2 ❌ 비권장

```
전략: Hybrid v2 (v1 + High-band)
결과: Test에서 Hybrid v1과 완전 동일 (추가 효과 없음)
결론: 사용 불필요
```

### 제품별 권장 전략

| 제품 | 권장 전략 | 이유 |
|------|----------|------|
| **100W** | XGBoost v3 단독 | Normal Recall 0.750 유지, Hybrid 효과 없음 |
| **200W** | **XGBoost v3 단독** | Hybrid는 Normal Recall=0으로 사용 불가 |

**결론**: 제품 구분 없이 **XGBoost v3 단독 사용**이 최적

---

## 📈 Phase 3 전체 성과 요약

### Phase 3-1 vs Phase 3-2 비교

| 단계 | 모델 | Test Abnormal Recall | Test Normal Recall | 평가 |
|------|------|---------------------|-------------------|------|
| **Phase 3-1** | **XGBoost v3** | **0.804** | **0.714** | **최고 성능** ✅ |
| Phase 3-2 | Hybrid v1 | 0.814 (+1%) | 0.643 (-10%) | Trade-off 불리 |
| Phase 3-2 | Hybrid v2 | 0.814 | 0.643 | v1과 동일 (효과 없음) |

**최종 선택**: **Phase 3-1 XGBoost v3 단독 사용**

### Phase 전체 흐름 정리

```
Phase 2 Step 2 (Baseline)
  └─ Test AUC: 0.811, Normal Recall: 0.691
     ↓
Phase 2 Step 3-4 (18개 특성)
  └─ 과적합 발생 (CV AUC 0.997, Test AUC 0.811)
  └─ Normal Recall: 0.567 (성능 하락) ❌
     ↓
Phase 3-0 (데이터 감사)
  └─ 누수 확인: 없음 ✅
     ↓
Phase 3-1 (핵심 Band RMS + GroupKFold)
  └─ Test AUC: 0.820 (+0.009)
  └─ Normal Recall: 0.714 (+0.023) ✅
  └─ 과적합 해결 ✅
     ↓
Phase 3-2 (Hybrid v2)
  └─ Validation 효과: 있음
  └─ Test 효과: 없음 (Phase 3-1과 동일 또는 악화)
  └─ 결론: Phase 3-1 사용 ✅
```

---

## 🔬 기술적 분석

### 왜 Hybrid v2가 실패했는가?

#### 1. XGBoost가 이미 최적화됨

```python
# XGBoost v3 특성에 이미 포함
selected_features = [
    'acc_Y_rms',        # Hybrid v1 규칙
    'acc_Y_rms_high',   # Hybrid v2 규칙
    # ... 기타 특성
]
```

- XGBoost는 비선형 결정 경계로 **이미 acc_Y_rms_high를 최적 활용**
- 단순 임계값 규칙(> 0.15)은 XGBoost보다 덜 정교함

#### 2. Validation 과적합

- Normal 4개로 threshold 결정 → Test 분포와 불일치
- Validation T=0.05에서 New TP 7개 → Test T=0.15에서 New TP 0개

#### 3. 규칙의 단순함

```python
# Hybrid v2 규칙
if acc_Y_rms_high > 0.15:  # 단순 임계값
    return Abnormal

# XGBoost 내부 (예시)
if (acc_Y_rms_high > 0.12) AND (Gyro_Y_rms_high < 0.08) AND ...:
    return Abnormal
```

- XGBoost는 **다변수 조건부 규칙** 학습 가능
- 단일 임계값은 이를 대체할 수 없음

---

## 📁 생성된 파일

```
claudedocs/phase3_results/phase3_2_hybrid_v2/
├── val_threshold_tuning.csv          # Validation threshold 튜닝 결과
├── product_threshold_analysis.csv    # 제품별 threshold 분석
├── test_strategy_comparison.csv      # Test 전략 비교 (3가지)
├── product_test_results.csv          # 제품별 Test 성능
├── phase3_2_config.json              # 실험 설정 및 권장사항
└── PHASE3_2_결과보고서.md             # 본 보고서
```

---

## 🎓 학습 포인트 (Lessons Learned)

### 1. 특성 vs 규칙

**교훈**: 특성으로 모델에 넣은 것을 다시 규칙으로 추가해도 효과 없음

- ✅ **좋은 접근**: 새로운 특성을 모델에 추가 → Phase 3-1
- ❌ **나쁜 접근**: 모델이 이미 사용하는 특성으로 규칙 만들기 → Phase 3-2

### 2. Validation 크기의 중요성

**교훈**: Validation이 너무 작으면 (Normal 4개) threshold 튜닝 불가능

- Validation 결과 ≠ Test 결과
- 과적합 위험 (특정 샘플에 맞춰짐)

**해결책**:
- Validation 샘플 늘리기 (최소 20-30개 이상)
- 또는 Cross-Validation 사용

### 3. 단순 규칙의 한계

**교훈**: XGBoost 같은 앙상블 모델은 단순 규칙보다 항상 우수

- XGBoost: 다변수, 비선형, 조건부 규칙
- 임계값 규칙: 단변수, 선형

**적용**:
- 규칙은 "모델이 학습하지 않은 도메인 지식" 추가 시에만 유효
- Phase 3-2는 도메인 지식이 아니라 기존 특성 재사용 → 실패

---

## 🚀 다음 단계 (Phase 4 후보)

Phase 3에서 최종 모델이 **XGBoost v3 (Phase 3-1)**로 확정되었습니다.

### Phase 4 후보 방향

#### Option A: 데이터 확보 및 재학습

```
목표: Normal 샘플 추가 확보
현황: Test Normal 14개 → 목표 50+ 개
기대 효과:
  - Normal Recall 신뢰도 향상
  - Validation 크기 증가 → threshold 튜닝 가능
  - 클래스 불균형 완화
```

#### Option B: 앙상블 모델 탐색

```
목표: XGBoost v3 + 다른 모델 결합
후보:
  - Random Forest
  - LightGBM
  - CatBoost
방법: Soft voting 또는 Stacking
기대 효과: AUC +0.01~0.02
```

#### Option C: Autoencoder 기반 이상 탐지

```
목표: 비지도 학습으로 보완
방법: Autoencoder로 Normal 패턴 학습 → 재구성 오차로 이상 탐지
기대 효과: Normal 데이터가 적어도 작동 가능
```

#### Option D: 운영 배포 및 모니터링

```
목표: 실제 환경 배포
구현:
  - XGBoost v3 API 서버
  - 실시간 예측 모니터링
  - 성능 추적 대시보드
효과: 실전 데이터로 모델 검증 및 재훈련
```

### 권장: Option D (배포) + Option A (데이터 확보)

1. **현재 XGBoost v3를 운영 배포**
2. **운영 중 Normal 샘플 추가 수집**
3. **충분한 데이터 확보 후 재학습 및 개선**

---

## 📌 최종 결론

### Phase 3-2 결론

❌ **Hybrid v2는 Test 성능 개선 효과 없음**
- acc_Y_rms_high 규칙 추가가 Test에서 작동하지 않음
- Validation 효과가 Test로 일반화되지 않음

✅ **Phase 3-1 XGBoost v3를 최종 모델로 확정**
- Test AUC: 0.820
- Normal Recall: 0.714 (최고)
- Abnormal Recall: 0.804
- 균형잡힌 성능

### 운영 권장사항

**기본 설정**: XGBoost v3 단독 (threshold=0.5)
**적용 대상**: 모든 제품 (100W, 200W)
**모드**: 균형 모드 (Normal/Abnormal 균형)

**선택 옵션**: Hybrid v1 (불량 민감 모드)
- 조건: Abnormal 탐지가 극도로 중요한 경우만
- 제약: 200W에서 Normal Recall=0 → **사용 불가**
- 결론: **사용 비권장**

### 프로젝트 전체 성과

| Phase | 핵심 성과 |
|-------|----------|
| Phase 2 | Baseline 구축 (AUC 0.811, Normal Recall 0.691) |
| Phase 3-0 | 데이터 무결성 검증 (누수 없음 확인) |
| **Phase 3-1** | **최종 모델 (AUC 0.820, Normal Recall 0.714)** ✅ |
| Phase 3-2 | Hybrid 한계 확인 (추가 개선 없음) |

---

**작성일**: 2025-11-19
**작성자**: Claude Code
**프로젝트**: Robot Arm Actuator Vibration Analysis (Phase 3 완료)
