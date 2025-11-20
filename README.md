# Robot Arm Actuator Vibration Analysis

로봇 암 액추에이터의 진동 데이터를 분석하여 정상(Normal)과 비정상(Abnormal) 상태를 자동으로 분류하는 머신러닝 프로젝트

## 📊 프로젝트 개요

- **목적**: 진동 데이터 기반 품질 관리 자동화 및 불량 조기 감지
- **기간**: 2025-11-17 ~ 2025-11-19
- **최종 모델**: XGBoost v3 (12개 핵심 특성)
- **성능**: Test AUC 0.820, Abnormal Recall 0.804, Normal Recall 0.714

## 🎯 주요 성과

### 최종 모델 성능
```
Test AUC: 0.820
Accuracy: 79.3%

Abnormal (Class 0):
  Precision: 0.951
  Recall: 0.804
  F1: 0.872

Normal (Class 1):
  Precision: 0.345
  Recall: 0.714 (전체 실험 중 최고)
  F1: 0.465
```

### 핵심 기술
- **특성 선택**: 18개 → 12개 (과적합 해결)
- **CV 방법론**: StratifiedGroupKFold (파일 단위 일반화)
- **데이터 분할**: 혼합 전략 (Normal 시간 기반, Abnormal 파일 기반)
- **밴드 RMS**: 주파수 대역별 특성 (1-10Hz, 10-50Hz, 50-150Hz)

## 📁 프로젝트 구조

```
.
├── README.md                   # 본 문서
├── docs/                       # 분석 결과 및 문서
│   ├── 프로젝트_종합_분석_보고서.md
│   ├── SESSION_SUMMARY_Phase3.md
│   ├── eda_results/           # Phase 1: 탐색적 분석
│   ├── phase2_results/        # Phase 2: 모델링 (6단계)
│   └── phase3_results/        # Phase 3: 최종 모델
│       └── phase3_1_xgboost_core/
│           └── xgboost_v3_core_band_rms.json  # 최종 모델
├── scripts/                   # Phase 2 실험 스크립트
├── phase3/                    # Phase 3 실험 스크립트
├── src/                       # 소스 코드
│   ├── config/               # 설정 파일
│   ├── features/             # 특성 추출
│   ├── models/               # 모델 학습
│   └── preprocess/           # 데이터 전처리
├── data/                      # 데이터 (gitignore)
└── notebooks/                 # Jupyter 노트북
```

## 🚀 시작하기

### 환경 설정

```bash
# Python 가상환경 생성
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 필요한 패키지 설치
pip install numpy pandas scikit-learn xgboost scipy matplotlib seaborn
```

### 최종 모델 사용

```python
import json
import xgboost as xgb
import numpy as np

# 모델 로드
with open('docs/phase3_results/phase3_1_xgboost_core/xgboost_v3_core_band_rms.json', 'r') as f:
    model = xgb.Booster()
    model.load_model('docs/phase3_results/phase3_1_xgboost_core/xgboost_v3_core_band_rms.json')

# 예측
# X: 12개 특성 (9개 기존 + 3개 밴드 RMS)
# features = ['acc_Y_rms', 'acc_X_rms', 'Gyro_Y_rms', 'Gyro_X_rms',
#            'acc_Y_peak', 'acc_Sum_peak', 'acc_Y_crest',
#            'acc_Y_kurtosis', 'acc_Sum_kurtosis',
#            'acc_Y_rms_high', 'Gyro_Y_rms_high', 'Gyro_Y_rms_low']

dtest = xgb.DMatrix(X)
predictions = model.predict(dtest)
```

## 📈 Phase별 진행 과정

### Phase 1: 탐색적 데이터 분석
- 60개 파일 (100W: 30개, 200W: 30개)
- 684개 윈도우 생성 (8초, 50% 오버랩)
- 특성 상관관계 및 판별력 분석

### Phase 2: 모델링 및 개선 (6단계)
1. **Step 1**: RMS 임계값 규칙 (Baseline, AUC 0.696)
2. **Step 2**: XGBoost 베이스라인 (9개 특성, AUC 0.708)
3. **Step 3-1**: Threshold 튜닝 (Recall 0.990, 실용성 부족)
4. **Step 3-2**: Hybrid 규칙 v1 (Recall 0.804)
5. **Step 3-3**: 밴드 RMS 특성 추출 (9개 추가)
6. **Step 3-4**: XGBoost v2 (18개 특성, 과적합 발견)

### Phase 3: 과적합 해결 및 최종 모델
1. **Phase 3-0**: 데이터 무결성 감사 (누수 없음 확인)
2. **Phase 3-1**: XGBoost v3 최종 모델 (12개 특성, AUC 0.820) ✅
3. **Phase 3-2**: Hybrid v2 실험 (실패, v3 단독 사용 권장)

## 🔍 주요 학습 포인트

1. **특성 선택의 중요성**: "더 많은 특성 ≠ 더 좋은 성능"
2. **CV 방법론**: StratifiedGroupKFold로 파일 단위 일반화
3. **Hybrid 규칙 조건**: 모델이 모르는 도메인 지식일 때만 유효
4. **Validation 크기**: Threshold 튜닝은 충분한 샘플 필요
5. **제품별 규칙 재검토**: 단일 모델이 최적

## 📊 데이터셋

### 최종 분포 (Phase 3, 609개 윈도우)
| 데이터셋 | 윈도우 수 | Normal | Abnormal |
|---------|----------|--------|----------|
| Train | 686 | 365 (oversampled) | 321 |
| Validation | 92 | 4 | 88 |
| Test | 111 | 14 | 97 |

### 특성 구조
- **기존 9개**: RMS, Peak, Crest, Kurtosis
- **밴드 RMS 3개**: acc_Y_rms_high, Gyro_Y_rms_high, Gyro_Y_rms_low
- **제외 6개**: 판별력 낮은 특성 제거

## 🎯 Phase 4 권장사항

### 즉시 실행 (Phase 4-1)
- 모델 인퍼런스 파이프라인 구축
- 로깅 및 모니터링 시스템
- 경계 케이스 관리 (0.4 < p < 0.6)

### 중기 계획 (Phase 4-2)
- Normal 데이터 50개 이상 확보
- 재학습 및 성능 개선
- Threshold 재튜닝

### 장기 연구 (Phase 4-3)
- Autoencoder 기반 이상 탐지 (추천)
- 앙상블 모델
- 딥러닝 (1D CNN/LSTM)

## 📝 문서

- **종합 분석 보고서**: `docs/프로젝트_종합_분석_보고서.md`
- **Phase 3 세션 요약**: `docs/SESSION_SUMMARY_Phase3.md`
- **데이터 검증 보고서**: `docs/data_validation_report.md`
- **각 단계별 상세 보고서**: `docs/phase2_results/*/`, `docs/phase3_results/*/`

## 🛠️ 기술 스택

- **언어**: Python 3.8+
- **ML**: XGBoost, scikit-learn
- **데이터**: NumPy, Pandas
- **시각화**: Matplotlib, Seaborn
- **신호처리**: SciPy

## 👥 기여

이 프로젝트는 데이터 기반 의사결정, 체계적 실험 설계, 과학적 문제 해결 방법론의 모범 사례를 제시합니다.

## 📄 라이선스

내부 연구 프로젝트

---

**작성일**: 2025-11-19
**최종 업데이트**: 2025-11-20
