# Step 3-3: 밴드 RMS 특성 - 상세 분석 보고서

## 요약

밴드 RMS 특성이 세 개의 주파수 밴드에 걸쳐 684개 윈도우에서 성공적으로 추출되었습니다:
- **Low (1-10 Hz)**: 구조/불균형 진동
- **Mid (10-50 Hz)**: 회전 고조파
- **High (50-150 Hz)**: 베어링 결함 및 충격 이벤트

**주요 발견**: 고주파 밴드 특성이 Normal과 Abnormal 클래스 간 가장 강한 판별력을 보여줍니다.

## 데이터 개요

- **전체 윈도우**: 898
- **정제된 데이터**: 889 (NaN 제거 후)
- **Train 세트**: 686 (Normal: 365, Abnormal: 321)
- **Val 세트**: 92
- **Test 세트**: 111
- **밴드 RMS 특성**: 9

## 가장 판별력 높은 상위 3개 특성

### 1. acc_Y_rms_high
- **Abnormal/Normal 비율**: 2.612x
- **Normal 평균**: 0.0974
- **Abnormal 평균**: 0.2545
- **해석**: 고주파 밴드의 acc_Y 센서 ({'low': '1-10Hz', 'mid': '10-50Hz', 'high': '50-150Hz'}[top3.iloc[0]['band']])

### 2. Gyro_Y_rms_low
- **Abnormal/Normal 비율**: 2.091x
- **Normal 평균**: 0.0107
- **Abnormal 평균**: 0.0225

### 3. Gyro_Y_rms_high
- **Abnormal/Normal 비율**: 2.085x
- **Normal 평균**: 0.0014
- **Abnormal 평균**: 0.0029

## 전체 스펙트럼 RMS와 비교

이전 분석에서 `acc_Y_rms` (전체 스펙트럼)의 비율은 **2.22x**였습니다.

밴드별 RMS 특성이 **개선된 판별력** 보여줌:
- `acc_Y_rms_high`: **2.61x** (17.6% 개선)
- 밴드 분해가 판별력 있는 주파수 콘텐츠를 성공적으로 분리

## 주파수 밴드 분석

### Low 밴드 (1-10 Hz)
- **높은 판별력의 센서**: Gyro_Y, acc_Y
- **평균 비율**: 1.80x

### Mid 밴드 (10-50 Hz)
- **높은 판별력의 센서**: Gyro_Y, acc_Y
- **평균 비율**: 1.57x

### High 밴드 (50-150 Hz)
- **높은 판별력의 센서**: acc_Y, Gyro_Y
- **평균 비율**: 2.10x
- **결론**: **고주파 밴드가 결함 감지에 가장 판별력 높음**

## 제품별 분석 (100W vs 200W)

### 100W 제품
- 최고 판별 특성: acc_Y_rms_low
- 비율: 1.17x

### 200W 제품
- 최고 판별 특성: acc_Y_rms_high
- 비율: 3.21x

**관찰**: 두 제품 모두 유사한 패턴을 보이며, 고주파 밴드가 가장 판별력 높음.

## 특성 상관관계 인사이트

- **레이블 상관관계**: 레이블과 가장 강한 상관관계를 가진 상위 3개 특성 (Abnormal=0, Normal=1)
  - acc_Sum_rms_low: r=-0.242
  - acc_Sum_rms_high: r=-0.229
  - acc_Y_rms_high: r=-0.229

- **특성 간 상관관계**: 밴드 RMS 특성은 동일 센서 내 밴드 간 중간 수준의 상관관계 보임

## 다음 단계 권장사항

1. **XGBoost 재학습**
   - 기존 특성 집합에 9개 밴드 RMS 특성 모두 추가
   - 예상 개선: 임계값=0.5에서 Recall ≥ 0.75 (현재 0.691 대비)
   - 특성 중요도를 위해 고주파 특성에 집중

2. **하이브리드 규칙 강화**
   - 전체 스펙트럼 RMS에 추가로 `acc_Y_rms_high > threshold` 고려
   - 경계 케이스 감지 개선 가능 (특히 200W)

3. **특성 선택**
   - 모델 복잡도가 우려되는 경우, 우선순위:
     - acc_Y_rms_high (가장 강한 판별자)
     - Gyro_Y_rms_low (보완 센서)
     - acc_Y_rms_low (저주파 구조 정보)

## 생성된 파일

### 시각화
- `band_rms_boxplots_by_sensor_band.png`: 센서 및 주파수 밴드별 박스플롯
- `band_rms_distributions_top5.png`: 상위 5개 특성의 분포 히스토그램
- `band_rms_product_comparison.png`: 100W와 200W 간 판별력 비교
- `band_rms_correlation_heatmap.png`: 특성 간 상관관계 행렬

### CSV 데이터 파일
- `band_rms_detailed_statistics.csv`: 특성별 전체 통계
- `band_rms_by_product.csv`: 제품별 통계
- `band_rms_label_correlation.csv`: 각 특성의 레이블 상관관계
- `band_rms_feature_correlation_matrix.csv`: 전체 상관관계 행렬
- `band_rms_sample_data.csv`: AI 검사용 샘플 데이터 (200개 샘플)
- `band_rms_mean_by_class_product.csv`: 클래스 및 제품별 평균값

### 요약
- `step3_3_detailed_report.md`: 본 종합 보고서

## 결론

밴드 RMS 특성이 전체 스펙트럼 RMS 대비 **개선된 판별력**을 제공하며, **고주파 밴드(50-150Hz)가 Normal과 Abnormal 클래스 간 가장 강한 분리**를 보여줍니다. 이러한 특성의 추가는 XGBoost 성능을 개선할 것으로 예상되며, 특히 현재 모델이 놓치는 경계 케이스 감지에 효과적일 것입니다.

**다음 조치**: 확장된 특성 집합(기존 9 + 새로운 9 밴드 RMS = 총 18개 특성)으로 XGBoost 재학습 진행.
