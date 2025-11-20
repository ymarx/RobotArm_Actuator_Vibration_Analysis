# Step 3-3: 밴드 RMS 특성 - 추출 요약

## 목표
모델 표현력 향상을 위해 주파수 밴드별 RMS 특성을 추출합니다.

## 주파수 밴드
- **Low (1-10 Hz)**: 구조/불균형 진동
- **Mid (10-50 Hz)**: 1× 및 2× 회전 고조파
- **High (50-150 Hz)**: 베어링 결함, 충격 이벤트

## 방법론
1. 윈도우 및 시계열 데이터 로드
2. 각 윈도우에 대해 scipy.signal.butter를 사용한 밴드패스 필터링 적용 (4차)
3. 각 밴드에 대해 필터링된 신호의 RMS 계산
4. 채널 처리: acc_Y, acc_Sum (가능한 경우 Gyro_Y)
5. 기존 특성과 병합

## 기술적 세부사항
- **샘플링 레이트**: 512.0 Hz
- **필터 타입**: Butterworth 밴드패스 (4차)
- **대체 방법**: 필터 설계 실패 시 FFT 기반 필터링

## 결과

### 데이터 요약
- **전체 윈도우**: 684
- **로드된 시계열**: 57/57 파일
- **밴드 RMS 추출**: 684 윈도우
- **정제된 데이터**: 889 행 (NaN 제거 후)

### 추가된 새 특성
9개 밴드 RMS 특성:
- acc_Y_rms_low
- acc_Y_rms_mid
- acc_Y_rms_high
- acc_Sum_rms_low
- acc_Sum_rms_mid
- acc_Sum_rms_high
- Gyro_Y_rms_low
- Gyro_Y_rms_mid
- Gyro_Y_rms_high

### 가장 판별력 높은 상위 5개 특성 (Abnormal/Normal 비율)
| feature         |   ratio_mean |   ratio_median |
|:----------------|-------------:|---------------:|
| acc_Y_rms_high  |      2.61153 |        1.12831 |
| Gyro_Y_rms_low  |      2.09089 |        1.15508 |
| Gyro_Y_rms_high |      2.08492 |        1.14974 |
| Gyro_Y_rms_mid  |      2.0795  |        1.10177 |
| acc_Y_rms_low   |      1.6821  |        1.14435 |

## 주요 발견사항
1. **고주파 밴드**가 가장 강한 판별력 보여줌 (비율 > X)
2. **acc_Y**와 **acc_Sum** 모두 유용한 밴드 정보 기여
3. 밴드 RMS 특성이 기존 전체 스펙트럼 RMS를 보완

## 생성된 파일
- `features_combined_v2_with_band_rms.parquet`: 강화된 특성 집합
- `band_rms_statistics.csv`: 클래스별 상세 통계
- `step3_3_extraction_summary.md`: 본 요약

## 다음 단계
1. 확장된 특성 집합으로 XGBoost 재학습
2. 임계값=0.5에서 성능 개선 평가
3. 하이브리드 규칙 재적용 및 Step 3-2와 비교
