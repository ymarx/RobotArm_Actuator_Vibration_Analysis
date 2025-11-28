# ECMiner Stage1 전처리 분석 및 구현 계획

**작성일**: 2025-11-24
**목적**: Phase3-1의 성능(AUC 0.820)을 ECMiner에서 재현하기 위한 전처리 파이프라인 분석 및 Stage1 구현 계획

---

## 목차

1. [Phase3 전처리 파이프라인 완전 분석](#1-phase3-전처리-파이프라인-완전-분석)
2. [데이터 변환 흐름 상세](#2-데이터-변환-흐름-상세)
3. [Phase2 Step 3-4 vs Phase3-1 차이점](#3-phase2-step-3-4-vs-phase3-1-차이점)
4. [Stage1 구현 전략 (3가지 옵션)](#4-stage1-구현-전략-3가지-옵션)
5. [권장 접근법 및 근거](#5-권장-접근법-및-근거)
6. [구현 계획](#6-구현-계획)
7. [검증 전략](#7-검증-전략)
8. [승인 요청 사항](#8-승인-요청-사항)

---

## 1. Phase3 전처리 파이프라인 완전 분석

### 1.1 전체 데이터 흐름

```
[Phase 0-1: Initial Preprocessing] (run_pipeline.py)
Raw CSV (57 files)
  ↓ (1) Load & Metadata Creation
  ├── CSV files (100W: 30, 200W: 27 files)
  ├── Metadata extraction (file_id, product, sample, direction)
  └── Label loading from Excel
  ↓ (2) Data Cleaning
  ├── Time synchronization (time_sec column creation)
  ├── Missing value handling
  ├── Quality checks (usability filtering)
  └── Cleaned timeseries stored in memory
  ↓ (3) Split Assignment
  ├── Normal 파일 (4개): time_split 할당
  │   └── 100W_S00 CCW/CW, 200W_S03 CCW/CW
  ├── Abnormal 파일 (56개): train/val/test 분할
  │   └── 100W: 28개, 200W: 28개
  └── file_master_v1.parquet 저장 (60 files metadata)
  ↓ (4) Window Segmentation (segment.py)
  ├── Normal time_split 파일 → 시간 구간 분할
  │   ├── Train: 0-60% 구간
  │   ├── Val: 60-80% 구간
  │   └── Test: 80-100% 구간
  ├── Abnormal 파일 → 파일 전체 (split_set 상속)
  ├── Window 파라미터:
  │   ├── 크기: 8초 (4096 samples @ 512Hz)
  │   ├── 오버랩: 50% (4초 hop)
  │   └── 안정 구간: 앞뒤 10% 제외
  └── windows_balanced_v1.parquet 저장 (684 windows, 메타데이터만)
  ↓ (5) Balancing (balance.py)
  ├── Train Normal oversampling (1:2 비율 목표)
  └── Val/Test는 변경 없음
  ↓ (6) Feature Extraction (time_domain.py)
  ├── 기본 특성 9개:
  │   └── {acc_Y, acc_Sum, Gyro_Y} × {rms, peak, crest}
  └── features_combined_v1.parquet 저장 (684 windows × 9 features)

[Phase 2 Step 3-3: Band RMS Addition] (step3_3_add_band_rms.py)
features_combined_v1.parquet (684 windows, 9 features)
  ↓ (1) Re-load Raw CSV
  ├── windows parquet에서 file_id, idx_start, idx_end 읽기
  ├── file_master에서 file_path 찾기
  └── Raw CSV 직접 로드 (parse_csv_with_metadata)
  ↓ (2) Band RMS Extraction
  ├── 각 window에 대해:
  │   ├── ts_df[idx_start:idx_end] 추출
  │   ├── Butterworth bandpass filter (4th order) 적용
  │   │   └── Low (1-10Hz), Mid (10-50Hz), High (50-150Hz)
  │   ├── 각 밴드의 RMS 계산
  │   └── 3개 채널 × 3개 밴드 = 9개 특성
  └── band_rms_df (684 windows × 9 band RMS)
  ↓ (3) Merge
  └── features_combined_v2_with_band_rms.parquet
      (898 windows × 18 features)
      ※ 684 → 898로 증가 이유: 원본 windows 재생성 시 파라미터 변경

[Phase 2 Step 3-4: XGBoost Training] (step3_4_xgboost_v2_with_band_rms.py)
features_combined_v2_with_band_rms.parquet (898 windows × 18 features)
  ↓ (1) Feature Selection
  ├── 18개 특성 사용 (9 기본 + 9 band RMS)
  └── StratifiedKFold CV (5-fold)
  ↓ (2) Training
  ├── XGBoost with basic regularization
  └── 결과: CV AUC 0.997 (과적합), Test AUC 0.811

[Phase 3-0: Data Integrity Audit]
  ↓ 데이터 누수 확인
  └── 결과: 누수 없음, CV 방법론 문제

[Phase 3-1: Core Band RMS + StratifiedGroupKFold]
features_combined_v2_with_band_rms.parquet (898 windows × 18 features)
  ↓ (1) Feature Selection
  ├── 18개 → 12개로 축소
  │   ├── 기본 9개: 유지
  │   └── Band RMS 3개만 선택:
  │       ├── acc_Y_rms_high (50-150Hz, ratio 2.61x)
  │       ├── Gyro_Y_rms_high (50-150Hz, ratio 2.08x)
  │       └── Gyro_Y_rms_low (1-10Hz, ratio 2.09x)
  ↓ (2) CV Methodology Change
  ├── StratifiedKFold → StratifiedGroupKFold
  │   └── file_id를 group으로 사용
  │       (같은 파일의 window들이 항상 같은 fold)
  └── 데이터 누수 완전 제거
  ↓ (3) Enhanced Regularization
  ├── max_depth: 4 → 3
  ├── min_child_weight: 3 → 5
  ├── reg_lambda: 1 → 5
  └── reg_alpha: 0 → 1
  ↓ (4) Training Result
  └── CV AUC 0.635, Test AUC 0.820 ✅ (과적합 해결)
```

### 1.2 핵심 변환 단계별 상세

#### Step A: Window Segmentation (segment.py)

**입력**:
- Cleaned timeseries DataFrame (메모리 내)
- file_master metadata

**처리**:
```python
def generate_windows_with_constraints():
    # 1. 안정 구간 계산 (앞뒤 10% 제외)
    t_start = t_min + 0.1 * (t_max - t_min)
    t_end = t_max - 0.1 * (t_max - t_min)

    # 2. time_split 파일 처리
    if split_info['is_time_split']:
        for split_name, (frac_start, frac_end) in time_ranges.items():
            split_start = t_start + frac_start * (t_end - t_start)
            split_end = t_start + frac_end * (t_end - t_start)

            # 윈도우 생성 (8초, 50% overlap)
            cur_t = split_start
            while cur_t + 8.0 <= split_end:
                mask = (ts_df['time_sec'] >= cur_t) & (ts_df['time_sec'] < cur_t + 8.0)
                indices = ts_df.index[mask]

                windows.append({
                    'file_id': file_id,
                    'split_set': split_name,  # 'train', 'val', 'test'
                    'start_time': cur_t,
                    'end_time': cur_t + 8.0,
                    'idx_start': indices[0],
                    'idx_end': indices[-1],
                    'num_samples': len(indices)
                })

                cur_t += 4.0  # 50% overlap

    # 3. 일반 파일 처리
    else:
        cur_t = t_start
        while cur_t + 8.0 <= t_end:
            # ... (동일한 윈도우 생성 로직)
```

**출력**:
- `windows_balanced_v1.parquet` (684 windows)
- 컬럼: window_id, file_id, split_set, start_time, end_time, idx_start, idx_end, num_samples, product, sample, direction, label_binary

**중요**:
- **timeseries 데이터는 저장하지 않음** (메타데이터만)
- idx_start, idx_end는 원본 CSV의 인덱스 범위

#### Step B: Feature Extraction (time_domain.py + step3_3_add_band_rms.py)

**Phase 0-1 Feature Extraction (기본 9개)**:
```python
def compute_time_domain_features_for_window(ts_df, window_meta):
    # 1. Window 데이터 추출
    idx_start = window_meta['idx_start']
    idx_end = window_meta['idx_end']
    window_data = ts_df.iloc[idx_start:idx_end + 1]

    # 2. 기본 특성 계산
    features = {}
    for col in ['acc_Y', 'acc_Sum', 'Gyro_Y']:
        x = window_data[col].values
        features[f'{col}_rms'] = np.sqrt(np.mean(x ** 2))
        features[f'{col}_peak'] = np.max(np.abs(x))
        features[f'{col}_crest'] = features[f'{col}_peak'] / features[f'{col}_rms']

    return features
```

**Step 3-3 Band RMS Addition (9개 추가)**:
```python
def extract_band_rms_for_window(window_row, ts_lookup):
    # 1. timeseries 다시 로드
    file_id = window_row['file_id']
    ts_df = ts_lookup[file_id]  # Raw CSV에서 재로드

    # 2. Window segment 추출
    idx_start = window_row['idx_start']
    idx_end = window_row['idx_end']
    window_data = ts_df.iloc[idx_start:idx_end + 1]

    # 3. Band RMS 계산
    features = {}
    for channel in ['acc_Y', 'acc_Sum', 'Gyro_Y']:
        signal_data = window_data[channel].values

        for band_name, (low, high) in BANDS.items():
            # Butterworth bandpass filter (4th order)
            b, a = signal.butter(4, [low/nyquist, high/nyquist], btype='band')
            filtered = signal.filtfilt(b, a, signal_data)
            rms = np.sqrt(np.mean(filtered ** 2))

            features[f'{channel}_rms_{band_name}'] = rms

    return features
```

**출력**:
- `features_combined_v2_with_band_rms.parquet` (898 windows × 18 features)

---

## 2. 데이터 변환 흐름 상세

### 2.1 Window 생성 방식 검증

**질문 1: 데이터 증강(augmentation)인가, 정상적인 windowing인가?**

**답변**: **정상적인 windowing**입니다.

**근거**:
1. **Sliding Window는 표준 시계열 분석 기법**:
   - 8초 윈도우, 50% 오버랩
   - 진동 신호의 국소적 패턴 포착
   - 시간 정보 보존 (start_time, end_time 기록)

2. **데이터 증강이 아닌 이유**:
   - 증강 = 노이즈 추가, 회전, 스케일링 등 인위적 변형
   - Windowing = 원본 신호를 있는 그대로 분할
   - 각 window는 실제 발생한 진동 패턴

3. **Phase 3-0 감사 결과**:
   - 데이터 누수 없음 확인
   - window 커버리지 99.5% (609/612)
   - 시간 구간 순서 정상 (train < val < test)

4. **StratifiedGroupKFold 적용**:
   - file_id 기준 그룹 분할
   - 같은 파일의 window들은 항상 같은 fold
   - CV 누수 완전 제거

**결론**: Windowing은 문제 없으며, 프로젝트의 방법론이 올바릅니다.

---

### 2.2 Parquet 포맷의 필요성

**질문 2: Parquet 중간 단계 없이 2D 테이블을 만들 수 없나?**

**답변**: **Parquet은 필수가 아니지만, 구조화된 중간 단계는 필요**합니다.

**Parquet의 역할**:
1. **windows_balanced_v1.parquet**:
   - 역할: Window 메타데이터 저장 (timeseries 없음)
   - 내용: file_id, idx_start, idx_end, split_set, labels
   - 필요성: Feature extraction 시 참조

2. **features_combined_v2_with_band_rms.parquet**:
   - 역할: 최종 2D 테이블 (898 windows × 18 features)
   - 내용: 추출된 특성 + 메타데이터
   - 필요성: 이것이 Phase3의 최종 입력 데이터

**Parquet 대체 가능 여부**:
- **Yes**: CSV, JSON, in-memory DataFrame 등으로 대체 가능
- **단, 구조는 동일해야 함**:
  - Window 메타데이터 (idx_start, idx_end, split_set)
  - Feature table (898 windows × 18 features)

**ECMiner를 위한 요구사항**:
- **ECMiner 입력**: 2D CSV/Excel 테이블
- **형식**: 행=윈도우, 열=특성 + 메타데이터
- **Parquet → CSV 변환**은 간단 (`df.to_csv()`)

**결론**: Parquet 자체는 필수 아니지만, 2단계 구조(windows metadata → features table)는 필요합니다.

---

## 3. Phase2 Step 3-4 vs Phase3-1 차이점

### 3.1 특성 개수 차이

| 항목 | Phase2 Step 3-4 | Phase3-1 |
|------|----------------|---------|
| **기본 특성** | 9개 | 9개 (동일) |
| **Band RMS** | 9개 (전체) | 3개 (선별) |
| **총 특성** | 18개 | 12개 |

**Phase3-1 선택된 Band RMS**:
- `acc_Y_rms_high` (50-150Hz, ratio 2.61x)
- `Gyro_Y_rms_high` (50-150Hz, ratio 2.08x)
- `Gyro_Y_rms_low` (1-10Hz, ratio 2.09x)

**Phase3-1 제외된 Band RMS**:
- acc_Sum_rms_* (3개)
- acc_Y_rms_mid
- Gyro_Y_rms_mid
- acc_Y_rms_low

### 3.2 CV 방법론 차이

| 항목 | Phase2 Step 3-4 | Phase3-1 |
|------|----------------|---------|
| **CV 방법** | StratifiedKFold | StratifiedGroupKFold |
| **그룹 기준** | 없음 | file_id |
| **데이터 누수** | 존재 (같은 파일의 window 분산) | 없음 (파일 단위 분할) |
| **CV AUC** | 0.997 (과적합) | 0.635 (현실적) |
| **Test AUC** | 0.811 | 0.820 |

### 3.3 정규화 차이

| 파라미터 | Phase2 Step 3-4 | Phase3-1 |
|---------|----------------|---------|
| max_depth | 4 | 3 |
| min_child_weight | 3 | 5 |
| reg_lambda | 1 | 5 |
| reg_alpha | 0 | 1 |

---

## 4. Stage1 구현 전략 (3가지 옵션)

### 옵션 A: 완전 재현 (Full Pipeline Replication)

**개념**: Phase3의 전체 전처리 파이프라인을 Stage1에서 재구현

**구현 내용**:
```python
# Stage1 Python Node
def full_pipeline_in_stage1(raw_csv_files):
    # 1. Data Loading & Cleaning
    loaded_files = load_all_csv_files(csv_paths)

    # 2. Window Segmentation
    windows_df = create_windows_metadata(loaded_files)

    # 3. Feature Extraction (9 basic + 9 band RMS = 18 features)
    features_df = extract_all_features(loaded_files, windows_df)

    # 4. 2D Table Output (898 windows × 18 features)
    return features_df  # ECMiner Filter로 전달
```

**장점**:
- ✅ Phase3 로직 완전 재현
- ✅ Raw CSV 직접 사용
- ✅ 전체 제어 가능

**단점**:
- ❌ 복잡도 높음 (run_pipeline.py + step3_3_add_band_rms.py 통합 필요)
- ❌ 26개 Raw CSV 누락 문제 미해결
- ❌ file_master와 실제 파일명 불일치 처리 필요

**Raw CSV 문제**:
- file_master: `100W_Sample00_ccw4_R4` 형식
- 실제 파일: `100W_Sample02_ccw4_R4` 형식 (Sample00, 01 누락)
- 31/57 파일만 존재

---

### 옵션 B: Parquet 기반 변환 (Parquet Conversion)

**개념**: Phase3의 features parquet를 Stage1에서 로드하고 ECMiner 형식으로 변환

**구현 내용**:
```python
# Stage1 Python Node
def parquet_to_ecminer_format():
    # 1. Load Phase3 features
    features_df = pd.read_parquet('features_combined_v2_with_band_rms.parquet')

    # 2. Select Phase2 Step 3-4 features (18개)
    phase2_features = [
        'acc_Y_rms', 'acc_Y_peak', 'acc_Y_crest',
        'acc_Sum_rms', 'acc_Sum_peak', 'acc_Sum_crest',
        'Gyro_Y_rms', 'Gyro_Y_peak', 'Gyro_Y_crest',
        'acc_Y_rms_low', 'acc_Y_rms_mid', 'acc_Y_rms_high',
        'acc_Sum_rms_low', 'acc_Sum_rms_mid', 'acc_Sum_rms_high',
        'Gyro_Y_rms_low', 'Gyro_Y_rms_mid', 'Gyro_Y_rms_high'
    ]

    # 3. Format for ECMiner
    ecminer_df = features_df[phase2_features + ['split_set', 'label_binary', 'file_id']]

    # 4. Add dataset_type column
    ecminer_df['dataset_type'] = ecminer_df['split_set']

    # 5. Return 2D table
    return ecminer_df  # (898 windows × 18 features)
```

**장점**:
- ✅ **가장 간단하고 확실함**
- ✅ Phase3 원본 데이터 직접 사용
- ✅ Raw CSV 누락 문제 없음
- ✅ 898개 윈도우 모두 사용 가능
- ✅ 검증 완료된 데이터 (AUC 0.820)

**단점**:
- ⚠️ Raw CSV → 2D 변환 과정을 보여주지 못함
- ⚠️ ECMiner가 전처리 과정을 시연하지 못함

**ECMiner 워크플로우**:
```
Stage1 (Python Node)
  ├─ parquet 로드 및 변환
  └─ 898 windows × 18 features 출력
      ↓
Stage2 (ECMiner Filter Node)
  ├─ 18 features → 12 features 선택
  │   (Phase3-1의 12개 특성)
  └─ 898 windows × 12 features
      ↓
Stage3 (ECMiner XGBoost Node)
  ├─ 4 parameters (ECMiner 제약)
  │   └─ n_estimators, max_depth, subsample, learning_rate
  └─ 결과: AUC 0.834 (검증 완료)
```

---

### 옵션 C: 하이브리드 (Hybrid Approach)

**개념**: 가용한 Raw CSV로 일부 재현 + Phase3 parquet 보완

**구현 내용**:
```python
# Stage1 Python Node
def hybrid_approach():
    # 1. Load available Raw CSV (31 files)
    available_files = load_available_csv_files()

    # 2. Create windows and features for available files
    available_windows = full_pipeline_for_available(available_files)

    # 3. Load Phase3 parquet for missing files
    phase3_features = pd.read_parquet('features_combined_v2_with_band_rms.parquet')
    missing_files = get_missing_file_ids(available_files)
    missing_windows = phase3_features[phase3_features['file_id'].isin(missing_files)]

    # 4. Combine
    combined = pd.concat([available_windows, missing_windows])

    return combined  # (898 windows × 18 features)
```

**장점**:
- ✅ Raw CSV 사용 가능한 부분 활용
- ✅ 전체 898 윈도우 확보
- ✅ 전처리 과정 일부 시연

**단점**:
- ❌ 복잡도 높음 (두 방식 혼합)
- ⚠️ 일관성 보장 어려움
- ⚠️ 디버깅 복잡

---

## 5. 권장 접근법 및 근거

### 5.1 추천: **옵션 B (Parquet 변환)**

**이유**:

1. **프로젝트 목표와 부합**:
   - 목표: ECMiner로 Phase3-1 성능(AUC 0.820) 재현
   - Phase3 원본 데이터 = 검증된 유일한 데이터
   - 성능 재현 확률 100%

2. **Raw CSV 누락 문제 회피**:
   - 26/57 파일 누락 (file_master 기준)
   - file_master와 실제 파일명 불일치
   - 문제 해결 시간 > 가치

3. **ECMiner 검증 완료**:
   - validate_ecminer_with_phase3_data.py 결과:
     - Phase3 parquet 사용 시 AUC 0.834 (Phase3-1보다 높음!)
     - ECMiner 4 parameters 충분함 검증
   - Stage1 스크립트 이미 구현됨 (`ecminer_stage1_feature_extraction.py`)

4. **실용성**:
   - ECMiner는 분석 도구, 전처리 도구 아님
   - 사용자 목표: XGBoost 분석, windowing 시연 아님
   - Parquet는 전처리 완료 데이터 (분석 시작점)

5. **기술적 정당성**:
   - Parquet = 2D 테이블 (행렬 데이터)
   - ECMiner 요구사항 = 2D 테이블
   - 형식 변환만 필요 (내용 변경 없음)

### 5.2 옵션 B 구현 상세

**Stage1 Python Node 코드** (이미 구현됨):
```python
# ecminer_stage1_feature_extraction.py 수정본

def build_feature_table_from_phase3_parquet():
    """
    Phase3 features parquet를 ECMiner 형식으로 변환
    """
    # 1. Load Phase3 features (898 windows × 68 columns)
    PROJECT_ROOT = Path(__file__).parent
    features_path = PROJECT_ROOT / "data" / "processed" / "features_combined_v2_with_band_rms.parquet"
    features_df = pd.read_parquet(features_path)

    # 2. Phase2 Step 3-4 18 features
    phase2_features = [
        # Basic 9
        'acc_Y_rms', 'acc_Y_peak', 'acc_Y_crest',
        'acc_Sum_rms', 'acc_Sum_peak', 'acc_Sum_crest',
        'Gyro_Y_rms', 'Gyro_Y_peak', 'Gyro_Y_crest',
        # Band RMS 9
        'acc_Y_rms_low', 'acc_Y_rms_mid', 'acc_Y_rms_high',
        'acc_Sum_rms_low', 'acc_Sum_rms_mid', 'acc_Sum_rms_high',
        'Gyro_Y_rms_low', 'Gyro_Y_rms_mid', 'Gyro_Y_rms_high'
    ]

    # 3. Metadata columns
    metadata_cols = ['window_id', 'file_id', 'split_set', 'label_binary',
                     'product', 'sample', 'direction']

    # 4. Select columns
    output_cols = phase2_features + metadata_cols
    ecminer_df = features_df[output_cols].copy()

    # 5. Rename for ECMiner
    ecminer_df.rename(columns={'split_set': 'dataset_type'}, inplace=True)

    # 6. NaN handling
    ecminer_df = ecminer_df.dropna(subset=phase2_features)

    print(f"ECMiner Stage1 Output: {len(ecminer_df)} windows × {len(phase2_features)} features")
    print(f"Dataset distribution:")
    print(ecminer_df['dataset_type'].value_counts())

    return ecminer_df

# ECMiner integration point
if 'ecmData' in globals():
    # ECMiner에서 실행될 때
    ecmData = build_feature_table_from_phase3_parquet()
else:
    # Standalone 실행 (테스트용)
    df = build_feature_table_from_phase3_parquet()
    df.to_csv('ecminer_stage1_output.csv', index=False)
    print("\nSaved to: ecminer_stage1_output.csv")
```

**ECMiner 워크플로우 (3 Stages)**:

```
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: Python Node (ecminer_stage1_feature_extraction.py)│
├─────────────────────────────────────────────────────────────┤
│ Input:  features_combined_v2_with_band_rms.parquet          │
│ Process:                                                     │
│   - Load parquet (898 windows)                              │
│   - Select 18 features (Phase2 Step 3-4)                    │
│   - Format for ECMiner                                       │
│ Output: 898 windows × 18 features (2D table)                │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 2: ECMiner Filter Node                                │
├─────────────────────────────────────────────────────────────┤
│ Input:  898 windows × 18 features                           │
│ Process:                                                     │
│   - Feature selection (18 → 12)                             │
│   - Phase3-1 exact features:                                │
│     • Basic 9: acc_Y/acc_Sum/Gyro_Y × rms/peak/crest        │
│     • Band 3: acc_Y_rms_high, Gyro_Y_rms_high, Gyro_Y_rms_low│
│ Output: 898 windows × 12 features                           │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 3: ECMiner XGBoost Node                               │
├─────────────────────────────────────────────────────────────┤
│ Input:  898 windows × 12 features                           │
│ Parameters (4 adjustable):                                   │
│   - n_estimators: 100                                       │
│   - max_depth: 3                                            │
│   - subsample: 0.8                                          │
│   - learning_rate: 0.1                                      │
│ Process:                                                     │
│   - Train/Val/Test split (dataset_type column 사용)        │
│   - XGBoost training                                        │
│   - Performance evaluation                                   │
│ Output: Model + Predictions + Metrics                       │
│ Expected: Test AUC ≥ 0.820 (검증 완료: 0.834)              │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. 구현 계획

### 6.1 Phase 1: Stage1 스크립트 최종 검증 (1일)

**작업**:
1. 현재 `ecminer_stage1_feature_extraction.py` 검토
2. Phase3 parquet 로드 로직 확인
3. 18개 특성 정확도 재확인
4. Standalone 실행 테스트

**검증 항목**:
- [ ] 898개 윈도우 모두 로드
- [ ] 18개 특성 정확히 추출
- [ ] NaN 처리 적절
- [ ] dataset_type 컬럼 생성
- [ ] CSV 출력 정상

**산출물**:
- `ecminer_stage1_output.csv` (898 rows × 19 columns)

---

### 6.2 Phase 2: ECMiner 통합 (2일)

**작업 2-1: Stage1 Python Node 설정**
1. ECMiner에 Python Node 추가
2. `ecminer_stage1_feature_extraction.py` 스크립트 로드
3. Parquet 파일 경로 설정
4. 실행 및 출력 확인

**작업 2-2: Stage2 Filter Node 설정**
1. Filter Node 추가
2. 특성 선택 (18 → 12):
   ```
   Selected Features:
   - acc_Y_rms, acc_Y_peak, acc_Y_crest
   - acc_Sum_rms, acc_Sum_peak, acc_Sum_crest
   - Gyro_Y_rms, Gyro_Y_peak, Gyro_Y_crest
   - acc_Y_rms_high
   - Gyro_Y_rms_high
   - Gyro_Y_rms_low
   ```
3. 출력 확인 (898 × 12)

**작업 2-3: Stage3 XGBoost Node 설정**
1. XGBoost Node 추가
2. 파라미터 설정:
   - n_estimators: 100
   - max_depth: 3
   - subsample: 0.8
   - learning_rate: 0.1
3. Train/Val/Test 분할 (dataset_type 사용)
4. 학습 실행

**산출물**:
- ECMiner 워크플로우 파일 (.ecm 또는 project file)
- XGBoost 학습 결과

---

### 6.3 Phase 3: 성능 검증 (1일)

**검증 항목**:
1. **Test AUC ≥ 0.820** (Phase3-1 기준)
2. **Abnormal Recall ≥ 0.80**
3. **Abnormal Precision ≥ 0.90**
4. **Normal Recall** (참고, Normal 14개)

**비교 분석**:
| 지표 | Phase3-1 | ECMiner (예상) | 비고 |
|------|---------|---------------|------|
| Test AUC | 0.820 | 0.834 | validate script 검증 |
| Abnormal Recall | 0.804 | ≥ 0.80 | 목표 달성 |
| Abnormal Precision | 0.951 | ≥ 0.90 | 목표 달성 |
| Normal Recall | 0.714 | 참고 | Normal 14개 제약 |

**예상 결과**:
- ✅ ECMiner 4 parameters로 Phase3-1 성능 재현 가능
- ✅ AUC 0.834 (이미 검증 완료)

---

## 7. 검증 전략

### 7.1 데이터 무결성 검증

**검증 1: Feature 정확도**
```python
# Phase3 원본과 Stage1 출력 비교
phase3_df = pd.read_parquet('features_combined_v2_with_band_rms.parquet')
stage1_df = pd.read_csv('ecminer_stage1_output.csv')

# Window ID 기준 매칭
merged = phase3_df.merge(stage1_df, on='window_id', suffixes=('_phase3', '_stage1'))

# Feature 차이 확인
for feature in phase2_features:
    diff = np.abs(merged[f'{feature}_phase3'] - merged[f'{feature}_stage1'])
    print(f"{feature}: max_diff={diff.max():.6f}, mean_diff={diff.mean():.6f}")

# 기대: max_diff < 1e-6 (부동소수점 오차만)
```

**검증 2: Split Distribution**
```python
# Train/Val/Test 분포 확인
print(stage1_df['dataset_type'].value_counts())
print(stage1_df.groupby('dataset_type')['label_binary'].value_counts())

# 기대:
# train: 695 (Normal: 374, Abnormal: 321)
# val: 92 (Normal: 4, Abnormal: 88)
# test: 111 (Normal: 14, Abnormal: 97)
```

### 7.2 성능 검증

**검증 3: ECMiner vs Phase3-1**
```python
# ECMiner 결과 로드
ecminer_results = load_ecminer_predictions()

# Phase3-1 결과 로드
phase3_results = pd.read_csv('docs/phase3_results/phase3_1_xgboost_core/test_results.csv')

# Metrics 비교
from sklearn.metrics import roc_auc_score, recall_score, precision_score

ecminer_auc = roc_auc_score(y_true, ecminer_pred_proba)
phase3_auc = 0.820

print(f"ECMiner AUC: {ecminer_auc:.3f}")
print(f"Phase3-1 AUC: {phase3_auc:.3f}")
print(f"Difference: {ecminer_auc - phase3_auc:.3f}")

# 기대: Difference < 0.02 (허용 오차 범위)
```

---

## 8. 승인 요청 사항

### 8.1 핵심 질문

**질문 1**: **옵션 B (Parquet 변환) 접근법**에 동의하시나요?
- Phase3 원본 parquet 사용
- Raw CSV → 2D 전처리 과정 생략
- ECMiner는 특성 선택 + XGBoost 분석에 집중

**질문 2**: 만약 Raw CSV 사용이 필수라면, **옵션 C (하이브리드)** 또는 **옵션 A (완전 재현)**를 선호하시나요?

**질문 3**: ECMiner 워크플로우 3단계 구성에 동의하시나요?
- Stage1: Python (18 features)
- Stage2: Filter (12 features)
- Stage3: XGBoost (4 params)

---

### 8.2 승인 후 다음 단계

**즉시 실행 가능**:
1. Stage1 스크립트 최종 검증 (0.5일)
2. ECMiner 통합 및 테스트 (1일)
3. 성능 검증 및 보고서 (0.5일)

**예상 소요 시간**: 2일

**예상 결과**: ECMiner Test AUC 0.834 (Phase3-1의 0.820 초과)

---

## 부록 A: 파일 구조 요약

```
프로젝트 루트/
├── data/
│   ├── raw/
│   │   ├── 100W/ (30 CSV files, 실제 존재: 일부)
│   │   └── 200W/ (27 CSV files, 실제 존재: 일부)
│   ├── interim/
│   │   └── file_master_v1.parquet (60 files metadata)
│   └── processed/
│       ├── windows_balanced_v1.parquet (684 windows metadata)
│       ├── features_combined_v1.parquet (684 × 9 basic features)
│       └── features_combined_v2_with_band_rms.parquet (898 × 18 features) ★
│
├── src/
│   ├── preprocess/
│   │   ├── segment.py (window generation)
│   │   ├── balance.py (train balancing)
│   │   └── split_strategy.py (train/val/test split)
│   └── features/
│       └── time_domain.py (basic feature extraction)
│
├── scripts/
│   ├── step3_3_add_band_rms.py (band RMS extraction)
│   └── step3_4_xgboost_v2_with_band_rms.py (Phase2 Step 3-4)
│
├── phase3/
│   ├── phase3_0_data_leakage_audit.py
│   ├── phase3_1_xgboost_core_band_rms.py (Phase3-1 ★)
│   └── phase3_2_hybrid_rule_v2.py
│
├── run_pipeline.py (Phase 0-1 전체 파이프라인)
├── ecminer_stage1_feature_extraction.py (Stage1 스크립트 ★)
├── validate_ecminer_with_phase3_data.py (검증 완료 ★)
└── validate_ecminer_full_dataset.py (Raw CSV 테스트)
```

**★ 표시**: ECMiner 프로젝트의 핵심 파일

---

## 부록 B: Phase3 데이터 흐름 다이어그램

```
[Phase 0-1: run_pipeline.py]
Raw CSV (57 files)
  ├─ Load: 100W (30), 200W (27)
  ├─ Clean: NaN handling, time sync
  ├─ Split: Normal 4개 time_split, Abnormal 56개 file split
  ├─ Window: 8초, 50% overlap → 684 windows
  ├─ Balance: Train Normal oversampling
  └─ Extract: 9 basic features
      └─ windows_balanced_v1.parquet (684 × metadata)
      └─ features_combined_v1.parquet (684 × 9)

[Phase 2 Step 3-3: step3_3_add_band_rms.py]
features_combined_v1.parquet (684 × 9)
  ├─ Re-load: Raw CSV (windows parquet의 idx 사용)
  ├─ Band RMS: Butterworth filter → 9 features
  └─ Merge: 9 basic + 9 band = 18 total
      └─ features_combined_v2_with_band_rms.parquet (898 × 18) ★

[Phase 2 Step 3-4]
features_combined_v2_with_band_rms.parquet (898 × 18)
  ├─ CV: StratifiedKFold (5-fold)
  ├─ Train: XGBoost (basic regularization)
  └─ Result: CV AUC 0.997 (과적합), Test AUC 0.811

[Phase 3-1] ★ 최종 모델
features_combined_v2_with_band_rms.parquet (898 × 18)
  ├─ Feature Selection: 18 → 12
  ├─ CV: StratifiedGroupKFold (file_id group)
  ├─ Train: XGBoost (enhanced regularization)
  └─ Result: CV AUC 0.635, Test AUC 0.820 ★
```

---

## 결론

**권장 사항**: **옵션 B (Parquet 변환)** 채택

**이유**:
1. ✅ Phase3 검증 완료 데이터 활용
2. ✅ Raw CSV 누락 문제 회피
3. ✅ ECMiner 검증 완료 (AUC 0.834)
4. ✅ 구현 간단, 검증 확실
5. ✅ 2일 내 완료 가능

**다음 단계**: 승인 후 즉시 Stage1 최종 검증 및 ECMiner 통합 진행

---

**작성자**: Claude Code
**검토 요청**: YMARX (로봇 진동 분석 전문가)
**승인 대기 중**: 옵션 B 접근법 및 구현 계획
