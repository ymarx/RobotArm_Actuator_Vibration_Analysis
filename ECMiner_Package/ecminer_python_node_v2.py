#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ECMiner 파이썬 연동 노드용 스크립트 v2 (하이브리드 방식)

이 스크립트는 ecminer_stage1_py_node.py의 검증된 입력 방식과
절대 경로 시스템을 결합한 안정적인 버전입니다.

사용 방법:
1. create_ecminer_input.py를 먼저 실행하여 ecminer_input_full.csv 생성
2. ECMiner에서 빈 노드 → Python 연동 노드 추가
3. 이 스크립트 내용을 편집기에 복사-붙여넣기
4. 아래 "경로 설정" 섹션의 PROJECT_ROOT만 수정
5. 실행 → 다음 노드로 결과 전달됨

주의사항:
- 이 스크립트는 ECMiner 임시 폴더에서 실행됩니다
- 절대 경로를 사용하여 안정적인 파일 접근을 보장합니다
- ecminer_input_full.csv가 미리 생성되어 있어야 합니다
"""

# ============================================================================
# 🔧 경로 설정 (사용자가 수정하는 영역)
# ============================================================================
#
# ECMiner 파이썬 연동 노드를 만들 때마다 아래 경로를 시스템에 맞게 수정하세요
#
# ============================================================================

# 프로젝트 루트 경로 (절대 경로)
# 이 경로 아래에 다음 파일/폴더가 있어야 함:
#   - ecminer_input_full.csv (create_ecminer_input.py로 생성)
#   - data/ (원본 데이터 폴더)
#   - ECMiner_Package/ (레이블 파일 및 설정)
#
# Windows 예시: "C:/Users/UserName/Projects/RobotArm"
# macOS 예시:   "/Users/UserName/Dropbox/RobotArm"
# Linux 예시:   "/home/username/projects/robotarm"

PROJECT_ROOT = "/Users/YMARX/Dropbox/2025_ECMiner/CP25_NeuroMecha/03_진행/[Analysis]RobotArm_Actuator_QT(Ociliation)"

# 출력 CSV 파일 저장 여부 (True/False)
# True: 파일로 저장 (디버깅/확인용)
# False: ECMiner 다음 노드로만 전달
SAVE_OUTPUT_FILE = True

# 출력 파일명 (SAVE_OUTPUT_FILE=True 인 경우만 사용)
OUTPUT_FILENAME = "ecminer_output.csv"

# ============================================================================
# ⚠️ 이 아래 코드는 수정하지 마세요
# ============================================================================

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import signal
from typing import Dict, List, Tuple, Optional
import warnings
import yaml
warnings.filterwarnings('ignore')

# 절대 경로 구성
PROJECT_ROOT = Path(PROJECT_ROOT)
INPUT_CSV_PATH = PROJECT_ROOT / "ecminer_input_full.csv"
DATA_ROOT = PROJECT_ROOT / "data"
LABEL_CSV_PATH = PROJECT_ROOT / "ECMiner_Package" / "ecminer_labels.csv"
CONFIG_YAML_PATH = PROJECT_ROOT / "ECMiner_Package" / "ecminer_config.yaml"
OUTPUT_CSV_PATH = PROJECT_ROOT / OUTPUT_FILENAME if SAVE_OUTPUT_FILE else None

print("=" * 60)
print("ECMiner 파이썬 연동 노드 v2: 진동 데이터 전처리")
print("=" * 60)
print(f"\nPROJECT_ROOT: {PROJECT_ROOT}")
print(f"INPUT_CSV: {INPUT_CSV_PATH}")
print(f"DATA_ROOT: {DATA_ROOT}")
print(f"LABEL_CSV: {LABEL_CSV_PATH}")
print(f"CONFIG_YAML: {CONFIG_YAML_PATH}")

# ============================================================================
# 1. 설정 및 상수 정의
# ============================================================================

# 샘플링 주파수 (Hz)
SAMPLING_FREQ = 512.0

# 윈도우 파라미터
WINDOW_SEC = 8.0      # 8초 윈도우
HOP_SEC = 4.0         # 4초 hop (50% 중첩)
STABLE_MARGIN = 0.1   # 앞/뒤 10% 안정 구간 제외

# 양품 파일 정의 (100W: Sample00만, 200W: Sample03만)
NORMAL_SAMPLES = {
    '100W': [0],
    '200W': [3]
}

# Train/Val/Test 시간 분할 비율 (양품 파일용)
TIME_SPLIT_RANGES = {
    'train': (0.0, 0.6),   # 0-60%
    'val': (0.6, 0.8),     # 60-80%
    'test': (0.8, 1.0)     # 80-100%
}

# 불량 파일 분할 비율
ABNORMAL_SPLIT_RATIOS = {
    'train': 0.7,
    'val': 0.15,
    'test': 0.15
}
RANDOM_SEED = 42

# 밴드 RMS 주파수 범위 (Hz)
FREQUENCY_BANDS = {
    'low': (1, 10),      # 저주파: 구조 진동
    'mid': (10, 50),     # 중주파: 회전 고조파
    'high': (50, 150)    # 고주파: 베어링 결함, 충격
}

# 추출할 특징 목록
BASIC_FEATURES = [
    'acc_Y_rms', 'acc_Y_peak', 'acc_Y_crest',
    'acc_Sum_rms', 'acc_Sum_peak', 'acc_Sum_crest',
    'Gyro_Y_rms', 'Gyro_Y_peak', 'Gyro_Y_crest'
]

BAND_RMS_FEATURES = [
    'acc_Y_rms_low', 'acc_Y_rms_mid', 'acc_Y_rms_high',
    'acc_Sum_rms_low', 'acc_Sum_rms_mid', 'acc_Sum_rms_high',
    'Gyro_Y_rms_low', 'Gyro_Y_rms_mid', 'Gyro_Y_rms_high'
]

# ============================================================================
# 2. 레이블 시스템 함수
# ============================================================================

def load_label_file(label_path: Path) -> pd.DataFrame:
    """
    레이블 파일 로드 (CSV 또는 Excel)

    Returns:
        DataFrame with columns: product, sample, label
    """
    if label_path.suffix == '.csv':
        return pd.read_csv(label_path, encoding='utf-8-sig')
    elif label_path.suffix in ['.xlsx', '.xls']:
        return pd.read_excel(label_path)
    else:
        raise ValueError(f"지원하지 않는 파일 형식: {label_path.suffix}")

def load_config(config_path: Path) -> Dict:
    """
    설정 파일 로드 (YAML)

    Returns:
        Config dictionary
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def apply_label_strategy(
    label_raw: str,
    product: str,
    sample: int,
    strategy: Dict
) -> Tuple[int, float]:
    """
    레이블 전략 적용: label_raw → (label_binary, label_weight)

    Args:
        label_raw: 원본 라벨 (정상/소음/진동/표기없음)
        product: 제품 (100W/200W)
        sample: 샘플 번호
        strategy: 전략 설정 딕셔너리

    Returns:
        (label_binary, label_weight) - (1=양품/0=불량, 가중치)
    """
    # 1. Special overrides 먼저 적용
    original_label = label_raw
    for override in strategy.get('special_overrides', []):
        if (product == override['product'] and
            sample == override['sample'] and
            label_raw == override['from']):
            label_raw = override['to']
            break

    # 2. PASS/FAIL 매핑
    if label_raw in strategy['PASS']:
        label_binary = 1  # 양품
    elif label_raw in strategy['FAIL']:
        label_binary = 0  # 불량
    else:
        # 정의되지 않은 라벨은 불량으로 처리
        label_binary = 0

    # 3. 가중치
    label_weight = strategy.get('weights', {}).get(original_label, 1.0)

    return label_binary, label_weight

# ============================================================================
# 3. 데이터 로드 및 파싱 함수
# ============================================================================

def parse_file_id(file_id: str) -> Dict:
    """
    파일 ID를 파싱하여 메타데이터 추출

    입력: "100W_Sample00_CW_20251107_034124" (타임스탬프 포함)
    출력: {'product': '100W', 'sample': 0, 'direction': 'CW'}
    """
    parts = file_id.split('_')
    product = parts[0]
    sample_str = parts[1].replace('Sample', '')
    sample = int(sample_str)
    direction = parts[2]
    # parts[3] 이후는 타임스탬프 (메타데이터 파싱에는 불필요)

    return {
        'product': product,
        'sample': sample,
        'direction': direction
    }

def is_normal_file(product: str, sample: int) -> bool:
    """
    양품 파일 여부 판단

    100W: Sample00만 양품
    200W: Sample03만 양품
    """
    return sample in NORMAL_SAMPLES.get(product, [])

def load_csv_file(file_path: str) -> pd.DataFrame:
    """
    CSV 파일 로드 (7개 채널)

    메타데이터를 건너뛰고 실제 데이터만 읽기
    채널: acc-X, acc-Y, acc-Z, acc-Sum, Gyro-X, Gyro-Y, Gyro-Z
    """
    # 메타데이터를 건너뛰고 실제 데이터만 읽기
    # "DataSet" 줄 이후의 데이터 읽기
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # "DataSet" 줄 찾기
    data_start_idx = None
    for i, line in enumerate(lines):
        if line.strip() == 'DataSet':
            data_start_idx = i + 1  # 다음 줄이 헤더
            break

    if data_start_idx is None:
        raise ValueError(f"'DataSet' 줄을 찾을 수 없습니다: {file_path}")

    # 헤더 + 데이터 읽기
    df = pd.read_csv(file_path, skiprows=data_start_idx)

    # TimeStamp 컬럼 제거 (사용하지 않음)
    if 'TimeStamp' in df.columns:
        df = df.drop(columns=['TimeStamp'])

    # 결측치 처리 (선형 보간)
    df = df.interpolate(method='linear', axis=0).fillna(method='bfill').fillna(method='ffill')

    return df

# ============================================================================
# 4. 윈도우 세그먼테이션 함수
# ============================================================================

def get_stable_range(total_length: int, margin: float = 0.1) -> Tuple[int, int]:
    """
    안정 구간 추출 (앞/뒤 margin% 제외)

    예: total_length=10000, margin=0.1 → (1000, 9000)
    """
    start_idx = int(total_length * margin)
    end_idx = int(total_length * (1.0 - margin))
    return start_idx, end_idx

def create_windows(timeseries: pd.DataFrame, fs: float, window_sec: float, hop_sec: float) -> List[pd.DataFrame]:
    """
    시계열 데이터를 윈도우로 분할

    Args:
        timeseries: 시계열 데이터 (N × 7)
        fs: 샘플링 주파수 (Hz)
        window_sec: 윈도우 크기 (초)
        hop_sec: hop 크기 (초)

    Returns:
        윈도우 리스트
    """
    window_size = int(window_sec * fs)
    hop_size = int(hop_sec * fs)

    # 안정 구간 추출
    start_idx, end_idx = get_stable_range(len(timeseries), STABLE_MARGIN)
    stable_ts = timeseries.iloc[start_idx:end_idx].reset_index(drop=True)

    windows = []
    for start in range(0, len(stable_ts) - window_size + 1, hop_size):
        end = start + window_size
        window = stable_ts.iloc[start:end].copy()
        windows.append(window)

    return windows

def assign_time_split(window_idx: int, total_windows: int) -> str:
    """
    윈도우의 시간 기반 데이터셋 할당 (양품 파일용)

    Args:
        window_idx: 윈도우 인덱스 (0부터 시작)
        total_windows: 전체 윈도우 개수

    Returns:
        'train', 'val', 'test' 중 하나
    """
    position = window_idx / total_windows

    if position < TIME_SPLIT_RANGES['train'][1]:
        return 'train'
    elif position < TIME_SPLIT_RANGES['val'][1]:
        return 'val'
    else:
        return 'test'

# ============================================================================
# 5. 특징 추출 함수
# ============================================================================

def compute_rms(x: np.ndarray) -> float:
    """RMS (Root Mean Square) 계산"""
    return np.sqrt(np.mean(x**2))

def compute_peak(x: np.ndarray) -> float:
    """Peak 값 계산 (절댓값의 최댓값)"""
    return np.max(np.abs(x))

def compute_crest_factor(x: np.ndarray) -> float:
    """Crest Factor 계산 (Peak / RMS)"""
    rms_val = compute_rms(x)
    peak_val = compute_peak(x)
    return peak_val / rms_val if rms_val > 0 else 0.0

def butterworth_bandpass(lowcut: float, highcut: float, fs: float, order: int = 4):
    """
    Butterworth 밴드패스 필터 설계

    Args:
        lowcut: 하한 주파수 (Hz)
        highcut: 상한 주파수 (Hz)
        fs: 샘플링 주파수 (Hz)
        order: 필터 차수

    Returns:
        필터 계수 (b, a)
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def compute_band_rms(signal_data: np.ndarray, fs: float, band: Tuple[float, float]) -> float:
    """
    특정 주파수 밴드의 RMS 계산

    Args:
        signal_data: 신호 데이터 (1D array)
        fs: 샘플링 주파수 (Hz)
        band: 주파수 범위 (lowcut, highcut)

    Returns:
        밴드 RMS 값
    """
    lowcut, highcut = band
    b, a = butterworth_bandpass(lowcut, highcut, fs, order=4)

    # 필터 적용
    filtered_signal = signal.filtfilt(b, a, signal_data)

    # RMS 계산
    rms_val = compute_rms(filtered_signal)

    return rms_val

def extract_basic_features(window: pd.DataFrame) -> Dict[str, float]:
    """
    기본 통계 특징 추출 (9개)

    채널: acc-Y, acc-Sum, Gyro-Y
    통계량: RMS, Peak, Crest Factor
    """
    features = {}

    # acc-Y
    acc_y = window['acc-Y'].values
    features['acc_Y_rms'] = compute_rms(acc_y)
    features['acc_Y_peak'] = compute_peak(acc_y)
    features['acc_Y_crest'] = compute_crest_factor(acc_y)

    # acc-Sum
    acc_sum = window['acc-Sum'].values
    features['acc_Sum_rms'] = compute_rms(acc_sum)
    features['acc_Sum_peak'] = compute_peak(acc_sum)
    features['acc_Sum_crest'] = compute_crest_factor(acc_sum)

    # Gyro-Y
    gyro_y = window['Gyro-Y'].values
    features['Gyro_Y_rms'] = compute_rms(gyro_y)
    features['Gyro_Y_peak'] = compute_peak(gyro_y)
    features['Gyro_Y_crest'] = compute_crest_factor(gyro_y)

    return features

def extract_band_rms_features(window: pd.DataFrame, fs: float) -> Dict[str, float]:
    """
    밴드 RMS 특징 추출 (9개)

    채널: acc-Y, acc-Sum, Gyro-Y
    밴드: Low (1-10 Hz), Mid (10-50 Hz), High (50-150 Hz)
    """
    features = {}

    # 각 채널에 대해
    for channel_name, column_name in [('acc_Y', 'acc-Y'), ('acc_Sum', 'acc-Sum'), ('Gyro_Y', 'Gyro-Y')]:
        signal_data = window[column_name].values

        # 각 주파수 밴드에 대해
        for band_name, band_range in FREQUENCY_BANDS.items():
            feature_name = f"{channel_name}_rms_{band_name}"
            features[feature_name] = compute_band_rms(signal_data, fs, band_range)

    return features

def extract_window_features(window: pd.DataFrame, fs: float) -> Dict[str, float]:
    """
    윈도우에서 모든 특징 추출 (18개)

    Returns:
        9개 기본 통계량 + 9개 밴드 RMS = 18개 특징
    """
    features = {}

    # 기본 통계 특징 (9개)
    basic_feat = extract_basic_features(window)
    features.update(basic_feat)

    # 밴드 RMS 특징 (9개)
    band_feat = extract_band_rms_features(window, fs)
    features.update(band_feat)

    return features

# ============================================================================
# 6. 메인 파이프라인
# ============================================================================

def process_file(
    row: pd.Series,
    fs: float,
    label_raw: str,
    label_binary: int,
    label_weight: float
) -> List[Dict]:
    """
    단일 파일 처리: CSV 로드 → 윈도우 생성 → 특징 추출

    Args:
        row: ecmData의 한 행 (file_path, file_id)
        fs: 샘플링 주파수
        label_raw: 원본 라벨 (정상/소음/진동/표기없음)
        label_binary: 이진 라벨 (1=양품, 0=불량)
        label_weight: 라벨 가중치 (학습용)

    Returns:
        윈도우별 특징 딕셔너리 리스트
    """
    file_path_str = row['file_path']
    file_id = row['file_id']

    # 파일 경로 처리 (상대 경로 → 절대 경로)
    file_path = Path(file_path_str)
    if not file_path.is_absolute():
        # 상대 경로는 PROJECT_ROOT 기준으로 변환
        file_path = PROJECT_ROOT / file_path

    # 메타데이터 파싱
    meta = parse_file_id(file_id)
    product = meta['product']
    sample = meta['sample']
    direction = meta['direction']

    # CSV 파일 로드
    timeseries = load_csv_file(str(file_path))

    # 윈도우 생성
    windows = create_windows(timeseries, fs, WINDOW_SEC, HOP_SEC)

    # 각 윈도우 처리
    window_features_list = []

    for window_idx, window in enumerate(windows):
        # 데이터셋 할당
        if is_normal_file(product, sample):
            # 양품: 시간 기반 분할
            dataset_type = assign_time_split(window_idx, len(windows))
        else:
            # 불량: train으로 기본 설정 (나중에 파일 단위로 재분할)
            dataset_type = 'train'

        # 특징 추출
        features = extract_window_features(window, fs)

        # 메타데이터 추가
        window_data = {
            'window_id': f"{file_id}_win{window_idx:03d}",
            'file_id': file_id,
            'dataset_type': dataset_type,
            'label_raw': label_raw,
            'label_binary': label_binary,
            'label_weight': label_weight,
            'product': product,
            'sample': sample,
            'direction': direction
        }
        window_data.update(features)

        window_features_list.append(window_data)

    return window_features_list

def assign_abnormal_splits(df: pd.DataFrame) -> pd.DataFrame:
    """
    불량 파일의 윈도우를 train/val/test로 분할

    Args:
        df: 전체 윈도우 데이터프레임

    Returns:
        dataset_type이 할당된 데이터프레임
    """
    # 양품은 이미 시간 기반으로 할당되어 있음
    normal_df = df[df['label_binary'] == 1].copy()

    # 불량 윈도우만 추출
    abnormal_df = df[df['label_binary'] == 0].copy()

    if len(abnormal_df) == 0:
        return df

    # 파일 ID별로 그룹화
    file_ids = abnormal_df['file_id'].unique()
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(file_ids)

    # 파일 단위로 train/val/test 분할
    n_files = len(file_ids)
    n_train = int(n_files * ABNORMAL_SPLIT_RATIOS['train'])
    n_val = int(n_files * ABNORMAL_SPLIT_RATIOS['val'])

    train_files = file_ids[:n_train]
    val_files = file_ids[n_train:n_train+n_val]
    test_files = file_ids[n_train+n_val:]

    # 파일 ID에 따라 dataset_type 할당
    abnormal_df.loc[abnormal_df['file_id'].isin(train_files), 'dataset_type'] = 'train'
    abnormal_df.loc[abnormal_df['file_id'].isin(val_files), 'dataset_type'] = 'val'
    abnormal_df.loc[abnormal_df['file_id'].isin(test_files), 'dataset_type'] = 'test'

    # 양품과 불량 합치기
    result_df = pd.concat([normal_df, abnormal_df], ignore_index=True)

    return result_df

# ============================================================================
# 7. 실행 코드
# ============================================================================

print("\n[1단계] 입력 CSV 로드")
print("-" * 60)

# 입력 CSV 로드 (ecminer_input_full.csv)
if not INPUT_CSV_PATH.exists():
    raise FileNotFoundError(
        f"입력 CSV 파일을 찾을 수 없습니다: {INPUT_CSV_PATH}\n"
        f"먼저 create_ecminer_input.py를 실행하여 입력 파일을 생성하세요."
    )

ecmData = pd.read_csv(INPUT_CSV_PATH)
print(f"✓ 입력 CSV 로드 완료: {len(ecmData)}개 파일")

# 필수 컬럼 검증
required_columns = ['file_path', 'file_id']
for col in required_columns:
    if col not in ecmData.columns:
        raise ValueError(f"입력 데이터에 '{col}' 컬럼이 없습니다.")

# ============================================================================
# 레이블 시스템 초기화
# ============================================================================

print("\n[2단계] 레이블 시스템 초기화")
print("-" * 60)

# 레이블 파일 로드
if not LABEL_CSV_PATH.exists():
    raise FileNotFoundError(f"레이블 파일을 찾을 수 없습니다: {LABEL_CSV_PATH}")

labels_df = load_label_file(LABEL_CSV_PATH)
print(f"✓ 레이블 파일 로드 완료: {len(labels_df)}개 레이블")

# 설정 파일 로드
if not CONFIG_YAML_PATH.exists():
    raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {CONFIG_YAML_PATH}")

config = load_config(CONFIG_YAML_PATH)
strategy_name = config['label_strategy']
strategy = config['strategies'][strategy_name]
print(f"✓ 레이블 전략: {strategy_name} - {strategy.get('description', '')}")

# ============================================================================
# 전체 파일 처리
# ============================================================================

print("\n[3단계] 파일 처리")
print("-" * 60)

all_windows = []
processed_count = 0
error_count = 0

for idx, row in ecmData.iterrows():
    try:
        # file_id에서 product, sample 추출
        meta = parse_file_id(row['file_id'])
        product = meta['product']
        sample = meta['sample']

        # 레이블 매칭
        label_row = labels_df[
            (labels_df['product'] == product) &
            (labels_df['sample'] == sample)
        ]

        if len(label_row) == 0:
            print(f"경고: 레이블 없음, 건너뜀: {product} Sample{sample:02d}")
            error_count += 1
            continue

        label_raw = label_row['label'].iloc[0]

        # 레이블 전략 적용
        label_binary, label_weight = apply_label_strategy(
            label_raw, product, sample, strategy
        )

        # 파일 처리
        windows = process_file(
            row, SAMPLING_FREQ,
            label_raw=label_raw,
            label_binary=label_binary,
            label_weight=label_weight
        )
        all_windows.extend(windows)
        processed_count += 1

        if processed_count % 10 == 0:
            print(f"  진행중: {processed_count}/{len(ecmData)} 파일 처리됨...")

    except Exception as e:
        print(f"경고: 파일 {row['file_id']} 처리 중 오류: {e}")
        error_count += 1
        continue

print(f"✓ 파일 처리 완료: {processed_count}개 성공, {error_count}개 실패")

# 데이터프레임 생성
output_df = pd.DataFrame(all_windows)

# 불량 파일 데이터셋 분할
output_df = assign_abnormal_splits(output_df)

# 결과 정렬 (train → val → test 순서)
dataset_order = {'train': 0, 'val': 1, 'test': 2}
output_df['_sort_key'] = output_df['dataset_type'].map(dataset_order)
output_df = output_df.sort_values('_sort_key').drop(columns=['_sort_key']).reset_index(drop=True)

# ============================================================================
# 결과 출력
# ============================================================================

print("\n[4단계] 처리 완료")
print("-" * 60)
print(f"  - 총 윈도우 수: {len(output_df)}")
print(f"  - Train: {len(output_df[output_df['dataset_type']=='train'])}")
print(f"  - Val: {len(output_df[output_df['dataset_type']=='val'])}")
print(f"  - Test: {len(output_df[output_df['dataset_type']=='test'])}")
print(f"  - 정상: {len(output_df[output_df['label_binary']==1])}")
print(f"  - 불량: {len(output_df[output_df['label_binary']==0])}")
print(f"  - 특징 개수: {len(BASIC_FEATURES + BAND_RMS_FEATURES)}개")

# NaN 검사
if output_df.isnull().any().any():
    print("경고: 결과에 NaN 값이 포함되어 있습니다.")
    nan_cols = output_df.columns[output_df.isnull().any()].tolist()
    print(f"  NaN 포함 컬럼: {nan_cols}")

# 중복 window_id 검사
if output_df['window_id'].duplicated().any():
    n_duplicates = output_df['window_id'].duplicated().sum()
    print(f"경고: {n_duplicates}개의 중복 window_id가 발견되었습니다.")
else:
    print("  ✓ 중복 window_id 없음")

# ECMiner가 다음 노드로 전달할 데이터
ecmData = output_df

# 파일로 저장 (선택사항)
if OUTPUT_CSV_PATH is not None:
    output_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\n✓ 파일 저장 완료: {OUTPUT_CSV_PATH}")

print("\n" + "=" * 60)
print("ECMiner 출력 준비 완료!")
print("=" * 60)
