"""
ECMiner Stage 1: 진동 데이터 전처리 및 특징 추출 (Python 노드용)
================================================================

입력 (ecmData):
    - file_path: 원본 CSV 파일 경로 (예: data/100W/Sample00_CW.csv)
    - file_id: 파일 고유 ID (product_sample_direction, 예: 100W_Sample00_CW)
    - label: 라벨 (정상, 소음, 진동, 표기없음)

출력 (ecmData):
    - window_id: 윈도우 고유 ID
    - file_id: 원본 파일 ID
    - dataset_type: 데이터셋 유형 (train/val/test)
    - label_binary: 이진 라벨 (1=정상, 0=불량)
    - product: 제품 (100W/200W)
    - sample: 시료 번호 (0-9)
    - direction: 회전 방향 (CW/CCW)
    - 18개 특징: 9개 기본 통계량 + 9개 밴드 RMS
"""

# ============================================================================
# --- ECMiner 진입점 (자동 처리) ---
# import pandas as pd
# ecmData = pd.read_csv('%temporary_input_csv_file%')
# ============================================================================

# --- 여기부터 사용자 코드 ---

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import signal
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

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
# 2. 데이터 로드 및 파싱 함수
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

def assign_label_binary(label: str, product: str, sample: int) -> int:
    """
    이진 라벨 할당 (1=정상, 0=불량)

    - 양품 파일: 1 (정상)
    - 불량 파일: 0 (불량)
    """
    if is_normal_file(product, sample):
        return 1  # 정상
    else:
        return 0  # 불량

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
# 3. 윈도우 세그먼테이션 함수
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
# 4. 특징 추출 함수
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
# 5. 메인 파이프라인
# ============================================================================

def process_file(row: pd.Series, fs: float) -> List[Dict]:
    """
    단일 파일 처리: CSV 로드 → 윈도우 생성 → 특징 추출

    Args:
        row: ecmData의 한 행 (file_path, file_id, label)
        fs: 샘플링 주파수

    Returns:
        윈도우별 특징 딕셔너리 리스트
    """
    file_path = row['file_path']
    file_id = row['file_id']
    label = row['label']

    # 메타데이터 파싱
    meta = parse_file_id(file_id)
    product = meta['product']
    sample = meta['sample']
    direction = meta['direction']

    # 이진 라벨 할당
    label_binary = assign_label_binary(label, product, sample)

    # CSV 파일 로드
    timeseries = load_csv_file(file_path)

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
            # 불량: 랜덤 분할 (파일 단위 - 여기서는 단순화를 위해 train으로 통일)
            # 실제로는 불량 파일들을 모아서 train/val/test로 분할해야 하지만,
            # ECMiner에서 이미 분할된 데이터를 받는다고 가정
            dataset_type = 'train'  # 기본값, 실제로는 외부에서 결정됨

        # 특징 추출
        features = extract_window_features(window, fs)

        # 메타데이터 추가
        window_data = {
            'window_id': f"{file_id}_win{window_idx:03d}",
            'file_id': file_id,
            'dataset_type': dataset_type,
            'label_binary': label_binary,
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
# 6. 실행 코드
# ============================================================================

# 입력 데이터 검증
required_columns = ['file_path', 'file_id', 'label']
for col in required_columns:
    if col not in ecmData.columns:
        raise ValueError(f"입력 데이터에 '{col}' 컬럼이 없습니다. ECMiner 입력 형식을 확인하세요.")

# 전체 파일 처리
all_windows = []

for idx, row in ecmData.iterrows():
    try:
        windows = process_file(row, SAMPLING_FREQ)
        all_windows.extend(windows)
    except Exception as e:
        print(f"경고: 파일 {row['file_id']} 처리 중 오류 발생: {e}")
        continue

# 데이터프레임 생성
output_df = pd.DataFrame(all_windows)

# 불량 파일 데이터셋 분할
output_df = assign_abnormal_splits(output_df)

# 결과 정렬 (train → val → test 순서)
dataset_order = {'train': 0, 'val': 1, 'test': 2}
output_df['_sort_key'] = output_df['dataset_type'].map(dataset_order)
output_df = output_df.sort_values('_sort_key').drop(columns=['_sort_key']).reset_index(drop=True)

# 출력 데이터 검증
print(f"\n처리 완료:")
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

# ecmData 덮어쓰기 (ECMiner 출력)
ecmData = output_df

# --- 여기까지 ---

# ============================================================================
# (ECMiner 내부 처리)
# ecmData.to_csv('%temporary_output_csv_file%', index=False)
# ============================================================================
