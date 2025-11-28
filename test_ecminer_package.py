"""
ECMiner 패키지 테스트 스크립트
ecminer_stage1_py_node.py 스크립트를 실행하여 출력 CSV 생성
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).parent

# ============================================================================
# 1. ECMiner 입력 CSV 로드
# ============================================================================

print("=" * 80)
print("ECMiner Stage 1 패키지 테스트")
print("=" * 80)

input_csv = PROJECT_ROOT / "ecminer_input_full.csv"

if not input_csv.exists():
    print(f"❌ 입력 CSV 파일이 없습니다: {input_csv}")
    print("먼저 create_ecminer_input.py를 실행하세요.")
    sys.exit(1)

print(f"\n[1/4] ECMiner 입력 CSV 로드: {input_csv}")
ecmData = pd.read_csv(input_csv)

print(f"  - 파일 수: {len(ecmData)}")
print(f"  - 정상 파일: {len(ecmData[ecmData['label'] == '정상'])}")
print(f"  - 불량 파일: {len(ecmData[ecmData['label'] == '표기없음'])}")

# ============================================================================
# 2. ECMiner 스크립트 실행 (ecminer_stage1_py_node.py 내용 임베드)
# ============================================================================

print(f"\n[2/4] 전처리 스크립트 실행 중...")

# ecminer_stage1_py_node.py 스크립트 실행
# (스크립트 내용을 직접 임베드하여 실행)

from scipy import signal
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# 설정
SAMPLING_FREQ = 512.0
WINDOW_SEC = 8.0
HOP_SEC = 4.0
STABLE_MARGIN = 0.1

NORMAL_SAMPLES = {
    '100W': [0],
    '200W': [3]
}

TIME_SPLIT_RANGES = {
    'train': (0.0, 0.6),
    'val': (0.6, 0.8),
    'test': (0.8, 1.0)
}

ABNORMAL_SPLIT_RATIOS = {
    'train': 0.7,
    'val': 0.15,
    'test': 0.15
}
RANDOM_SEED = 42

FREQUENCY_BANDS = {
    'low': (1, 10),
    'mid': (10, 50),
    'high': (50, 150)
}

# 함수 정의
def parse_file_id(file_id: str) -> Dict:
    parts = file_id.split('_')
    product = parts[0]
    sample_str = parts[1].replace('Sample', '')
    sample = int(sample_str)
    direction = parts[2]
    return {'product': product, 'sample': sample, 'direction': direction}

def is_normal_file(product: str, sample: int) -> bool:
    return sample in NORMAL_SAMPLES.get(product, [])

def assign_label_binary(label: str, product: str, sample: int) -> int:
    if is_normal_file(product, sample):
        return 1
    else:
        return 0

def load_csv_file(file_path: str) -> pd.DataFrame:
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

    # 결측치 처리
    df = df.interpolate(method='linear', axis=0).fillna(method='bfill').fillna(method='ffill')

    return df

def get_stable_range(total_length: int, margin: float = 0.1) -> Tuple[int, int]:
    start_idx = int(total_length * margin)
    end_idx = int(total_length * (1.0 - margin))
    return start_idx, end_idx

def create_windows(timeseries: pd.DataFrame, fs: float, window_sec: float, hop_sec: float) -> List[pd.DataFrame]:
    window_size = int(window_sec * fs)
    hop_size = int(hop_sec * fs)

    start_idx, end_idx = get_stable_range(len(timeseries), STABLE_MARGIN)
    stable_ts = timeseries.iloc[start_idx:end_idx].reset_index(drop=True)

    windows = []
    for start in range(0, len(stable_ts) - window_size + 1, hop_size):
        end = start + window_size
        window = stable_ts.iloc[start:end].copy()
        windows.append(window)

    return windows

def assign_time_split(window_idx: int, total_windows: int) -> str:
    position = window_idx / total_windows

    if position < TIME_SPLIT_RANGES['train'][1]:
        return 'train'
    elif position < TIME_SPLIT_RANGES['val'][1]:
        return 'val'
    else:
        return 'test'

def compute_rms(x: np.ndarray) -> float:
    return np.sqrt(np.mean(x**2))

def compute_peak(x: np.ndarray) -> float:
    return np.max(np.abs(x))

def compute_crest_factor(x: np.ndarray) -> float:
    rms_val = compute_rms(x)
    peak_val = compute_peak(x)
    return peak_val / rms_val if rms_val > 0 else 0.0

def butterworth_bandpass(lowcut: float, highcut: float, fs: float, order: int = 4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def compute_band_rms(signal_data: np.ndarray, fs: float, band: Tuple[float, float]) -> float:
    lowcut, highcut = band
    b, a = butterworth_bandpass(lowcut, highcut, fs, order=4)
    filtered_signal = signal.filtfilt(b, a, signal_data)
    rms_val = compute_rms(filtered_signal)
    return rms_val

def extract_basic_features(window: pd.DataFrame) -> Dict[str, float]:
    features = {}

    acc_y = window['acc-Y'].values
    features['acc_Y_rms'] = compute_rms(acc_y)
    features['acc_Y_peak'] = compute_peak(acc_y)
    features['acc_Y_crest'] = compute_crest_factor(acc_y)

    acc_sum = window['acc-Sum'].values
    features['acc_Sum_rms'] = compute_rms(acc_sum)
    features['acc_Sum_peak'] = compute_peak(acc_sum)
    features['acc_Sum_crest'] = compute_crest_factor(acc_sum)

    gyro_y = window['Gyro-Y'].values
    features['Gyro_Y_rms'] = compute_rms(gyro_y)
    features['Gyro_Y_peak'] = compute_peak(gyro_y)
    features['Gyro_Y_crest'] = compute_crest_factor(gyro_y)

    return features

def extract_band_rms_features(window: pd.DataFrame, fs: float) -> Dict[str, float]:
    features = {}

    for channel_name, column_name in [('acc_Y', 'acc-Y'), ('acc_Sum', 'acc-Sum'), ('Gyro_Y', 'Gyro-Y')]:
        signal_data = window[column_name].values

        for band_name, band_range in FREQUENCY_BANDS.items():
            feature_name = f"{channel_name}_rms_{band_name}"
            features[feature_name] = compute_band_rms(signal_data, fs, band_range)

    return features

def extract_window_features(window: pd.DataFrame, fs: float) -> Dict[str, float]:
    features = {}
    basic_feat = extract_basic_features(window)
    features.update(basic_feat)
    band_feat = extract_band_rms_features(window, fs)
    features.update(band_feat)
    return features

def process_file(row: pd.Series, fs: float) -> List[Dict]:
    file_path = row['file_path']
    file_id = row['file_id']
    label = row['label']

    meta = parse_file_id(file_id)
    product = meta['product']
    sample = meta['sample']
    direction = meta['direction']

    label_binary = assign_label_binary(label, product, sample)

    # 파일 경로 절대 경로로 변환
    abs_file_path = PROJECT_ROOT / file_path

    timeseries = load_csv_file(str(abs_file_path))
    windows = create_windows(timeseries, fs, WINDOW_SEC, HOP_SEC)

    window_features_list = []

    for window_idx, window in enumerate(windows):
        if is_normal_file(product, sample):
            dataset_type = assign_time_split(window_idx, len(windows))
        else:
            dataset_type = 'train'

        features = extract_window_features(window, fs)

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
    normal_df = df[df['label_binary'] == 1].copy()
    abnormal_df = df[df['label_binary'] == 0].copy()

    if len(abnormal_df) == 0:
        return df

    file_ids = abnormal_df['file_id'].unique()
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(file_ids)

    n_files = len(file_ids)
    n_train = int(n_files * ABNORMAL_SPLIT_RATIOS['train'])
    n_val = int(n_files * ABNORMAL_SPLIT_RATIOS['val'])

    train_files = file_ids[:n_train]
    val_files = file_ids[n_train:n_train+n_val]
    test_files = file_ids[n_train+n_val:]

    abnormal_df.loc[abnormal_df['file_id'].isin(train_files), 'dataset_type'] = 'train'
    abnormal_df.loc[abnormal_df['file_id'].isin(val_files), 'dataset_type'] = 'val'
    abnormal_df.loc[abnormal_df['file_id'].isin(test_files), 'dataset_type'] = 'test'

    result_df = pd.concat([normal_df, abnormal_df], ignore_index=True)

    return result_df

# 전체 파일 처리
all_windows = []

for idx, row in ecmData.iterrows():
    try:
        windows = process_file(row, SAMPLING_FREQ)
        all_windows.extend(windows)
        if (idx + 1) % 10 == 0:
            print(f"  진행: {idx + 1}/{len(ecmData)} 파일 처리 완료")
    except Exception as e:
        print(f"  경고: 파일 {row['file_id']} 처리 중 오류: {e}")
        continue

print(f"  완료: {len(ecmData)}개 파일 처리")

# 출력 데이터프레임 생성
output_df = pd.DataFrame(all_windows)
output_df = assign_abnormal_splits(output_df)

# 정렬
dataset_order = {'train': 0, 'val': 1, 'test': 2}
output_df['_sort_key'] = output_df['dataset_type'].map(dataset_order)
output_df = output_df.sort_values('_sort_key').drop(columns=['_sort_key']).reset_index(drop=True)

# ecmData 덮어쓰기 (ECMiner 출력)
ecmData = output_df

# ============================================================================
# 3. 출력 검증
# ============================================================================

print(f"\n[3/4] 출력 CSV 검증")
print(f"  - 총 윈도우 수: {len(ecmData)}")
print(f"  - Train: {len(ecmData[ecmData['dataset_type']=='train'])}")
print(f"  - Val: {len(ecmData[ecmData['dataset_type']=='val'])}")
print(f"  - Test: {len(ecmData[ecmData['dataset_type']=='test'])}")
print(f"  - 정상 (label=1): {len(ecmData[ecmData['label_binary']==1])}")
print(f"  - 불량 (label=0): {len(ecmData[ecmData['label_binary']==0])}")

# NaN 검사
if ecmData.isnull().any().any():
    print("  ⚠️  NaN 값 발견")
    nan_cols = ecmData.columns[ecmData.isnull().any()].tolist()
    print(f"     NaN 포함 컬럼: {nan_cols}")
else:
    print("  ✓ NaN 없음")

# 중복 검사
if ecmData['window_id'].duplicated().any():
    n_duplicates = ecmData['window_id'].duplicated().sum()
    print(f"  ⚠️  중복 window_id: {n_duplicates}개")
else:
    print("  ✓ 중복 window_id 없음")

# 특징 개수
feature_cols = [col for col in ecmData.columns if col not in
                ['window_id', 'file_id', 'dataset_type', 'label_binary', 'product', 'sample', 'direction']]
print(f"  - 특징 개수: {len(feature_cols)}")

# CSV 저장
output_csv = PROJECT_ROOT / "ecminer_output_test.csv"
ecmData.to_csv(output_csv, index=False)
print(f"\n  ✅ 출력 CSV 저장: {output_csv}")

print("\n" + "=" * 80)
print("✅ ECMiner Stage 1 패키지 테스트 완료")
print("=" * 80)
