#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ECMiner 파이썬 연동 노드용 스크립트

사용 방법:
1. ECMiner에서 빈 노드 → Python 연동 노드 추가
2. 이 스크립트 내용을 편집기에 복사-붙여넣기
3. 아래 "경로 설정" 섹션의 PROJECT_ROOT만 수정
4. 실행 → 다음 노드로 결과 전달됨

주의사항:
- 이 스크립트는 ECMiner 임시 폴더에서 실행됩니다
- 모든 파일 참조는 절대 경로를 사용합니다
- 상대 경로(SCRIPT_DIR 등)는 사용하지 않습니다
"""

# ============================================================================
# 🔧 경로 설정 (사용자가 수정하는 영역)
# ============================================================================
#
# ECMiner 파이썬 연동 노드를 만들 때마다 아래 경로를 시스템에 맞게 수정하세요
#
# ============================================================================

# 프로젝트 루트 경로 (절대 경로)
# 이 경로 아래에 data/, ECMiner_Package/ 폴더가 있어야 함
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
import re
from datetime import datetime

warnings.filterwarnings('ignore')

# 절대 경로 구성
PROJECT_ROOT = Path(PROJECT_ROOT)
DATA_ROOT = PROJECT_ROOT / "data"
LABEL_CSV_PATH = PROJECT_ROOT / "ECMiner_Package" / "ecminer_labels.csv"
CONFIG_YAML_PATH = PROJECT_ROOT / "ECMiner_Package" / "ecminer_config.yaml"
OUTPUT_CSV_PATH = PROJECT_ROOT / OUTPUT_FILENAME if SAVE_OUTPUT_FILE else None

print("=" * 60)
print("ECMiner 파이썬 연동 노드: 진동 데이터 전처리")
print("=" * 60)
print(f"\nPROJECT_ROOT: {PROJECT_ROOT}")
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

# 데이터셋 분할 비율
TIME_SPLIT_RATIOS = {
    'train': 0.6,
    'val': 0.2,
    'test': 0.2
}

ABNORMAL_SPLIT_RATIOS = {
    'train': 0.7,
    'val': 0.15,
    'test': 0.15
}

RANDOM_SEED = 42

# 특징 이름 정의
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

# 주파수 밴드 정의 (Hz)
FREQ_BANDS = {
    'low': (0, 50),
    'mid': (50, 150),
    'high': (150, 256)  # Nyquist 주파수의 절반
}

# ============================================================================
# 2. 레이블 시스템 함수
# ============================================================================

def load_label_file(label_path: Path) -> pd.DataFrame:
    """레이블 파일 로드 (CSV 또는 Excel)"""
    if label_path.suffix == '.csv':
        return pd.read_csv(label_path, encoding='utf-8-sig')
    elif label_path.suffix in ['.xlsx', '.xls']:
        return pd.read_excel(label_path)
    else:
        raise ValueError(f"지원하지 않는 파일 형식: {label_path.suffix}")

def load_config(config_path: Path) -> Dict:
    """설정 파일 로드 (YAML)"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def apply_label_strategy(
    label_raw: str,
    product: str,
    sample: int,
    strategy: Dict
) -> Tuple[int, float]:
    """레이블 전략 적용: label_raw → (label_binary, label_weight)"""
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

    return {
        'product': product,
        'sample': sample,
        'direction': direction
    }

def parse_filename(filename: str, product: str) -> Optional[Dict]:
    """
    파일명에서 메타데이터 추출

    예: "100W_Sample00 cw4_2025-11-07 03-41-24.csv"
    → product=100W, sample=0, direction=CW, timestamp=20251107_034124
    """
    # Sample 번호 추출
    sample_match = re.search(r'Sample(\d+)', filename, re.IGNORECASE)
    if not sample_match:
        return None

    sample = int(sample_match.group(1))

    # 회전 방향 추출 (cw/ccw)
    direction_match = re.search(r'(cw|ccw)', filename, re.IGNORECASE)
    if not direction_match:
        return None

    direction = direction_match.group(1).upper()

    # 타임스탬프 추출
    timestamp_match = re.search(r'(\d{4})-(\d{2})-(\d{2})\s+(\d{2})-(\d{2})-(\d{2})', filename)
    if timestamp_match:
        y, m, d, h, mi, s = timestamp_match.groups()
        timestamp = f"{y}{m}{d}_{h}{mi}{s}"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    return {
        'product': product,
        'sample': sample,
        'direction': direction,
        'timestamp': timestamp
    }

def is_normal_file(product: str, sample: int) -> bool:
    """양품 파일 여부 판단"""
    return sample in NORMAL_SAMPLES.get(product, [])

def load_csv_file(file_path: Path) -> pd.DataFrame:
    """
    CSV 파일 로드 (7개 채널)
    메타데이터를 건너뛰고 실제 데이터만 읽기
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # "DataSet" 줄 찾기
    data_start_idx = None
    for i, line in enumerate(lines):
        if line.strip() == 'DataSet':
            data_start_idx = i + 1
            break

    if data_start_idx is None:
        raise ValueError(f"'DataSet' 줄을 찾을 수 없습니다: {file_path}")

    # 헤더 + 데이터 읽기
    df = pd.read_csv(file_path, skiprows=data_start_idx)

    # TimeStamp 컬럼 제거
    if 'TimeStamp' in df.columns:
        df = df.drop(columns=['TimeStamp'])

    # 결측치 처리
    df = df.interpolate(method='linear', axis=0).fillna(method='bfill').fillna(method='ffill')

    return df

# ============================================================================
# 4. 윈도우 세그먼테이션 함수
# ============================================================================

def get_stable_range(total_length: int, margin: float = 0.1) -> Tuple[int, int]:
    """안정 구간 추출 (앞/뒤 margin% 제외)"""
    margin_samples = int(total_length * margin)
    start = margin_samples
    end = total_length - margin_samples
    return start, end

def create_windows(data: pd.DataFrame, fs: float, window_sec: float, hop_sec: float) -> List[np.ndarray]:
    """슬라이딩 윈도우 생성 (안정 구간만 사용)"""
    stable_start, stable_end = get_stable_range(len(data), STABLE_MARGIN)
    stable_data = data.iloc[stable_start:stable_end]

    window_samples = int(window_sec * fs)
    hop_samples = int(hop_sec * fs)

    windows = []
    start = 0
    while start + window_samples <= len(stable_data):
        window = stable_data.iloc[start:start+window_samples].values
        windows.append(window)
        start += hop_samples

    return windows

def assign_time_split(window_idx: int, total_windows: int) -> str:
    """시간 순서대로 train/val/test 분할"""
    train_end = int(total_windows * TIME_SPLIT_RATIOS['train'])
    val_end = train_end + int(total_windows * TIME_SPLIT_RATIOS['val'])

    if window_idx < train_end:
        return 'train'
    elif window_idx < val_end:
        return 'val'
    else:
        return 'test'

# ============================================================================
# 5. 특징 추출 함수
# ============================================================================

def extract_window_features(window: np.ndarray, fs: float) -> Dict:
    """윈도우에서 18개 특징 추출"""
    features = {}

    # 채널 선택 (acc-Y, acc-Sum, Gyro-Y)
    channels = {
        'acc_Y': 1,
        'acc_Sum': 3,
        'Gyro_Y': 5
    }

    for ch_name, ch_idx in channels.items():
        signal_data = window[:, ch_idx]

        # 기본 통계량
        features[f'{ch_name}_rms'] = np.sqrt(np.mean(signal_data**2))
        features[f'{ch_name}_peak'] = np.max(np.abs(signal_data))
        features[f'{ch_name}_crest'] = features[f'{ch_name}_peak'] / (features[f'{ch_name}_rms'] + 1e-10)

        # 밴드별 RMS
        for band_name, (f_low, f_high) in FREQ_BANDS.items():
            sos = signal.butter(4, [f_low, f_high], btype='band', fs=fs, output='sos')
            filtered = signal.sosfilt(sos, signal_data)
            features[f'{ch_name}_rms_{band_name}'] = np.sqrt(np.mean(filtered**2))

    return features

# ============================================================================
# 6. 메인 파이프라인
# ============================================================================

def process_file(
    file_path: Path,
    file_id: str,
    fs: float,
    label_raw: str,
    label_binary: int,
    label_weight: float
) -> List[Dict]:
    """단일 파일 처리: CSV 로드 → 윈도우 생성 → 특징 추출"""
    # 메타데이터 파싱
    meta = parse_file_id(file_id)
    product = meta['product']
    sample = meta['sample']
    direction = meta['direction']

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
    """불량 파일의 윈도우를 train/val/test로 분할"""
    normal_df = df[df['label_binary'] == 1].copy()
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

def scan_data_folder(data_root: Path, product: str) -> List[Dict]:
    """
    데이터 폴더를 스캔하여 파일 목록 생성

    Args:
        data_root: data 폴더 경로
        product: "100W" 또는 "200W"

    Returns:
        파일 정보 리스트 [{'file_path': Path, 'file_id': str}, ...]
    """
    product_folder = data_root / product

    if not product_folder.exists():
        print(f"경고: {product_folder} 폴더가 없습니다.")
        return []

    files = []
    csv_files = list(product_folder.glob("*.csv"))

    for csv_file in csv_files:
        filename = csv_file.name

        # 파일명 파싱
        meta = parse_filename(filename, product)
        if meta is None:
            print(f"경고: 파일명 파싱 실패, 건너뜀: {filename}")
            continue

        # file_id 생성
        file_id = f"{meta['product']}_Sample{meta['sample']:02d}_{meta['direction']}_{meta['timestamp']}"

        files.append({
            'file_path': csv_file,
            'file_id': file_id
        })

    return files

# ============================================================================
# 7. 실행 코드
# ============================================================================

print("\n[1단계] 데이터 폴더 스캔")
print("-" * 60)

# 100W, 200W 폴더 스캔
all_files = []
all_files.extend(scan_data_folder(DATA_ROOT, "100W"))
all_files.extend(scan_data_folder(DATA_ROOT, "200W"))

print(f"✓ 총 {len(all_files)}개 파일 발견")

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

for file_info in all_files:
    try:
        file_path = file_info['file_path']
        file_id = file_info['file_id']

        # file_id에서 product, sample 추출
        meta = parse_file_id(file_id)
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
            file_path, file_id, SAMPLING_FREQ,
            label_raw=label_raw,
            label_binary=label_binary,
            label_weight=label_weight
        )
        all_windows.extend(windows)
        processed_count += 1

        if processed_count % 10 == 0:
            print(f"  진행중: {processed_count}/{len(all_files)} 파일 처리됨...")

    except Exception as e:
        print(f"경고: 파일 {file_info['file_id']} 처리 중 오류: {e}")
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
