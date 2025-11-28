"""
ECMiner 파이썬 연동 노드 - Stage 1: Feature Extraction
========================================================

목적:
- 로봇 암 액추에이터 진동 CSV 파일을 읽어 윈도우 단위로 특성 추출
- ECMiner가 처리할 수 있는 2D Tabular 데이터 생성
- XGBoost v3 모델 학습을 위한 18개 특성 추출

입력 (ecmData):
- 필수 컬럼:
  * file_path: Raw CSV 파일 경로
  * label: 파일 레벨 레이블 (Normal=1, Abnormal=0)
  * dataset_type: 'train' 또는 'test' (ECMiner에서 사전 분류)
- 선택 컬럼:
  * file_id: 파일 고유 ID (없으면 자동 생성)

출력 (ecmData):
- 윈도우 단위 Feature Table (2D DataFrame)
- 컬럼:
  * 메타: file_id, window_idx, start_idx, end_idx, label, dataset_type, product, serial, condition, load
  * 기본 특성 (9개): acc_Y_rms, acc_X_rms, Gyro_Y_rms, Gyro_X_rms,
                      acc_Y_peak, acc_Sum_peak,
                      acc_Y_crest,
                      acc_Y_kurtosis, acc_Sum_kurtosis
  * Band RMS (9개): acc_Y_rms_low/mid/high, acc_Sum_rms_low/mid/high, Gyro_Y_rms_low/mid/high

참고:
- ECMiner Stage2에서 Filter 노드로 18개 → 12개 특성 선택 (XGBoost v3)
- 선택 특성: 위 9개 + acc_Y_rms_high, Gyro_Y_rms_high, Gyro_Y_rms_low
"""

import os
from typing import Dict, Tuple, List
from io import StringIO

import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.stats import kurtosis as scipy_kurtosis

# ============================================================================
# 설정값
# ============================================================================

# 윈도우 파라미터
FS = 512  # 샘플링 레이트 [Hz]
WINDOW_SEC = 8.0
WINDOW_SIZE = int(FS * WINDOW_SEC)  # 4096
WINDOW_STEP = WINDOW_SIZE // 2      # 50% overlap

# 주파수 대역 [Hz]
BAND_LOW = (1.0, 10.0)    # 구조적 진동, 언밸런스
BAND_MID = (10.0, 50.0)   # 회전 고조파 (1×, 2× RPM)
BAND_HIGH = (50.0, 150.0) # 베어링 결함, 충격 이벤트

# ============================================================================
# 유틸 함수
# ============================================================================

def read_raw_csv_with_metadata(path: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    프로젝트 원본 CSV 형식 파싱

    형식:
      - 상단: "키,값" 형태의 메타데이터
      - 빈 줄 + 'DataSet' 마커
      - 헤더: TimeStamp,acc-X,acc-Y,acc-Z,Gyro-X,Gyro-Y,Gyro-Z,acc-Sum
      - 이후: 센서 데이터

    Returns:
      - data_df: 센서 시계열 DataFrame
      - meta: 메타데이터 dict
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    header_idx = None
    meta: Dict[str, str] = {}

    for idx, line in enumerate(lines):
        striped = line.strip()
        if not striped:
            continue

        if striped.lower() == "dataset":
            header_idx = idx + 1  # 다음 줄이 헤더
            break

        # 메타데이터 파싱
        if "," in line:
            parts = line.split(",", 1)
            if len(parts) == 2:
                key, value = parts
                key = key.strip()
                value = value.strip()
                if key and value and key.lower() != "dataset":
                    meta[key] = value

    # 헤더 찾기 (폴백)
    if header_idx is None:
        for idx, line in enumerate(lines):
            if line.startswith("TimeStamp"):
                header_idx = idx
                break

    if header_idx is None:
        raise ValueError(f"헤더 라인을 찾을 수 없습니다: {path}")

    # 데이터 파싱
    data_str = "".join(lines[header_idx:])
    data_df = pd.read_csv(StringIO(data_str))

    # 컬럼 이름 정리 (하이픈 → 언더스코어)
    rename_map = {
        "acc-X": "acc_X",
        "acc-Y": "acc_Y",
        "acc-Z": "acc_Z",
        "Gyro-X": "Gyro_X",
        "Gyro-Y": "Gyro_Y",
        "Gyro-Z": "Gyro_Z",
        "acc-Sum": "acc_Sum",
    }
    data_df = data_df.rename(columns=rename_map)

    return data_df, meta


def parse_file_id_from_meta(meta: Dict[str, str], fallback_path: str) -> str:
    """
    메타데이터에서 file_id 구성

    형식: {Product}_S{Sample}_{ Direction}_R{Run}
    예: 100W_S03_CW_R4
    """
    product = meta.get("Product", "").strip()  # 100W / 200W
    serial = meta.get("Serial", "").strip()    # "Sample03 cw4"
    condition = meta.get("Condition", "").strip().upper()  # "CW4" / "CCW4"

    sample_num = None
    direction = None
    run = None

    # Serial에서 추출
    if serial:
        parts = serial.replace(",", " ").split()
        for token in parts:
            t = token.strip()
            if t.lower().startswith("sample"):
                try:
                    sample_num = int(t.lower().replace("sample", ""))
                except:
                    pass
            # cw4, ccw4 등
            if "cw" in t.lower():
                t_low = t.lower()
                if "ccw" in t_low:
                    direction = "CCW"
                    run = "".join(ch for ch in t_low.split("ccw")[-1] if ch.isdigit())
                else:
                    direction = "CW"
                    run = "".join(ch for ch in t_low.split("cw")[-1] if ch.isdigit())

    # Condition에서 보조 정보
    if condition:
        if "CCW" in condition:
            direction = "CCW"
        elif "CW" in condition:
            direction = "CW"
        digits = "".join(ch for ch in condition if ch.isdigit())
        if digits:
            run = digits

    # 파일명에서 보조 추출
    if sample_num is None:
        base = os.path.basename(fallback_path)
        if "Sample" in base:
            try:
                tmp = base.split("Sample", 1)[1]
                num_str = ""
                for ch in tmp:
                    if ch.isdigit():
                        num_str += ch
                    else:
                        break
                sample_num = int(num_str)
            except:
                pass

    if direction is None:
        base = os.path.basename(fallback_path).lower()
        if "ccw" in base:
            direction = "CCW"
        elif "cw" in base:
            direction = "CW"

    if run is None:
        base = os.path.basename(fallback_path)
        digits = "".join(ch for ch in base if ch.isdigit())
        if digits:
            run = digits[-1]

    if not product:
        base = os.path.basename(fallback_path)
        if "_" in base:
            product = base.split("_", 1)[0]

    # 안전 장치
    sample_str = f"{sample_num:02d}" if sample_num is not None else "XX"
    direction = direction or "UNK"
    run = run or "0"

    file_id = f"{product}_S{sample_str}_{direction}_R{run}"
    return file_id


def sliding_windows(n_samples: int, window_size: int, step: int) -> List[Tuple[int, int]]:
    """
    슬라이딩 윈도우 인덱스 생성

    Returns:
      [(start_idx, end_idx), ...]  # end_idx는 exclusive (Python 슬라이스)
    """
    windows = []
    start = 0
    while start + window_size <= n_samples:
        end = start + window_size
        windows.append((start, end))
        start += step
    return windows


# ============================================================================
# 특성 추출 함수
# ============================================================================

def rms(x: np.ndarray) -> float:
    """Root Mean Square"""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.nan
    return float(np.sqrt(np.nanmean(x ** 2)))


def peak_abs(x: np.ndarray) -> float:
    """Peak (maximum absolute value)"""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.nan
    return float(np.nanmax(np.abs(x)))


def crest_factor(x: np.ndarray) -> float:
    """Crest Factor = Peak / RMS"""
    r = rms(x)
    p = peak_abs(x)
    if r <= 0 or np.isnan(r):
        return np.nan
    return float(p / r)


def kurtosis_excess(x: np.ndarray) -> float:
    """
    Kurtosis (excess, 정규분포 기준 0)
    scipy.stats.kurtosis 사용
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size < 4:
        return np.nan
    return float(scipy_kurtosis(x, fisher=True, bias=False, nan_policy='omit'))


def band_rms_butterworth(x: np.ndarray, fs: float, f_low: float, f_high: float) -> float:
    """
    Butterworth bandpass filter 기반 Band RMS 계산

    프로젝트 검증된 방식:
    1. Butterworth 4차 bandpass filter 설계
    2. filtfilt (zero-phase) 적용
    3. 필터링된 신호의 RMS 계산
    4. 실패 시 FFT 폴백
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan

    # Butterworth filter 설계
    nyquist = fs / 2.0
    low_normalized = f_low / nyquist
    high_normalized = f_high / nyquist

    # Edge case 처리
    if low_normalized <= 0:
        low_normalized = 0.01
    if high_normalized >= 1:
        high_normalized = 0.99

    try:
        # Butterworth 4차 bandpass filter
        b, a = signal.butter(4, [low_normalized, high_normalized], btype='band')
        filtered_signal = signal.filtfilt(b, a, x)
    except Exception:
        # FFT 폴백
        N = len(x)
        fft_values = fft(x)
        freqs = fftfreq(N, 1/fs)

        mask = (np.abs(freqs) >= f_low) & (np.abs(freqs) <= f_high)
        fft_filtered = fft_values.copy()
        fft_filtered[~mask] = 0

        filtered_signal = np.fft.ifft(fft_filtered).real

    return rms(filtered_signal)


def extract_features_from_window(win_df: pd.DataFrame) -> Dict[str, float]:
    """
    하나의 윈도우에 대해 18개 특성 계산 (Phase2 Step 3-4와 동일)

    특성 (18개):
    - 기본 9개: acc_Y_rms, acc_Y_peak, acc_Y_crest,
                acc_Sum_rms, acc_Sum_peak, acc_Sum_crest,
                Gyro_Y_rms, Gyro_Y_peak, Gyro_Y_crest
    - Band RMS 9개: {acc_Y, acc_Sum, Gyro_Y} × {low, mid, high}

    Phase3-1 Filter: 18개 → 12개 (acc_Sum_rms, acc_Sum_crest,
                                   acc_Y_crest 제외된 Band RMS 6개 제거)
    """
    acc_y = win_df["acc_Y"].values
    acc_sum = win_df["acc_Sum"].values
    gyro_y = win_df["Gyro_Y"].values

    features = {}

    # === 기본 특성 9개 (Phase2 Step 3-4) ===

    # acc_Y 관련 (3개)
    features["acc_Y_rms"] = rms(acc_y)
    features["acc_Y_peak"] = peak_abs(acc_y)
    features["acc_Y_crest"] = crest_factor(acc_y)

    # acc_Sum 관련 (3개)
    features["acc_Sum_rms"] = rms(acc_sum)
    features["acc_Sum_peak"] = peak_abs(acc_sum)
    features["acc_Sum_crest"] = crest_factor(acc_sum)

    # Gyro_Y 관련 (3개)
    features["Gyro_Y_rms"] = rms(gyro_y)
    features["Gyro_Y_peak"] = peak_abs(gyro_y)
    features["Gyro_Y_crest"] = crest_factor(gyro_y)

    # === Band RMS 특성 9개 (Phase2 Step 3-4) ===

    # acc_Y Band RMS (3개)
    features["acc_Y_rms_low"] = band_rms_butterworth(acc_y, FS, BAND_LOW[0], BAND_LOW[1])
    features["acc_Y_rms_mid"] = band_rms_butterworth(acc_y, FS, BAND_MID[0], BAND_MID[1])
    features["acc_Y_rms_high"] = band_rms_butterworth(acc_y, FS, BAND_HIGH[0], BAND_HIGH[1])

    # acc_Sum Band RMS (3개)
    features["acc_Sum_rms_low"] = band_rms_butterworth(acc_sum, FS, BAND_LOW[0], BAND_LOW[1])
    features["acc_Sum_rms_mid"] = band_rms_butterworth(acc_sum, FS, BAND_MID[0], BAND_MID[1])
    features["acc_Sum_rms_high"] = band_rms_butterworth(acc_sum, FS, BAND_HIGH[0], BAND_HIGH[1])

    # Gyro_Y Band RMS (3개)
    features["Gyro_Y_rms_low"] = band_rms_butterworth(gyro_y, FS, BAND_LOW[0], BAND_LOW[1])
    features["Gyro_Y_rms_mid"] = band_rms_butterworth(gyro_y, FS, BAND_MID[0], BAND_MID[1])
    features["Gyro_Y_rms_high"] = band_rms_butterworth(gyro_y, FS, BAND_HIGH[0], BAND_HIGH[1])

    return features


def build_feature_table_for_file(file_row: pd.Series) -> pd.DataFrame:
    """
    ecmData의 한 row (한 개 파일)에 대해 윈도우 단위 Feature Table 생성

    처리:
    1. Raw CSV 로드
    2. 메타데이터 파싱 및 file_id 생성
    3. 슬라이딩 윈도우 생성
    4. 각 윈도우마다 18개 특성 계산
    5. 메타 컬럼 추가 (file_id, window_idx, label, dataset_type, product...)
    """
    path = str(file_row["file_path"])
    label = int(file_row["label"])
    dataset_type = str(file_row.get("dataset_type", "unknown"))  # 'train' or 'test'

    file_id = file_row.get("file_id")
    if pd.isna(file_id):
        file_id = None
    else:
        file_id = str(file_id)

    # CSV 파싱
    data_df, meta = read_raw_csv_with_metadata(path)

    if not file_id:
        file_id = parse_file_id_from_meta(meta, path)

    # 윈도우 슬라이딩
    n_samples = len(data_df)
    win_indices = sliding_windows(n_samples, WINDOW_SIZE, WINDOW_STEP)
    total_windows = len(win_indices)

    records = []
    for win_idx, (start, end) in enumerate(win_indices):
        win_df = data_df.iloc[start:end].reset_index(drop=True)

        # 18개 특성 추출
        feats = extract_features_from_window(win_df)

        # dataset_type 결정
        # time_split 파일인 경우, 윈도우 위치에 따라 train/val/test 분할
        if dataset_type == 'time_split':
            # 프로젝트 방법론: train(0-60%), val(60-80%), test(80-100%)
            frac = win_idx / total_windows
            if frac < 0.6:
                window_dataset_type = 'train'
            elif frac < 0.8:
                window_dataset_type = 'val'
            else:
                window_dataset_type = 'test'
        else:
            # Abnormal 파일은 이미 train/test/val로 할당됨
            window_dataset_type = dataset_type

        # 레코드 구성
        rec = {
            "file_id": file_id,
            "window_idx": win_idx,
            "start_idx": start,
            "end_idx": end,
            "label": label,
            "dataset_type": window_dataset_type,
            # 메타데이터
            "product": meta.get("Product", ""),
            "serial": meta.get("Serial", ""),
            "condition": meta.get("Condition", ""),
            "load": meta.get("Load", ""),
        }
        rec.update(feats)
        records.append(rec)

    if not records:
        # 윈도우가 하나도 안 나온 경우 (파일 길이 < 8초)
        return pd.DataFrame()

    return pd.DataFrame.from_records(records)


def build_feature_table_from_ecmdata(ecm_df: pd.DataFrame) -> pd.DataFrame:
    """
    ECMiner 상위 노드에서 전달된 ecmData (파일 목록 + 레이블)를
    윈도우 단위 Feature Table로 변환

    입력 ecmData 필수 컬럼:
    - file_path: Raw CSV 경로
    - label: Normal=1, Abnormal=0
    - dataset_type: 'train' or 'test'

    출력:
    - 윈도우 × (18개 특성 + 메타 컬럼)
    """
    feature_tables = []

    for idx, row in ecm_df.iterrows():
        try:
            ft = build_feature_table_for_file(row)
            if not ft.empty:
                feature_tables.append(ft)
        except Exception as e:
            print(f"[WARN] 파일 처리 중 오류 (index={idx}, path={row.get('file_path')}): {e}")

    if not feature_tables:
        return pd.DataFrame()

    all_features = pd.concat(feature_tables, axis=0, ignore_index=True)

    return all_features


# ============================================================================
# ECMiner 진입점
# ============================================================================

# ECMiner는 이 스크립트 앞뒤에 ecmData 로딩/저장을 자동으로 붙여줌
# 여기서는 ecmData를 윈도우 단위 Feature Table로 덮어씀

# ECMiner 환경에서만 실행 (테스트/모듈 import시에는 실행하지 않음)
if 'ecmData' in globals():
    ecmData = build_feature_table_from_ecmdata(ecmData)

    # 최종 데이터 정보 출력 (ECMiner 로그 확인용)
    print(f"[INFO] Feature Extraction 완료:")
    print(f"  - 총 윈도우: {len(ecmData)}")
    print(f"  - 특성 개수: {len(ecmData.columns)}")
    if 'dataset_type' in ecmData.columns:
        print(f"  - Train: {len(ecmData[ecmData['dataset_type']=='train'])}")
        print(f"  - Test: {len(ecmData[ecmData['dataset_type']=='test'])}")
    print(f"  - Normal: {len(ecmData[ecmData['label']==1])}")
    print(f"  - Abnormal: {len(ecmData[ecmData['label']==0])}")
