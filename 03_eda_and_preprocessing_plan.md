# 01_eda_and_preprocessing_plan.md

Created: 2025년 11월 17일 오전 11:04

**파일명 예시:** `01_eda_and_preprocessing_plan.md`

```markdown
# EDA 및 데이터 전처리·세그먼트 상세 설계 (클로드 코드용)

이 문서는 **클로드 코드(코딩 에이전트)**가 실제 파이썬 코드를 작성할 때
참조해야 하는 **상세 실행 계획**이다.

특히 **다음 작업을 명시적으로 구현**해야 한다:

1. 기존 CSV에서 메타데이터 분리
2. 엑셀 라벨/정보 매핑
3. 이상치(outlier) 처리 및 데이터 정제
4. 세그먼트(윈도우) 분할
5. 1차 분석용 시간영역 feature 추출의 기본 골격

> 주의:
> 이 문서는 **탐색적 분석만을 위한 문서가 아니다.**
> 이후 본 분석(XGBoost, 오토인코더)을 그대로 사용할 수 있도록
> **모듈화/재사용 가능한 구조**를 염두에 두고 설계한다.

---

## 1. 로우 데이터 로딩 및 메타데이터 분리

### 1.1 CSV 구조 가정

각 CSV 파일은 대략 다음과 같은 형태로 구성되어 있다.

```text
xRMS,0.123456
yRMS,0.234567
zRMS,0.345678
sumRMS,0.456789
xPeak,1.2345
...
MeasFreq,166.67
Resampling,512
LowFreq,1
HighFreq,150
...
DataSet
TimeStamp,acc-X,acc-Y,acc-Z,Gyro-X,Gyro-Y,Gyro-Z,acc-Sum
0.000000, ...
0.001953, ...
...
1.2 메타데이터 분리 로직
목표:
각 CSV 파일에 대해:

metadata_df: 파일당 1행 메타데이터 행

timeseries_df: 파일당 여러 행의 시계열 데이터 (TimeStamp + 센서 값)

구현 규칙 (의사코드)

python
코드 복사
def parse_csv_with_metadata(path: Path) -> tuple[dict, pd.DataFrame]:
    """
    path에 있는 CSV 파일을 읽어
    - header 메타데이터 (dict)
    - DataSet 이후의 시계열 DataFrame
    을 반환한다.
    """
    lines = path.read_text(encoding="utf-8").splitlines()

    metadata = {}
    data_lines = []
    in_dataset = False

    for line in lines:
        if not in_dataset:
            if line.strip().lower().startswith("dataset"):
                in_dataset = True
                continue
            # 메타데이터 라인: "Key,Value" 형식 가정
            # 빈 줄은 건너뛰기
            if not line.strip():
                continue
            key, value = line.split(",", 1)
            metadata[key.strip()] = value.strip()
        else:
            # DataSet 이후의 라인: 헤더 + 데이터
            data_lines.append(line)

    # 데이터 부분을 판다스로 파싱
    ts_df = pd.read_csv(io.StringIO("\\n".join(data_lines)))

    return metadata, ts_df
1.3 메타데이터 통합 테이블 생성
모든 CSV 파일에 대해 위 함수를 적용하여:

metadata_list: 각 파일의 metadata dict + file_id를 병합

timeseries_list: 각 파일의 ts_df에 file_id 컬럼을 추가

이후:

python
코드 복사
metadata_df = pd.DataFrame(metadata_list)
timeseries_df = pd.concat(timeseries_list, ignore_index=True)
metadata_df에는 적어도 다음 컬럼이 있어야 한다.

file_id (파일명에서 추출하거나 별도 정의)

xRMS, yRMS, zRMS, sumRMS, xPeak, ... (존재하는 경우)

MeasFreq, Resampling, LowFreq, HighFreq

그 외 헤더에 있는 정보

2. 엑셀 라벨 및 시료 정보 매핑
2.1 엑셀 구조 가정
엑셀 파일(예: 시험전 시료 표기내용.xlsx)에는:

sample_id 혹은 파일명/시료명

product_power (100W / 200W)

direction (CW / CCW)

label (정상 / 소음 / 진동 / 표기없음)

기타 메모

2.2 매핑 전략
엑셀을 DataFrame으로 로드: labels_df

CSV 파일명과 엑셀의 sample/파일명을 일관된 규칙으로 매칭:

예: file_id = "100W_Sample03_cw4" 형태로 통일

metadata_df에 labels_df를 left join:

python
코드 복사
file_master_df = metadata_df.merge(
    labels_df,
    on="file_id",
    how="left",
    validate="one_to_one"
)
file_master_df는 이후 모든 단계에서 파일 단위 메타정보의 기준이 된다.

여기에는:

file_id

product_power (100/200)

direction (CW/CCW)

label (정상/소음/진동/표기없음)

메타데이터(MeasFreq, Resampling, LowFreq, HighFreq, RMS/Peak 등)

이 테이블을 data/interim/file_master.parquet 등으로 저장.

3. 시계열 데이터 정제 및 이상치 처리
3.1 TimeStamp 변환
timeseries_df에 대해:

TimeStamp는 보통 시작이 0 또는 작은 값부터 시작 → **상대 시간(초)**로 맞춘다.

python
코드 복사
def normalize_timestamp(ts_df: pd.DataFrame) -> pd.DataFrame:
    t0 = ts_df["TimeStamp"].iloc[0]
    ts_df = ts_df.copy()
    ts_df["time_sec"] = ts_df["TimeStamp"] - t0
    return ts_df
이후 분석에서는 TimeStamp 대신 time_sec 사용을 기본으로 한다.

3.2 결측치/NaN 처리
각 파일(file_id)별로:

센서 컬럼: acc-X, acc-Y, acc-Z, Gyro-X, Gyro-Y, Gyro-Z, acc-Sum

NaN 비율 계산

처리 규칙(초기안):

한 파일에서 특정 채널의 NaN 비율 > 5%:

해당 채널은 해당 파일에서 사용하지 않거나 (마스킹)

간단한 선형 보간(interpolate) 후 사용 여부를 옵션으로 둔다.

NaN 비율이 매우 크거나, 대부분 NaN인 파일:

파일 자체를 제외 목록에 넣고, file_master_df에 is_usable=False 플래그 추가.

구현 예시:

python
코드 복사
def handle_missing_values(ts_df: pd.DataFrame, max_nan_ratio: float = 0.05):
    sensor_cols = ["acc-X","acc-Y","acc-Z","Gyro-X","Gyro-Y","Gyro-Z","acc-Sum"]
    nan_ratio = ts_df[sensor_cols].isna().mean()

    bad_channels = nan_ratio[nan_ratio > max_nan_ratio].index.tolist()
    # 우선은 bad_channels는 로깅만 해두고, 나중에 실제로 제외할지 여부는 설정에서 결정
    # 나머지 NaN은 간단히 보간
    ts_df = ts_df.copy()
    ts_df[sensor_cols] = ts_df[sensor_cols].interpolate().ffill().bfill()
    return ts_df, bad_channels
3.3 센서 이상치(스파이크, 클리핑) 탐지
스파이크/클리핑은 센서가 순간적으로 비정상적인 큰 값 또는 같은 값을 반복하는 문제를 의미한다.

초기안:

값 범위 기반 체크

각 채널별로 전체 파일에서

평균, 표준편차, 최소, 최대를 계산

|x - mean| > k * std (예: k=8)인 포인트를 스파이크로 간주

또는 센서 물리 범위(있다면) 밖의 값은 바로 이상치로 처리

연속된 동일값 체크

센서가 죽으면 일정값만 계속 나오는 경우가 있으므로,

윈도우 내에서 동일값이 길게 이어지면(예: 100샘플 이상) 해당 구간을 결측으로 간주

처리 전략:

개별 포인트 스파이크:

해당 포인트만 NaN 처리 후, 보간

특정 파일/채널에서 스파이크 비율이 너무 크면:

파일 또는 채널을 제외(is_usable=False 또는 use_channel=False 플래그)

의사코드 예시

python
코드 복사
def detect_and_fix_spikes(ts_df: pd.DataFrame, z_thresh: float = 8.0):
    sensor_cols = ["acc-X","acc-Y","acc-Z","Gyro-X","Gyro-Y","Gyro-Z","acc-Sum"]
    ts_df = ts_df.copy()
    spike_mask_total = pd.DataFrame(False, index=ts_df.index, columns=sensor_cols)

    for col in sensor_cols:
        x = ts_df[col]
        m = x.mean()
        s = x.std()
        if s == 0:
            continue
        z = (x - m) / s
        spike_mask = z.abs() > z_thresh
        spike_mask_total[col] = spike_mask_total[col] | spike_mask
        # 스파이크를 NaN으로 마스킹
        ts_df.loc[spike_mask, col] = np.nan

    # 마스킹된 값 보간
    ts_df[sensor_cols] = ts_df[sensor_cols].interpolate().ffill().bfill()

    return ts_df, spike_mask_total
주의:
z-threshold 등은 EDA에서 실제 분포를 보고 조정할 수 있도록
config 파일(params_stage1.yaml)에 넣고 쉽게 바꿀 수 있게 한다.

3.4 채널/파일 품질 요약 리포트
각 파일별로 다음 정보를 요약해 data/interim/quality_report.csv 등으로 저장:

file_id

각 채널별 NaN 비율, 스파이크 비율

is_usable (초기에는 모두 True, 문제 심각시 False로 전환)

메모(이상이 심한 파일에 대한 코멘트)

EDA에서는 이 리포트를 보고:

어떤 파일을 본 분석에서 제외할지 결정한다.

4. 세그먼트(윈도우) 분할 설계
4.1 기본 개념
한 파일의 시계열을

윈도우 길이 (window_sec)

hop 길이 (hop_sec)
로 여러 개의 세그먼트로 분할한다.

각 세그먼트가 모델 학습/분류/이상탐지에 쓰이는 1개의 샘플이다.

4.2 안정 구간만 사용하기
시동/정지 구간은 진동 특성이 크게 달라질 수 있으므로:

전체 시간 범위 [t_min, t_max]에서

예: 앞 10%, 뒤 10%는 버리고

가운데 80% 구간만 윈도우 생성에 사용

이를 위한 함수 예시:

python
코드 복사
def get_stable_time_range(ts_df: pd.DataFrame, margin_ratio: float = 0.1):
    t_min = ts_df["time_sec"].min()
    t_max = ts_df["time_sec"].max()
    span = t_max - t_min
    t_start = t_min + margin_ratio * span
    t_end = t_max - margin_ratio * span
    return t_start, t_end
4.3 윈도우 생성 함수 설계
python
코드 복사
def generate_windows(ts_df: pd.DataFrame,
                     file_id: str,
                     window_sec: float,
                     hop_sec: float,
                     margin_ratio: float = 0.1) -> list[dict]:
    """
    한 파일의 정제된 시계열 ts_df를 받아
    - 안정 구간만 선택한 뒤
    - [window_sec] 길이로 [hop_sec] 간격으로 나누어
    윈도우별 메타정보와 인덱스를 반환한다.

    반환 예시: [{"file_id":..., "start_time":..., "end_time":..., "idx_start":..., "idx_end":...}, ...]
    """
    t_start, t_end = get_stable_time_range(ts_df, margin_ratio)
    # 안정 구간 데이터만 필터
    ts_stable = ts_df[(ts_df["time_sec"] >= t_start) & (ts_df["time_sec"] <= t_end)].reset_index(drop=True)

    windows = []
    t_min = ts_stable["time_sec"].iloc[0]
    t_max = ts_stable["time_sec"].iloc[-1]

    cur_start = t_min
    while cur_start + window_sec <= t_max:
        cur_end = cur_start + window_sec
        # 인덱스 범위 선택
        mask = (ts_stable["time_sec"] >= cur_start) & (ts_stable["time_sec"] < cur_end)
        idx = ts_stable.index[mask]
        if len(idx) == 0:
            break
        windows.append({
            "file_id": file_id,
            "start_time": cur_start,
            "end_time": cur_end,
            "idx_start": int(idx[0]),
            "idx_end": int(idx[-1]),
        })
        cur_start += hop_sec

    return windows
이 함수의 결과는:

나중에 feature 추출 함수가 시계열의 해당 구간만 잘라서 feature를 계산할 수 있도록 인덱스 정보를 제공한다.

5. 데이터 분할 및 증강(세그먼트 관점) 전략 반영
5.1 파일 단위 분할 → 윈도우 생성 순서
file_master_df를 바탕으로,

제품(100W/200W) 단위로 분리

각 제품 내에서 label(정상/불량) 분포를 고려하여 파일을:

train_files, val_files, test_files로 나눈다.

정상 파일이 매우 적다면:

시간 기반 분할(앞/중간/뒤 구간)로 train/val/test를 나누는 전략 사용.

각 세트에 대해:

포함된 파일 목록만 대상으로

CSV 로드 → 메타 분리 → 정제 → 세그먼트 생성

이렇게 하면 파일 단위 데이터 누수 방지가 가능하다.

5.2 세그먼트 수 및 클래스 불균형
파일별 세그먼트 수를 너무 치우치지 않도록:

한 파일에서 추출할 최대 윈도우 개수를 제한 (예: 200개)

전체적으로 normal : abnormal 비율이

대략 1:1 ~ 1:3 범위가 되도록:

필요시:

정상 세그먼트 oversampling

불량 세그먼트 undersampling

6. 1차 단계 시간영역 feature 추출 골격
이 문서에서는 시간영역 feature의 구조만 정의한다.
실제 구현은 src/features/time_domain.py에서 하고,
EDA와 본 분석에서 공통으로 사용한다.

6.1 사용 채널
가속도: acc-X, acc-Y, acc-Z, acc-Sum

자이로: Gyro-X, Gyro-Y, Gyro-Z

6.2 기본 feature 목록 (채널별)
각 윈도우 × 채널에 대해:

mean

std

RMS

Peak (max |x|)

crest_factor = Peak / RMS

kurtosis

skewness

6.3 구현 예시 스켈레톤
python
코드 복사
from scipy.stats import kurtosis, skew

SENSOR_COLS = ["acc-X","acc-Y","acc-Z","Gyro-X","Gyro-Y","Gyro-Z","acc-Sum"]

def compute_time_domain_features_for_window(ts_df: pd.DataFrame,
                                            idx_start: int,
                                            idx_end: int,
                                            file_meta: dict) -> dict:
    """
    ts_df의 [idx_start:idx_end] 구간에 대해
    시간영역 feature를 계산하고, 파일/라벨 메타와 함께 dict로 반환.
    """
    window = ts_df.iloc[idx_start:idx_end+1]
    feat = {}

    for col in SENSOR_COLS:
        x = window[col].values
        col_prefix = col.replace("-", "_")
        feat[f"{col_prefix}_mean"] = float(x.mean())
        feat[f"{col_prefix}_std"] = float(x.std())
        # RMS
        feat[f"{col_prefix}_rms"] = float(np.sqrt(np.mean(x**2)))
        # Peak (절대값 기준)
        feat[f"{col_prefix}_peak"] = float(np.max(np.abs(x)))
        # crest factor (RMS가 0이면 0 처리)
        feat[f"{col_prefix}_crest"] = float(feat[f"{col_prefix}_peak"] / feat[f"{col_prefix}_rms"]) if feat[f"{col_prefix}_rms"] > 0 else 0.0
        # kurtosis, skewness
        feat[f"{col_prefix}_kurtosis"] = float(kurtosis(x, fisher=True, bias=False))
        feat[f"{col_prefix}_skewness"] = float(skew(x, bias=False))

    # 파일/라벨 메타 추가
    feat["file_id"] = file_meta["file_id"]
    feat["product_power"] = file_meta["product_power"]   # 100 or 200
    feat["direction"] = file_meta["direction"]           # "CW" or "CCW"
    feat["label"] = file_meta["label"]                   # "정상" / "소음" / "진동" / "표기없음"
    feat["start_time"] = float(window["time_sec"].iloc[0])
    feat["end_time"] = float(window["time_sec"].iloc[-1])

    return feat
모든 윈도우에 대해 위 함수를 적용해

features_stage1.parquet 같은 feature 테이블을 만든다.

이 테이블은:

EDA(히스토그램, 박스플롯, scatter plot)

1차 XGBoost 모델 학습
에서 그대로 사용한다.

7. EDA에서 먼저 확인해야 할 것들 (1차 단계 관점)
코드를 작성한 후, EDA 노트북에서 다음을 반드시 확인한다.

메타데이터/품질

file_master_df에서:

제품별(100W/200W), 방향별(CW/CCW), 라벨별(정상/소음/진동/표기없음) 파일 수

MeasFreq, Resampling, LowFreq, HighFreq 값이 일관적인지

quality_report에서:

NaN/스파이크 비율이 높은 파일/채널 확인

제외해야 할 파일 후보 기록

시간영역 feature 분포

features_stage1에서:

sum_rms (acc-Sum RMS) 기준 정상/비정상 분포

축별 RMS/Peak/kurtosis 등 박스플롯

제품/방향별로 분포가 어떻게 다른지 확인:

100W vs 200W

CW vs CCW

라벨 간 분리 가능성

간단한 scatter (예: acc_Sum_rms vs acc_Sum_kurtosis)로

정상 vs 소음/진동이 대략 분리되는지 눈으로 확인

이 결과를 바탕으로:

1차 XGBoost 모델을 어떻게 설정할지,

2차 단계에서 어떤 주파수/오더 feature를 추가할지

를 결정한다.

perl
코드 복사

위 두 개가 “항상 참조용 상위 설계 문서”와 “클로드 코드용 상세 계획서” 초안이야.
그대로 `.md` 파일로 저장해서, 이후 단계(코드 작성·EDA·1차 분석)에 참고하면 된다.
::contentReference[oaicite:0]{index=0}

```