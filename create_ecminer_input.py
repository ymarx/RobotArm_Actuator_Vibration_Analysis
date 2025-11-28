"""
ECMiner 입력 CSV 생성 스크립트
data/ 폴더의 모든 CSV 파일을 스캔하여 ECMiner 입력 형식으로 변환
"""

import pandas as pd
from pathlib import Path
import re

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).parent

# 데이터 폴더
DATA_100W = PROJECT_ROOT / "data" / "100W"
DATA_200W = PROJECT_ROOT / "data" / "200W"

# 라벨 매핑 (기존 프로젝트 규칙)
LABEL_MAPPING = {
    '100W': {
        0: '정상',  # Sample00만 정상
    },
    '200W': {
        3: '정상',  # Sample03만 정상
    }
}

def parse_filename(filename: str, product: str) -> dict:
    """
    파일명에서 메타데이터 추출

    예: "100W_Sample00 cw4_2025-11-07 03-41-24.csv"
    → product=100W, sample=0, direction=CW
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

    return {
        'product': product,
        'sample': sample,
        'direction': direction
    }

def assign_label(product: str, sample: int) -> str:
    """
    라벨 할당

    양품: 100W Sample00, 200W Sample03
    불량: 나머지 전부 '표기없음'
    """
    normal_samples = LABEL_MAPPING.get(product, {})

    if sample in normal_samples:
        return normal_samples[sample]
    else:
        return '표기없음'

def scan_directory(data_dir: Path, product: str) -> list:
    """
    디렉토리를 스캔하여 파일 정보 수집
    """
    files = []

    csv_files = list(data_dir.glob("*.csv"))

    for csv_file in csv_files:
        filename = csv_file.name

        # 파일명 파싱
        meta = parse_filename(filename, product)
        if meta is None:
            print(f"경고: 파일명 파싱 실패 - {filename}")
            continue

        # 라벨 할당
        label = assign_label(meta['product'], meta['sample'])

        # 파일 경로 (프로젝트 루트 기준 상대 경로)
        file_path = f"data/{product}/{filename}"

        # 타임스탬프 추출 (파일명에서 날짜/시간 부분)
        # 예: "100W_Sample00 cw4_2025-11-07 03-41-24.csv" → "20251107_034124"
        timestamp_match = re.search(r'(\d{4})-(\d{2})-(\d{2})\s+(\d{2})-(\d{2})-(\d{2})', filename)
        if timestamp_match:
            y, m, d, h, mi, s = timestamp_match.groups()
            timestamp = f"{y}{m}{d}_{h}{mi}{s}"
        else:
            # 타임스탬프가 없으면 파일명 전체를 해시
            import hashlib
            timestamp = hashlib.md5(filename.encode()).hexdigest()[:8]

        # 파일 ID (ECMiner 형식, 타임스탬프 포함하여 고유성 보장)
        file_id = f"{meta['product']}_Sample{meta['sample']:02d}_{meta['direction']}_{timestamp}"

        files.append({
            'file_path': file_path,
            'file_id': file_id,
            'label': label
        })

    return files

def main():
    """
    메인 실행: 100W, 200W 데이터 스캔 후 ECMiner 입력 CSV 생성
    """
    all_files = []

    # 100W 스캔
    print("100W 데이터 스캔 중...")
    files_100w = scan_directory(DATA_100W, "100W")
    print(f"  - {len(files_100w)}개 파일 발견")
    all_files.extend(files_100w)

    # 200W 스캔
    print("200W 데이터 스캔 중...")
    files_200w = scan_directory(DATA_200W, "200W")
    print(f"  - {len(files_200w)}개 파일 발견")
    all_files.extend(files_200w)

    # DataFrame 생성
    df = pd.DataFrame(all_files)

    # 정렬 (product → sample → direction)
    df = df.sort_values(['file_path']).reset_index(drop=True)

    # CSV 저장
    output_path = PROJECT_ROOT / "ecminer_input_full.csv"
    df.to_csv(output_path, index=False)

    print(f"\n✅ ECMiner 입력 CSV 생성 완료: {output_path}")
    print(f"   - 총 파일 수: {len(df)}")
    print(f"   - 정상 파일: {len(df[df['label'] == '정상'])}")
    print(f"   - 불량 파일: {len(df[df['label'] == '표기없음'])}")

    # 샘플 출력
    print("\n첫 5개 파일:")
    print(df.head(5).to_string(index=False))

    return df

if __name__ == "__main__":
    df = main()
