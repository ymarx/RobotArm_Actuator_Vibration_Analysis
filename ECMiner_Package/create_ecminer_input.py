"""
ECMiner 입력 CSV 생성 스크립트
data/ 폴더의 모든 CSV 파일을 스캔하여 ECMiner 입력 형식으로 변환

이 스크립트는 ecminer_python_node.py 실행 전에 필수로 실행해야 합니다.
파일명 정규화 문제를 해결하고 고유한 file_id를 생성합니다.

사용 방법:
1. ECMiner_Package 폴더에서 실행
2. 또는 프로젝트 루트에서 실행 (자동 탐색)
"""

import pandas as pd
from pathlib import Path
import re
from datetime import datetime

# ============================================================================
# 경로 설정 (자동 탐색)
# ============================================================================

# 스크립트 위치 감지
SCRIPT_DIR = Path(__file__).parent if '__file__' in globals() else Path('.')

# 프로젝트 루트 자동 탐색
if SCRIPT_DIR.name == 'ECMiner_Package':
    # ECMiner_Package 폴더에서 실행
    PROJECT_ROOT = SCRIPT_DIR.parent
else:
    # 프로젝트 루트에서 실행
    PROJECT_ROOT = SCRIPT_DIR

# 데이터 폴더
DATA_ROOT = PROJECT_ROOT / "data"
DATA_100W = DATA_ROOT / "100W"
DATA_200W = DATA_ROOT / "200W"

# 출력 파일 경로
OUTPUT_CSV = PROJECT_ROOT / "ecminer_input_full.csv"

print("=" * 60)
print("ECMiner 입력 CSV 생성 스크립트")
print("=" * 60)
print(f"\n프로젝트 루트: {PROJECT_ROOT}")
print(f"데이터 폴더: {DATA_ROOT}")
print(f"출력 파일: {OUTPUT_CSV}")

# ============================================================================
# 파일명 파싱 함수
# ============================================================================

def parse_filename(filename: str, product: str) -> dict:
    """
    파일명에서 메타데이터 추출

    예: "100W_Sample00 cw4_2025-11-07 03-41-24.csv"
    → product=100W, sample=0, direction=CW, timestamp=20251107_034124

    예: "200W_Sample3 ccw4_2025-11-07 11-23-08.csv"
    → product=200W, sample=3, direction=CCW, timestamp=20251107_112308
    """
    # Sample 번호 추출 (0 패딩 여부 무관)
    sample_match = re.search(r'Sample(\d+)', filename, re.IGNORECASE)
    if not sample_match:
        return None

    sample = int(sample_match.group(1))

    # 회전 방향 추출 (cw/ccw, 뒤의 숫자는 무시)
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
        # 타임스탬프가 없으면 현재 시간 사용
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    return {
        'product': product,
        'sample': sample,
        'direction': direction,
        'timestamp': timestamp
    }

def scan_directory(data_dir: Path, product: str) -> list:
    """
    디렉토리를 스캔하여 파일 정보 수집

    Args:
        data_dir: 데이터 폴더 경로 (예: data/100W)
        product: 제품명 (100W/200W)

    Returns:
        파일 정보 리스트 [{'file_path': str, 'file_id': str}, ...]
    """
    if not data_dir.exists():
        print(f"  ⚠️  경고: {data_dir} 폴더가 없습니다.")
        return []

    files = []
    csv_files = list(data_dir.glob("*.csv"))

    for csv_file in csv_files:
        filename = csv_file.name

        # 파일명 파싱
        meta = parse_filename(filename, product)
        if meta is None:
            print(f"  ⚠️  경고: 파일명 파싱 실패 - {filename}")
            continue

        # 파일 경로 (프로젝트 루트 기준 상대 경로)
        file_path = f"data/{product}/{filename}"

        # 파일 ID (고유성 보장: 0 패딩 + 타임스탬프)
        # 예: 100W_Sample00_CW_20251107_034124
        file_id = f"{meta['product']}_Sample{meta['sample']:02d}_{meta['direction']}_{meta['timestamp']}"

        files.append({
            'file_path': file_path,
            'file_id': file_id
        })

    return files

# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """
    메인 실행: 100W, 200W 데이터 스캔 후 ECMiner 입력 CSV 생성
    """
    print("\n[1단계] 데이터 폴더 스캔")
    print("-" * 60)

    all_files = []

    # 100W 스캔
    print("  📁 100W 폴더 스캔 중...")
    files_100w = scan_directory(DATA_100W, "100W")
    print(f"     → {len(files_100w)}개 파일 발견")
    all_files.extend(files_100w)

    # 200W 스캔
    print("  📁 200W 폴더 스캔 중...")
    files_200w = scan_directory(DATA_200W, "200W")
    print(f"     → {len(files_200w)}개 파일 발견")
    all_files.extend(files_200w)

    if len(all_files) == 0:
        print("\n❌ 오류: CSV 파일을 찾을 수 없습니다.")
        print(f"   데이터 폴더를 확인하세요: {DATA_ROOT}")
        return None

    # DataFrame 생성
    df = pd.DataFrame(all_files)

    # 정렬 (file_path 기준)
    df = df.sort_values(['file_path']).reset_index(drop=True)

    print(f"\n[2단계] CSV 파일 생성")
    print("-" * 60)

    # CSV 저장
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"  ✅ ECMiner 입력 CSV 생성 완료!")
    print(f"     경로: {OUTPUT_CSV}")
    print(f"     파일 수: {len(df)}개")

    # 샘플 출력
    print(f"\n[3단계] 결과 미리보기 (처음 5개)")
    print("-" * 60)
    print(df.head(5).to_string(index=False))

    print("\n" + "=" * 60)
    print("✅ 완료! 이제 ecminer_python_node.py를 실행하세요.")
    print("=" * 60)

    return df

if __name__ == "__main__":
    df = main()
