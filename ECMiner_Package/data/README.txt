이 폴더에 원본 CSV 파일을 배치하세요
========================================

폴더 구조:
  data/
  ├── 100W/
  │   ├── Sample00_CW.csv
  │   ├── Sample00_CCW.csv
  │   ├── Sample01_CW.csv
  │   └── ...
  └── 200W/
      ├── Sample00_CW.csv
      ├── Sample00_CCW.csv
      └── ...

파일 형식:
- CSV 파일 (7개 채널, 헤더 포함)
- 샘플링 주파수: 512 Hz
- 컬럼: acc-X, acc-Y, acc-Z, acc-Sum, Gyro-X, Gyro-Y, Gyro-Z

참고:
- 원본 데이터는 절대 삭제하거나 수정하지 마세요
- ECMiner 스크립트는 상대 경로로 파일을 참조합니다
