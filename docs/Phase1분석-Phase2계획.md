# Phase 1 분석 완료 및 Phase 2 실행 계획

**작성일**: 2025-11-17  
**상태**: Step 1 실행 준비 완료

---

## 📊 Phase 1 핵심 발견 (RMS 분석)

**핵심 통찰**:
> "불량은 일부 구간에서 RMS가 유난히 튀어나온다" (평균↑ but 중앙값 비슷)

- acc_Y_rms: 불량 2.22배 (가장 강력)
- 극단값: 불량 max 2.587 vs 정상 max 0.234 (11배)
- → Autoencoder 이상치 탐지 유효

---

## 🎯 Phase 2 단계별 계획

### Step 1: RMS Threshold 룰 실험 ← **다음 실행**
- acc_Y_rms threshold 탐색
- Baseline 성능 측정

### Step 2: XGBoost Baseline (9 features)
- RMS 4개 + Peak + Crest + Kurtosis
- 목표: CV AUC > 0.75

### Step 3: 대역별 RMS 추출
- rms_1_10, rms_10_50, rms_50_150
- 하이브리드 룰 기반 구축

### Step 4: Autoencoder 이상치 탐지
### Step 5: Hybrid Rule & Ensemble

---

**참조**: EDA_SUMMARY_v2.md, EDA_RMS_ANALYSIS.md
