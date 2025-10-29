# 딥페이크 탐지 대회 - 최신 데이터 기반 전략

## 📋 대회 핵심 정보

### 평가 데이터 특징
- **최신 생성 모델 사용**: 리더보드 상위 모델로 생성
- **지속적 업데이트**: 주최측이 최신 모델로 계속 갱신
- **4가지 딥페이크 유형 중심**

---

## 🎯 4가지 딥페이크 유형

### 1. 생성형 영상 (Generation) - 40%
```
Text-to-Image 모델로 완전 새로운 인물 생성
- FLUX.1, Midjourney, Stable Diffusion
- 평가 데이터의 핵심!
```

### 2. 페이스 제어 (Face Reenactment) - 20%
```
표정/움직임 전이
- LivePortrait (2024 최신)
- V-Express
- First Order Motion Model
```

### 3. 얼굴 바꾸기 (Face Swap) - 30%
```
얼굴 교체
- ReActor (ComfyUI)
- Roop, SimSwap
- DeepFaceLive
```

### 4. 립싱크 (Lip Sync) - 10%
```
입모양 동기화
- Wav2Lip
- SadTalker
- Video-Retalking
```

---

## 🔥 핵심 전략

### ✅ 해야 할 것
1. **리더보드 상위 모델로 데이터 생성** ⭐⭐⭐
   - FLUX.1 (7위, 오픈소스)
   - Leonardo.Ai, Recraft V3
   - 온라인 갤러리 수집

2. **최신 도구 중심** (2024-2025)
   - LivePortrait, ReActor
   - Wav2Lip, SadTalker

3. **대규모 데이터셋 구축**
   - 목표: 50,000개 (Fake 25K + Real 25K)

### ❌ 하지 말아야 할 것
1. **구식 데이터셋 위주 학습**
   - FaceForensics++ (2019) - 보조만
   - Celeb-DF - 검증용만

2. **오래된 기법**
   - StyleGAN, ProGAN
   - 2020년 이전 방법

---

## 📊 데이터셋 구성

```
총 50,000개 샘플
├── Fake: 25,000개
│   ├── Generation (리더보드):     10,000개 (40%)
│   ├── Face Swap:                  7,500개 (30%)
│   ├── Face Reenactment:           5,000개 (20%)
│   └── Lip Sync:                   2,500개 (10%)
│
└── Real: 25,000개
    ├── FFHQ:                      10,000개
    ├── CelebA-HQ:                  8,000개
    └── YouTube 일반 영상:          7,000개
```

---

## 🚀 3주 실행 계획

### Week 1: FLUX.1 + Face Swap
- ComfyUI + FLUX.1 설치
- FLUX.1 생성: 3,000개
- ReActor Face Swap: 3,000개

### Week 2: 리더보드 모델 수집
- Leonardo.Ai, Recraft V3
- 온라인 크롤링 (Reddit, CivitAI)
- LivePortrait 생성

### Week 3: 통합 + 학습
- 전체 데이터 전처리
- 최종 모델 학습
- 제출

---

## 📈 예상 성능

```
현재 (EXP-003-Fixed):
- F1 Score: 0.5600
- 베이스라인 모델 (FF++ 학습)

목표 (리더보드 기반):
- F1 Score: 0.65~0.75 (+16~34%)
- 평가 데이터와 동일한 생성 모델 사용
```

---

## 📁 관련 문서

- [LEADERBOARD_ANALYSIS.md](LEADERBOARD_ANALYSIS.md) - 리더보드 상세 분석
- [DATA_GENERATION_PLAN.md](DATA_GENERATION_PLAN.md) - 데이터 생성 세부 계획
- [EXECUTION_SCHEDULE.md](EXECUTION_SCHEDULE.md) - 주차별 실행 일정
- [TOOLS_SETUP.md](TOOLS_SETUP.md) - 도구 설치 가이드

---

**작성일**: 2025-10-30  
**현재 최고 성적**: 0.5600 (EXP-003-Fixed)  
**목표**: 0.70+ (리더보드 기반 학습)

