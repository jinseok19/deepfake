# 딥페이크 탐지 대회 전략 문서

> 리더보드 기반 최신 데이터 생성 전략

---

## 📚 문서 목록

| 문서 | 설명 | 우선순위 |
|------|------|----------|
| [OVERVIEW.md](OVERVIEW.md) | 전체 전략 개요 | ⭐⭐⭐ 필독 |
| [LEADERBOARD_ANALYSIS.md](LEADERBOARD_ANALYSIS.md) | 리더보드 상세 분석 | ⭐⭐⭐ 필독 |
| [DATA_GENERATION_PLAN.md](DATA_GENERATION_PLAN.md) | 데이터 생성 계획 | ⭐⭐ 참고 |
| [EXECUTION_SCHEDULE.md](EXECUTION_SCHEDULE.md) | 3주 실행 일정 | ⭐⭐ 참고 |
| [TOOLS_SETUP.md](TOOLS_SETUP.md) | 도구 설치 가이드 | ⭐ 필요시 |

---

## 🎯 핵심 전략 요약

### 문제 인식
```
❌ 기존 방식: FaceForensics++ (2019) 위주 학습
   → 평가 데이터(2024-2025)와 5년 차이
   → 낮은 성능

✅ 새로운 방식: 리더보드 상위 모델로 데이터 생성
   → 평가 데이터와 동일한 생성 패턴
   → 높은 성능 기대
```

### 4가지 딥페이크 유형
1. **Generation** (40%): FLUX.1, Midjourney 등
2. **Face Swap** (30%): ReActor, Roop
3. **Face Reenactment** (20%): LivePortrait
4. **Lip Sync** (10%): Wav2Lip

### 목표
- **데이터**: 50,000개 (Fake 25K + Real 25K)
- **기간**: 3주
- **목표 성능**: F1 0.70+ (현재 0.56)

---

## 🚀 빠른 시작

### 1. 문서 읽기 (30분)
```bash
1. OVERVIEW.md - 전체 이해
2. LEADERBOARD_ANALYSIS.md - 타겟 확인
3. EXECUTION_SCHEDULE.md - 일정 파악
```

### 2. 환경 구축 (4시간)
```bash
# TOOLS_SETUP.md 참고
1. ComfyUI 설치
2. FLUX.1 다운로드
3. ReActor 설치
```

### 3. 생성 시작 (Day 2~)
```bash
# DATA_GENERATION_PLAN.md 참고
1. FLUX.1 테스트 (100개)
2. 자동화 스크립트
3. 대량 생성 (3,000개)
```

---

## 📊 현재 상태

### 실험 결과
| 실험 | 방법 | F1 Score | 변화 |
|------|------|----------|------|
| Baseline | 베이스라인 모델 | 0.5489 | - |
| EXP-002 | 프레임 40개 | 0.5506 | +0.31% |
| EXP-003-Fixed | TTA | **0.5600** | +2.01% ⭐ |
| EXP-004 | FF++ 파인튜닝 | 진행중 | ? |

### 다음 단계
- **EXP-005**: 리더보드 데이터 (10K) → 예상 F1 0.62~0.65
- **EXP-006**: 전체 데이터 (50K) → 목표 F1 0.70+

---

## 📅 타임라인

```
Week 1 (Day 1-7):
└─ FLUX.1 (3K) + ReActor (3K)

Week 2 (Day 8-14):
└─ 리더보드 수집 + LivePortrait

Week 3 (Day 15-21):
└─ 완성 + 학습 + 제출

대회 마감: 2024-11-20
```

---

## 🔗 관련 링크

### 외부 리소스
- [리더보드](https://artificialanalysis.ai/image/leaderboard/text-to-image)
- [FLUX.1 HuggingFace](https://huggingface.co/black-forest-labs)
- [ComfyUI GitHub](https://github.com/comfyanonymous/ComfyUI)

### 프로젝트 문서
- [../docs/](../docs/) - 전체 프로젝트 문서
- [../log/](../log/) - 실험 로그
- [../baseline/](../baseline/) - 실험 코드

---

## ❓ FAQ

### Q: 왜 FF++를 안 쓰나요?
A: FF++는 2019년 데이터입니다. 평가 데이터는 2024-2025 최신 모델로 생성되므로, 5년의 기술 차이가 있습니다. 리더보드 상위 모델(FLUX.1 등)로 생성한 데이터가 더 효과적입니다.

### Q: 50,000개 생성 가능한가요?
A: 네. FLUX.1 (3K) + 온라인 수집 (3K) + Face Swap (7.5K) + 기타 (11.5K) = 25K Fake. Real 25K는 공개 데이터셋(FFHQ, CelebA-HQ)에서 확보. 3주면 충분합니다.

### Q: GPU가 부족하면?
A: FLUX.1 schnell (4GB), Colab Pro, 또는 클라우드 GPU(RunPod, Vast.ai) 활용. 최소 RTX 3060 (12GB) 권장.

### Q: 어떤 것부터 시작?
A: FLUX.1부터! 리더보드 7위, 오픈소스, ComfyUI 완벽 지원. 가장 중요하고 접근성 좋음.

---

## 📝 업데이트 로그

```
2025-10-30:
- 초기 전략 문서 작성
- 리더보드 분석 완료
- 3주 일정 수립

다음 업데이트:
- Day 7: Week 1 회고
- Day 14: Week 2 회고
- Day 21: 최종 결과
```

---

## 🎯 목표 달성 지표

```
현재:  F1 0.5600 (EXP-003-Fixed)
Week 1: F1 0.58~0.60 (첫 테스트)
Week 2: F1 0.62~0.65 (중간 평가)
Week 3: F1 0.68~0.72 (최종 목표)

성공 기준: F1 > 0.70 (베이스라인 대비 +27%)
```

---

**작성자**: AI Assistant  
**작성일**: 2025-10-30  
**버전**: 1.0  
**상태**: 실행 준비 완료 ✅

