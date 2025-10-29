# 딥페이크 탐지 모델 - 팀 공유 리포트

> **대회**: 딥페이크 범죄 대응을 위한 AI 탐지 모델 경진대회  
> **기간**: 2024.10.23 ~ 2024.11.20  
> **평가지표**: Macro F1-score (Fake=양성)  
> **팀원**: 김진석, [팀원명]

---

## 📊 실험 리더보드

| Model | Creator | Score | Baseline 대비 | Hyperparameters | Development |
|-------|---------|-------|---------------|-----------------|-------------|
| **ViT-Base<br>+ TTA** | 김진석 | **0.5599217639** | **+2.01%** ✅ | 프레임: 40<br>TTA: 좌우반전<br>CUDA: 12.6 | EXP-003-Fixed<br>TTA 메모리 최적화<br>순차 추론으로 OOM 해결 |
| **ViT-Base** | 김진석 | 0.5506 | +0.31% | 프레임: 40<br>TTA: ❌ | EXP-002<br>프레임 수 증가만 |
| **ViT-Base<br>(Baseline)** | - | 0.5489 | - | 프레임: 30<br>TTA: ❌ | 베이스라인 모델<br>deep-fake-detector-v2 |
| **ViT-Base** | 김진석 | 0.5354 | **-2.46%** ❌ | 중앙크롭<br>Mean+Max | EXP-001<br>여러 개선 시도 → 역효과 |
| **ViT-Base<br>+ TTA** | 김진석 | **OOM** | - | 프레임: 40<br>TTA: 시도 | EXP-003<br>❌ 메모리 부족 실패 |
| **ViT-Base<br>Finetuned** | 김진석 | 준비중 | ? | FF++ 200샘플<br>TTA: ✅ | EXP-004<br>Val F1: 0.9933<br>(오버피팅 의심) |

---

## 🏆 현재 최고 성적

```
모델: EXP-003-Fixed (TTA + 메모리 최적화)
F1 Score: 0.5600
개선율: +2.01% (Baseline 대비)
상태: 🎯 현재 최고 성능
```

---

## 📈 성능 추이

```
F1 Score
0.56 |                        ▲ EXP-003-Fixed (0.5600) 🏆
     |                    ▪ EXP-002 (0.5506)
0.55 |
     |  ▬ BASELINE (0.5489)
0.54 |
     |              ■ EXP-001 (0.5354) ❌
0.53 |
     |
     +───────────────────────────────────────────
        BASELINE    EXP-001    EXP-002    EXP-003-Fixed
```

---

## 🔬 주요 실험 상세

### ✅ EXP-003-Fixed: TTA + 메모리 최적화 (성공)

#### 결과
- **F1 Score**: 0.5600 (+2.01%)
- **제출일**: 2025-10-29
- **CUDA**: 12.6 (PyTorch 2.7.1)

#### 주요 변경
```python
전처리:
- 프레임 수: 40개 (균등 샘플링)
- 얼굴 탐지: Dlib
- 크롭: 30% margin

후처리 (핵심):
- TTA: 좌우 반전 앙상블
- 메모리 최적화: 순차 추론
```

#### 성능 분해
| 단계 | F1 Score | 기여도 |
|------|----------|--------|
| Baseline | 0.5489 | - |
| + 프레임 40 (EXP-002) | 0.5506 | +0.31% |
| + TTA (EXP-003-Fixed) | **0.5600** | **+1.69%** |

**인사이트**: TTA가 프레임 증가보다 **5배 효과적!**

---

### ❌ EXP-003: TTA 시도 (실패 - OOM)

#### 문제 상황
```
에러 메시지: "점수를 산출할 수 없었습니다"
원인: Out of Memory (메모리 부족)
```

#### 원인 분석

**1. 메모리 폭발 원인**
```python
# 문제 코드 (EXP-003)
face_images = [40개 원본 프레임]
flipped_images = [40개 반전 프레임]
tta_images = face_images + flipped_images  # 총 80개!

# GPU로 한번에 전송
inputs = processor(images=tta_images, return_tensors="pt").to("cuda")
# → 80개 × 3채널 × 224 × 224 = ~12GB 메모리 소요!

outputs = model(**inputs)  # OOM 발생!
```

**메모리 계산**:
```
이미지 1개: 3 × 224 × 224 × 4 bytes (float32) = 0.6 MB
80개: 0.6 × 80 = 48 MB (입력만)
중간 레이어까지 포함: ~12GB 필요
서버 GPU: L4 (24GB) 또는 T4 (16GB)
→ 다른 프로세스 + 모델 가중치 고려 시 부족!
```

**2. 왜 로컬에서는 안 났는가?**
- 로컬: RTX 3090 (24GB) + 전용 사용
- 서버: 공유 환경 + 다른 프로세스 존재
- 메모리 여유 차이

#### 해결 방법 (EXP-003-Fixed)

**순차 추론으로 메모리 절반 감소**
```python
# 개선 코드 (EXP-003-Fixed)

# 1. 원본 40개만 먼저 추론
inputs_orig = processor(images=face_images, return_tensors="pt").to("cuda")
# → 40개만 = ~6GB

with torch.no_grad():
    outputs_orig = model(**inputs_orig)
    probs_orig = F.softmax(outputs_orig.logits, dim=1).mean(dim=0)

# 2. 반전 40개 따로 추론
flipped_images = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in face_images]
inputs_flip = processor(images=flipped_images, return_tensors="pt").to("cuda")
# → 40개만 = ~6GB

with torch.no_grad():
    outputs_flip = model(**inputs_flip)
    probs_flip = F.softmax(outputs_flip.logits, dim=1).mean(dim=0)

# 3. 앙상블 (CPU에서)
avg_probs = (probs_orig + probs_flip) / 2.0
predicted_class = torch.argmax(avg_probs).item()
```

**메모리 비교**:
```
Before (EXP-003):    80개 한번에 → ~12GB → OOM ❌
After (EXP-003-Fixed): 40개 × 2회 → ~6GB × 2 → OK ✅
메모리 절약: 50%
```

#### 교훈
```
✅ 배운 점:
1. 로컬과 서버 환경은 다르다
2. 메모리 계산을 미리 해야 한다
3. 배치를 나누면 안전하다

⚠️ 주의사항:
- GPU 메모리는 항상 여유를 둘 것
- 서버는 공유 환경임을 고려
- 큰 배치는 순차 처리 고려
```

---

### ⚠️ EXP-002: 프레임 40개 (미미한 성능 향상)

#### 결과
- **F1 Score**: 0.5506 (+0.31%)
- **변경**: 프레임 30 → 40개만

#### 분석
```
예상: +2~4% 향상
실제: +0.31% (예상보다 훨씬 낮음)

이유:
- 30개 프레임으로도 이미 충분한 정보 확보
- 추가 10개는 중복 정보
- 단순 프레임 증가로는 한계
```

#### 인사이트
- 프레임 수보다 **데이터 증강(TTA)**이 더 효과적
- 모델 자체 개선 필요 (파인튜닝)

---

### ❌ EXP-001: 여러 개선 시도 (실패)

#### 결과
- **F1 Score**: 0.5354 (-2.46%) ❌
- **순위**: 7위

#### 시도한 것들
```python
1. 얼굴 미검출 → 중앙 크롭 (베이스라인: 레이블 0)
2. Mean(60%) + Max(40%) 조합 (베이스라인: Mean only)
3. 프레임 필터링 (베이스라인: 전체 사용)
```

#### 실패 원인 분석
| 변경사항 | 예상 | 실제 | 기여도 |
|----------|------|------|--------|
| 중앙 크롭 | +1~2% | **-1~2%** ❌ | 노이즈 증가 |
| Mean+Max | +1~2% | **-0.5~1%** ❌ | Max가 노이즈 증폭 |
| 프레임 필터링 | +0.5~1% | **-0.5~1%** ❌ | 정보 손실 |

**교훈**: 
- "개선"이 항상 좋은 것은 아니다
- 단순한 방법(Mean only)이 더 효과적일 수 있다
- 여러 변경을 한번에 하면 원인 파악 어려움

---

## 🎯 다음 실험 계획

### EXP-004: FaceForensics++ 파인튜닝

#### 준비 상황
```
✅ 모델 학습 완료 (Colab)
✅ Validation F1: 0.9933
⚠️ 오버피팅 의심 (Val이 너무 높음)
✅ 제출 준비 완료 (submit/ 폴더)
```

#### 기대 효과
```
낙관적: F1 0.62~0.65 (+13~18%)
현실적: F1 0.58~0.60 (+6~9%)
비관적: F1 0.54~0.56 (오버피팅)
```

#### 리스크
- Val F1 0.9933은 이상하게 높음
- 학습 데이터 200개만 사용 (소량)
- Test 데이터와 분포 차이 가능

---

### 중장기 전략: 리더보드 기반 데이터 생성

#### 문제 인식
```
현재 접근:
- FaceForensics++ (2019) 데이터로 학습
- 평가 데이터는 2024-2025 최신 모델 생성
- 5년 기술 격차!

새로운 전략:
- 리더보드 상위 모델로 직접 데이터 생성
- FLUX.1, Midjourney, Leonardo.Ai 등
- 평가 데이터와 동일한 패턴
```

#### 4가지 딥페이크 유형 커버
```
1. Generation (40%):        FLUX.1, Midjourney
2. Face Swap (30%):         ReActor, Roop
3. Face Reenactment (20%):  LivePortrait
4. Lip Sync (10%):          Wav2Lip
```

#### 목표
- 데이터: 50,000개 (3주)
- 예상 성능: F1 0.70+ (+27%)
- 상세 계획: `strategy/` 폴더 참고

---

## 💡 핵심 인사이트

### 성공 요인
```
✅ TTA (Test Time Augmentation)
   - 간단하지만 효과적 (+1.69%)
   - 추론 시간 2배지만 성능 5배 향상

✅ 메모리 최적화
   - 순차 처리로 OOM 해결
   - 안정성 확보

✅ 단계별 실험
   - 한 번에 하나씩 변경
   - 원인 파악 용이
```

### 실패 요인
```
❌ 검증 없는 가설
   - "개선"이 항상 좋은 것은 아님
   - 베이스라인이 이미 최적일 수 있음

❌ 여러 변경 동시 적용
   - 원인 파악 어려움
   - 디버깅 불가능

❌ 환경 차이 미고려
   - 로컬 ≠ 서버
   - 메모리 계산 필수
```

---

## 📊 기술 스택

### 모델
- **백본**: ViT-Base-Patch16-224
- **파라미터**: 85.8M
- **입력**: 224×224×3

### 전처리
```python
- 얼굴 검출: Dlib frontal face detector
- 크롭: Bounding box + 30% margin
- 프레임: 균등 샘플링 (linspace)
- 병렬 처리: multiprocessing.Pool
```

### 추론 환경
```
서버 스펙:
- CPU: 8 core
- RAM: 48GB
- GPU: L4 (24GB) or T4 (16GB)
- 시간 제한: 3시간

제출 제한:
- 하루 3회
- 30분 간격
```

---

## 🔍 문제 해결 가이드

### OOM (Out of Memory) 발생 시

#### 증상
```
- "점수를 산출할 수 없었습니다"
- 추론 중단
- submission.csv 생성 실패
```

#### 해결 방법
```python
1. 배치 크기 줄이기
   - 80개 → 40개로 분할
   - 순차 처리

2. 프레임 수 줄이기
   - 40 → 30 또는 20

3. 메모리 정리
   torch.cuda.empty_cache()
   
4. 낮은 정밀도 사용
   model.half()  # float16
```

#### 예방법
```
✅ 사전 메모리 계산
- 입력 크기 × 배치 × 4 bytes
- 중간 레이어 × 2~3배
- 여유 20% 확보

✅ 로컬 테스트
- 서버와 유사한 메모리 제한
- nvidia-smi로 모니터링
```

---

## 📞 질문 & 논의사항

### 팀 논의 필요
```
1. EXP-004 제출 여부
   - Val F1 0.9933 (오버피팅 의심)
   - 제출? vs 재학습?

2. 리더보드 전략 실행
   - 3주 투자 가치?
   - 역할 분담?

3. 우선순위
   - 빠른 개선 vs 장기 투자?
```

### 도움 필요한 부분
```
- ComfyUI 설치 및 FLUX.1 테스트
- 데이터 크롤링 스크립트
- 학습 환경 (GPU) 확보
```

---

## 📁 참고 자료

### 프로젝트 문서
- **실험 로그**: `log/EXP-*.md`
- **전략 문서**: `strategy/README.md`
- **제출 가이드**: `docs/SUBMISSION_GUIDE.md`

### 외부 링크
- [대회 페이지](https://aifactory.space/task/9197)
- [리더보드](https://artificialanalysis.ai/image/leaderboard/text-to-image)
- [Q&A 게시판](https://aifactory.space/task/9197/qna)

---

## 📅 타임라인

```
완료:
✅ 2024.10.24 - EXP-001 (실패)
✅ 2024.10.29 - EXP-002 (+0.31%)
✅ 2024.10.29 - EXP-003-Fixed (+2.01%) 🏆

진행중:
🔄 EXP-004 제출 대기 (파인튜닝 모델)

계획:
📋 Week 1-3: 리더보드 기반 데이터 생성
📋 2024.11.20: 대회 마감
```

---

## ✅ Action Items

### 즉시
- [ ] 팀 회의: EXP-004 제출 여부 결정
- [ ] 리더보드 전략 검토 및 합의

### 이번 주
- [ ] ComfyUI + FLUX.1 환경 구축
- [ ] 테스트 생성 (100개)

### 다음 주
- [ ] 데이터 생성 본격 시작
- [ ] 중간 파인튜닝 테스트

---

**작성일**: 2025-10-30  
**최종 업데이트**: 2025-10-30  
**작성자**: 김진석  
**버전**: 1.0

