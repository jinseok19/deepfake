# Text-to-Image 리더보드 분석

> 출처: https://artificialanalysis.ai/image/leaderboard/text-to-image

---

## 📊 상위 10개 모델 (2024-2025)

| 순위 | 모델 | 제공자 | ELO | 출시일 | 접근성 |
|------|------|--------|-----|--------|--------|
| 1 | Seedream 4.0 | ByteDance | 1,242 | Sep 2025 | 🔴 상용 |
| 2 | Imagen 4 Ultra Preview | Google | 1,203 | Jun 2025 | 🟡 제한적 |
| 3 | Gemini 2.5 Flash (Nano-Banana) | Google | 1,187 | Aug 2025 | 🟡 제한적 |
| 4 | Imagen 4 Preview | Google | 1,175 | Jun 2025 | 🟡 제한적 |
| 5 | Kolors 2.1 | KlingAI | 1,156 | Jul 2025 | 🟡 상용 |
| 6 | Lucid Origin Ultra | Leonardo.Ai | 1,146 | Aug 2025 | 🟢 무료체험 |
| 7 | FLUX.1 Kontext [max] | Black Forest Labs | 1,140 | May 2025 | 🟢 오픈소스 |
| 8 | Recraft V3 | Recraft | 1,136 | Oct 2024 | 🟢 무료체험 |
| 9 | Dreamina 3.1 | Bytedance | 1,130 | Aug 2025 | 🟡 상용 |

---

## 🎯 Tier 분류

### Tier 1: 직접 생성 가능 (최우선) ⭐⭐⭐

#### FLUX.1 (7위)
```
제공자: Black Forest Labs
ELO: 1,140
출시: May 2025

✅ 장점:
- 오픈소스 (완전 무료)
- ComfyUI 완벽 지원
- 로컬 실행 가능
- 고품질 (리더보드 7위)

📦 설치:
- ComfyUI Manager
- FLUX.1 schnell (빠른 버전, 4GB VRAM)
- FLUX.1 dev (고품질 버전, 12GB VRAM)

🎯 생성 목표: 3,000개
우선순위: ⭐⭐⭐ (최우선!)
```

#### Recraft V3 (8위)
```
제공자: Recraft
ELO: 1,136
출시: Oct 2024

✅ 장점:
- 무료 체험 제공
- API 사용 가능
- 고품질

⚠️ 제약:
- 일일 생성 제한

🎯 생성 목표: 1,000개
우선순위: ⭐⭐
```

#### Leonardo.Ai - Lucid Origin Ultra (6위)
```
제공자: Leonardo.Ai
ELO: 1,146
출시: Aug 2025

✅ 장점:
- 무료 체험 (일일 150 토큰)
- 웹 인터페이스 편리
- API 제공

🎯 생성 목표: 1,000개
우선순위: ⭐⭐
```

---

### Tier 2: 온라인 수집 가능 ⭐⭐

#### Seedream 4.0 (1위)
```
제공자: ByteDance
ELO: 1,242 (1위!)

📥 수집 방법:
- Reddit r/StableDiffusion
- AI 아트 커뮤니티
- 공개 갤러리

🎯 수집 목표: 1,000개
우선순위: ⭐⭐
```

#### Imagen 4 (2,4위)
```
제공자: Google
ELO: 1,175~1,203

📥 수집 방법:
- Google AI Test Kitchen (제한적)
- 온라인 공개 샘플
- Reddit, Twitter

🎯 수집 목표: 500개
우선순위: ⭐
```

#### Gemini 2.5 Flash (3위)
```
제공자: Google
ELO: 1,187

📥 수집 방법:
- 공개 이미지 수집
- AI 갤러리

🎯 수집 목표: 500개
우선순위: ⭐
```

#### Kolors 2.1 (5위)
```
제공자: KlingAI
ELO: 1,156

📥 수집 방법:
- KlingAI 갤러리
- 중국 AI 커뮤니티

🎯 수집 목표: 500개
우선순위: ⭐
```

#### Dreamina 3.1 (9위)
```
제공자: Bytedance
ELO: 1,130

📥 수집 방법:
- 중국 서비스 (VPN 필요)
- 공개 갤러리

🎯 수집 목표: 500개
우선순위: ⭐ (선택적)
```

---

## 📊 데이터 생성 계획

### 직접 생성 (7,000개)

```python
우선순위 1: FLUX.1
- ComfyUI 로컬 생성
- 목표: 3,000개
- 시간: 3-5일

우선순위 2: Leonardo.Ai
- 무료 체험 활용
- 목표: 1,000개
- 시간: 7-10일 (일일 제한)

우선순위 3: Recraft V3
- 무료 체험 / API
- 목표: 1,000개
- 시간: 7-10일

우선순위 4: Stable Diffusion XL
- 보조 생성 (리더보드 외)
- 목표: 2,000개
- 시간: 2-3일

─────────────────────────────
소계: 7,000개
```

### 온라인 수집 (3,000개)

```python
크롤링 소스:
├── Reddit
│   ├── r/StableDiffusion (FLUX, Seedream)
│   ├── r/midjourney
│   └── r/aiArt
│
├── CivitAI
│   ├── FLUX 갤러리
│   └── SDXL 갤러리
│
├── Leonardo.Ai
│   └── 공개 갤러리
│
└── Artstation
    └── AI Art 태그

목표: 3,000개
시간: 3-5일 (크롤링 스크립트)
```

---

## 🔧 기술 스택

### 직접 생성
```
ComfyUI + FLUX.1
├── FLUX.1 schnell (빠른 생성)
├── FLUX.1 dev (고품질)
└── 자동화 스크립트 (Python)
```

### 수집 크롤링
```python
import requests
from bs4 import BeautifulSoup
import praw  # Reddit API

tools = [
    "requests + BeautifulSoup",
    "praw (Reddit API)",
    "gallery-dl (범용 크롤러)",
    "selenium (동적 페이지)"
]
```

---

## 📈 예상 효과

```
리더보드 상위 모델 사용 시:
├── 평가 데이터와 동일한 생성 패턴
├── 최신 기법 커버 (2024-2025)
└── 탐지 난이도 높음 (고품질)

예상 성능 향상:
- 현재: F1 0.5600
- 목표: F1 0.65~0.75 (+16~34%)
```

---

## ⚠️ 주의사항

### 저작권/라이선스
```
✅ 허용:
- 연구/교육 목적 사용
- 공개 갤러리 이미지
- 오픈소스 모델 생성물

⚠️ 확인 필요:
- 상용 서비스 생성물 (ToS 확인)
- 인물 초상권 (합성 얼굴은 OK)
```

### 데이터 다양성
```
✅ 확보해야 할 것:
- 다양한 인종/연령/성별
- 다양한 조명/각도
- 다양한 배경/환경
- 다양한 표정/포즈
```

---

**작성일**: 2025-10-30  
**출처**: https://artificialanalysis.ai/image/leaderboard/text-to-image  
**다음 업데이트**: 리더보드 변경 시

