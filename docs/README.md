# 📚 문서 인덱스

> 딥페이크 탐지 모델 경진대회 - 모든 문서의 중앙 허브

---

## 🚀 빠른 시작

처음 시작하시나요? 이 순서대로 읽어보세요:

1. **[QUICK_START.md](QUICK_START.md)** - 5분 안에 시작하기
2. **[ENV_SETUP_GUIDE.md](ENV_SETUP_GUIDE.md)** - 환경 설정
3. **[DLIB_INSTALL.md](DLIB_INSTALL.md)** - dlib 설치 (필요시)
4. **[SUBMISSION_GUIDE.md](SUBMISSION_GUIDE.md)** - 제출 방법

---

## 📂 문서 카테고리

### 🎯 대회 정보
| 문서 | 설명 | 중요도 |
|------|------|--------|
| [CompetitionDescription.md](CompetitionDescription.md) | 대회 개요, 규칙, 일정 | ⭐⭐⭐ |
| [DATASET_SEMINAR.md](DATASET_SEMINAR.md) | 데이터셋 구조, 평가 방법 | ⭐⭐⭐ |

### ⚙️ 설치 및 설정
| 문서 | 설명 | 중요도 |
|------|------|--------|
| [QUICK_START.md](QUICK_START.md) | 빠른 시작 가이드 | ⭐⭐⭐ |
| [ENV_SETUP_GUIDE.md](ENV_SETUP_GUIDE.md) | 환경 설정 상세 가이드 | ⭐⭐ |
| [DLIB_INSTALL.md](DLIB_INSTALL.md) | dlib 로컬 설치 (3가지 방법) | ⭐⭐ |

### 📤 제출 및 전략
| 문서 | 설명 | 중요도 |
|------|------|--------|
| [SUBMISSION_GUIDE.md](SUBMISSION_GUIDE.md) | 제출 방법 상세 가이드 | ⭐⭐⭐ |
| [STRATEGY.md](STRATEGY.md) | 딥페이크 탐지 전략 | ⭐⭐ |

### 📊 실험 및 로그
| 문서 | 설명 | 중요도 |
|------|------|--------|
| **→ [../log/LOG_SUMMARY.md](../log/LOG_SUMMARY.md)** | **전체 실험 요약 (최신!)** | ⭐⭐⭐ |
| **→ [../log/PIPELINE_GUIDE.md](../log/PIPELINE_GUIDE.md)** | **파이프라인 가이드** | ⭐⭐⭐ |
| [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) | 실험 로그 (아카이브) | ⭐ |

> **💡 팁**: 실험 로그는 `../log/` 폴더가 최신 버전입니다!

### 🏗️ 프로젝트 구조
| 문서 | 설명 | 중요도 |
|------|------|--------|
| [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) | 프로젝트 구조 및 워크플로우 | ⭐⭐ |

---

## 📋 상황별 추천 문서

### 😊 처음 시작할 때
```
1. QUICK_START.md
2. ENV_SETUP_GUIDE.md
3. CompetitionDescription.md
```

### 🔧 환경 설정 문제
```
1. ENV_SETUP_GUIDE.md
2. DLIB_INSTALL.md (dlib 오류 시)
3. PROJECT_STRUCTURE.md (구조 이해)
```

### 🚀 제출 준비
```
1. SUBMISSION_GUIDE.md
2. ../log/LOG_SUMMARY.md (최신 실험 확인)
3. PROJECT_STRUCTURE.md (폴더 구조 확인)
```

### 🧪 실험 계획
```
1. ../log/LOG_SUMMARY.md (기존 실험 분석)
2. ../log/PIPELINE_GUIDE.md (파이프라인 이해)
3. STRATEGY.md (전략 참고)
```

### ❓ 문제 해결
```
1. ENV_SETUP_GUIDE.md (환경 문제)
2. DLIB_INSTALL.md (dlib 문제)
3. SUBMISSION_GUIDE.md (제출 문제)
```

---

## 🗂️ 전체 프로젝트 구조

```
deepfake/
├── baseline/              # 베이스라인 코드
├── dev/                   # 개발용 스크립트
├── docs/                  # 📚 모든 문서 (현재 위치)
│   ├── CompetitionDescription.md
│   ├── DATASET_SEMINAR.md
│   ├── DLIB_INSTALL.md
│   ├── ENV_SETUP_GUIDE.md
│   ├── EXPERIMENT_LOG.md (아카이브)
│   ├── PROJECT_STRUCTURE.md
│   ├── QUICK_START.md
│   ├── README.md (이 파일)
│   ├── STRATEGY.md
│   └── SUBMISSION_GUIDE.md
├── log/                   # 📊 실험 로그 (최신!)
│   ├── EXP-001.md
│   ├── EXP-002.md
│   ├── LOG_SUMMARY.md ⭐
│   ├── PIPELINE_GUIDE.md ⭐
│   ├── README.md
│   └── TEMPLATE.md
├── samples/               # 샘플 데이터
├── scripts/               # 유틸리티 스크립트
├── submit/                # 🚀 제출용 폴더
├── README.md              # 프로젝트 메인 README
└── requirements.txt       # Python 의존성
```

> 📌 **주의**: `log/` 폴더가 최신 실험 로그입니다. `docs/EXPERIMENT_LOG.md`는 아카이브입니다.

---

## 📝 문서 작성 규칙

### 파일명
- PascalCase + .md (예: `CompetitionDescription.md`)
- 설명적이고 명확한 이름

### 내용
- 마크다운 형식
- 이모지 활용 (가독성)
- 코드 블록 사용 (예제)
- 목차 포함 (긴 문서)

### 업데이트
- 날짜 명시
- 변경 이력 기록 (중요 문서)

---

## 🔍 문서 검색 팁

### Windows 탐색기 검색
```
docs 폴더에서 Ctrl+F → 내용 검색
```

### VS Code 검색
```
Ctrl+Shift+F → docs 폴더만 선택 → 검색
```

### grep (터미널)
```bash
cd docs
grep -r "검색어" .
```

---

## 💡 꿀팁

### 자주 보는 문서는 북마크!
- VS Code: 파일 우클릭 → "Pin to Quick Open"
- 브라우저: 즐겨찾기 추가

### 문서 간 링크 활용
- 각 문서 하단에 관련 문서 링크 있음
- "참고 자료" 섹션 확인

### 최신 정보는 log/ 폴더!
- `docs/EXPERIMENT_LOG.md` - 아카이브
- `log/LOG_SUMMARY.md` - 최신! ⭐

---

## 📞 추가 정보

- **프로젝트 메인 README**: [../README.md](../README.md)
- **실험 로그 허브**: [../log/README.md](../log/README.md)
- **대회 페이지**: https://aifactory.space/task/9197

---

**마지막 업데이트**: 2025.10.29  
**문서 개수**: 10개  
**상태**: 최신 ✅
