## 딥페이크 판별 대회 요약 정리 (Markdown)

### 대회 개요
- **주제**: 이미지/동영상 속 얼굴의 딥페이크 여부 이진 분류
- **주최/주관/운영**: 행정안전부, NIA, 국과수 / 인공지능팩토리, 웨슬리퀘스트
- **일정**
  - 접수: 9/25(목) 10:00 ~ 11/20(목) 17:00
  - 운영: 10/23(목) 10:00 ~ 11/20(목) 17:00
  - 팀병합 마감: ~ 11/6(목) 17:00
  - 모델 검증: 11/21(금) ~ 11/26(수)
  - 발표: 11/27(목) 17:00, 시상식: 12/3(수) (변동 가능)
- **참가 대상**: 대한민국 국민, 개인/최대 5인 팀 (미성년자 별도 서식 제출)

### 시상
- **총상금 9,200만원 / 7팀**
  - 대상(1): 3,000만원
  - 최우수상(2): 각 1,500만원
  - 우수상(4): 각 800만원

---

## 평가/제출 방식

### 평가 방법
- **비공개 테스트셋 자동 채점**
- **지표**: Macro F1-score (양성=Fake=1, 음성=Real=0)
- **시간 제한**: 최대 3시간/제출

### 제출 규칙
- **모델 제출(추론 자동화)**: `task.ipynb`에서 추론/저장/제출까지 수행
- **하루 3회**, 재제출은 이전 결과 확인 후 **30분 간격**
- **평가 입력 경로**: `./data/` (jpg/png/mp4 혼합, 하위 폴더 없음)
- **출력 파일**: 현재 경로에 `submission.csv` 저장

---

## 데이터셋/입출력 규격

### 입력
- 이미지 또는 동영상(mp4)
- 각 데이터에는 **한 명의 얼굴**이 명확히 포함되어야 함
- 동영상은 프레임 추출 후 이미지 입력 가능

### 출력
- 각 파일별 **단일 예측**: 0(Real) 또는 1(Fake)
- 동영상도 **단일 값**(프레임 다수 분석 가능하나 최종 0/1)

### 평가 데이터 특성
- 포맷: `.png`, `.jpg`, `.mp4`
- 평균 5초 내외 동영상, 다양한 인종/연령, 다양한 생성/스왑/립싱크 기법 혼합

### 샘플 데이터
- Fake 이미지 7개, Fake 동영상 5개 제공

---

## submission.csv 형식

- 컬럼: `filename`(string, 확장자 포함), `label`(int; Real=0, Fake=1)
- 순서는 무작위여도 무관

```csv
filename,label
image_001.jpg,0
video_003.mp4,1
...
```

채점은 제출된 `submission.csv`와 내부 `answer.csv`로 Macro F1-score 계산.

---

## 베이스라인/폴더 구성

- 제출 압축 대상: `task.ipynb`가 위치한 폴더 및 하위 전부
- **모델/소스는 `task.ipynb` 하위 경로**에 배치, **상대경로("./")** 사용 권장
  - 예: 모델 경로 `./model/best.pth`
- 불필요 파일(.git, 개인정보 등) 포함 금지 (용량/시간 이슈)
- 베이스라인 구성 예시
  - `task.ipynb` (제출용)
  - `model/deep-fake-detector-v2-model/` (베이스라인 모델, 결과 생성 약 70분)

---

## task.ipynb 구성/예시

### 1) 라이브러리 설치
- `!pip install`로 `aifactory` 및 모델 추론에 필요한 의존성 설치

### 2) 추론 코드
- 모델 로드 → 전처리 → 예측 → `submission.csv` 저장 (현재 경로)

### 3) 제출 호출
```python
import aifactory.score as aif

aif.submit(
  model_name="my_model_name",
  key="MY-KEY"  # 마이페이지 활동히스토리에서 CUDA 환경별 Key 복사
)
```

---

## CUDA/환경 안내

### 지원되는 조합
| Python | CUDA | torch     |
|--------|------|-----------|
| 3.8    | 10.2 | 1.6.0     |
| 3.9    | 11.8 | 1.8.0     |
| 3.10   | 12.6 | 2.7.1     |

- 기본 환경에 torch가 준비되어 있으나, **호환 버전으로 재설치 가능**
- **제출 시 선택한 CUDA 버전에 맞는 Key 사용**

### 환경별 기본 라이브러리 세팅(요약)
- **CUDA 11.8**
  - torch==1.8.0+cu111, torchvision==0.9.0+cu111
  - numpy==1.23.5, scipy==1.11.4, scikit-learn==1.3.2
  - opencv-python-headless==4.9.0.80, pandas==2.0.3, Pillow
- **CUDA 12.6**
  - torch==2.7.1, torchvision==0.22.1
  - numpy==1.26.4, scipy==1.11.4, scikit-learn==1.3.2
  - opencv-python-headless==4.10.0.82, pandas, Pillow
- **CUDA 10.2**
  - torch==1.6.0, torchvision==0.7.0
  - numpy==1.23.5, scipy==1.10.1, scikit-learn==1.1.3
  - opencv-python-headless==4.9.0.80, pandas==1.5.3, Pillow

---

## Key 발급/사용
- 마이페이지 → 활동히스토리 → 대회 선택 → **키복사**
- 한 대회 내 **CUDA 10.2/11.8/12.6 별로 총 3개 Key** 생성
- `aif.submit(..., key="...")`에 정확히 입력

---

## 추론 자동화 작동/유의

### 작동 원리
1) `aif.submit()` 실행 시 현재 폴더/하위 전부 압축 전송  
2) 대기열 후 보안 클라우드 컨테이너에서 추론  
3) 완료/오류 결과가 리더보드로 전송 (할당/실행에 시간 소요 가능)

### 리소스/제한
- 환경: CPU 8core, RAM 48GB, L4 GPU(대기 시 T4), 스토리지 수 GB
- 시간 제한: **최대 3시간**
- 에러 "점수를 산출할 수 없었습니다…": **OOM** 또는 **Disk 초과** 가능성 큼
  - DataLoader CPU 워커 ≤ 8, 배치 크기 감소 권장

### 디버깅
- 리더보드 제출이력의 **에러보기**로 Traceback 확인
- 애매하면 QnA 문의: [Q&A 게시판](https://aifactory.space/task/9197/qna)

---

## 체크리스트 (빠른 제출용)
- [ ] `task.ipynb` 내 의존성 `!pip install`  
- [ ] 모델/코드 `task.ipynb` 하위에 배치, 상대경로 사용  
- [ ] `./data/` 전체 순회 → 예측 → `submission.csv` 생성  
- [ ] `filename,label` 스키마/형/경로/단일 예측 확인  
- [ ] CUDA 버전 선택과 **Key 일치** 확인  
- [ ] 3시간/메모리/디스크 제한 고려(워커/배치 조절)

---

참고/출처: `docs/CompetitionDescription.md`  
문의: 운영사무국 `cs@aifactory.page`, [Q&A 게시판](https://aifactory.space/task/9197/qna)
