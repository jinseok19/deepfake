# 딥페이크 탐지 대회 전략서

## 목표
- 비공개 테스트셋 기반 추론 자동화 환경에서 Macro F1 최대화(양성=Fake=1)
- 3시간 시간 제한 내 안정 추론 + 높은 재현성

## 핵심 요약(영상 사전설명회 + 리포지터리 분석)
- 데이터: 평가셋은 이미지(.jpg/.png) + 동영상(.mp4) 혼합, 한 이미지/영상 당 명확한 단일 얼굴
- 제출: task.ipynb에서 !pip 설치 → 추론 → submission.csv 생성 → aifactory.submit
- 지표/정책: Macro F1(Real=0/Fake=1), 하루 3회 제출(간격 존재), 리더보드 에러로 디버깅 가능
- 리소스: CPU 8C, RAM 48GB, L4/T4, 수 GB 스토리지, 3시간 제한
- 내부 검증 결과: Mean only 집계가 안정적, 얼굴 미검출 시 보수적 처리 권장, 과도한 필터링/Max 가중치는 하락 경향

## 데이터 전략
- 학습데이터 미제공 → 공개 데이터 + 온라인 후처리 분포 정합
  - 권장: FaceForensics++(c23/c40), Celeb-DF v2, DFDC preview, YouTube/SNS 재인코딩 샘플
  - 립싱크/스왑/생성형 혼합 커버리지 확보, 한 얼굴만 명확한 데이터 위주

## 전처리 파이프라인
- 얼굴 검출: dlib 우선(실패 시 중앙 크롭 백업), 224x224 리사이즈
- 동영상: 균등 샘플링 NUM_FRAMES=30(기본), 20~40 범위에서 시간 예산 내 조정
- 프레임 필터링: 기본 비권장(정보 손실 위험). 전 프레임 사용 후 집계에서 안정화

## 모델/집계
- 모델: ViT 사전학습 분류기(리포지터리 베이스라인과 일치)
- 집계: 프레임 확률 단순 평균(Mean only)로 최종 클래스 결정
- 임계값 튜닝: 내부 밸리데이션으로 Macro F1 최대화 threshold 산정(클래스 불균형 대비)

## 성능/시간 최적화
- BATCH_SIZE=16, num_workers≤8(환경 한도 내 상승 시도)
- 3시간 초과 방지: NUM_FRAMES 30 → 20(타임아웃 조짐 시), 필요 시 BATCH_SIZE 증가로 보상
- OOM/DISK: 배치 축소, 워커 축소, 중간 산출물/캐시 저장 금지

## 제출 안정화 규칙
- 의존성은 task.ipynb에서 !pip 설치, 경로는 상대("./model/..."), 불필요 파일(.git 등) 제외
- aifactory.submit 직전 submission.csv 스키마 확인: filename(string), label(int)
- 리더보드 에러로 ImportError/메모리/디스크 초과 1차 진단

## 제출 스케줄(하루 3회 가정)
1) 안정 확보기: 베이스라인 회귀(Mean only, 전 프레임, 얼굴 미검출 보수 처리)
2) 단일 변수 실험: NUM_FRAMES 30→40(나머지 고정) 제출
3) 미세 조정: 배치/워커 또는 threshold 튜닝 제출(내부 검증값 반영)

## 파라미터 프리셋
- 기본(균형): NUM_FRAMES=30, BATCH_SIZE=16, 집계=Mean only, workers=8
- 속도: NUM_FRAMES=20, BATCH_SIZE=32, workers=8
- 정확: NUM_FRAMES=40, BATCH_SIZE=16, workers=8

## 리스크/대응
- 얼굴 미검출 남용(중앙 크롭 과다) → 노이즈 증가: 실패 시 Real(0) 백오프 고려
- Max 가중 집계 → 노이즈 증폭: Mean only 유지
- dlib 설치 이슈 → dlib-bin 휠 대체, 제출 환경에는 dlib 기본 제공

## 체크리스트
- [ ] task.ipynb 내 !pip 설치 셀 정상 실행
- [ ] 모델/소스 상대경로 및 불필요 파일 제거
- [ ] ./data/ 전 파일 순회 → submission.csv 저장 완료
- [ ] filename,label 스키마/자료형/단일 예측 보장(영상도 0/1 하나)
- [ ] CUDA 키/버전 일치 확인, 3시간/메모리/디스크 제한 고려

## 부록: 실행 순서(로컬)
1) venv 활성화 → requirements 설치 완료
2) python dev/improved_inference.py로 샘플 검증
3) scripts/prepare_submission.bat 실행 → submit/task.ipynb 오픈
4) Key 입력 후 모든 셀 실행 → 제출/리더보드 확인
