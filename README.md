# Fake Catcher

RGB 및 RGB+FFT 이미지 판별 모델, Grad-CAM 판단 근거 시각화, 로컬 웹 앱입니다.

## Windows에서 실행

Python 3.12를 설치하고 이 저장소를 내려받아 압축을 푼 뒤 프로젝트 폴더에서 실행하세요.

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1 -Device cpu
.\run.cmd
```

브라우저에서 http://127.0.0.1:8765 를 엽니다. 검토된 예비 학습 모델 `models/best.pt`가 포함되어 있어 재학습 없이 실행할 수 있습니다. GPU 없이도 CPU에서 웹 분석이 가능합니다. 최초 패키지 설치에는 인터넷이 필요합니다. 종료는 서버 창에서 Ctrl+C입니다.

## Linux에서 실행

Python 3.12와 venv를 준비하고 실행하세요.

```bash
bash scripts/setup_linux.sh cpu
.venv/bin/python -m uvicorn fakecatcher.server:app --host 127.0.0.1 --port 8765
```

GPU 학습을 하려면 서버 드라이버 호환성을 확인하고 `bash scripts/setup_linux.sh cu124`로 설치합니다. Slurm 사용 환경은 `scripts/train.slurm`의 자원 설정을 서버 규칙에 맞게 수정하세요.

## 예비 성능

CIFAKE 표본: 학습 8,000장 / 검증 1,000장 / 테스트 2,000장. 각 클래스 균형, seed 42, 각 모델 5 epoch. 검증 AUROC가 가장 높은 모델을 평가했습니다.

| 모델 | 테스트 정확도 | F1 | AUROC |
|---|---:|---:|---:|
| RGB | 95.75% | 0.956962 | 0.990637 |
| RGB+FFT | 94.05% | 0.942816 | 0.991235 |

**32×32 CIFAKE 이미지를 확대한 예비 실험입니다. 최신 생성기·고해상도·SNS 압축 이미지에 대한 일반화 성능은 검증하지 않았습니다.** 이번 실험은 FFT 추가의 정확도 향상을 입증하지 않았습니다. 점수는 보정되지 않은 모델 출력이며 진위의 확정 증거가 아닙니다. Grad-CAM은 판정에 참고한 RGB 영역을 보여주며 실제 위조 위치의 정답이 아닙니다.

상세 보고서: `reports/cifake-pilot.html`, 수치: `reports/comparison.json`.

## 학습 데이터 및 재학습

데이터 이미지와 학습 재개용 원본 체크포인트는 이 저장소에 포함하지 않았습니다. 저자가 공개한 CIFAKE에서 표본을 준비할 수 있습니다.

```powershell
.\.venv\Scripts\python.exe -m fakecatcher.prepare_data
.\.venv\Scripts\python.exe -m fakecatcher.experiment --device cuda --epochs 5
```

GPU 환경이 없으면 `--device cpu`를 사용할 수 있지만 더 오래 걸립니다. 원본 출처·버전·중복 제거 기록은 데이터 준비 시 생성됩니다. 생성된 `runs/cifake_pilot`에는 최고 검증 성능의 `best.pt`와 재개용 `last.pt`가 저장됩니다. 같은 명령에서 `--epochs 10`으로 늘리면 해당 PC에서 저장한 마지막 완료 회차부터 이어갑니다. 테스트 결과를 보고 반복해서 설정을 고르면 최종 평가는 별도 테스트셋으로 해야 합니다.

새 모델을 웹에 연결하려면 `FAKE_CATCHER_CHECKPOINT`에 해당 `best.pt` 경로를 지정하고 서버를 재시작하세요.

## 검증과 개인정보

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

이미지 분석은 실행 중인 PC에서 처리합니다. 업로드 이미지는 디스크에 저장하지 않습니다. 개인 파일, 인증정보, 원본 PC 환경 보고서, 개인 경로, 전체 패키지 목록, 원본 파일 시간은 저장소에 포함하지 않습니다. 포함된 모델은 CPU 형식으로 변환했으며 원본 장치 설정·학습 시간·CUDA 난수 상태를 제거했습니다. 커밋에도 개인 이름이나 개인 이메일을 사용하지 않습니다.

## 데이터 출처

CIFAKE: Bird & Lotfi (2024), 실제 이미지: CIFAR-10, Krizhevsky & Hinton (2009). 데이터 출처 및 저자가 명시한 라이선스는 `THIRD_PARTY_NOTICES.md`를 참고하세요. 본 저장소에는 공개 데이터에서 학습한 예비 가중치를 포함합니다.
