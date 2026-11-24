# 밝기·노이즈 인식 기반 저조도 이미지 복원

영상처리 기말 프로젝트의 실행 가능한 초기 구현입니다. Zero-DCE를 직접 학습하고, 고정 필터 기반 밝기/노이즈 모듈의 효과를 동일 조건에서 비교합니다. 실제 LOL 성능 개선은 아직 검증하지 않았습니다.

## 구성

| 파일 | 기능 |
|---|---|
| dataset.py | LOL 쌍 탐색, 고정 분할, 정규화·크롭·좌우 반전 |
| model.py | 7층 DCE-Net, 8회 곡선 적용, none/bright/noise/both |
| loss.py | 공간 일관성·노출·색·곡선 TV 손실 |
| train.py | 학습, 검증 손실, 실행 설정·분할 기록, 체크포인트 재개 |
| eval.py | PSNR/SSIM·보조지표, 추론 시간·파라미터, 비교 그림·확대 패치·맵 |
| colab/experiment.ipynb | Drive 연결과 네 구성 실행 예시 |
| docs/experiment_plan.md | 요구사항 구분, 실험 순서와 남은 작업 |

## 준비

Python 3.10 이상을 사용합니다.

```bash
python -m pip install -r requirements.txt
```

LOL 데이터를 별도로 확보하고 공식 학습/평가 폴더를 유지하세요. 학습용 high는 학습 손실에 사용하지 않습니다. 검증·평가에서만 참조합니다. low/high 파일의 상대 경로와 확장자를 제외한 이름이 같아야 합니다.

```text
data/LOL/
  our485/low/
  our485/high/
  eval15/low/
  eval15/high/
```

위 폴더 이름은 보유 데이터의 Drive 구조와 같습니다. 데이터는 저장소에 포함하지 않습니다.

### LOL 데이터 출처와 검증 상태 (TASK-004)

- **보유 데이터:** [공유 LOL 폴더](https://drive.google.com/drive/folders/1EVxcAMhHSJes04PDHd1U_vwCCtk51V8b)의 `our485`·`eval15` 구조와 수량·해상도는 원본 LOL(LOL-v1) 설명에 부합합니다. 실제 다운로드 출처가 이 배포본인지는 업로더의 확인이 필요합니다.
- **원저자 출처:** [Wei 등의 LOL 배포 페이지](https://daooshee.github.io/BMVC2018website/)에 원본 쌍 데이터 다운로드 링크가 있습니다. [원 논문](https://bmvc2018.org/contents/papers/0451.pdf)은 485개 학습 쌍, 15개 평가 쌍, 400×600 이미지로 설명합니다. 보유 파일도 2026-11-24 확인 결과 이 수량과 해상도입니다.
- **접근·사용 조건:** 공유 폴더 원본 파일 읽기는 확인했습니다. 데이터 자체의 명시적인 라이선스·재배포 조건과 실제 다운로드 경로는 미확인입니다. 코드 저장소의 라이선스를 이미지 데이터에 적용한다고 가정하지 마세요. 데이터는 저장소에 올리지 않습니다.
- **현재 검증 결과:** 학습 485쌍, 공식 평가 15쌍이 이름으로 모두 대응하고, 전체 1,000장이 600×400 PNG입니다. 원본 바이트의 PNG 서명·청크 CRC·종료 구조 오류와 분할 간 동일 바이트 중복은 0건입니다. 학습 `high`에는 서로 다른 파일명으로 저장된 동일 바이트 이미지가 66그룹 있어 분할 시 묶습니다. 2026-11-24 Colab에서 `dataset.py` CLI로 전체 1,000장의 Pillow 디코딩·쌍·분할 감사를 통과했습니다(종료 코드 0, `issues: []`). 상세 증거는 `results/TASK-004-audit.txt`와 `results/TASK-004-005-colab-audit.txt`에 있습니다.

공유 폴더를 Colab 또는 로컬의 위 구조로 배치한 뒤 프로젝트 루트에서 다음 명령을 실행하세요. `python` 명령이 없는 환경에서는 `.venv/bin/python`처럼 프로젝트 Python 경로를 사용합니다.

```bash
python dataset.py \
  --train-low data/LOL/our485/low --train-high data/LOL/our485/high \
  --test-low data/LOL/eval15/low --test-high data/LOL/eval15/high \
  --crop-size 128 --val-ratio 0.1 --split-seed 0
```

이 명령은 공식 학습/평가 폴더의 파일 수·이름 중복·누락 쌍·해상도 분포·크기 불일치·손상·지원하지 않는 확장자·학습 크롭 가능 여부를 확인합니다. `partition`에는 공식 학습 쌍에서 나눈 train/val 장수와 공식 test 장수, 세 분할 사이의 같은 경로·**바이트 내용이 같은 파일** 수가 기록됩니다. `issues`가 비어 있고 종료 코드가 0이어야 자동 검사 통과입니다. 파일명이 같은 영상이 실제로 같은 장면인지, 서로 다른 파일의 유사 장면이 두 분할에 섞였는지는 대표 쌍을 직접 열어 추가 확인하세요. 검사 출력과 실제 다운로드 주소·버전·사용 조건을 결과 기록에 보관하세요.

### 분할·전처리·배치 규칙 (TASK-005)

`train.py`는 별도의 공식 test 경로를 필수로 받습니다. 공식 학습 쌍만 고정된 `split_seed`로 약 90% train·10% val로 나누며, low 또는 high 파일 내용이 동일한 쌍은 같은 분할에 둡니다. `split_seed=0`으로 보유 파일의 목록과 내용 중복 그룹을 적용한 결과 train 437쌍·val 48쌍·test 15쌍, 분할 간 동일 바이트 중복 0건입니다. 명단은 `results/TASK-005-manifest.json`에 있습니다. test는 재분할하지 않습니다.

학습 입력은 RGB `float32` [0,1] 텐서 `[B,3,H,W]`이며 원본 영상에서 랜덤 크롭과 좌우 반전을 적용합니다. 학습 손실에는 high를 공급하지 않습니다. val/test는 증강·크롭 없이 원본 크기로 읽고 배치 크기 1을 사용합니다. 학습 배치는 셔플하며 마지막 작은 배치를 버리지 않습니다. 실제 파일 8장으로 train 첫 배치 `[8,3,128,128]`을, 대표 val/test 쌍으로 원본 크기 `[1,3,400,600]`을 확인했습니다. Colab 전체 데이터 CLI에서 train/val/test 437/48/15와 분할 간 동일 경로·동일 바이트 내용 중복 0건을 확인했습니다.

## 학습

```bash
python train.py --low-dir data/LOL/our485/low --high-dir data/LOL/our485/high --test-low-dir data/LOL/eval15/low --test-high-dir data/LOL/eval15/high --mode none --epochs 10 --lr 0.0001 --crop-size 128 --batch-size 8 --split-seed 0 --seed 0 --max_hours 3 --output results/runs
```

`--mode`만 `bright`, `noise`, `both`로 바꾸어 반복합니다. 10에폭·128크롭은 실행 예시이며 본 실험의 확정값이 아닙니다. 가장 느린 구성의 T4 예비 측정 후 모든 구성의 에폭·크롭·배치를 동일하게 확정하세요. `--output`을 Drive 경로로 지정하면 에폭마다 Drive에 저장됩니다.

- `--split-seed`: 데이터 분할 고정용. 모든 시드 실험에서 유지합니다.
- `--seed`: 모델 초기화와 증강·셔플용. 가산점 실험에서는 0, 1, 2로 변경합니다.
- 평가 폴더는 필수로 지정합니다. 학습 데이터 일부만 검증용으로 분리합니다.
- 크롭보다 작은 영상은 오류로 알립니다. 평가 배치는 1이므로 서로 다른 원본 크기를 처리합니다.
- 같은 출력 폴더를 덮어쓰지 않습니다. 재개하려면 동일 명령에 `--resume`을 추가합니다.
- 재개는 마지막 완료 에폭부터입니다. 부분 에폭은 폐기하며, 시간 초과 실행은 최종 비교에서 제외합니다. 예산은 저장된 누적 시간 기준입니다. 한 연산 실행 중에는 즉시 중단할 수 없으므로 실제 시간에 여유를 두세요.
- 정확한 에폭 경계 RNG 재개를 위해 기본 `--num-workers 0`을 권장합니다. 작업자가 여러 개이면 작업자 내부 증강 RNG까지 복원하지 않으므로 재개 전후의 완전 동일성을 보장하지 않습니다.

모듈 기본값은 가설상의 초깃값입니다. 검증 세트로 선정한 설정을 JSON으로 저장해 **네 구성 모두에 같은** `--module-config 경로.json`을 전달하세요. 설정 키는 model.py의 EnhancementConfig를 참고하세요.

## 평가

튜닝에는 먼저 검증 분할을 사용합니다. 하이퍼파라미터 확정 후 테스트를 실행합니다.

```bash
python eval.py --runs results/runs --compare none bright noise both --split val
python eval.py --runs results/runs --compare none bright noise both --split test
```

시드 3개 학습이 완료되면 `--seeds 0 1 2`로 평균·표준편차를 계산합니다. 테스트 결과를 보며 설정을 다시 선택하지 마세요.

평가 폴더에 summary.json, 시드별 per_image.csv, 입력·모델 출력·정답 비교 그림, 밝고 어두운 위치의 확대 패치, 보조 맵, 기본 모델 대비 PSNR 차이 순위가 저장됩니다. 순위 상하위는 성공/실패 사례 선정 후보입니다. 개선폭의 해석에는 PSNR dB 차이와 원시 지표를 함께 사용하세요.

SSIM은 RGB 채널별 Gaussian 11×11, sigma=1.5, 유효 영역 평균입니다. PSNR은 RGB MSE, data_range=1이며 동일 이미지의 무한대 대신 최대 120dB로 표기합니다. 과노출 비율(Y≥0.98), 정상조도 기준 평탄 영역 잔차 표준편차, 채널 평균 편차는 팀 설계 보조지표이며 노이즈·색 왜곡의 완전한 측정이 아닙니다. 추론 시간은 배치 1, 이미지별 워밍업 후 forward 평균으로 I/O·지표 계산을 제외합니다. 최종 보고는 T4에서 측정하세요.

## 확인

```bash
python -m unittest discover -s tests -v
```

합성 영상으로 네 모드의 출력 범위·역전파·동일 파라미터 수, 분할 고정, 학습·재개·평가 및 미완료 실행 거부를 확인합니다. 합성 영상 결과를 프로젝트 성능 수치로 사용하지 않습니다.

## 출처와 구현 범위

[Zero-DCE 논문](https://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_Zero-Reference_Deep_Curve_Estimation_for_Low-Light_Image_Enhancement_CVPR_2020_paper.pdf)과 [저자 저장소](https://github.com/Li-Chongyi/Zero-DCE)를 참고해 코드를 작성했습니다. 논문 보고 성능이나 사전학습 가중치를 비교 기준으로 사용하지 않습니다. 본 구현은 공식 모델 가중치 키와 호환되지 않습니다.

공식 학습 코드 기준의 손실 가중치 spatial=1, exposure=10, color=5, TV=200, 노출 목표 0.6을 사용합니다. TV는 공식 코드처럼 채널 합을 유지합니다. 곡선 부호는 공식 구현의 `I + r*(I²-I)`를 따릅니다(제안서 `I + α*I*(1-I)`의 α=-r). 개선 연산은 제안서의 밝기 혼합과 노이즈 선택적 평활화를 구현했습니다. 보고서에는 참고 출처와 직접 작성한 범위를 표시하세요.
