# qlt-etf-single-swing-v1

ETF single swing 전략의 학습, 백테스트, 랭킹 추론을 위한 독립 프로젝트입니다.

제약:

- `quantylab` 파이썬 모듈에 의존하지 않습니다.
- 데이터 확보는 Quantylab REST API만 사용합니다.
- 주문 실행 및 계좌 연동 기능은 포함하지 않습니다.

## 포함 범위

- PPO 기반 ETF single swing 학습
- 백테스트
- 모델 랭킹/신호 추론
- Quantylab REST API 기반 데이터셋 생성

## 설치

```bash
cd qlt-etf-single-swing-v1
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 환경변수

```bash
export QUANTYLAB_API_KEY=...
```

## 1. 데이터셋 생성

```bash
qlt-etf-build-dataset \
  --output-dir data/etf_api_20260607 \
  --start-date 20180101 \
  --end-date 20260607
```

생성 파일:

- `environment.csv`
- `training_selected.csv`
- `training_scaled.csv`
- `etf_codes.csv`
- `dataset_meta.json`

## 2. 학습

```bash
qlt-etf-train \
  --dataset etf_api_20260607 \
  --trading-method swing \
  --network-type mamba \
  --episodes 300 \
  --output-dir output/train/etf-swing-v1 \
  --log-dir output/train/etf-swing-v1
```

학습 결과 모델은 `policy_best.pt`, `value_best.pt` 형태로 저장됩니다.

## 3. 백테스트

```bash
qlt-etf-backtest \
  --dataset etf_api_20260607 \
  --model /absolute/path/to/etf-swing-v1
```

`--model`은 절대경로 모델 디렉토리 또는 `models/` 하위 이름 둘 다 받을 수 있습니다.

## 4. 최신 랭킹/신호 추론

데이터셋 없이 API에서 최신 피처 벡터를 직접 받아 추론할 수 있습니다.

```bash
qlt-etf-predict \
  --model /absolute/path/to/etf-swing-v1 \
  --api-token "$QUANTYLAB_API_KEY"
```

또는 기존 데이터셋으로도 가능합니다.

```bash
qlt-etf-predict \
  --model /absolute/path/to/etf-swing-v1 \
  --dataset etf_api_20260607
```

## 디렉토리 구조

```text
qlt-etf-single-swing-v1/
├── pyproject.toml
├── README.md
├── data/
├── models/
├── output/
└── src/
```

## 참고

- 데이터셋 생성은 공개 REST API의 `feature-vectors`와 `stock-candles`를 함께 사용합니다.
- `training_scaled.csv`는 API에서 받은 scaled `feature-vectors`를 그대로 저장합니다.
- 사용자가 자체 OHLCV 데이터로 feature vector를 만들고 싶다면 `dataset_builder.build_feature_vectors_from_candles()`를 선택적으로 사용할 수 있습니다.
- 최신 추론은 REST API의 scaled `feature-vectors`를 그대로 사용합니다.
- 주문 실행 및 실계좌 관련 기능은 제외되어 있습니다.
