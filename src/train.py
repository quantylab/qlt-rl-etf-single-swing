"""
학습 실행 모듈: ETF Trading

거래 방식:
- day: 시초가 매수 → 종가 매도 (지도학습 — 일중 수익률 예측)
- swing: 종가 기준 포지션 비율 리밸런싱 (RL — PPO)
"""
import os
import json
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import torch

from environment import SwingTradingEnvironment
from agent import TradingAgent
from target_etfs import TARGET_ETFS
from network import (
    ContinuousPolicyNetwork,
    LSTMContinuousPolicyNetwork,
    ValueNetwork,
    LSTMValueNetwork,
    GRNPolicyNetwork,
    GRNValueNetwork,
    FTTransformerPolicyNetwork,
    FTTransformerValueNetwork,
    gMLPPolicyNetwork,
    gMLPValueNetwork,
    MambaPolicyNetwork,
    MambaValueNetwork,
    MambaRegressionNetwork,
)
from trainer import PPOTrainer
from supervised_trainer import SupervisedDayTrainer


def load_data(env_data_path: str, training_data_path: str):
    """데이터 로드"""
    print(f"환경 데이터 로드: {env_data_path}")
    env_data = pd.read_csv(env_data_path)
    print(f"  - 데이터 크기: {len(env_data):,}행")
    print(f"  - 컬럼: {list(env_data.columns)}")

    print(f"\n학습 데이터 로드: {training_data_path}")
    training_data = pd.read_csv(training_data_path).values
    print(f"  - 데이터 크기: {training_data.shape}")

    assert len(env_data) == len(training_data), \
        f"환경 데이터({len(env_data)})와 학습 데이터({len(training_data)})의 길이가 다릅니다."

    return env_data, training_data


def create_networks(input_dim: int, network_type: str = 'grn',
                    device: str = 'cpu', min_concentration: float = 1.5,
                    d_model: int = 128, n_blocks: int = 3, d_state: int = 16,
                    policy_dropout: float = 0.15, value_dropout: float = 0.15):
    """신경망 생성 (연속 행동)"""
    print(f"\n신경망 생성 (타입: {network_type})")
    print(f"  - 입력 차원: {input_dim}")
    print(f"  - 연산 장치: {device}")

    min_conc = min_concentration
    if network_type == 'lstm':
        policy_net = LSTMContinuousPolicyNetwork(
            input_dim, hidden_dim=128, num_layers=2,
            dropout=policy_dropout, min_concentration=min_conc,
        )
        value_net = LSTMValueNetwork(
            input_dim, hidden_dim=128, num_layers=2,
            dropout=value_dropout,
        )
    elif network_type == 'grn':
        policy_net = GRNPolicyNetwork(
            input_dim, hidden_dim=128, num_blocks=2,
            dropout=policy_dropout, min_concentration=min_conc,
        )
        value_net = GRNValueNetwork(
            input_dim, hidden_dim=128, num_blocks=2,
            dropout=value_dropout,
        )
    elif network_type == 'ft_transformer':
        policy_net = FTTransformerPolicyNetwork(
            input_dim, d_token=64, n_heads=4, n_layers=2,
            dropout=policy_dropout, min_concentration=min_conc,
        )
        value_net = FTTransformerValueNetwork(
            input_dim, d_token=64, n_heads=4, n_layers=2,
            dropout=value_dropout,
        )
    elif network_type == 'gmlp':
        policy_net = gMLPPolicyNetwork(
            input_dim, d_model=64, d_ffn=128, n_blocks=2,
            dropout=policy_dropout, min_concentration=min_conc,
        )
        value_net = gMLPValueNetwork(
            input_dim, d_model=64, d_ffn=128, n_blocks=2,
            dropout=value_dropout,
        )
    elif network_type == 'mamba':
        policy_net = MambaPolicyNetwork(
            input_dim, d_model=d_model, d_state=d_state, n_blocks=n_blocks,
            dropout=policy_dropout, min_concentration=min_conc,
        )
        value_net = MambaValueNetwork(
            input_dim, d_model=d_model, d_state=d_state, n_blocks=n_blocks,
            dropout=value_dropout,
        )
    else:  # standard (ResidualBlock MLP)
        policy_net = ContinuousPolicyNetwork(
            input_dim, hidden_dim=256, num_blocks=3,
            dropout=policy_dropout, min_concentration=min_conc,
        )
        value_net = ValueNetwork(
            input_dim, hidden_dim=256, num_blocks=3,
            dropout=value_dropout,
        )

    policy_params = sum(p.numel() for p in policy_net.parameters())
    value_params = sum(p.numel() for p in value_net.parameters())
    print(f"  - 정책 신경망 파라미터: {policy_params:,}개")
    print(f"  - 가치 신경망 파라미터: {value_params:,}개")

    return policy_net, value_net


def main():
    parser = argparse.ArgumentParser(description='ETF Trading 학습')

    parser.add_argument(
        '--base-path',
        type=str,
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')),
    )
    parser.add_argument('--dataset', type=str, default='etf_20260410')
    parser.add_argument('--env-data', type=str, default=None)
    parser.add_argument('--training-data', type=str, default=None)
    parser.add_argument('--trading-method', type=str, default='swing',
                        choices=['day', 'swing'],
                        help='거래 방식 (day=지도학습, swing=RL)')

    # 공통 설정
    parser.add_argument('--max-steps', type=int, default=0,
                        help='최대 거래일 수 (0=전체)')
    parser.add_argument('--start-step', type=int, default=0)
    parser.add_argument('--start-chunk', type=int, default=0,
                        help='이 청크 인덱스(1-based)부터 학습 재개. 0이면 처음부터')
    parser.add_argument('--trading-fee', type=float, default=0.00015,
                        help='ETF 편도 수수료 (0.015%%)')
    parser.add_argument('--trading-tax', type=float, default=0.0,
                        help='거래세 (일반 주식: 0.002=0.2%%, ETF: 0)')
    parser.add_argument('--slippage', type=float, default=0.0003,
                        help='슬리피지 (0.03%%)')
    parser.add_argument('--network-type', type=str, default='mamba',
                        choices=['standard', 'lstm', 'grn', 'ft_transformer', 'gmlp', 'mamba'])
    parser.add_argument('--log-dir', type=str, default='output/train')
    parser.add_argument('--output-dir', type=str, default='output/train')
    parser.add_argument('--val-start', type=int, default=-1,
                        help='Validation 데이터 시작 스텝 (-1=비활성)')
    parser.add_argument('--clean-run', action='store_true',
                        help='기존 output 디렉토리를 삭제하고 처음부터 학습')
    parser.add_argument('--update', action='store_true',
                        help='기존 모델을 불러와서 추가 학습')
    parser.add_argument('--base-model', type=str, default='etf-swing-v5',
                        help='base 모델 디렉토리 경로 (미지정 시 --output-dir 사용)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--seed', type=int, default=42)

    # 지도학습 설정 (day trading)
    parser.add_argument('--epochs', type=int, default=100,
                        help='지도학습 에폭 수')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='미니배치 크기')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='지도학습 학습률')

    # RL 설정 (swing trading)
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--update-interval', type=int, default=64,
                        help='PPO 업데이트 간격')
    parser.add_argument('--initial-balance', type=float, default=10_000_000.0)
    parser.add_argument('--action-scale', type=float, default=1.0)
    parser.add_argument('--reward-scale', type=float, default=30.0)
    parser.add_argument('--fee-penalty-scale', type=float, default=15.0)
    parser.add_argument('--reward-terminal-scale', type=float, default=30.0)
    parser.add_argument('--inaction-penalty', type=float, default=10.0)
    parser.add_argument('--hold-threshold', type=float, default=0.2)
    parser.add_argument('--reward-clip', type=float, default=5.0)
    parser.add_argument('--drawdown-penalty-scale', type=float, default=25.0,
                        help='낙폭 패널티 강도 (0=비활성)')
    parser.add_argument('--drawdown-penalty-threshold', type=float, default=0.12,
                        help='낙폭 패널티 시작 MDD 수준')
    parser.add_argument('--rolling-sharpe-window', type=int, default=20,
                        help='롤링 Sharpe 윈도우')
    parser.add_argument('--rolling-sharpe-scale', type=float, default=2.0,
                        help='롤링 Sharpe 보너스 강도 (0=비활성)')
    parser.add_argument('--loss-aversion', type=float, default=1.2,
                        help='손실 비대칭 배율 (>1: 손실 패널티 강화, 1=대칭)')
    parser.add_argument('--min-concentration', type=float, default=1.5)
    parser.add_argument('--d-model', type=int, default=128,
                        help='Mamba d_model 차원 (기본: 128)')
    parser.add_argument('--n-blocks', type=int, default=3,
                        help='Mamba 블록 수 (기본: 3)')
    parser.add_argument('--d-state', type=int, default=16,
                        help='Mamba SSM 상태 차원 (기본: 16)')
    parser.add_argument('--policy-dropout', type=float, default=0.15,
                        help='정책 신경망 드롭아웃 비율')
    parser.add_argument('--value-dropout', type=float, default=0.15,
                        help='가치 신경망 드롭아웃 비율')
    parser.add_argument('--lr-policy', type=float, default=0.0001)
    parser.add_argument('--lr-value', type=float, default=0.0003)
    parser.add_argument('--policy-weight-decay', type=float, default=1e-4,
                        help='정책 Optimizer weight decay')
    parser.add_argument('--value-weight-decay', type=float, default=3e-4,
                        help='가치 Optimizer weight decay')
    parser.add_argument('--gamma', type=float, default=0.995)
    parser.add_argument('--epsilon', type=float, default=0.2)
    parser.add_argument('--action-mix-prob', type=float, default=0.05)
    parser.add_argument('--concentration-target', type=float, default=4.0)
    parser.add_argument('--concentration-penalty-coef', type=float, default=0.01)
    parser.add_argument('--entropy-coef-start', type=float, default=0.05)
    parser.add_argument('--entropy-coef-end', type=float, default=0.01)
    parser.add_argument('--entropy-decay-episodes', type=int, default=300)
    parser.add_argument('--target-bias-low', type=float, default=0.10)
    parser.add_argument('--target-bias-high', type=float, default=0.90)
    parser.add_argument('--trade-rate-threshold', type=float, default=0.15)
    parser.add_argument('--entropy-boost-factor', type=float, default=1.15)
    parser.add_argument('--low-policy-std-threshold', type=float, default=0.08)
    parser.add_argument('--action-mix-start', type=float, default=0.05)
    parser.add_argument('--action-mix-end', type=float, default=0.01)
    parser.add_argument('--val-min-trades', type=int, default=1)
    parser.add_argument('--validation-interval', type=int, default=5,
                        help='Validation 평가 주기 (에피소드)')
    parser.add_argument('--early-stop-patience', type=int, default=35,
                        help='Validation score 미개선 허용 횟수 (0=비활성)')
    parser.add_argument('--early-stop-min-delta', type=float, default=0.1,
                        help='Validation score 최소 개선폭')
    parser.add_argument('--early-stop-warmup-episodes', type=int, default=120,
                        help='조기 종료를 적용하기 전 최소 에피소드')
    parser.add_argument('--chunk-years', type=int, default=1,
                        help='청크당 연수 (0=분할 안 함, 1=ETF별 1년씩 분할)')
    parser.add_argument('--update-lr-scale', type=float, default=0.5)
    parser.add_argument('--visualize', action='store_true', default=True)
    parser.add_argument('--no-visualize', action='store_false', dest='visualize')
    parser.add_argument('--viz-interval', type=int, default=10)

    args = parser.parse_args()

    # 시드
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = ('cuda' if torch.cuda.is_available() else 'cpu') \
        if args.device == 'auto' else args.device

    # Clean run
    if args.clean_run:
        import shutil
        log_dir_abs = args.log_dir if os.path.isabs(args.log_dir) \
            else os.path.join(args.base_path, args.log_dir)
        output_dir_abs = args.output_dir if os.path.isabs(args.output_dir) \
            else os.path.join(args.base_path, args.output_dir)
        for d in [log_dir_abs, output_dir_abs]:
            if os.path.exists(d):
                shutil.rmtree(d)
                print(f"  삭제: {d}")
        print("  Clean run: 기존 데이터 삭제 완료")

    # 데이터 경로
    if args.env_data and args.training_data:
        env_data_path = args.env_data if os.path.isabs(args.env_data) \
            else os.path.join(args.base_path, args.env_data)
        training_data_path = args.training_data if os.path.isabs(args.training_data) \
            else os.path.join(args.base_path, args.training_data)
    else:
        dataset_dir = os.path.join(args.base_path, 'data', args.dataset)
        env_data_path = os.path.join(dataset_dir, 'environment.csv')
        training_data_path = os.path.join(dataset_dir, 'training_scaled.csv')

    env_data_full, training_data_full = load_data(env_data_path, training_data_path)

    # 라우팅
    if args.trading_method == 'day':
        train_day(args, env_data_full, training_data_full, device)
    else:
        train_swing(args, env_data_full, training_data_full, device)

    print("\n프로그램 종료")


# ═══════════════════════════════════════════════════════
#  Day Trading — 지도학습 (일중 수익률 예측)
# ═══════════════════════════════════════════════════════

def train_day(args, env_data_full, training_data_full, device):
    """Day Trading 지도학습: (close - open) / open 회귀"""
    print("=" * 80)
    print("ETF Day Trading 지도학습 (일중 수익률 예측)")
    print("=" * 80)

    total_len = len(env_data_full)

    # Train 데이터 슬라이싱
    start = args.start_step
    end = min(start + args.max_steps, total_len) if args.max_steps > 0 else total_len

    # Validation 분리
    if args.val_start >= 0:
        # val_start 이전까지를 train으로, val_start부터를 val로
        train_end = min(end, args.val_start)
        val_start = args.val_start
        val_end = min(val_start + args.max_steps, total_len) \
            if args.max_steps > 0 else total_len
    else:
        train_end = end
        val_start = -1
        val_end = -1

    # Train
    train_env_data = env_data_full.iloc[start:train_end].reset_index(drop=True)
    train_features = training_data_full[start:train_end]
    train_opens = train_env_data['open'].values
    train_closes = train_env_data['close'].values
    train_labels = (train_closes - train_opens) / (train_opens + 1e-8)

    print(f"\n데이터셋: {args.dataset}")
    print(f"  Train: [{start}:{train_end}] ({len(train_features):,}행)")
    print(f"  Features: {train_features.shape[1]}개 (포트폴리오 피처 없음)")
    print(f"  레이블: (종가-시가)/시가 = 일중 수익률")

    # 통계
    avg_ret = float(train_labels.mean()) * 100
    std_ret = float(train_labels.std()) * 100
    win_days = int((train_labels > 0).sum())
    total_days = len(train_labels)
    bnh_return = float(np.prod(1 + train_labels) - 1) * 100
    round_trip = (args.trading_fee + args.slippage) + (args.trading_fee + args.trading_tax + args.slippage)

    print(f"  평균 일중 수익률: {avg_ret:+.4f}% (σ={std_ret:.4f}%)")
    print(f"  양의 수익률 비율: {win_days}/{total_days} ({win_days/total_days*100:.1f}%)")
    print(f"  B&H 일중 수익률: {bnh_return:+.2f}%")
    print(f"  Round-trip 비용: {round_trip*100:.4f}%")

    # Validation
    val_features = None
    val_labels_np = None
    val_env_data = None
    if val_start >= 0 and val_end - val_start >= 30:
        has_overlap = not (val_end <= start or val_start >= train_end)
        if has_overlap:
            print(f"\n  Validation 구간이 학습 구간과 겹침 → 비활성화")
        else:
            val_env_data = env_data_full.iloc[val_start:val_end].reset_index(drop=True)
            val_features = training_data_full[val_start:val_end]
            val_opens = val_env_data['open'].values
            val_closes = val_env_data['close'].values
            val_labels_np = (val_closes - val_opens) / (val_opens + 1e-8)
            print(f"  Validation: [{val_start}:{val_end}] ({len(val_features):,}행)")

    # 신경망 (회귀)
    input_dim = train_features.shape[1]
    print(f"\n신경망 생성 (회귀, 타입: {args.network_type})")
    print(f"  - 입력 차원: {input_dim}")
    model = MambaRegressionNetwork(input_dim, d_model=64, d_state=16, n_blocks=2)
    model_params = sum(p.numel() for p in model.parameters())
    print(f"  - 파라미터: {model_params:,}개")

    # 모델 로드 (추가 학습)
    lr = args.lr
    if args.update:
        if args.base_model:
            base_dir = args.base_model if os.path.isabs(args.base_model) \
                else os.path.join(args.base_path, 'models', args.base_model)
        else:
            base_dir = args.output_dir if os.path.isabs(args.output_dir) \
                else os.path.join(args.base_path, args.output_dir)

        model_path = os.path.join(base_dir, 'model_best.pt')
        if os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['state_dict'])
            lr *= args.update_lr_scale
            print(f"  모델 로드: {model_path}")
            print(f"  학습률 조정: {lr} (×{args.update_lr_scale})")
        else:
            print(f"  모델 파일 없음 ({model_path}) - 처음부터 학습")

    # 경로
    log_dir = args.log_dir if os.path.isabs(args.log_dir) \
        else os.path.join(args.base_path, args.log_dir)
    output_dir = args.output_dir if os.path.isabs(args.output_dir) \
        else os.path.join(args.base_path, args.output_dir)

    print(f"\n학습 결과 저장")
    print(f"  - 로그: {log_dir}")
    print(f"  - 출력: {output_dir}")

    # 트레이너
    trainer = SupervisedDayTrainer(
        model=model,
        train_features=train_features,
        train_labels=train_labels,
        train_env_data=train_env_data,
        val_features=val_features,
        val_labels=val_labels_np,
        val_env_data=val_env_data,
        learning_rate=lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        log_dir=log_dir,
        output_dir=output_dir,
        device=device,
        trading_fee=args.trading_fee,
        trading_tax=args.trading_tax,
        slippage=args.slippage,
    )

    print("\n" + "=" * 80)
    print("학습 시작")
    print("=" * 80)

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\n학습 중단 (사용자 인터럽트)")


# ═══════════════════════════════════════════════════════
#  Swing Trading — 강화학습 (PPO)
# ═══════════════════════════════════════════════════════

def train_swing(args, env_data_full, training_data_full, device):
    """Swing Trading 강화학습 (PPO) — 청크 기반 학습 지원"""
    print("=" * 80)
    print("ETF Swing Trading 강화학습 (PPO)")
    print("=" * 80)

    total_len = len(env_data_full)
    start = args.start_step
    end = min(start + args.max_steps, total_len) if args.max_steps > 0 else total_len
    env_data = env_data_full.iloc[start:end].reset_index(drop=True)
    training_data = training_data_full[start:end]
    if start > 0 or end < total_len:
        print(f"\n슬라이싱: [{start}:{end}] ({total_len:,} → {len(env_data):,})")

    if len(env_data) < 60:
        raise ValueError("데이터 길이가 너무 짧습니다. 최소 60 거래일 이상 필요합니다.")

    # ── ETF 경계 탐지 ──
    etf_segments = []  # [(start_idx, end_idx, etf_code)]
    if 'etf_code' in env_data.columns:
        codes = env_data['etf_code'].values
        seg_start = 0
        for i in range(1, len(codes)):
            if codes[i] != codes[i - 1]:
                etf_segments.append((seg_start, i, codes[seg_start]))
                seg_start = i
        etf_segments.append((seg_start, len(codes), codes[seg_start]))
    else:
        etf_segments.append((0, len(env_data), 'unknown'))
    num_total_etfs = len(etf_segments)

    # ── 청크 분할: ETF 1개씩 × chunk_years 년 단위 ──
    chunk_years = args.chunk_years
    if chunk_years > 0 and 'date' in env_data.columns:
        import random
        chunks = []  # [(start_idx, end_idx, etf_code, date_from, date_to)]
        dates = env_data['date'].values
        for seg_start, seg_end, etf_code in etf_segments:
            seg_dates = dates[seg_start:seg_end].astype(str)
            # 연도 경계 찾기
            first_year = int(str(seg_dates[0])[:4])
            last_year = int(str(seg_dates[-1])[:4])
            for y_start in range(first_year, last_year + 1, chunk_years):
                y_end = y_start + chunk_years
                y_start_str = f"{y_start}0101"
                y_end_str = f"{y_end}0101"
                # 이 연도 구간에 해당하는 인덱스
                mask = (seg_dates >= y_start_str) & (seg_dates < y_end_str)
                idxs = np.where(mask)[0]
                if len(idxs) < 30:  # 너무 짧은 구간 스킵
                    continue
                cs = seg_start + idxs[0]
                ce = seg_start + idxs[-1] + 1
                chunks.append((cs, ce, etf_code, str(seg_dates[idxs[0]]), str(seg_dates[idxs[-1]])))
        # 랜덤 셔플 (다양한 ETF/기간 고르게 학습)
        random.seed(42)
        random.shuffle(chunks)
        print(f"\n청크 분할: {num_total_etfs}개 ETF × {chunk_years}년 → {len(chunks)}개 청크")
        for idx, (cs, ce, ec, df, dt) in enumerate(chunks[:10]):
            print(f"  Chunk {idx+1}: {ec} [{df}~{dt}] ({ce-cs:,} days)")
        if len(chunks) > 10:
            print(f"  ... 외 {len(chunks)-10}개")
    else:
        chunks = [(0, len(env_data), 'all',
                   str(env_data['date'].iloc[0]) if 'date' in env_data.columns else '',
                   str(env_data['date'].iloc[-1]) if 'date' in env_data.columns else '')]
        print(f"\n청크 미분할: {num_total_etfs}개 ETF, {len(env_data):,} days")

    # ── Validation 환경 ──
    val_env = None
    if args.val_start >= 0:
        val_start_abs = args.val_start
        val_end = min(val_start_abs + args.max_steps, total_len) \
            if args.max_steps > 0 else total_len
        if val_end - val_start_abs >= 60:
            has_overlap = not (val_end <= start or val_start_abs >= end)
            if has_overlap:
                print(f"\n  Validation 구간 겹침 → 비활성화")
            else:
                val_env_data = env_data_full.iloc[val_start_abs:val_end].reset_index(drop=True)
                val_training_data = training_data_full[val_start_abs:val_end]
                val_env = SwingTradingEnvironment(
                    env_data=val_env_data,
                    training_data=val_training_data,
                    initial_balance=args.initial_balance,
                    trading_fee=args.trading_fee,
                    trading_tax=args.trading_tax,
                    slippage=args.slippage,
                    action_scale=args.action_scale,
                    reward_clip=args.reward_clip,
                    reward_scale=args.reward_scale,
                    fee_penalty_scale=args.fee_penalty_scale,
                    reward_terminal_scale=args.reward_terminal_scale,
                    inaction_penalty=args.inaction_penalty,
                    hold_threshold=args.hold_threshold,
                    drawdown_penalty_scale=args.drawdown_penalty_scale,
                    drawdown_penalty_threshold=args.drawdown_penalty_threshold,
                    rolling_sharpe_window=args.rolling_sharpe_window,
                    rolling_sharpe_scale=args.rolling_sharpe_scale,
                    loss_aversion=args.loss_aversion,
                )
                print(f"\n  Validation: [{val_start_abs}:{val_end}] ({val_end - val_start_abs:,}일)")

    # ── 신경망 & 에이전트 (전체 청크에서 공유) ──
    # 첫 번째 청크로 input_dim 결정
    first_chunk_env = SwingTradingEnvironment(
        env_data=env_data.iloc[chunks[0][0]:chunks[0][1]].reset_index(drop=True),
        training_data=training_data[chunks[0][0]:chunks[0][1]],
        initial_balance=args.initial_balance,
    )
    input_dim = first_chunk_env.num_features
    del first_chunk_env

    policy_net, value_net = create_networks(
        input_dim, args.network_type, device, args.min_concentration,
        d_model=args.d_model, n_blocks=args.n_blocks, d_state=args.d_state,
        policy_dropout=args.policy_dropout, value_dropout=args.value_dropout,
    )

    lr_policy = args.lr_policy * args.update_lr_scale if args.update else args.lr_policy
    lr_value = args.lr_value * args.update_lr_scale if args.update else args.lr_value

    print("\n에이전트 생성")
    agent = TradingAgent(
        policy_network=policy_net,
        value_network=value_net,
        lr_policy=lr_policy,
        lr_value=lr_value,
        gamma=args.gamma,
        epsilon=args.epsilon,
        reward_clip=args.reward_clip,
        action_mix_prob=args.action_mix_prob,
        concentration_target=args.concentration_target,
        concentration_penalty_coef=args.concentration_penalty_coef,
        policy_weight_decay=args.policy_weight_decay,
        value_weight_decay=args.value_weight_decay,
        device=device,
        use_lstm=(args.network_type == 'lstm'),
    )

    # ── 모델 로드 ──
    if args.update:
        print("\n모델 로드 시도 (추가 학습 모드)")
        if args.base_model:
            base_dir = args.base_model if os.path.isabs(args.base_model) \
                else os.path.join(args.base_path, 'models', args.base_model)
        else:
            base_dir = args.output_dir if os.path.isabs(args.output_dir) \
                else os.path.join(args.base_path, args.output_dir)

        policy_path = os.path.join(base_dir, 'policy_best.pt')
        value_path = os.path.join(base_dir, 'value_best.pt')

        if os.path.exists(policy_path) and os.path.exists(value_path):
            try:
                agent.load(policy_path, value_path)
                print(f"  정책 모델 로드: {policy_path}")
                print(f"  가치 모델 로드: {value_path}")
            except Exception as e:
                print(f"  모델 로드 실패: {e}")
        else:
            print(f"  모델 파일 없음 - 새로운 모델로 학습")

    # ── 경로 설정 ──
    iteration_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_dir = args.log_dir if os.path.isabs(args.log_dir) \
        else os.path.join(args.base_path, args.log_dir)
    output_dir = args.output_dir if os.path.isabs(args.output_dir) \
        else os.path.join(args.base_path, args.output_dir)

    print(f"\n학습 결과 저장")
    print(f"  - 로그: {log_dir}")
    print(f"  - 출력: {output_dir}")

    # ── 학습 설정 저장 ──
    os.makedirs(output_dir, exist_ok=True)
    train_config = {
        "dataset": args.dataset,
        "trading_method": "swing",
        "episodes": args.episodes,
        "lr_policy": args.lr_policy,
        "lr_value": args.lr_value,
        "chunk_years": args.chunk_years,
        "val_start": args.val_start,
        "data_length": len(env_data_full),
        "base_model": args.base_model,
        "drawdown_penalty_scale": args.drawdown_penalty_scale,
        "drawdown_penalty_threshold": args.drawdown_penalty_threshold,
        "rolling_sharpe_window": args.rolling_sharpe_window,
        "rolling_sharpe_scale": args.rolling_sharpe_scale,
        "loss_aversion": args.loss_aversion,
        "hold_threshold": args.hold_threshold,
        "min_concentration": args.min_concentration,
        "gamma": args.gamma,
        "network_type": args.network_type,
        "d_model": args.d_model,
        "n_blocks": args.n_blocks,
        "d_state": args.d_state,
        "policy_dropout": args.policy_dropout,
        "value_dropout": args.value_dropout,
        "policy_weight_decay": args.policy_weight_decay,
        "value_weight_decay": args.value_weight_decay,
        "validation_interval": args.validation_interval,
        "early_stop_patience": args.early_stop_patience,
        "early_stop_min_delta": args.early_stop_min_delta,
        "early_stop_warmup_episodes": args.early_stop_warmup_episodes,
        "trained_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    config_path = os.path.join(output_dir, "train_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(train_config, f, ensure_ascii=False, indent=2)
    print(f"  - 설정: {config_path}")

# ── 청크별 학습 루프 ──
    resume_from = max(0, args.start_chunk - 1)  # 1-based → 0-based
    if resume_from > 0:
        print(f"\n청크 {resume_from + 1}/{len(chunks)}부터 재개 (앞 {resume_from}개 청크 스킵)")

    for chunk_idx, (chunk_start, chunk_end, chunk_etf_code, chunk_date_from, chunk_date_to) in enumerate(chunks):
        if chunk_idx < resume_from:
            continue
        chunk_env_data = env_data.iloc[chunk_start:chunk_end].reset_index(drop=True)
        chunk_training_data = training_data[chunk_start:chunk_end]

        chunk_etf_name = TARGET_ETFS.get(str(chunk_etf_code).zfill(6), '')
        print(f"\n{'=' * 80}")
        print(f"청크 {chunk_idx+1}/{len(chunks)}: {chunk_etf_code} {chunk_etf_name} [{chunk_date_from}~{chunk_date_to}] ({chunk_end-chunk_start:,} days)")
        print(f"{'=' * 80}")

        chunk_env = SwingTradingEnvironment(
            env_data=chunk_env_data,
            training_data=chunk_training_data,
            initial_balance=args.initial_balance,
            trading_fee=args.trading_fee,
            trading_tax=args.trading_tax,
            slippage=args.slippage,
            action_scale=args.action_scale,
            reward_clip=args.reward_clip,
            reward_scale=args.reward_scale,
            fee_penalty_scale=args.fee_penalty_scale,
            reward_terminal_scale=args.reward_terminal_scale,
            inaction_penalty=args.inaction_penalty,
            hold_threshold=args.hold_threshold,
            drawdown_penalty_scale=args.drawdown_penalty_scale,
            drawdown_penalty_threshold=args.drawdown_penalty_threshold,
            rolling_sharpe_window=args.rolling_sharpe_window,
            rolling_sharpe_scale=args.rolling_sharpe_scale,
            loss_aversion=args.loss_aversion,
        )

        # 통계
        closes = chunk_env.env_data['close'].values
        daily_rets = np.diff(closes) / (closes[:-1] + 1e-8)
        bnh_return = float((closes[-1] / closes[0] - 1) * 100)
        avg_ret = float(daily_rets.mean()) * 100
        win_days = int((daily_rets > 0).sum())
        total_days = len(closes)

        print(f"  - 거래일 수: {chunk_env.total_ticks:,}일, B&H: {bnh_return:+.2f}%")

        chunk_info = {
            'iteration_name': iteration_name,
            'start_step': int(start + chunk_start),
            'end_step': int(start + chunk_end),
            'chunk_size': int(chunk_end - chunk_start),
            'chunk_idx': int(chunk_idx),
            'total_chunks': int(len(chunks)),
            'etf_code': str(chunk_etf_code),
            'etf_name': str(chunk_etf_name),
            'start_date': str(chunk_date_from),
            'end_date': str(chunk_date_to),
            'dataset': str(args.dataset),
        }

        trainer = PPOTrainer(
            env=chunk_env,
            agent=agent,
            num_episodes=args.episodes,
            update_interval=args.update_interval,
            log_dir=log_dir,
            output_dir=output_dir,
            visualize=args.visualize,
            viz_interval=args.viz_interval,
            entropy_coef_start=args.entropy_coef_start,
            entropy_coef_end=args.entropy_coef_end,
            entropy_decay_episodes=args.entropy_decay_episodes,
            target_bias_low=args.target_bias_low,
            target_bias_high=args.target_bias_high,
            trade_rate_threshold=args.trade_rate_threshold,
            entropy_boost_factor=args.entropy_boost_factor,
            low_policy_std_threshold=args.low_policy_std_threshold,
            action_mix_start=args.action_mix_start,
            action_mix_end=args.action_mix_end,
            val_min_trades=args.val_min_trades,
            validation_interval=args.validation_interval,
            early_stop_patience=args.early_stop_patience,
            early_stop_min_delta=args.early_stop_min_delta,
            early_stop_warmup_episodes=args.early_stop_warmup_episodes,
            val_env=val_env,
            chunk_info=chunk_info,
        )

        try:
            trainer.train()
        except KeyboardInterrupt:
            print(f"\n\n청크 {chunk_idx+1} 학습 중단 (사용자 인터럽트)")
            break

        print(f"\n청크 {chunk_idx+1}/{len(chunks)} 학습 완료")


if __name__ == '__main__':
    main()
