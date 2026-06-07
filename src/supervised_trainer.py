"""
지도학습 트레이너: ETF Day Trading 일중 수익률 예측

시초매수→마감매도 전략을 위한 지도학습:
  - 레이블: (종가 - 시가) / 시가 = 일중 수익률
  - 입력: 정규화된 시장 피처 (포트폴리오 피처 불필요 — 매일 독립)
  - 손실: Huber Loss (이상치에 강건)
  - 검증: 시뮬레이션 수익률로 Best 모델 선정
"""
import os
import json
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class SupervisedDayTrainer:
    """지도학습 Day Trading 트레이너

    매일 독립적인 시초매수→마감매도이므로 RL이 불필요.
    (close - open) / open 을 직접 회귀하고,
    예측값이 거래비용 초과 시 매수하는 전략.
    """

    def __init__(
        self,
        model: nn.Module,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        train_env_data: pd.DataFrame,
        val_features: np.ndarray = None,
        val_labels: np.ndarray = None,
        val_env_data: pd.DataFrame = None,
        learning_rate: float = 0.001,
        batch_size: int = 256,
        num_epochs: int = 100,
        log_dir: str = 'logs',
        output_dir: str = 'output',
        device: str = 'cpu',
        trading_fee: float = 0.00015,
        trading_tax: float = 0.0,
        slippage: float = 0.0003,
    ):
        self.model = model.to(device)
        self.device = device
        self.num_epochs = num_epochs
        self.log_dir = log_dir
        self.output_dir = output_dir
        self.trading_fee = trading_fee
        self.trading_tax = trading_tax
        self.slippage = slippage

        # Round-trip cost: 매수 (fee+slippage) + 매도 (fee+tax+slippage)
        self.round_trip_cost = (trading_fee + slippage) + (trading_fee + trading_tax + slippage)

        # Train DataLoader (셔플 — 매일 독립 IID)
        self.train_dataset = TensorDataset(
            torch.FloatTensor(train_features),
            torch.FloatTensor(train_labels),
        )
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True,
        )
        self.train_labels_np = train_labels
        self.train_env_data = train_env_data

        # Validation
        if val_features is not None and val_labels is not None:
            self.val_features = torch.FloatTensor(val_features).to(device)
            self.val_labels_np = val_labels
            self.val_labels = torch.FloatTensor(val_labels).to(device)
            self.val_env_data = val_env_data
        else:
            self.val_features = None
            self.val_labels = None
            self.val_labels_np = None
            self.val_env_data = None

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=1e-4,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs,
        )
        self.criterion = nn.HuberLoss(delta=0.005)

        # Best tracking
        self.best_val_profit = -float('inf')
        self.best_train_profit = -float('inf')
        self.best_epoch = 0

        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        self._setup_logger()

    def _setup_logger(self):
        log_file = os.path.join(self.log_dir, 'epochs.jsonl')
        self.logger = logging.getLogger(f'SupervisedTrainer_{id(self)}')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []

        fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        fh.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(fh)

        meta = {
            'type': 'meta',
            'trading_method': 'day',
            'num_epochs': self.num_epochs,
            'batch_size': self.train_loader.batch_size,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'train_samples': len(self.train_dataset),
            'val_samples': len(self.val_labels) if self.val_labels is not None else 0,
            'round_trip_cost': round(self.round_trip_cost, 6),
        }
        self.logger.info(json.dumps(meta))

    def _log_json(self, data: dict):
        data['timestamp'] = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        self.logger.info(json.dumps(data, ensure_ascii=False))

    def train(self):
        pbar = tqdm(range(1, self.num_epochs + 1), desc="Training", ncols=130)

        for epoch in pbar:
            # ── Training ──
            self.model.train()
            train_loss = 0.0
            n_batches = 0

            for features, labels in self.train_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                predictions = self.model(features)
                loss = self.criterion(predictions, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                train_loss += loss.item()
                n_batches += 1

            train_loss /= n_batches
            self.scheduler.step()
            lr = self.optimizer.param_groups[0]['lr']

            # Train 시뮬레이션
            self.model.eval()
            with torch.no_grad():
                train_all = torch.FloatTensor(
                    self.train_dataset.tensors[0]
                ).to(self.device)
                train_pred = self.model(train_all).cpu().numpy()
            train_sim = self._simulate_trading(train_pred, self.train_labels_np)

            # ── Validation ──
            val_loss = None
            val_sim = None
            is_best = False

            if self.val_features is not None:
                with torch.no_grad():
                    val_pred = self.model(self.val_features)
                    val_loss = self.criterion(val_pred, self.val_labels).item()
                    val_pred_np = val_pred.cpu().numpy()

                val_sim = self._simulate_trading(val_pred_np, self.val_labels_np)

                if val_sim['profit_rate'] > self.best_val_profit:
                    self.best_val_profit = val_sim['profit_rate']
                    self.best_epoch = epoch
                    is_best = True
                    self._save_model('best')
            else:
                # No validation → use train profit
                if train_sim['profit_rate'] > self.best_train_profit:
                    self.best_train_profit = train_sim['profit_rate']
                    self.best_epoch = epoch
                    is_best = True
                    self._save_model('best')

            self._save_model('final')

            # ── Progress ──
            desc_parts = [f"Loss={train_loss:.6f}"]
            desc_parts.append(
                f"Prf={train_sim['profit_rate']:+.2f}% "
                f"WR={train_sim['win_rate']:.0f}% "
                f"T={train_sim['num_trades']}"
            )
            if val_sim is not None:
                desc_parts.append(
                    f"V={val_sim['profit_rate']:+.2f}%"
                )
            if is_best:
                desc_parts.append("*")
            pbar.set_postfix_str(" ".join(desc_parts))

            # ── Log ──
            log_data = {
                'type': 'epoch',
                'epoch': epoch,
                'train_loss': round(train_loss, 8),
                'lr': round(lr, 8),
                'is_best': is_best,
                'train': {
                    'profit_rate': round(train_sim['profit_rate'], 4),
                    'num_trades': train_sim['num_trades'],
                    'trade_ratio': train_sim['trade_ratio'],
                    'win_rate': round(train_sim['win_rate'], 2),
                    'avg_return': round(train_sim['avg_return'], 6),
                    'sharpe': round(train_sim['sharpe'], 4),
                },
            }
            if val_loss is not None:
                log_data['val_loss'] = round(val_loss, 8)
            if val_sim is not None:
                log_data['val'] = {
                    'profit_rate': round(val_sim['profit_rate'], 4),
                    'num_trades': val_sim['num_trades'],
                    'trade_ratio': val_sim['trade_ratio'],
                    'win_rate': round(val_sim['win_rate'], 2),
                    'avg_return': round(val_sim['avg_return'], 6),
                    'sharpe': round(val_sim['sharpe'], 4),
                }
            self._log_json(log_data)

        pbar.close()

        best_metric = self.best_val_profit if self.val_features is not None \
            else self.best_train_profit
        print(f"\nBest epoch: {self.best_epoch} (profit: {best_metric:+.2f}%)")

    def _simulate_trading(self, predictions: np.ndarray, labels: np.ndarray) -> dict:
        """예측 기반 Day Trading 시뮬레이션

        전략: predicted_return > round_trip_cost 이면 전액 매수
        """
        cost = self.round_trip_cost
        trades = 0
        wins = 0
        returns_list = []

        for pred, actual_return in zip(predictions, labels):
            if pred > cost:
                net_return = actual_return - cost
                trades += 1
                returns_list.append(net_return)
                if net_return > 0:
                    wins += 1

        # 거래당 평균 수익률 × 거래 비율로 일평균 수익률 산출
        avg_return = float(np.mean(returns_list)) if returns_list else 0.0
        std_return = float(np.std(returns_list)) if len(returns_list) > 1 else 1.0
        win_rate = (wins / trades * 100) if trades > 0 else 0.0
        total_days = len(predictions)
        trade_ratio = trades / total_days if total_days > 0 else 0.0
        # 일평균 수익률 = 거래당 평균수익률 × 거래비율, 연환산 수익률
        daily_return = avg_return * trade_ratio
        profit_rate = daily_return * 252 * 100

        sharpe = (avg_return / std_return * np.sqrt(252)) if std_return > 1e-8 else 0.0

        return {
            'profit_rate': float(profit_rate),
            'num_trades': int(trades),
            'trade_ratio': round(float(trade_ratio), 4),
            'win_rate': float(win_rate),
            'avg_return': float(avg_return),
            'sharpe': float(sharpe),
        }

    def _save_model(self, name: str):
        path = os.path.join(self.output_dir, f'model_{name}.pt')
        # input_dim 추출
        first_param = None
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                first_param = module
                break
        input_dim = first_param.in_features if first_param else 0

        torch.save({
            'state_dict': self.model.state_dict(),
            'input_dim': input_dim,
            'model_type': 'regression',
        }, path)
