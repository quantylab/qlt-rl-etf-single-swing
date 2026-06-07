"""
백테스팅 모듈: ETF Trading 백테스트 (Day / Swing)

사용법:
    python src/backtest.py --dataset etf_20260317
    python src/backtest.py --dataset etf_20260317 --sequential --chunk-size 120
    python src/backtest.py --model etf-swing-v1 --sequential --chunk-size 21600
"""
import os
import sys
import argparse
import csv
import json
import math
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from tabulate import tabulate

from environment import DayTradingEnvironment, SwingTradingEnvironment
from agent import TradingAgent
from network import (
    LSTMContinuousPolicyNetwork,
    LSTMValueNetwork,
    ContinuousPolicyNetwork,
    ValueNetwork,
    MambaPolicyNetwork,
    MambaValueNetwork,
)
from visualizer import TradingVisualizer
from target_etfs import TARGET_ETFS


def is_cross_sectional_dataset(env_data: pd.DataFrame) -> bool:
    """일자별 복수 ETF가 있는 단면 데이터셋인지 판별"""
    if 'etf_code' not in env_data.columns or 'date' not in env_data.columns:
        return False
    return env_data['date'].nunique() < len(env_data)


def _resolve_model_dir(base_path: str, model_arg: str) -> str:
    if not model_arg:
        return os.path.join(base_path, 'models')
    if os.path.isabs(model_arg):
        return model_arg
    if os.path.isdir(model_arg):
        return os.path.abspath(model_arg)
    return os.path.join(base_path, 'models', model_arg)


def _build_extra_features(
    portfolio_value: float,
    initial_balance: float,
    peak_portfolio_value: float,
    daily_returns: list,
    consecutive_wins: int,
    consecutive_losses: int,
    current_ratio: float,
) -> np.ndarray:
    """환경 포트폴리오 보조 피처(최대 6개) 생성"""
    cumulative_return = (portfolio_value - initial_balance) / (initial_balance + 1e-8)
    cumulative_return_scaled = np.clip(cumulative_return * 20.0, -3.0, 3.0)

    drawdown = (portfolio_value - peak_portfolio_value) / (peak_portfolio_value + 1e-8)
    drawdown_scaled = np.clip(drawdown * 50.0, -3.0, 0.0)

    if len(daily_returns) >= 5:
        recent_wins = sum(1 for r in daily_returns[-5:] if r > 0) / 5.0
    elif len(daily_returns) > 0:
        recent_wins = sum(1 for r in daily_returns if r > 0) / len(daily_returns)
    else:
        recent_wins = 0.5
    win_rate_scaled = (recent_wins - 0.5) * 4.0

    streak = consecutive_wins - consecutive_losses
    streak_scaled = np.clip(streak / 5.0, -2.0, 2.0)

    if len(daily_returns) >= 5:
        vol = np.std(daily_returns[-20:]) if len(daily_returns) >= 20 else np.std(daily_returns[-5:])
    else:
        vol = 0.01
    vol_scaled = np.clip(vol * 100.0, 0.0, 3.0)

    pos_scaled = np.clip(current_ratio, 0.0, 1.0) * 2.0 - 1.0
    return np.array([
        cumulative_return_scaled,
        drawdown_scaled,
        win_rate_scaled,
        streak_scaled,
        vol_scaled,
        pos_scaled,
    ], dtype=np.float32)


def _align_state_dim(features: np.ndarray, model_input_dim: int, extra_features: np.ndarray) -> np.ndarray:
    """모델 입력 차원에 맞춰 상태 벡터 정렬"""
    if model_input_dim is None:
        return np.nan_to_num(features.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    base_dim = features.shape[0]
    if model_input_dim <= base_dim:
        state = features[:model_input_dim]
    else:
        need = model_input_dim - base_dim
        append = extra_features[:need]
        if append.shape[0] < need:
            append = np.concatenate([append, np.zeros(need - append.shape[0], dtype=np.float32)])
        state = np.concatenate([features, append])
    return np.nan_to_num(state.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def run_selector_backtest(
    env_data: pd.DataFrame,
    training_data: np.ndarray,
    agent: TradingAgent,
    trading_method: str,
    initial_balance: float,
    trading_fee: float,
    trading_tax: float,
    slippage: float,
    hold_threshold: float,
    model_input_dim: int,
    min_value=None,
    min_concentration=None,
    max_buy_per_day: int = 5,
    max_holdings: int = 10,
    stop_loss_pct: float = 0.0,
    drawdown_reduce_pct: float = 0.0,
    drawdown_pause_pct: float = 0.0,
    sell_threshold: float = None,
    trailing_stop_pct: float = 0.0,
    max_exposure: float = 1.0,
    vol_target: float = 0.0,
    min_hold_days: int = 0,
    output_dir: str = None,
):
    """멀티 ETF 종목 선택/매도 의사결정 백테스트
    
    Args:
        hold_threshold: 거래 최소 기준값 (절댓값, e.g., 0.05 = 5% 이상 차이때만 거래)
        output_dir: 지정 시 trade_log.csv, backtest_history.csv를 즉시 스트리밍 기록
    """
    df = env_data.copy().reset_index(drop=True)
    if 'etf_code' not in df.columns or 'date' not in df.columns:
        raise ValueError("selector 백테스트는 env_data에 date/etf_code 컬럼이 필요합니다.")

    df['etf_code'] = df['etf_code'].astype(str).str.zfill(6)
    df['_idx'] = np.arange(len(df))
    grouped = df.groupby('date', sort=True)
    if len(grouped) < 2:
        raise ValueError("selector 백테스트를 위한 거래일이 부족합니다.")

    cash = float(initial_balance)
    holdings = {}  # {etf_code: shares} 다중 보유
    avg_buy_cost = {}  # {etf_code: avg_price} 종목별 평균 매수가
    peak_price = {}  # {etf_code: max_price_since_buy} trailing stop용
    hold_since = {}  # {etf_code: tick_idx} 매수 시점 (최소 보유기간 체크)
    prev_close_price = {}  # ETF별 전일 종가 추적
    portfolio_value = float(initial_balance)
    peak_portfolio_value = float(initial_balance)

    total_trade_amount = 0.0
    total_fee_paid = 0.0
    num_buy = 0
    num_sell = 0
    num_hold = 0

    consecutive_wins = 0
    consecutive_losses = 0
    daily_returns = []
    pv_history = [portfolio_value]
    history = []
    trade_log = []  # 개별 매매 로그
    policy_outputs = []
    values = []

    # 스트리밍 CSV 기록
    _trade_log_file = None
    _trade_log_writer = None
    _history_file = None
    _history_writer = None
    _trade_log_fields = [
        'date', 'action', 'code', 'name', 'shares', 'price',
        'amount', 'fee', 'balance', 'portfolio_value', 'sell_return_pct',
    ]
    _history_fields = [
        'date', 'tick', 'selected_etf', 'held_etf', 'num_holdings',
        'position_size', 'position_ratio', 'num_shares', 'traded',
        'open_price', 'close_price', 'intraday_return', 'fee_paid',
        'balance', 'portfolio_value', 'prev_portfolio_value', 'reward',
    ]
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        _trade_log_file = open(os.path.join(output_dir, 'trade_log.csv'), 'w', newline='')
        _trade_log_writer = csv.DictWriter(_trade_log_file, fieldnames=_trade_log_fields)
        _trade_log_writer.writeheader()
        _history_file = open(os.path.join(output_dir, 'backtest_history.csv'), 'w', newline='')
        _history_writer = csv.DictWriter(_history_file, fieldnames=_history_fields)
        _history_writer.writeheader()

    def _flush_trade(entry):
        trade_log.append(entry)
        if _trade_log_writer:
            _trade_log_writer.writerow(entry)
            _trade_log_file.flush()

    def _flush_history(entry):
        history.append(entry)
        if _history_writer:
            _history_writer.writerow(entry)
            _history_file.flush()

    all_dates = sorted(grouped.groups.keys())
    _rotation_cooldown = 3  # 거래일 간격으로 rotation 허용
    _last_rotation_tick = -_rotation_cooldown  # 첫 날부터 허용

    for tick_idx, date in enumerate(all_dates):
        day_rows = grouped.get_group(date)
        open_map = {r['etf_code']: float(r['open']) for _, r in day_rows.iterrows()}
        close_map = {r['etf_code']: float(r['close']) for _, r in day_rows.iterrows()}

        # 포트폴리오 가치 계산 (전일 종가 기준)
        current_stock_value = 0.0
        for h_code, h_shares in holdings.items():
            if h_code in prev_close_price:
                current_stock_value += h_shares * prev_close_price[h_code]
        pv_open = cash + current_stock_value
        if pv_open <= 0:
            pv_open = 1e-8

        # Trailing stop: 보유 종목 고점 갱신
        for h_code in list(holdings.keys()):
            if h_code in open_map:
                peak_price[h_code] = max(peak_price.get(h_code, 0), open_map[h_code])

        current_ratio = current_stock_value / pv_open if pv_open > 0 else 0.0
        extra = _build_extra_features(
            portfolio_value=portfolio_value,
            initial_balance=initial_balance,
            peak_portfolio_value=peak_portfolio_value,
            daily_returns=daily_returns,
            consecutive_wins=consecutive_wins,
            consecutive_losses=consecutive_losses,
            current_ratio=current_ratio,
        )

        candidates = []
        for _, row in day_rows.iterrows():
            idx = int(row['_idx'])
            state = _align_state_dim(training_data[idx], model_input_dim, extra)
            action, _, policy_output = agent.get_action(state, training=False)
            value = agent.get_value(state)
            concentration = float(policy_output[0] + policy_output[1])

            if min_value is not None and value < min_value:
                continue
            if min_concentration is not None and concentration < min_concentration:
                continue

            candidates.append({
                'code': row['etf_code'],
                'open': float(row['open']),
                'close': float(row['close']),
                'target_ratio': float(np.clip(action, 0.0, 1.0)),
                'value': float(value),
                'concentration': concentration,
                'policy_output': policy_output,
            })

        candidates.sort(key=lambda x: (x['target_ratio'], x['value']), reverse=True)
        selected = candidates[0] if candidates else None

        # 하이브리드 방어 모드: 낙폭 트리거 + 모멘텀 해제
        current_dd = (portfolio_value - peak_portfolio_value) / (peak_portfolio_value + 1e-8)
        _defense_threshold = -drawdown_pause_pct if drawdown_pause_pct > 0 else -0.30
        in_severe_drawdown = current_dd < _defense_threshold if drawdown_pause_pct > 0 else False
        recovering = False
        if in_severe_drawdown and len(daily_returns) >= 7:
            recent_7d = sum(daily_returns[-7:])
            recovering = recent_7d > 0.015
        # 추가: 보유 종목 수가 이미 3개 이하이면 추가 청산 불필요
        defense_active = in_severe_drawdown and not recovering

        traded = False
        fee_paid = 0.0
        trade_amount = 0.0
        selected_code = selected['code'] if selected else ''
        target_ratio = selected['target_ratio'] if selected else 0.0

        # Day: 당일 종목 선택 후 시가매수/종가청산
        if trading_method == 'day':
            traded = False
            intraday_return = 0.0
            
            if selected:
                # target_ratio가 hold_threshold보다 크면 매수 신호
                if target_ratio > hold_threshold:
                    # 시가에 slippage 적용한 매수가
                    buy_price = selected['open'] * (1 + slippage)
                    invest_amount = cash * target_ratio
                    
                    if invest_amount > 0 and invest_amount <= cash:
                        # 수수료 계산: 투자액 = 실제매수금액 + 수수료
                        buy_fee_amount = invest_amount * trading_fee / (1 + trading_fee)
                        net_invest = invest_amount - buy_fee_amount
                        
                        if net_invest > 0:
                            buy_shares = int(net_invest / buy_price)
                            if buy_shares > 0:
                                # 정수 주 기준으로 실제 투자금 재계산
                                actual_buy_amount = buy_shares * buy_price
                                invest_amount = actual_buy_amount + actual_buy_amount * trading_fee
                                buy_fee_amount = invest_amount - actual_buy_amount
                                
                                # 종가에 slippage 적용한 매도가
                                sell_price = selected['close'] * (1 - slippage)
                                sell_revenue = buy_shares * sell_price
                                
                                # 매도 수수료 및 세금
                                sell_fee = sell_revenue * trading_fee
                                sell_tax = sell_revenue * trading_tax
                                net_revenue = sell_revenue - sell_fee - sell_tax
                                
                                cash = cash - invest_amount + net_revenue
                                fee_paid = buy_fee_amount + sell_fee + sell_tax
                                trade_amount = invest_amount + sell_revenue
                                total_fee_paid += fee_paid
                                total_trade_amount += trade_amount
                                num_buy += 1
                                num_sell += 1
                                traded = True
                                intraday_return = (selected['close'] - selected['open']) / (selected['open'] + 1e-8)
                                sell_return = (sell_price - buy_price) / (buy_price + 1e-8) * 100
                                _flush_trade({
                                    'date': date, 'action': 'BUY', 'code': selected['code'],
                                    'name': TARGET_ETFS.get(selected['code'], ''),
                                    'shares': buy_shares, 'price': buy_price,
                                    'amount': invest_amount, 'fee': buy_fee_amount,
                                    'balance': cash + sell_revenue - sell_fee - sell_tax,
                                    'portfolio_value': cash,
                                    'sell_return_pct': 0.0,
                                })
                                _flush_trade({
                                    'date': date, 'action': 'SELL', 'code': selected['code'],
                                    'name': TARGET_ETFS.get(selected['code'], ''),
                                    'shares': buy_shares, 'price': sell_price,
                                    'amount': sell_revenue, 'fee': sell_fee + sell_tax,
                                    'balance': cash,
                                    'portfolio_value': cash,
                                    'sell_return_pct': sell_return,
                                })
            
            if not traded:
                num_hold += 1

            holdings = {}
            portfolio_value = cash
            position_ratio_close = 0.0
            if not intraday_return:
                intraday_return = (selected['close'] - selected['open']) / (selected['open'] + 1e-8) if selected else 0.0

        # Swing: 다중 ETF 보유 (자체 신호 기반 매도 + 현금 여력시 신규 매수)
        else:
            candidate_map = {c['code']: c for c in candidates}
            sold_today = set()  # 당일 매도한 종목 (whipsaw 방지)

            # 1) 보유 종목별 매도 결정 (히스테리시스: 매도 기준은 매수 기준보다 낮게)
            _sell_th = sell_threshold if sell_threshold is not None else hold_threshold * 0.7

            # 방어 모드: 최약 보유 종목 매도하여 현금 확보
            if defense_active and len(holdings) > 3:
                held_signals = []
                for code in list(holdings.keys()):
                    cand = candidate_map.get(code)
                    sig = cand['target_ratio'] if cand else 0.0
                    held_signals.append((code, sig))
                held_signals.sort(key=lambda x: x[1])  # 약한 순
                # 하루 최대 1종목씩 점진적으로 줄임
                weak_code, _ = held_signals[0]
                if weak_code in open_map and weak_code not in sold_today:
                    shares = holdings[weak_code]
                    if shares > 0:
                        sell_price = open_map[weak_code] * (1 - slippage)
                        sell_revenue = shares * sell_price
                        sell_fee_val = sell_revenue * trading_fee
                        sell_tax_val = sell_revenue * trading_tax
                        net_revenue = sell_revenue - sell_fee_val - sell_tax_val
                        buy_cost = avg_buy_cost.get(weak_code, sell_price)
                        sell_ret = (sell_price - buy_cost) / (buy_cost + 1e-8) * 100
                        cash += net_revenue
                        fee_paid += sell_fee_val + sell_tax_val
                        trade_amount += sell_revenue
                        total_fee_paid += sell_fee_val + sell_tax_val
                        total_trade_amount += sell_revenue
                        num_sell += 1
                        traded = True
                        _flush_trade({
                            'date': date, 'action': 'SELL', 'code': weak_code,
                            'name': TARGET_ETFS.get(weak_code, ''),
                            'shares': shares, 'price': sell_price,
                            'amount': sell_revenue, 'fee': sell_fee_val + sell_tax_val,
                            'balance': cash,
                            'portfolio_value': portfolio_value,
                            'sell_return_pct': sell_ret,
                        })
                        del holdings[weak_code]
                        avg_buy_cost.pop(weak_code, None)
                        peak_price.pop(weak_code, None)
                        hold_since.pop(weak_code, None)
                        sold_today.add(weak_code)
            sold_codes = []
            for code in list(holdings.keys()):
                if code not in open_map:
                    continue  # 당일 데이터 없음 → 유지
                open_price = open_map[code]
                held_cand = candidate_map.get(code)
                avg_cost = avg_buy_cost.get(code, open_price)
                stop_loss_triggered = stop_loss_pct > 0 and open_price < avg_cost * (1 - stop_loss_pct)
                trailing_triggered = trailing_stop_pct > 0 and open_price < peak_price.get(code, open_price) * (1 - trailing_stop_pct)
                force_sell = stop_loss_triggered or trailing_triggered
                signal_weak = held_cand is None or held_cand['target_ratio'] <= _sell_th
                min_hold_ok = min_hold_days <= 0 or (tick_idx - hold_since.get(code, 0)) >= min_hold_days
                if force_sell or (min_hold_ok and signal_weak):
                    # 신호 약화 또는 손절 → 전량 매도
                    shares = holdings[code]
                    if open_price > 0 and shares > 0:
                        sell_price = open_price * (1 - slippage)
                        sell_revenue = shares * sell_price
                        sell_fee = sell_revenue * trading_fee
                        sell_tax = sell_revenue * trading_tax
                        net_revenue = sell_revenue - sell_fee - sell_tax
                        buy_cost = avg_buy_cost.get(code, sell_price)
                        sell_ret = (sell_price - buy_cost) / (buy_cost + 1e-8) * 100
                        cash += net_revenue
                        fee_paid += sell_fee + sell_tax
                        trade_amount += sell_revenue
                        total_fee_paid += sell_fee + sell_tax
                        total_trade_amount += sell_revenue
                        num_sell += 1
                        traded = True
                        _flush_trade({
                            'date': date, 'action': 'SELL', 'code': code,
                            'name': TARGET_ETFS.get(code, ''),
                            'shares': shares, 'price': sell_price,
                            'amount': sell_revenue, 'fee': sell_fee + sell_tax,
                            'balance': cash,
                            'portfolio_value': portfolio_value,
                            'sell_return_pct': sell_ret,
                        })
                    sold_codes.append(code)
                    sold_today.add(code)
            for code in sold_codes:
                del holdings[code]
                avg_buy_cost.pop(code, None)
                peak_price.pop(code, None)
                hold_since.pop(code, None)

            # 2) 보유 종목 리밸런싱 (집중 리스크 관리: 개별 종목 비중 상한 초과 시 트리밍)
            if holdings and len(holdings) >= 2:
                pv_rebal = cash + sum(
                    holdings[c] * open_map.get(c, prev_close_price.get(c, 0))
                    for c in holdings
                )
                if pv_rebal > 0:
                    # 동적 상한: 종목수 기반 + 여유 (예: 5종목→30%, 3종목→50%)
                    max_single_weight = min(1.0 / max(len(holdings), 1) + 0.10, 0.50)

                    # Pass 1: 과대 비중 종목 트리밍 (max_single_weight 초과분 매도)
                    for code in list(holdings.keys()):
                        if code not in open_map or open_map[code] <= 0:
                            continue
                        current_weight = holdings[code] * open_map[code] / pv_rebal
                        if current_weight > max_single_weight:
                            open_price = open_map[code]
                            excess = (current_weight - max_single_weight) * pv_rebal
                            sell_shares = int(excess / (open_price * (1 - slippage) + 1e-8))
                            sell_shares = min(sell_shares, holdings[code])
                            if sell_shares > 0:
                                sell_price = open_price * (1 - slippage)
                                sell_revenue = sell_shares * sell_price
                                sell_fee_amt = sell_revenue * trading_fee
                                sell_tax_amt = sell_revenue * trading_tax
                                net_revenue = sell_revenue - sell_fee_amt - sell_tax_amt
                                cash += net_revenue
                                holdings[code] -= sell_shares
                                fee_paid += sell_fee_amt + sell_tax_amt
                                trade_amount += sell_revenue
                                total_fee_paid += sell_fee_amt + sell_tax_amt
                                total_trade_amount += sell_revenue
                                num_sell += 1
                                traded = True
                                buy_cost = avg_buy_cost.get(code, sell_price)
                                sell_ret = (sell_price - buy_cost) / (buy_cost + 1e-8) * 100
                                _flush_trade({
                                    'date': date, 'action': 'SELL', 'code': code,
                                    'name': TARGET_ETFS.get(code, ''),
                                    'shares': sell_shares, 'price': sell_price,
                                    'amount': sell_revenue, 'fee': sell_fee_amt + sell_tax_amt,
                                    'balance': cash,
                                    'portfolio_value': portfolio_value,
                                    'sell_return_pct': sell_ret,
                                })
                                if holdings[code] <= 0:
                                    del holdings[code]
                                    avg_buy_cost.pop(code, None)
                                    peak_price.pop(code, None)
                                    hold_since.pop(code, None)

                    # Pass 2: 트리밍 현금을 기존 저비중 보유종목에 재배분 (최소 거래금액 조건)
                    _min_rebal_amount = pv_rebal * 0.02  # 포트폴리오의 2% 이상만 거래
                    if cash > pv_rebal * 0.05 and holdings:
                        equal_weight = 1.0 / max(len(holdings), 1)
                        underweight = []
                        for code in list(holdings.keys()):
                            if code not in open_map or open_map[code] <= 0:
                                continue
                            w = holdings[code] * open_map[code] / pv_rebal
                            if w < equal_weight * 0.8:  # 동일비중 대비 80% 미만만 재분배
                                underweight.append((code, w))
                        if underweight:
                            underweight.sort(key=lambda x: x[1])  # 비중 낮은 순
                            reinvest_cash = cash * 0.5
                            per_etf = reinvest_cash / len(underweight)
                            for code, _ in underweight:
                                buy_price = open_map[code] * (1 + slippage)
                                buy_budget = min(per_etf, cash)
                                if buy_budget < _min_rebal_amount:
                                    continue  # 최소 거래금액 미만 → 스킵
                                buy_fee = buy_budget * trading_fee
                                net_buy = max(buy_budget - buy_fee, 0.0)
                                buy_shares = int(net_buy / (buy_price + 1e-8))
                                if buy_shares > 0:
                                    prev_shares = holdings.get(code, 0)
                                    prev_cost = avg_buy_cost.get(code, 0.0)
                                    actual_buy_amount = buy_shares * buy_price
                                    buy_budget = actual_buy_amount + actual_buy_amount * trading_fee
                                    buy_fee = buy_budget - actual_buy_amount
                                    total_cost = prev_cost * prev_shares + buy_budget
                                    holdings[code] = prev_shares + buy_shares
                                    avg_buy_cost[code] = total_cost / (holdings[code] + 1e-8)
                                    cash -= buy_budget
                                    fee_paid += buy_fee
                                    trade_amount += buy_budget
                                    total_fee_paid += buy_fee
                                    total_trade_amount += buy_budget
                                    num_buy += 1
                                    traded = True
                                    _flush_trade({
                                        'date': date, 'action': 'BUY', 'code': code,
                                        'name': TARGET_ETFS.get(code, ''),
                                        'shares': buy_shares, 'price': buy_price,
                                        'amount': buy_budget, 'fee': buy_fee,
                                        'balance': cash,
                                        'portfolio_value': portfolio_value,
                                        'sell_return_pct': 0.0,
                                    })

            # 3) 종목 교체: 약한 보유 종목 → 강한 미보유 후보 (모델 신호 기반 rotation)
            if holdings:
                held_signals = []
                for code in list(holdings.keys()):
                    cand = candidate_map.get(code)
                    ratio = cand['target_ratio'] if cand else 0.0
                    held_signals.append((code, ratio))
                held_signals.sort(key=lambda x: x[1])  # 약한 순

                non_held_strong = [c for c in candidates
                                   if c['code'] not in holdings
                                   and c['code'] not in sold_today
                                   and c['target_ratio'] > hold_threshold
                                   and c['code'] in open_map]

                rotation_gap = hold_threshold * 2  # 교체 최소 gap (10%)
                max_rotations = 1  # 일일 최대 교체 수 (쿨다운으로 주간 제한)
                rotated = 0

                for weak_code, weak_ratio in held_signals:
                    if rotated >= max_rotations or not non_held_strong:
                        break
                    if tick_idx - _last_rotation_tick < _rotation_cooldown:
                        break  # 쿨다운 기간
                    if min_hold_days > 0 and (tick_idx - hold_since.get(weak_code, 0)) < min_hold_days:
                        continue  # 최소 보유기간 미충족
                    best = non_held_strong[0]
                    if best['target_ratio'] - weak_ratio < rotation_gap:
                        break

                    # 약한 종목 매도
                    shares = holdings[weak_code]
                    if weak_code not in open_map or shares <= 0:
                        continue
                    sell_price = open_map[weak_code] * (1 - slippage)
                    sell_revenue = shares * sell_price
                    sell_fee_val = sell_revenue * trading_fee
                    sell_tax_val = sell_revenue * trading_tax
                    net_revenue = sell_revenue - sell_fee_val - sell_tax_val
                    rot_buy_cost = avg_buy_cost.get(weak_code, sell_price)
                    sell_ret = (sell_price - rot_buy_cost) / (rot_buy_cost + 1e-8) * 100
                    cash += net_revenue
                    fee_paid += sell_fee_val + sell_tax_val
                    trade_amount += sell_revenue
                    total_fee_paid += sell_fee_val + sell_tax_val
                    total_trade_amount += sell_revenue
                    num_sell += 1
                    traded = True
                    _flush_trade({
                        'date': date, 'action': 'SELL', 'code': weak_code,
                        'name': TARGET_ETFS.get(weak_code, ''),
                        'shares': shares, 'price': sell_price,
                        'amount': sell_revenue, 'fee': sell_fee_val + sell_tax_val,
                        'balance': cash,
                        'portfolio_value': portfolio_value,
                        'sell_return_pct': sell_ret,
                    })
                    del holdings[weak_code]
                    avg_buy_cost.pop(weak_code, None)
                    peak_price.pop(weak_code, None)
                    hold_since.pop(weak_code, None)
                    sold_today.add(weak_code)

                    # 강한 후보 매수 (매도 대금으로)
                    buy_price = best['open'] * (1 + slippage)
                    rot_budget = net_revenue
                    buy_fee_val = rot_budget * trading_fee
                    net_buy = max(rot_budget - buy_fee_val, 0.0)
                    buy_shares = int(net_buy / (buy_price + 1e-8))
                    if buy_shares > 0:
                        actual_buy_amount = buy_shares * buy_price
                        rot_budget = actual_buy_amount + actual_buy_amount * trading_fee
                        buy_fee_val = rot_budget - actual_buy_amount
                        holdings[best['code']] = buy_shares
                        avg_buy_cost[best['code']] = rot_budget / (buy_shares + 1e-8)
                        hold_since[best['code']] = tick_idx
                        peak_price[best['code']] = best['open']
                        cash -= rot_budget
                        fee_paid += buy_fee_val
                        trade_amount += rot_budget
                        total_fee_paid += buy_fee_val
                        total_trade_amount += rot_budget
                        num_buy += 1
                        _flush_trade({
                            'date': date, 'action': 'BUY', 'code': best['code'],
                            'name': TARGET_ETFS.get(best['code'], ''),
                            'shares': buy_shares, 'price': buy_price,
                            'amount': rot_budget, 'fee': buy_fee_val,
                            'balance': cash,
                            'portfolio_value': portfolio_value,
                            'sell_return_pct': 0.0,
                        })
                    non_held_strong.pop(0)
                    rotated += 1
                    _last_rotation_tick = tick_idx

            # 4) 현금 여력 시 신규 매수 (최대 max_buy_per_day개, 총 보유 max_holdings)
            pv_after = cash + sum(
                holdings[c] * open_map.get(c, prev_close_price.get(c, 0))
                for c in holdings
            )
            cash_ratio = cash / (pv_after + 1e-8) if pv_after > 0 else 1.0
            slots_available = max_holdings - len(holdings)

            # 현재 낙폭 계산 (드로다운 방어, 0 이면 비활성화)
            buy_scale = 1.0
            if drawdown_pause_pct > 0 or drawdown_reduce_pct > 0:
                current_drawdown = (pv_after - peak_portfolio_value) / (peak_portfolio_value + 1e-8)
                if drawdown_pause_pct > 0 and current_drawdown < -drawdown_pause_pct:
                    buy_scale = 0.0
                elif drawdown_reduce_pct > 0 and current_drawdown < -drawdown_reduce_pct:
                    buy_scale = 0.5

            # 방어 모드 시 신규 매수 억제
            if defense_active:
                buy_scale = 0.0  # 방어 모드에서는 추가 매수 안함

            # 변동성 기반 비중 조절 (buy_scale에만 적용, 노출도 제한과 분리)
            vol_scale = 1.0
            if vol_target > 0 and len(daily_returns) >= 20:
                recent_vol = np.std(daily_returns[-20:])
                if recent_vol > 0:
                    vol_scale = float(np.clip(vol_target / recent_vol, 0.5, 1.5))
                    buy_scale *= vol_scale

            # 최대 노출도 제한 (단독 적용, 곱셈 누적 제거)
            stock_ratio = 1.0 - cash_ratio
            if stock_ratio >= max_exposure:
                buy_scale = 0.0

            if cash_ratio > 0.1 and slots_available > 0 and buy_scale > 0:
                # 미보유 + 신호 강한 후보 선별
                buyable = [c for c in candidates
                           if c['code'] not in holdings
                           and c['code'] not in sold_today
                           and c['target_ratio'] > hold_threshold
                           and c['code'] in open_map]
                if buyable:
                    best_ratio = buyable[0]['target_ratio']
                    similar = [c for c in buyable
                               if c['target_ratio'] >= best_ratio * 0.8]
                    num_to_buy = min(len(similar), max_buy_per_day, slots_available)
                    to_buy = similar[:num_to_buy]

                    if to_buy:
                        per_etf_cash = cash / len(to_buy) * buy_scale
                        for c in to_buy:
                            buy_budget = per_etf_cash * c['target_ratio']
                            if buy_budget > 0 and buy_budget <= cash:
                                buy_price = c['open'] * (1 + slippage)
                                buy_fee = buy_budget * trading_fee
                                net_buy = max(buy_budget - buy_fee, 0.0)
                                buy_shares = int(net_buy / (buy_price + 1e-8))
                                if buy_shares > 0:
                                    prev_shares = holdings.get(c['code'], 0)
                                    prev_cost = avg_buy_cost.get(c['code'], 0.0)
                                    actual_buy_amount = buy_shares * buy_price
                                    buy_budget = actual_buy_amount + actual_buy_amount * trading_fee
                                    buy_fee = buy_budget - actual_buy_amount
                                    total_cost = prev_cost * prev_shares + buy_budget
                                    holdings[c['code']] = prev_shares + buy_shares
                                    avg_buy_cost[c['code']] = total_cost / (holdings[c['code']] + 1e-8)
                                    if prev_shares == 0:
                                        hold_since[c['code']] = tick_idx
                                        peak_price[c['code']] = c['open']
                                    cash -= buy_budget
                                    fee_paid += buy_fee
                                    trade_amount += buy_budget
                                    total_fee_paid += buy_fee
                                    total_trade_amount += buy_budget
                                    num_buy += 1
                                    traded = True
                                    _flush_trade({
                                        'date': date, 'action': 'BUY', 'code': c['code'],
                                        'name': TARGET_ETFS.get(c['code'], ''),
                                        'shares': buy_shares, 'price': buy_price,
                                        'amount': buy_budget, 'fee': buy_fee,
                                        'balance': cash,
                                        'portfolio_value': portfolio_value,
                                        'sell_return_pct': 0.0,
                                    })

            if not traded:
                num_hold += 1

            # 종일 종료: 포트폴리오 가치 (당일 종가 기준)
            stock_value_close = 0.0
            for code, shares in holdings.items():
                if code in close_map:
                    stock_value_close += shares * close_map[code]
                elif code in prev_close_price:
                    stock_value_close += shares * prev_close_price[code]
            portfolio_value = cash + stock_value_close
            position_ratio_close = stock_value_close / (portfolio_value + 1e-8) if portfolio_value > 0 else 0.0
            # 보유 종목 가중 평균 일중수익률
            intraday_return = 0.0
            if holdings:
                weighted_return = 0.0
                total_weight = 0.0
                for code, shares in holdings.items():
                    if code in open_map and code in close_map:
                        weight = shares * open_map[code]
                        ret = (close_map[code] - open_map[code]) / (open_map[code] + 1e-8)
                        weighted_return += weight * ret
                        total_weight += weight
                if total_weight > 0:
                    intraday_return = weighted_return / total_weight

        day_return = (portfolio_value - pv_open) / (pv_open + 1e-8)
        daily_returns.append(day_return)
        if day_return > 0:
            consecutive_wins += 1
            consecutive_losses = 0
        elif day_return < 0:
            consecutive_losses += 1
            consecutive_wins = 0

        peak_portfolio_value = max(peak_portfolio_value, portfolio_value)
        pv_history.append(portfolio_value)

        if selected is not None:
            policy_outputs.append(selected['policy_output'])
            values.append(selected['value'])
        else:
            policy_outputs.append(np.array([0.0, 0.0], dtype=np.float32))
            values.append(0.0)

        _flush_history({
            'date': date,
            'tick': len(history),
            'selected_etf': selected_code,
            'held_etf': ','.join(sorted(holdings.keys())) if holdings else '',
            'num_holdings': len(holdings),
            'position_size': float(target_ratio),
            'position_ratio': float(position_ratio_close),
            'num_shares': float(sum(holdings.values())) if holdings else 0.0,
            'traded': bool(traded),
            'open_price': float(selected['open']) if selected else np.nan,
            'close_price': float(selected['close']) if selected else np.nan,
            'intraday_return': float(intraday_return),
            'fee_paid': float(fee_paid),
            'balance': float(cash),
            'portfolio_value': float(portfolio_value),
            'prev_portfolio_value': float(pv_open),
            'reward': float(np.clip(day_return * 30.0, -5.0, 5.0)),
        })

        # 전일 종가 업데이트
        for code, row in day_rows.iterrows():
            prev_close_price[row['etf_code']] = float(row['close'])

    profit = portfolio_value - initial_balance
    profit_rate = (profit / (initial_balance + 1e-8)) * 100

    pv_arr = np.array(pv_history)
    peak_arr = np.maximum.accumulate(pv_arr)
    drawdowns = (pv_arr - peak_arr) / (peak_arr + 1e-8)
    max_drawdown = float(drawdowns.min()) * 100
    trough_arr = np.minimum.accumulate(pv_arr)
    draw_ups = (pv_arr - trough_arr) / (trough_arr + 1e-8)
    max_draw_up = float(draw_ups.max()) * 100

    traded_returns = [r for r in daily_returns if abs(r) > 1e-10]
    win_rate = (sum(1 for r in traded_returns if r > 0) / max(len(traded_returns), 1) * 100) if traded_returns else 0.0
    avg_win = (np.mean([r for r in traded_returns if r > 0]) * 100) if any(r > 0 for r in traded_returns) else 0.0
    avg_loss = (np.mean([r for r in traded_returns if r < 0]) * 100) if any(r < 0 for r in traded_returns) else 0.0
    sharpe = float(np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * math.sqrt(252)) if len(daily_returns) > 1 else 0.0

    # CAGR = (최종자산 / 초기자산)^(1 / 투자기간(년)) - 1, 투자기간은 실제 달력 날짜 기준
    if len(all_dates) >= 2:
        start_dt = datetime.strptime(str(int(all_dates[0])), '%Y%m%d')
        end_dt = datetime.strptime(str(int(all_dates[-1])), '%Y%m%d')
        n_years = max((end_dt - start_dt).days / 365.25, 0.01)
    else:
        n_years = max(len(all_dates) / 252.0, 0.01)
    cagr = ((portfolio_value / initial_balance) ** (1.0 / n_years) - 1.0) * 100
    calmar = abs(cagr / max_drawdown) if max_drawdown != 0 else 0.0
    profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0

    stats = {
        'num_buy': num_buy,
        'num_sell': num_sell,
        'num_hold': num_hold,
        'initial_balance': initial_balance,
        'portfolio_value': portfolio_value,
        'profit': profit,
        'profit_rate': profit_rate,
        'total_trade_amount': total_trade_amount,
        'total_fee_paid': total_fee_paid,

        'max_drawdown': max_drawdown,
        'max_draw_up': max_draw_up,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'sharpe_ratio': sharpe,
        'cagr': cagr,
        'calmar_ratio': calmar,
        'profit_loss_ratio': profit_loss_ratio,
        'total_days': len(history),
        'traded_days': int(sum(1 for h in history if h['traded'])),
        'trade_ratio': float(sum(1 for h in history if h['traded']) / max(len(history), 1)),
        'avg_position_size': float(np.mean([h['position_size'] for h in history])) if history else 0.0,
        'avg_intraday_return': float(np.mean([h['intraday_return'] for h in history])) * 100 if history else 0.0,
        'avg_fee_per_trade': float(total_fee_paid / max(num_buy + num_sell, 1)),
        'avg_target_ratio': float(np.mean([h['position_size'] for h in history])) if history else 0.0,
        'std_target_ratio': float(np.std([h['position_size'] for h in history])) if history else 0.0,
        'avg_position_ratio': float(np.mean([h['position_ratio'] for h in history])) if history else 0.0,
        'std_position_ratio': float(np.std([h['position_ratio'] for h in history])) if history else 0.0,
        'num_rebalance': int(num_buy + num_sell),
        'rebalance_ratio': float((num_buy + num_sell) / max(len(history), 1)),
        'days_in_position': int(sum(1 for h in history if h['position_ratio'] > 0)),
        'position_ratio_pct': float(sum(1 for h in history if h['position_ratio'] > 0) / max(len(history), 1)),
        'avg_num_holdings': float(np.mean([h.get('num_holdings', 1) for h in history])) if history else 0.0,
    }

    # 스트리밍 CSV 파일 닫기
    if _trade_log_file:
        _trade_log_file.close()
    if _history_file:
        _history_file.close()

    return stats, history, pv_history, policy_outputs, values, trade_log


def load_data(env_data_path: str, training_data_path: str):
    """데이터 로드"""
    env_data = pd.read_csv(env_data_path)
    training_data = pd.read_csv(training_data_path).values
    assert len(env_data) == len(training_data), \
        f"환경 데이터({len(env_data)})와 학습 데이터({len(training_data)})의 길이가 다릅니다."
    return env_data, training_data


def compute_yearly_stats(history, trade_log):
    """연속 백테스트 history에서 연도별 통계 산출"""
    if not history:
        return []
    yearly = {}
    for h in history:
        year = str(h['date'])[:4]
        if year not in yearly:
            yearly[year] = {'entries': [], 'start_pv': h['prev_portfolio_value']}
        yearly[year]['entries'].append(h)

    results = []
    for year in sorted(yearly.keys()):
        entries = yearly[year]['entries']
        start_pv = yearly[year]['start_pv']
        end_pv = entries[-1]['portfolio_value']
        profit_rate = (end_pv - start_pv) / (start_pv + 1e-8) * 100

        pvs = [start_pv] + [e['portfolio_value'] for e in entries]
        pv_arr = np.array(pvs)
        peak_arr = np.maximum.accumulate(pv_arr)
        drawdowns = (pv_arr - peak_arr) / (peak_arr + 1e-8)
        max_drawdown = float(drawdowns.min()) * 100

        daily_rets = [
            (e['portfolio_value'] - e['prev_portfolio_value']) / (e['prev_portfolio_value'] + 1e-8)
            for e in entries
        ]
        traded_rets = [r for r in daily_rets if abs(r) > 1e-10]
        win_rate = (sum(1 for r in traded_rets if r > 0) / max(len(traded_rets), 1) * 100) if traded_rets else 0.0
        avg_pos = float(np.mean([e['position_ratio'] for e in entries]))

        year_trades = [t for t in trade_log if str(t['date'])[:4] == year]
        num_buy = sum(1 for t in year_trades if t['action'] == 'BUY')
        num_sell = sum(1 for t in year_trades if t['action'] == 'SELL')
        total_fee = sum(float(t['fee']) for t in year_trades)

        sharpe = float(np.mean(daily_rets) / (np.std(daily_rets) + 1e-8) * math.sqrt(252)) if len(daily_rets) > 1 else 0.0
        calmar = abs(profit_rate / max_drawdown) if max_drawdown != 0 else 0.0

        results.append({
            'year': year,
            'profit_rate': profit_rate,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_position_ratio': avg_pos,
            'sharpe_ratio': sharpe,
            'calmar_ratio': calmar,
            'num_buy': num_buy,
            'num_sell': num_sell,
            'num_rebalance': num_buy + num_sell,
            'total_fee': total_fee,
            'start_pv': start_pv,
            'end_pv': end_pv,
            'total_days': len(entries),
            'start_step': entries[0]['tick'],
            'end_step': entries[-1]['tick'],
        })
    return results


def write_summary_markdown(output_dir, yearly_stats, stats, dataset, trading_method, timestamp):
    """백테스트 결과를 markdown 파일로 저장"""
    lines = [
        f'# Backtest Summary',
        f'',
        f'| 항목 | 값 |',
        f'|------|-----|',
        f'| Dataset | {dataset} |',
        f'| Trading Method | {trading_method} |',
        f'| Timestamp | {timestamp} |',
        f'| 초기 자산 | {stats["initial_balance"]:,.0f} 원 |',
        f'| 최종 자산 | {stats["portfolio_value"]:,.0f} 원 |',
        f'| 총 손익 | {stats["profit"]:+,.0f} 원 |',
        f'| 수익률 | {stats["profit_rate"]:+.2f}% |',
        f'| MDD | {stats["max_drawdown"]:.2f}% |',
        f'| 승률 | {stats["win_rate"]:.0f}% |',
        f'| Sharpe Ratio | {stats["sharpe_ratio"]:.4f} |',
        f'| CAGR | {stats.get("cagr", 0):+.2f}% |',
        f'| Calmar Ratio | {stats.get("calmar_ratio", 0):.4f} |',
        f'| 손익비 | {stats.get("profit_loss_ratio", 0):.2f} |',
        f'| 총 거래일 | {stats["total_days"]:,} |',
        f'| 총 리밸런싱 | {stats["num_rebalance"]:,} |',
        f'| 총 수수료 | {stats["total_fee_paid"]:,.0f} 원 |',
        f'',
        f'## 연도별 성과',
        f'',
        f'| 연도 | 수익률 | MDD | Sharpe | Calmar | 승률 | 주식비율 | 리밸런싱 | 구간 |',
        f'|------|--------|-----|-------|--------|------|---------|---------|------|',
    ]
    for ys in yearly_stats:
        seg = f'[{ys["start_step"]:,}~{ys["end_step"]:,}]'
        lines.append(
            f'| {ys["year"]} | {ys["profit_rate"]:+.4f}% '
            f'| {ys["max_drawdown"]:.4f}% '
            f'| {ys.get("sharpe_ratio", 0):.2f} '
            f'| {ys.get("calmar_ratio", 0):.2f} '
            f'| {ys["win_rate"]:.0f}% '
            f'| {ys["avg_position_ratio"]:.0%} '
            f'| {ys["num_rebalance"]} '
            f'| {seg} |'
        )
    n = len(yearly_stats)
    if n > 0:
        avg_pr = sum(y['profit_rate'] for y in yearly_stats) / n
        avg_mdd = sum(y['max_drawdown'] for y in yearly_stats) / n
        avg_wr = sum(y['win_rate'] for y in yearly_stats) / n
        avg_pos = sum(y['avg_position_ratio'] for y in yearly_stats) / n
        total_rebal = sum(y['num_rebalance'] for y in yearly_stats)
        avg_sharpe = sum(y.get('sharpe_ratio', 0) for y in yearly_stats) / n
        avg_calmar = sum(y.get('calmar_ratio', 0) for y in yearly_stats) / n
        lines.append(f'| **평균** | **{avg_pr:+.4f}%** | **{avg_mdd:.4f}%** | **{avg_sharpe:.2f}** | **{avg_calmar:.2f}** | **{avg_wr:.0f}%** | **{avg_pos:.0%}** | **{total_rebal}** | 전체 {n}구간 |')

    md_path = os.path.join(output_dir, 'backtest_summary.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    return md_path


def write_pv_chart_html(output_dir, history, stats, yearly_stats=None):
    """포트폴리오 가치 변화 차트를 HTML 파일로 저장 (Canvas 기반, 외부 의존성 없음)"""
    dates = [str(h['date']) for h in history]
    pvs = [float(h['portfolio_value']) for h in history]
    if not dates:
        return None

    # 날짜 포맷
    fmt_dates = []
    for d in dates:
        if len(d) == 8:
            fmt_dates.append(f"{d[:4]}-{d[4:6]}-{d[6:8]}")
        else:
            fmt_dates.append(d)

    # drawdown 계산
    peak = pvs[0]
    drawdowns = []
    for pv in pvs:
        peak = max(peak, pv)
        dd = (pv - peak) / (peak + 1e-8) * 100
        drawdowns.append(round(dd, 2))

    # 연도 경계선
    year_boundaries = []
    if yearly_stats:
        for i, ys in enumerate(yearly_stats):
            if i == 0:
                continue
            year_boundaries.append({
                'idx': int(ys['start_step']),
                'year': str(ys['year']),
            })

    # JSON 인라인 데이터
    import json as _json
    data_json = _json.dumps({
        'dates': fmt_dates,
        'pvs': [round(v, 0) for v in pvs],
        'drawdowns': drawdowns,
        'initial': round(float(stats['initial_balance']), 0),
        'final': round(float(stats['portfolio_value']), 0),
        'profit_rate': round(float(stats['profit_rate']), 2),
        'mdd': round(float(stats.get('max_drawdown', 0)), 2),
        'year_boundaries': year_boundaries,
    }, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Portfolio Value Chart</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #0f0f0f; color: #e0e0e0; padding: 20px; }}
  .header {{ margin-bottom: 16px; }}
  .header h1 {{ font-size: 18px; font-weight: 600; color: #f0f0f0; margin-bottom: 8px; }}
  .stats {{ display: flex; gap: 24px; flex-wrap: wrap; margin-bottom: 16px; }}
  .stat {{ font-size: 13px; }}
  .stat .label {{ color: #888; margin-right: 4px; }}
  .stat .value {{ font-weight: 600; }}
  .positive {{ color: #ef4444; }}
  .negative {{ color: #3b82f6; }}
  .chart-container {{ position: relative; background: #1a1a1a; border-radius: 8px;
                      padding: 16px; border: 1px solid #2a2a2a; }}
  canvas {{ display: block; width: 100%; }}
  .tooltip {{ position: absolute; display: none; background: rgba(30,30,30,0.95);
              border: 1px solid #444; border-radius: 6px; padding: 8px 12px;
              font-size: 12px; pointer-events: none; z-index: 10; white-space: nowrap; }}
  .tooltip .tt-date {{ color: #aaa; margin-bottom: 4px; }}
  .tooltip .tt-pv {{ font-weight: 600; font-size: 13px; }}
  .tooltip .tt-dd {{ color: #3b82f6; font-size: 11px; }}
  .legend {{ display: flex; gap: 20px; margin-top: 10px; font-size: 12px; color: #888; }}
  .legend span {{ display: inline-flex; align-items: center; gap: 4px; }}
  .legend .dot {{ width: 10px; height: 3px; border-radius: 1px; display: inline-block; }}
</style>
</head>
<body>
<div class="header">
  <h1>Portfolio Value</h1>
  <div class="stats" id="stats"></div>
</div>
<div class="chart-container">
  <canvas id="pvChart" height="100"></canvas>
  <canvas id="ddChart" height="30"></canvas>
  <div class="tooltip" id="tooltip"></div>
  <div class="legend">
    <span><span class="dot" style="background:#10b981;"></span> Portfolio Value</span>
    <span><span class="dot" style="background:#3b82f6;"></span> Drawdown</span>
  </div>
</div>
<script>
const D = {data_json};
const statsEl = document.getElementById('stats');
const profitClass = D.profit_rate >= 0 ? 'positive' : 'negative';
statsEl.innerHTML = `
  <div class="stat"><span class="label">초기</span><span class="value">${"{D.initial.toLocaleString()}"} 원</span></div>
  <div class="stat"><span class="label">최종</span><span class="value ${{profitClass}}">${"{D.final.toLocaleString()}"} 원</span></div>
  <div class="stat"><span class="label">수익률</span><span class="value ${{profitClass}}">${"{D.profit_rate > 0 ? '+' : ''}"}${{D.profit_rate}}%</span></div>
  <div class="stat"><span class="label">MDD</span><span class="value negative">${{D.mdd}}%</span></div>
  <div class="stat"><span class="label">거래일</span><span class="value">${{D.dates.length.toLocaleString()}}</span></div>
`;

function drawChart(canvasId, values, color, fillColor, yLabel, isPercent, refLine) {{
  const canvas = document.getElementById(canvasId);
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  const W = rect.width, H = rect.height;
  const pad = {{ l: 80, r: 20, t: 10, b: 24 }};
  const cW = W - pad.l - pad.r, cH = H - pad.t - pad.b;

  let mn = Math.min(...values), mx = Math.max(...values);
  if (refLine !== undefined) {{ mn = Math.min(mn, refLine); mx = Math.max(mx, refLine); }}
  const range = mx - mn || 1;
  mn -= range * 0.05; mx += range * 0.05;
  const yRange = mx - mn;

  const toX = i => pad.l + (i / (values.length - 1)) * cW;
  const toY = v => pad.t + (1 - (v - mn) / yRange) * cH;

  // year boundaries
  ctx.strokeStyle = '#333'; ctx.lineWidth = 1; ctx.setLineDash([4,4]);
  for (const yb of D.year_boundaries) {{
    const x = toX(yb.idx);
    ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, pad.t + cH); ctx.stroke();
    ctx.fillStyle = '#555'; ctx.font = '10px sans-serif'; ctx.textAlign = 'center';
    ctx.fillText(yb.year, x, pad.t + cH + 14);
  }}
  ctx.setLineDash([]);

  // reference line
  if (refLine !== undefined) {{
    ctx.strokeStyle = '#444'; ctx.lineWidth = 1; ctx.setLineDash([2,4]);
    const ry = toY(refLine);
    ctx.beginPath(); ctx.moveTo(pad.l, ry); ctx.lineTo(pad.l + cW, ry); ctx.stroke();
    ctx.setLineDash([]);
  }}

  // Y axis labels
  ctx.fillStyle = '#666'; ctx.font = '11px sans-serif'; ctx.textAlign = 'right';
  for (let i = 0; i <= 4; i++) {{
    const v = mn + (yRange * i / 4);
    const y = toY(v);
    const label = isPercent ? v.toFixed(1) + '%' : (v / 1e6).toFixed(1) + 'M';
    ctx.fillText(label, pad.l - 8, y + 4);
    ctx.strokeStyle = '#222'; ctx.lineWidth = 0.5;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + cW, y); ctx.stroke();
  }}

  // fill
  ctx.beginPath();
  ctx.moveTo(toX(0), toY(refLine !== undefined ? refLine : mn));
  for (let i = 0; i < values.length; i++) ctx.lineTo(toX(i), toY(values[i]));
  ctx.lineTo(toX(values.length - 1), toY(refLine !== undefined ? refLine : mn));
  ctx.closePath();
  ctx.fillStyle = fillColor; ctx.fill();

  // line
  ctx.beginPath();
  for (let i = 0; i < values.length; i++) {{
    const x = toX(i), y = toY(values[i]);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }}
  ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.stroke();

  return {{ toX, toY, pad, cW, cH, mn, mx: mn + yRange }};
}}

const pvInfo = drawChart('pvChart', D.pvs, '#10b981', 'rgba(16,185,129,0.08)', 'PV', false, D.initial);
const ddInfo = drawChart('ddChart', D.drawdowns, '#3b82f6', 'rgba(59,130,246,0.1)', 'DD', true, 0);

// Tooltip
const container = document.querySelector('.chart-container');
const tooltip = document.getElementById('tooltip');
const pvCanvas = document.getElementById('pvChart');

container.addEventListener('mousemove', e => {{
  const rect = pvCanvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const idx = Math.round((x - pvInfo.pad.l) / pvInfo.cW * (D.pvs.length - 1));
  if (idx < 0 || idx >= D.pvs.length) {{ tooltip.style.display = 'none'; return; }}
  const pv = D.pvs[idx];
  const dd = D.drawdowns[idx];
  const ret = ((pv - D.initial) / D.initial * 100).toFixed(2);
  const retClass = ret >= 0 ? 'positive' : 'negative';
  tooltip.innerHTML = `<div class="tt-date">${{D.dates[idx]}}</div>
    <div class="tt-pv ${{retClass}}">${{pv.toLocaleString()}} 원 (${{ret > 0 ? '+' : ''}}${{ret}}%)</div>
    <div class="tt-dd">DD: ${{dd}}%</div>`;
  tooltip.style.display = 'block';
  const tx = e.clientX - container.getBoundingClientRect().left;
  tooltip.style.left = (tx + 16) + 'px';
  tooltip.style.top = '40px';
  if (tx + tooltip.offsetWidth + 20 > container.offsetWidth)
    tooltip.style.left = (tx - tooltip.offsetWidth - 16) + 'px';
}});
container.addEventListener('mouseleave', () => {{ tooltip.style.display = 'none'; }});

window.addEventListener('resize', () => {{
  drawChart('pvChart', D.pvs, '#10b981', 'rgba(16,185,129,0.08)', 'PV', false, D.initial);
  drawChart('ddChart', D.drawdowns, '#3b82f6', 'rgba(59,130,246,0.1)', 'DD', true, 0);
}});
</script>
</body>
</html>"""

    html_path = os.path.join(output_dir, 'portfolio_value.html')
    with open(html_path, 'w') as f:
        f.write(html)
    return html_path


def create_networks(
    input_dim: int,
    network_type: str = 'standard',
    device: str = 'cpu',
    d_model: int = 128,
    n_blocks: int = 3,
    d_state: int = 16,
    min_concentration: float = 1.5,
    policy_dropout: float = 0.15,
    value_dropout: float = 0.15,
):
    """신경망 생성"""
    if network_type == 'lstm':
        policy_net = LSTMContinuousPolicyNetwork(
            input_dim, hidden_dim=128, num_layers=2,
            dropout=policy_dropout, min_concentration=min_concentration,
        )
        value_net = LSTMValueNetwork(
            input_dim, hidden_dim=128, num_layers=2,
            dropout=value_dropout,
        )
    elif network_type == 'mamba':
        policy_net = MambaPolicyNetwork(
            input_dim, d_model=d_model, d_state=d_state, n_blocks=n_blocks,
            dropout=policy_dropout, min_concentration=min_concentration,
        )
        value_net = MambaValueNetwork(
            input_dim, d_model=d_model, d_state=d_state, n_blocks=n_blocks,
            dropout=value_dropout,
        )
    else:
        policy_net = ContinuousPolicyNetwork(
            input_dim, hidden_dim=256, num_blocks=3,
            dropout=policy_dropout, min_concentration=min_concentration,
        )
        value_net = ValueNetwork(
            input_dim, hidden_dim=256, num_blocks=3,
            dropout=value_dropout,
        )
    return policy_net, value_net


def detect_network_type(policy_path: str) -> str:
    """저장된 모델 가중치에서 네트워크 타입 자동 감지"""
    checkpoint = torch.load(policy_path, map_location='cpu', weights_only=True)
    state_dict = checkpoint.get('state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
    keys = set(state_dict.keys())
    if any('ssm_gate' in k for k in keys):
        return 'mamba'
    if any('lstm' in k.lower() for k in keys):
        return 'lstm'
    return 'standard'


def detect_input_dim(policy_path: str) -> int:
    """저장된 모델에서 입력 차원 감지"""
    checkpoint = torch.load(policy_path, map_location='cpu', weights_only=True)
    if isinstance(checkpoint, dict) and 'input_dim' in checkpoint:
        return checkpoint['input_dim']
    state_dict = checkpoint.get('state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
    # input_proj (mamba/standard) 또는 lstm 에서 추출
    for key in ['input_proj.0.weight', 'feature_layers.0.weight', 'lstm.weight_ih_l0']:
        if key in state_dict:
            return state_dict[key].shape[1]
    return None


def run_backtest(env, agent, pad_dim=0, min_value=None, min_concentration=None):
    """백테스트 실행 (deterministic: Beta 분포의 mean 사용)
    
    min_value: value network 예측 >= min_value 일 때만 거래
    min_concentration: Beta 분포 집중도 (alpha+beta) >= min_concentration 일 때만 거래
    """
    state = env.reset()
    agent.reset_stats()
    done = False

    policy_outputs = []
    values = []

    def _get_current_ratio():
        """현재 포지션 비율 반환 (거래 억제용)"""
        if hasattr(env, 'num_shares') and hasattr(env, 'balance'):
            open_price = env.env_data.iloc[env.tick]['open']
            stock_value = env.num_shares * open_price
            return stock_value / (env.balance + stock_value + 1e-8)
        return 0.0  # Day Trading: hold

    while not done:
        if pad_dim > 0:
            state = np.concatenate([state, np.zeros(pad_dim)])
        action, _, policy_output = agent.get_action(state, training=False)
        value = agent.get_value(state)

        # 가치 기반 필터
        if min_value is not None and value < min_value:
            action = _get_current_ratio()

        # 집중도 기반 필터 (alpha + beta >= threshold)
        if min_concentration is not None:
            concentration = policy_output[0] + policy_output[1]
            if concentration < min_concentration:
                action = _get_current_ratio()

        policy_outputs.append(policy_output)
        values.append(value)

        next_state, reward, done, info = env.step(action)
        state = next_state

    return policy_outputs, values


def compute_metrics(env, env_data, trading_method='day', start_tick=0, end_tick=None):
    """백테스트 결과 메트릭 계산"""
    stats = env.get_stats()
    history = env.history

    if not history:
        return stats

    history_df = pd.DataFrame(history)

    stats['total_days'] = len(history)
    stats['traded_days'] = int(history_df['traded'].sum())
    stats['trade_ratio'] = stats['traded_days'] / max(stats['total_days'], 1)

    # 포트폴리오 가치 시계열
    pv_array = np.array(env.pv_history)

    # 수익률 통계
    pv_returns = np.diff(pv_array) / (pv_array[:-1] + 1e-8)
    if len(pv_returns) > 0:
        stats['mean_return'] = float(pv_returns.mean())
        stats['std_return'] = float(pv_returns.std())
    else:
        stats['mean_return'] = 0.0
        stats['std_return'] = 0.0

    # 거래 분석
    traded = history_df[history_df['traded'] == True]
    if len(traded) > 0:
        stats['avg_position_size'] = float(traded['position_size'].mean())
        stats['avg_intraday_return'] = float(traded['intraday_return'].mean()) * 100
        stats['avg_fee_per_trade'] = float(traded['fee_paid'].mean())
    else:
        stats['avg_position_size'] = 0.0
        stats['avg_intraday_return'] = 0.0
        stats['avg_fee_per_trade'] = 0.0

    stats['avg_target_ratio'] = float(history_df['position_size'].mean())
    stats['std_target_ratio'] = float(history_df['position_size'].std())

    # Swing 전용 메트릭
    if trading_method == 'swing' and 'position_ratio' in history_df.columns:
        stats['avg_position_ratio'] = float(history_df['position_ratio'].mean())
        stats['std_position_ratio'] = float(history_df['position_ratio'].std())
        # 리밸런싱 횟수 (실제 거래 발생)
        stats['num_rebalance'] = int(history_df['traded'].sum())
        stats['rebalance_ratio'] = stats['num_rebalance'] / max(stats['total_days'], 1)
        # 포지션 유지 구간 분석
        if 'num_shares' in history_df.columns:
            in_position = (history_df['num_shares'] > 0).astype(int)
            stats['days_in_position'] = int(in_position.sum())
            stats['position_ratio_pct'] = stats['days_in_position'] / max(stats['total_days'], 1)

    return stats


def print_report(stats, trading_method='day', dataset_name='', segment_info=''):
    """백테스트 결과 리포트 출력"""
    method_label = 'Swing Trading' if trading_method == 'swing' else 'Day Trading'

    print("\n" + "=" * 70)
    print(f"  ETF {method_label} 백테스트 결과")
    if dataset_name:
        print(f"  데이터셋: {dataset_name}")
    if segment_info:
        print(f"  구간: {segment_info}")
    print("=" * 70)

    print(f"\n{'─' * 40}")
    print(f"  수익률")
    print(f"{'─' * 40}")
    print(f"  포트폴리오 수익률 : {stats['profit_rate']:>+10.4f}%")
    if 'bnh_return' in stats:
        print(f"  B&H 수익률        : {stats['bnh_return']:>+10.4f}%")
        print(f"  초과 수익률        : {stats.get('excess_bnh', 0):>+10.4f}%")
    print(f"  최대 낙폭 (MDD)   : {stats.get('max_drawdown', 0):>10.4f}%")
    print(f"  Sharpe Ratio      : {stats.get('sharpe_ratio', 0):>10.4f}")
    if 'cagr' in stats:
        print(f"  CAGR              : {stats['cagr']:>+10.4f}%")
    if 'calmar_ratio' in stats:
        print(f"  Calmar Ratio      : {stats['calmar_ratio']:>10.4f}")
    if 'profit_loss_ratio' in stats:
        print(f"  손익비            : {stats['profit_loss_ratio']:>10.2f}")
    print(f"  승률              : {stats.get('win_rate', 0):>10.1f}%")

    print(f"\n{'─' * 40}")
    print(f"  포트폴리오")
    print(f"{'─' * 40}")
    print(f"  초기 자산 : {stats['initial_balance']:>15,.0f} 원")
    print(f"  최종 자산 : {stats['portfolio_value']:>15,.0f} 원")
    print(f"  손익      : {stats['profit']:>+15,.0f} 원")

    print(f"\n{'─' * 40}")
    print(f"  거래 통계")
    print(f"{'─' * 40}")
    print(f"  총 거래일        : {stats.get('total_days', 0):>8d}")

    if trading_method == 'swing':
        print(f"  리밸런싱 횟수    : {stats.get('num_rebalance', stats['num_buy']):>8d}")
        print(f"  리밸런싱 비율    : {stats.get('rebalance_ratio', 0):>8.1%}")
        print(f"  포지션 보유일    : {stats.get('days_in_position', 0):>8d}")
        print(f"  포지션 보유비율  : {stats.get('position_ratio_pct', 0):>8.1%}")
        print(f"  평균 보유종목수  : {stats.get('avg_num_holdings', 0):>8.1f}")
        print(f"  평균 목표비율    : {stats.get('avg_target_ratio', 0):>8.1%}")
        print(f"  평균 실제비율    : {stats.get('avg_position_ratio', 0):>8.1%}")
    else:
        print(f"  매수일 (거래)    : {stats['num_buy']:>8d}")
        print(f"  미매수일 (관망)  : {stats['num_hold']:>8d}")
        print(f"  평균 투입비율    : {stats.get('avg_position_size', 0):>8.2%}")
        print(f"  평균 일중수익률  : {stats.get('avg_intraday_return', 0):>+8.4f}%")

    print(f"  총 거래 금액     : {stats['total_trade_amount']:>15,.0f} 원")
    print(f"  총 수수료        : {stats['total_fee_paid']:>15,.0f} 원")
    print(f"  평균 수수료/거래 : {stats.get('avg_fee_per_trade', 0):>15,.0f} 원")

    print(f"\n{'=' * 70}\n")


def main():
    parser = argparse.ArgumentParser(description='백테스트')

    parser.add_argument(
        '--base-path',
        type=str,
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')),
    )
    parser.add_argument('--dataset', type=str, default='etf_20260410',
                        help='데이터셋 이름')
    parser.add_argument('--env-data', type=str, default=None)
    parser.add_argument('--training-data', type=str, default=None)

    parser.add_argument('--policy', type=str, default=None)
    parser.add_argument('--value', type=str, default=None)
    parser.add_argument('--model', type=str, default='etf-swing-v5',
                        help='모델 이름 또는 모델 디렉토리 절대경로')

    # 환경 설정
    parser.add_argument('--initial-balance', type=float, default=10_000_000.0)
    parser.add_argument('--trading-fee', type=float, default=0.00015)
    parser.add_argument('--trading-tax', type=float, default=0.0,
                        help='거래세 (일반 주식: 0.002=0.2%%, ETF: 0)')
    parser.add_argument('--slippage', type=float, default=0.0003)
    parser.add_argument('--action-scale', type=float, default=1.0)
    parser.add_argument('--hold-threshold', type=float, default=0.10,
                        help='매수 기준 (ratio_diff > threshold 일 때만 거래, 기본 0.10=10%%)')
    parser.add_argument('--min-value', type=float, default=None,
                        help='가치 기반 필터 (value >= min_value 일 때만 거래, 예: 0.0)')
    parser.add_argument('--min-concentration', type=float, default=None,
                        help='집중도 필터 (alpha+beta >= threshold 일 때만 거래, 예: 8.0)')
    parser.add_argument('--selector-mode', type=str, default='auto',
                        choices=['auto', 'off', 'on'],
                        help='멀티 ETF 종목선정 백테스트 모드 (auto: 데이터셋 자동 감지)')
    parser.add_argument('--max-buy-per-day', type=int, default=5,
                        help='일일 최대 신규 매수 종목 수 (기본 5)')
    parser.add_argument('--max-holdings', type=int, default=5,
                        help='최대 보유 종목 수 (기본 5)')
    parser.add_argument('--stop-loss-pct', type=float, default=0.0,
                        help='종목별 손절 비율 (0=비활성화, 예: 0.15=15%%)')
    parser.add_argument('--drawdown-reduce-pct', type=float, default=0.10,
                        help='신규 매수 축소 기준 낙폭 (0=비활성화, 예: 0.10=10%%)')
    parser.add_argument('--drawdown-pause-pct', type=float, default=0.25,
                        help='신규 매수 중단 기준 낙폭 (0=비활성화, 예: 0.25=25%%)')
    parser.add_argument('--sell-threshold', type=float, default=None,
                        help='매도 기준 (기본: hold_threshold*0.5, 매수보다 낮게 설정하여 whipsaw 방지)')
    parser.add_argument('--trailing-stop-pct', type=float, default=0.0,
                        help='트레일링 스톱 비율 (0=비활성화, 예: 0.10=고점대비 10%%%% 하락시 매도)')
    parser.add_argument('--max-exposure', type=float, default=1.0,
                        help='최대 주식 노출 비율 (기본 1.0=100%%%%, 예: 0.85=85%%%%)')
    parser.add_argument('--vol-target', type=float, default=0.0,
                        help='목표 일일 변동성 (0=비활성화, 예: 0.01=1%%%%)')
    parser.add_argument('--min-hold-days', type=int, default=3,
                        help='최소 보유 거래일 수 (0=비활성화, 예: 5)')

    # 슬라이싱
    parser.add_argument('--start-step', type=int, default=0)
    parser.add_argument('--max-steps', type=int, default=0)

    # 네트워크
    parser.add_argument('--network-type', type=str, default='auto',
                        choices=['auto', 'standard', 'lstm', 'mamba'])

    # 거래 방식 (auto: train_config에서 감지, day/swing: 강제 지정)
    parser.add_argument('--trading-method', type=str, default='auto',
                        choices=['auto', 'day', 'swing'])

    # 출력
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--no-visualize', action='store_true')

    # 장치
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'])

    # 순차 백테스트
    parser.add_argument('--sequential', action='store_true',
                        help='전체 데이터를 청크로 순차 백테스트')
    parser.add_argument('--sequential-mode', type=str, default='year',
                        choices=['year'],
                        help='순차 분할 방식 (연도별)')

    args = parser.parse_args()

    # 재현성을 위해 시드 고정 (GPU 사용 시 비결정적일 수 있음)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') \
        if args.device == 'auto' else args.device
    torch.manual_seed(42)
    np.random.seed(42)

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

    # 모델 경로
    if args.model:
        model_dir = _resolve_model_dir(args.base_path, args.model)
        policy_path = args.policy or os.path.join(model_dir, 'policy_best.pt')
        value_path = args.value or os.path.join(model_dir, 'value_best.pt')
    else:
        policy_path = args.policy or os.path.join(args.base_path, 'models', 'policy_best.pt')
        value_path = args.value or os.path.join(args.base_path, 'models', 'value_best.pt')
    if not os.path.isabs(policy_path):
        policy_path = os.path.join(args.base_path, policy_path)
    if not os.path.isabs(value_path):
        value_path = os.path.join(args.base_path, value_path)

    # 모델 설정 로드 (trading_method 자동 감지)
    trading_method = 'day'  # 기본값
    train_config = {}
    if args.model:
        config_path = os.path.join(model_dir, 'train_config.json')
        if os.path.exists(config_path):
            with open(config_path) as f:
                train_config = json.load(f)
            trading_method = train_config.get('trading_method', 'day')
            # train_config에서 dataset 자동 설정
            if 'dataset' in train_config and args.dataset == 'etf_20260317':
                args.dataset = train_config['dataset']
            print(f"  모델 설정 로드: trading_method={trading_method}")

    # --trading-method 옵션으로 강제 지정
    if args.trading_method != 'auto':
        trading_method = args.trading_method

    EnvClass = SwingTradingEnvironment if trading_method == 'swing' else DayTradingEnvironment

    # 출력 디렉토리
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join(
        args.base_path, 'output', 'backtest', f'backtest_{timestamp}'
    )
    os.makedirs(output_dir, exist_ok=True)

    method_label = 'Swing Trading' if trading_method == 'swing' else 'Day Trading'

    print("=" * 70)
    print(f"  ETF {method_label} RL 백테스트")
    print("=" * 70)
    print(f"  거래 방식       : {method_label}")
    print(f"  데이터셋       : {args.dataset}")
    print(f"  정책 모델      : {policy_path}")
    print(f"  가치 모델      : {value_path}")
    print(f"  결과 저장      : {output_dir}")
    print(f"  매수 기준      : {args.hold_threshold:.0%}")
    if args.min_value is not None:
        print(f"  가치 필터      : value >= {args.min_value}")
    if args.min_concentration is not None:
        print(f"  집중도 필터    : alpha+beta >= {args.min_concentration}")
    if args.trailing_stop_pct > 0:
        print(f"  트레일링 스톱  : {args.trailing_stop_pct:.0%}")
    if args.max_exposure < 1.0:
        print(f"  최대 노출도    : {args.max_exposure:.0%}")
    if args.vol_target > 0:
        print(f"  목표 변동성    : {args.vol_target:.2%}")
    if args.min_hold_days > 0:
        print(f"  최소 보유일    : {args.min_hold_days}일")

    # 모델 존재 확인
    if not os.path.exists(policy_path):
        print(f"\n  정책 모델을 찾을 수 없습니다: {policy_path}")
        sys.exit(1)
    if not os.path.exists(value_path):
        print(f"\n  가치 모델을 찾을 수 없습니다: {value_path}")
        sys.exit(1)

    # 네트워크 타입 자동 감지
    if args.network_type == 'auto':
        args.network_type = detect_network_type(policy_path)
        print(f"  네트워크 타입   : {args.network_type} (자동 감지)")

    # 모델 입력 차원 감지 (환경과 다를 경우 패딩)
    model_input_dim = detect_input_dim(policy_path)

    # 데이터 로드
    print(f"\n데이터 로드 중...")
    env_data_full, training_data_full = load_data(env_data_path, training_data_path)
    print(f"  전체 데이터: {len(env_data_full):,}행, {training_data_full.shape[1]}개 피처")

    use_selector = False
    if args.selector_mode == 'on':
        use_selector = True
    elif args.selector_mode == 'auto':
        use_selector = is_cross_sectional_dataset(env_data_full)

    if use_selector:
        print("  종목선정 모드   : ON (일자별 ETF 선택 + 보유/매도 의사결정)")
        if not args.no_visualize:
            print("  시각화          : selector 모드에서는 지원하지 않아 자동 비활성화")
            args.no_visualize = True

    if args.sequential:
        print("\n순차 백테스트 모드 (연도별 분할)")
        total_len = len(env_data_full)
        all_results = []
        all_histories = []
        all_trade_logs = []
        chunk_idx = 0
        rolling_balance = float(args.initial_balance)

        if use_selector:
            sort_df = env_data_full[['date', 'etf_code']].copy()
            sort_df['etf_code'] = sort_df['etf_code'].astype(str).str.zfill(6)
            sorted_index = sort_df.sort_values(['date', 'etf_code']).index.to_numpy()
            env_data_full = env_data_full.iloc[sorted_index].reset_index(drop=True)
            training_data_full = training_data_full[sorted_index]

        if 'date' not in env_data_full.columns:
            print("  date 컬럼이 없어 연도별 백테스트를 실행할 수 없습니다.")
            sys.exit(1)

        chunk_ranges = []
        date_str = env_data_full['date'].astype(str)
        year_series = date_str.str.slice(0, 4)
        year_values = year_series.to_numpy()
        start_row = 0
        for i in range(1, total_len):
            if year_values[i] != year_values[i - 1]:
                chunk_ranges.append({
                    'start': start_row,
                    'end': i,
                    'date_start': date_str.iloc[start_row],
                    'date_end': date_str.iloc[i - 1],
                    'year': year_values[i - 1],
                })
                start_row = i
        if total_len > 0:
            chunk_ranges.append({
                'start': start_row,
                'end': total_len,
                'date_start': date_str.iloc[start_row],
                'date_end': date_str.iloc[total_len - 1],
                'year': year_values[total_len - 1],
            })

        for info in chunk_ranges:
            start = info['start']
            end = info['end']
            if end - start < 30:
                break

            chunk_idx += 1
            env_data_chunk = env_data_full.iloc[start:end].reset_index(drop=True)
            training_data_chunk = training_data_full[start:end]

            print(f"\n{'─' * 50}")
            print(f"  청크 {chunk_idx}: [{start:,} ~ {end:,}] ({end - start:,}일)")
            if info['date_start'] is not None and info['date_end'] is not None:
                print(f"  날짜 범위: [{info['date_start']} ~ {info['date_end']}]")
            if info['year'] is not None:
                print(f"  연도      : {info['year']}")
            print(f"{'─' * 50}")

            if use_selector:
                base_input_dim = training_data_chunk.shape[1]
                net_input_dim = model_input_dim if model_input_dim else base_input_dim
            else:
                env = EnvClass(
                    env_data=env_data_chunk,
                    training_data=training_data_chunk,
                    initial_balance=rolling_balance,
                    trading_fee=args.trading_fee,
                    trading_tax=args.trading_tax,
                    slippage=args.slippage,
                    action_scale=args.action_scale,
                    hold_threshold=args.hold_threshold,
                )
                input_dim = env.num_features
                net_input_dim = model_input_dim if model_input_dim and model_input_dim > input_dim else input_dim

            policy_net, value_net = create_networks(
                net_input_dim, args.network_type, device,
                d_model=train_config.get('d_model', 64),
                n_blocks=train_config.get('n_blocks', 2),
                d_state=train_config.get('d_state', 16),
                min_concentration=train_config.get('min_concentration', 1.5),
                policy_dropout=train_config.get('policy_dropout', 0.15),
                value_dropout=train_config.get('value_dropout', 0.15),
            )
            agent = TradingAgent(
                policy_network=policy_net,
                value_network=value_net,
                device=device,
                use_lstm=(args.network_type == 'lstm'),
            )
            agent.load(policy_path, value_path)
            agent.policy_net.eval()
            agent.value_net.eval()

            if use_selector:
                stats, history, _, policy_outputs, values, chunk_trade_log = run_selector_backtest(
                    env_data=env_data_chunk,
                    training_data=training_data_chunk,
                    agent=agent,
                    trading_method=trading_method,
                    initial_balance=rolling_balance,
                    trading_fee=args.trading_fee,
                    trading_tax=args.trading_tax,
                    slippage=args.slippage,
                    hold_threshold=args.hold_threshold,
                    model_input_dim=model_input_dim,
                    min_value=args.min_value,
                    min_concentration=args.min_concentration,
                    max_buy_per_day=args.max_buy_per_day,
                    max_holdings=args.max_holdings,
                    stop_loss_pct=args.stop_loss_pct,
                    drawdown_reduce_pct=args.drawdown_reduce_pct,
                    drawdown_pause_pct=args.drawdown_pause_pct,
                    sell_threshold=args.sell_threshold,
                    trailing_stop_pct=args.trailing_stop_pct,
                    max_exposure=args.max_exposure,
                    vol_target=args.vol_target,
                    min_hold_days=args.min_hold_days,
                )
                for h in history:
                    h_copy = dict(h)
                    h_copy['tick'] += start
                    all_histories.append(h_copy)
                all_trade_logs.extend(chunk_trade_log)
            else:
                pad_dim = max(0, model_input_dim - input_dim) if model_input_dim else 0
                policy_outputs, values = run_backtest(env, agent, pad_dim=pad_dim, min_value=args.min_value, min_concentration=args.min_concentration)
                stats = compute_metrics(env, env_data_chunk, trading_method)
                for h in env.history:
                    h_copy = dict(h)
                    h_copy['tick'] += start
                    all_histories.append(h_copy)
                all_trade_logs.extend(env.trade_log)

            chunk_initial = rolling_balance
            rolling_balance = float(stats['portfolio_value'])
            stats['chunk'] = chunk_idx
            stats['start_step'] = start
            stats['end_step'] = end
            stats['segment_year'] = info['year']
            stats['date_start'] = info['date_start']
            stats['date_end'] = info['date_end']
            stats['chunk_initial_balance'] = chunk_initial
            stats['cumulative_profit'] = rolling_balance - args.initial_balance
            stats['cumulative_profit_rate'] = ((rolling_balance - args.initial_balance) / (args.initial_balance + 1e-8)) * 100.0
            all_results.append(stats)

            print_report(stats, trading_method, args.dataset, f"[{start:,} ~ {end:,}]")

            if not args.no_visualize and not use_selector:
                visualizer = TradingVisualizer(save_dir=output_dir)
                visualizer.plot_episode(
                    episode=0,
                    env_data=env_data_chunk,
                    history=env.history,
                    policy_outputs=policy_outputs,
                    values=values,
                    initial_balance=args.initial_balance,
                    filename=f'backtest_chunk_{chunk_idx}.html',
                )

        if all_results:
            print("\n" + "=" * 70)
            print("  전체 순차 백테스트 요약")
            print("=" * 70)
            total_profit = rolling_balance - args.initial_balance
            total_fee = sum(r['total_fee_paid'] for r in all_results)
            total_buys = sum(r['num_buy'] for r in all_results)
            total_holds = sum(r['num_hold'] for r in all_results)

            print(f"\n  총 청크 수       : {len(all_results)}")
            print(f"  총 손익          : {total_profit:>+15,.0f} 원")
            print(f"  총 수수료        : {total_fee:>15,.0f} 원")
            print(f"  최종 자산        : {rolling_balance:>15,.0f} 원")
            if trading_method == 'swing':
                total_rebal = sum(r.get('num_rebalance', r['num_buy']) for r in all_results)
                print(f"  총 리밸런싱      : {total_rebal}")
            else:
                print(f"  총 매수/관망     : {total_buys} / {total_holds}")

            table_data = []
            for r in all_results:
                seg = f"[{r['start_step']:,}~{r['end_step']:,}]"
                if trading_method == 'swing':
                    last_col = f"{r.get('avg_position_ratio', 0):.0%}"
                else:
                    last_col = f"{r['num_buy']}/{r['num_hold']}"
                table_data.append([
                    r.get('segment_year', '-'),
                    f"{r['profit_rate']:+.4f}%",
                    f"{r.get('max_drawdown', 0):.4f}%",
                    f"{r.get('sharpe_ratio', 0):.2f}",
                    f"{r.get('calmar_ratio', 0):.2f}",
                    f"{r.get('win_rate', 0):.0f}%",
                    last_col,
                    seg,
                ])

            n = len(all_results)
            avg_profit = sum(r['profit_rate'] for r in all_results) / n
            avg_mdd = sum(r.get('max_drawdown', 0) for r in all_results) / n
            avg_wr = sum(r.get('win_rate', 0) for r in all_results) / n
            avg_sharpe = sum(r.get('sharpe_ratio', 0) for r in all_results) / n
            avg_calmar = sum(r.get('calmar_ratio', 0) for r in all_results) / n

            if trading_method == 'swing':
                avg_last = f"{sum(r.get('avg_position_ratio', 0) for r in all_results) / n:.0%}"
            else:
                avg_last = f"{total_buys}/{total_holds}"

            last_header = '주식비율' if trading_method == 'swing' else '매수/관망'
            table_data.append(['─' * 4] * 8)
            table_data.append([
                '평균',
                f"{avg_profit:+.4f}%",
                f"{avg_mdd:.4f}%",
                f"{avg_sharpe:.2f}",
                f"{avg_calmar:.2f}",
                f"{avg_wr:.0f}%",
                avg_last,
                f'전체 {n}구간',
            ])
            headers = ['연도', '수익률', 'MDD', 'Sharpe', 'Calmar', '승률', last_header, '구간']
            print(tabulate(table_data, headers=headers, stralign='right', numalign='right'))

            summary_path = os.path.join(output_dir, 'backtest_summary.json')
            with open(summary_path, 'w') as f:
                json.dump({
                    'dataset': args.dataset,
                    'trading_method': trading_method,
                    'policy_model': policy_path,
                    'timestamp': timestamp,
                    'total_profit': total_profit,
                    'total_fee': total_fee,
                    'chunks': all_results,
                }, f, indent=2, default=str)
            print(f"\n  결과 저장: {summary_path}")

            history_path = os.path.join(output_dir, 'backtest_history.csv')
            pd.DataFrame(all_histories).to_csv(history_path, index=False)
            print(f"  히스토리 저장: {history_path}")

            if all_trade_logs:
                trade_log_path = os.path.join(output_dir, 'trade_log.csv')
                pd.DataFrame(all_trade_logs).to_csv(trade_log_path, index=False)
                print(f"  매매 로그 저장: {trade_log_path}")

    else:
        # 단일 백테스트
        total_len = len(env_data_full)
        start = args.start_step
        end = min(start + args.max_steps, total_len) if args.max_steps > 0 else total_len

        env_data = env_data_full.iloc[start:end].reset_index(drop=True)
        training_data = training_data_full[start:end]

        if start > 0 or end < total_len:
            print(f"  슬라이싱: [{start:,} ~ {end:,}] ({total_len:,} → {len(env_data):,})")

        if len(env_data) < 30:
            print(f"  데이터가 너무 짧습니다: {len(env_data)}행")
            sys.exit(1)

        if use_selector:
            # date/etf_code 기준 정렬
            sort_df = env_data[['date', 'etf_code']].copy()
            sort_df['etf_code'] = sort_df['etf_code'].astype(str).str.zfill(6)
            sorted_index = sort_df.sort_values(['date', 'etf_code']).index.to_numpy()
            env_data = env_data.iloc[sorted_index].reset_index(drop=True)
            training_data = training_data[sorted_index]
            print(f"  거래일: {env_data['date'].nunique():,}일 (멀티 ETF)")
            base_input_dim = training_data.shape[1]
            net_input_dim = model_input_dim if model_input_dim else base_input_dim
        else:
            env = EnvClass(
                env_data=env_data,
                training_data=training_data,
                initial_balance=args.initial_balance,
                trading_fee=args.trading_fee,
                trading_tax=args.trading_tax,
                slippage=args.slippage,
                action_scale=args.action_scale,
                hold_threshold=args.hold_threshold,
            )
            print(f"  거래일: {env.total_ticks:,}일")
            input_dim = env.num_features
            net_input_dim = model_input_dim if model_input_dim and model_input_dim > input_dim else input_dim

        policy_net, value_net = create_networks(
                net_input_dim, args.network_type, device,
                d_model=train_config.get('d_model', 64),
                n_blocks=train_config.get('n_blocks', 2),
                d_state=train_config.get('d_state', 16),
                min_concentration=train_config.get('min_concentration', 1.5),
                policy_dropout=train_config.get('policy_dropout', 0.15),
                value_dropout=train_config.get('value_dropout', 0.15),
            )
        agent = TradingAgent(
            policy_network=policy_net,
            value_network=value_net,
            device=device,
            use_lstm=(args.network_type == 'lstm'),
        )
        agent.load(policy_path, value_path)
        agent.policy_net.eval()
        agent.value_net.eval()
        print(f"  모델 로드 완료")

        print(f"\n백테스트 실행 중...")
        if use_selector:
            stats, history, pv_history, policy_outputs, values, selector_trade_log = run_selector_backtest(
                env_data=env_data,
                training_data=training_data,
                agent=agent,
                trading_method=trading_method,
                initial_balance=args.initial_balance,
                trading_fee=args.trading_fee,
                trading_tax=args.trading_tax,
                slippage=args.slippage,
                hold_threshold=args.hold_threshold,
                model_input_dim=model_input_dim,
                min_value=args.min_value,
                min_concentration=args.min_concentration,
                max_buy_per_day=args.max_buy_per_day,
                max_holdings=args.max_holdings,
                stop_loss_pct=args.stop_loss_pct,
                drawdown_reduce_pct=args.drawdown_reduce_pct,
                drawdown_pause_pct=args.drawdown_pause_pct,
                sell_threshold=args.sell_threshold,
                trailing_stop_pct=args.trailing_stop_pct,
                max_exposure=args.max_exposure,
                vol_target=args.vol_target,
                min_hold_days=args.min_hold_days,
                output_dir=output_dir,
            )
        else:
            pad_dim = max(0, model_input_dim - input_dim) if model_input_dim else 0
            policy_outputs, values = run_backtest(env, agent, pad_dim=pad_dim, min_value=args.min_value, min_concentration=args.min_concentration)
            stats = compute_metrics(env, env_data, trading_method)

        print_report(stats, trading_method, args.dataset,
                     f"[{start:,} ~ {end:,}]" if (start > 0 or end < total_len) else "전체")

        # 연도별 통계 및 요약 테이블 출력
        if use_selector and history:
            yearly_stats = compute_yearly_stats(history, selector_trade_log)
            if yearly_stats:
                print("\n" + "=" * 70)
                print("  연도별 성과 요약")
                print("=" * 70)
                total_profit = stats['profit']
                total_fee = stats['total_fee_paid']
                print(f"\n  총 손익          : {total_profit:>+15,.0f} 원")
                print(f"  총 수수료        : {total_fee:>15,.0f} 원")
                print(f"  최종 자산        : {stats['portfolio_value']:>15,.0f} 원")
                total_rebal = stats['num_rebalance']
                print(f"  총 리밸런싱      : {total_rebal}")

                table_data = []
                for ys in yearly_stats:
                    seg = f"[{ys['start_step']:,}~{ys['end_step']:,}]"
                    table_data.append([
                        ys['year'],
                        f"{ys['profit_rate']:+.4f}%",
                        f"{ys['max_drawdown']:.4f}%",
                        f"{ys.get('sharpe_ratio', 0):.2f}",
                        f"{ys.get('calmar_ratio', 0):.2f}",
                        f"{ys['win_rate']:.0f}%",
                        f"{ys['avg_position_ratio']:.0%}",
                        seg,
                    ])
                n = len(yearly_stats)
                avg_pr = sum(y['profit_rate'] for y in yearly_stats) / n
                avg_mdd = sum(y['max_drawdown'] for y in yearly_stats) / n
                avg_wr = sum(y['win_rate'] for y in yearly_stats) / n
                avg_pos = sum(y['avg_position_ratio'] for y in yearly_stats) / n
                avg_sharpe = sum(y.get('sharpe_ratio', 0) for y in yearly_stats) / n
                avg_calmar = sum(y.get('calmar_ratio', 0) for y in yearly_stats) / n

                # compute_yearly_stats 결과는 출력용으로만 사용; CAGR은 전체 기간 복리 기준
                # (stats['cagr']는 위에서 이미 올바르게 계산됨)

                table_data.append(['─' * 4] * 8)
                table_data.append([
                    '평균',
                    f"{avg_pr:+.4f}%",
                    f"{avg_mdd:.4f}%",
                    f"{avg_sharpe:.2f}",
                    f"{avg_calmar:.2f}",
                    f"{avg_wr:.0f}%",
                    f"{avg_pos:.0%}",
                    f'전체 {n}구간',
                ])
                headers = ['연도', '수익률', 'MDD', 'Sharpe', 'Calmar', '승률', '주식비율', '구간']
                print(tabulate(table_data, headers=headers, stralign='right', numalign='right'))

                # backtest_summary.md 작성
                md_path = write_summary_markdown(
                    output_dir, yearly_stats, stats, args.dataset, trading_method, timestamp)
                print(f"\n  마크다운 요약: {md_path}")

                # PV 차트 HTML 생성
                chart_path = write_pv_chart_html(output_dir, history, stats, yearly_stats)
                if chart_path:
                    print(f"  PV 차트: {chart_path}")

        if not args.no_visualize and not use_selector:
            visualizer = TradingVisualizer(save_dir=output_dir)
            html_path = visualizer.plot_episode(
                episode=0,
                env_data=env_data,
                history=env.history,
                policy_outputs=policy_outputs,
                values=values,
                initial_balance=args.initial_balance,
                filename='backtest.html',
            )
            print(f"  시각화 저장: {html_path}")

            import shutil
            monitor_path = os.path.join(args.base_path, 'logs', 'backtest.html')
            if os.path.exists(html_path):
                os.makedirs(os.path.dirname(monitor_path), exist_ok=True)
                shutil.copy2(html_path, monitor_path)

        result_path = os.path.join(output_dir, 'backtest_result.json')
        with open(result_path, 'w') as f:
            json.dump({
                'dataset': args.dataset,
                'trading_method': trading_method,
                'policy_model': policy_path,
                'timestamp': timestamp,
                'start_step': start,
                'end_step': end,
                **stats,
            }, f, indent=2, default=str)
        print(f"  결과 저장: {result_path}")

        # selector 모드: CSV는 run_selector_backtest에서 이미 스트리밍 저장됨
        if not use_selector:
            history_path = os.path.join(output_dir, 'backtest_history.csv')
            pd.DataFrame(env.history).to_csv(history_path, index=False)
            print(f"  히스토리 저장: {history_path}")
            if env.trade_log:
                trade_log_path = os.path.join(output_dir, 'trade_log.csv')
                pd.DataFrame(env.trade_log).to_csv(trade_log_path, index=False)
                print(f"  매매 로그 저장: {trade_log_path}")


if __name__ == '__main__':
    main()
