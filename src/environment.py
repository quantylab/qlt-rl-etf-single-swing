"""
환경 모듈: Trading 환경

두 가지 투자 방식 지원:
  1. Day Trading (시초매수 → 마감매도)
     - 매일 시초가에 매수, 장 마감 전에 매도
     - 야간 포지션 없음 (매일 청산)
  2. Swing Trading (포지션 유지)
     - 종가 기준 포지션 비율 조정
     - 포지션을 여러 날에 걸쳐 유지 가능

공통:
  - 액션: 목표 포지션 비율 ∈ [0, 1] (0=현금, 1=전액 투입)
  - 1 step = 1 거래일
  - 보상: 수익률 기반 - 수수료
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict
from target_etfs import TARGET_ETFS


class DayTradingEnvironment:
    """Day Trading 환경

    행동: position_size ∈ [0.0, 1.0]
        0.0 = 전액 현금 (미매수), 1.0 = 전액 매수
    관찰: 전일 종가 후 데이터 (look-ahead 없음)
    거래: 당일 시초가 매수 → 종가 매도 (일중 완결)
    """

    PORTFOLIO_FEATURE_NUM = 5

    def __init__(
        self,
        env_data: pd.DataFrame,
        training_data: np.ndarray,
        initial_balance: float = 10_000_000.0,
        trading_fee: float = 0.00015,      # 거래 수수료 (온라인 ~0.015%)
        trading_tax: float = 0.0,           # 거래세 (일반 주식: 0.002, ETF: 0)
        slippage: float = 0.0003,           # 시초가/종가 슬리피지
        min_trading_price: float = 10000.0,
        action_scale: float = 1.0,
        reward_clip: float = 5.0,
        reward_scale: float = 30.0,         # 수익률 보상 배율
        fee_penalty_scale: float = 15.0,    # 수수료 패널티 배율
        reward_terminal_scale: float = 30.0,   # 터미널 보상 배율
        inaction_penalty: float = 10.0,     # 미매수 보상/패널티 배율 (대칭: 손실회피+기회비용)
        hold_threshold: float = 0.1,        # 포지션 비율 이하 미매수 (0.1=10% 이하 hold)
    ):
        self.env_data = env_data.reset_index(drop=True)
        self.training_data = training_data
        self.initial_balance = initial_balance
        self.trading_fee = trading_fee
        self.trading_tax = trading_tax
        self.slippage = slippage
        self.min_trading_price = min_trading_price
        self.action_scale = action_scale
        self.reward_clip = reward_clip
        self.reward_scale = reward_scale
        self.fee_penalty_scale = fee_penalty_scale
        self.reward_terminal_scale = reward_terminal_scale
        self.inaction_penalty = inaction_penalty
        self.hold_threshold = hold_threshold

        assert len(self.env_data) == len(self.training_data), \
            "환경 데이터와 학습 데이터의 길이가 일치하지 않습니다."

        self.total_ticks = len(self.env_data)
        self.num_steps = self.total_ticks - 1  # 첫날은 피처 확보용 (전일 데이터 기반 판단)
        self.num_features = training_data.shape[1] + self.PORTFOLIO_FEATURE_NUM

        self.reset()
        self.history = []

    # ──────────────────── 리셋 ────────────────────

    def reset(self) -> np.ndarray:
        """환경 초기화"""
        self.tick = 1  # 0일차 피처를 보고 1일차부터 거래 (전일 데이터 기반)
        self.balance = self.initial_balance
        self.portfolio_value = self.initial_balance
        self.peak_portfolio_value = self.initial_balance

        # 통계
        self.num_buy = 0          # 매수 실행 횟수
        self.num_hold = 0         # 미매수 (현금 보유) 횟수
        self.num_sell = 0         # 매도 횟수 (= num_buy, 매일 청산)
        self.total_trade_amount = 0.0
        self.total_fee_paid = 0.0

        # 히스토리
        self.pv_history = [self.initial_balance]
        self.history = []
        self.trade_log = []  # 개별 매매 로그

        # 연속 승/패 추적
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.daily_returns = []

        return self._get_state()

    # ──────────────────── 상태 ────────────────────

    def _get_state(self) -> np.ndarray:
        """현재 거래일 기준 상태 반환

        포트폴리오 상태 피처 (5개):
          1. 누적 수익률 (정규화)
          2. 최대 낙폭 (드로다운)
          3. 최근 5일 승률
          4. 연속 승/패 수
          5. 변동성 (최근 수익률 std)
        """
        features = self.training_data[self.tick - 1].copy()  # 전일 피처 (look-ahead 방지)

        # 1. 누적 수익률
        cumulative_return = (self.portfolio_value - self.initial_balance) / self.initial_balance
        cumulative_return_scaled = np.clip(cumulative_return * 20.0, -3.0, 3.0)

        # 2. 최대 낙폭
        drawdown = (self.portfolio_value - self.peak_portfolio_value) / (self.peak_portfolio_value + 1e-8)
        drawdown_scaled = np.clip(drawdown * 50.0, -3.0, 0.0)

        # 3. 최근 5일 승률
        if len(self.daily_returns) >= 5:
            recent_wins = sum(1 for r in self.daily_returns[-5:] if r > 0) / 5.0
        elif len(self.daily_returns) > 0:
            recent_wins = sum(1 for r in self.daily_returns if r > 0) / len(self.daily_returns)
        else:
            recent_wins = 0.5
        win_rate_scaled = (recent_wins - 0.5) * 4.0  # [-2, 2]

        # 4. 연속 승/패
        streak = self.consecutive_wins - self.consecutive_losses
        streak_scaled = np.clip(streak / 5.0, -2.0, 2.0)

        # 5. 최근 변동성
        if len(self.daily_returns) >= 5:
            vol = np.std(self.daily_returns[-20:]) if len(self.daily_returns) >= 20 else np.std(self.daily_returns[-5:])
        else:
            vol = 0.01
        vol_scaled = np.clip(vol * 100.0, 0.0, 3.0)

        state = np.concatenate([
            features,
            [cumulative_return_scaled, drawdown_scaled, win_rate_scaled,
             streak_scaled, vol_scaled]
        ])
        state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
        return state.astype(np.float32)

    # ──────────────────── 스텝 ────────────────────

    def step(self, position_size: float) -> Tuple[np.ndarray, float, bool, Dict]:
        """한 거래일 실행: 시초가에 position_size만큼 매수 → 종가에 전량 매도

        Args:
            position_size: 자본 투입 비율 ∈ [0, 1]

        Returns:
            next_state, reward, done, info
        """
        position_size = float(np.clip(position_size, 0.0, 1.0))

        # 액션 스케일링
        if self.action_scale != 1.0:
            position_size = 0.5 + (position_size - 0.5) * self.action_scale
            position_size = float(np.clip(position_size, 0.0, 1.0))

        # 당일 OHLCV
        open_price = self.env_data.loc[self.tick, 'open']
        close_price = self.env_data.loc[self.tick, 'close']

        prev_pv = self.portfolio_value

        # 일중 수익률 (시가→종가)
        intraday_return = (close_price - open_price) / (open_price + 1e-8)

        # ── 매매 실행 ──
        traded = False
        trade_amount = 0.0
        fee_paid = 0.0
        realized_pnl = 0.0

        invest_amount = self.balance * position_size
        if position_size > self.hold_threshold and invest_amount >= self.min_trading_price:
            # 시초가 매수 (슬리피지 적용)
            buy_price = open_price * (1 + self.slippage)
            buy_fee = invest_amount * self.trading_fee
            actual_invest = invest_amount - buy_fee
            num_shares = int(actual_invest / buy_price)
            if num_shares <= 0:
                self.num_hold += 1
            else:
                # 정수 주 기준으로 실제 투자금 재계산
                actual_invest = num_shares * buy_price
                invest_amount = actual_invest + actual_invest * self.trading_fee
                buy_fee = invest_amount - actual_invest

                # 종가 매도 (슬리피지 적용)
                sell_price = close_price * (1 - self.slippage)
                sell_revenue = num_shares * sell_price
                sell_fee = sell_revenue * self.trading_fee
                sell_tax = sell_revenue * self.trading_tax

                net_revenue = sell_revenue - sell_fee - sell_tax
                realized_pnl = net_revenue - invest_amount
                fee_paid = buy_fee + sell_fee + sell_tax
                trade_amount = invest_amount + sell_revenue

                self.balance += realized_pnl
                self.num_buy += 1
                self.num_sell += 1
                self.total_trade_amount += trade_amount
                self.total_fee_paid += fee_paid
                traded = True
                sell_return = (sell_price - buy_price) / (buy_price + 1e-8) * 100
                date = self.env_data.loc[self.tick, 'date'] if 'date' in self.env_data.columns else self.tick
                code = self.env_data.loc[self.tick, 'etf_code'] if 'etf_code' in self.env_data.columns else ''
                self.trade_log.append({
                    'date': date, 'action': 'BUY', 'code': code,
                    'name': TARGET_ETFS.get(str(code).zfill(6), ''),
                    'shares': num_shares, 'price': buy_price,
                    'amount': invest_amount, 'fee': buy_fee,
                    'balance': self.balance - realized_pnl,
                    'portfolio_value': self.portfolio_value,
                    'sell_return_pct': 0.0,
                })
                self.trade_log.append({
                    'date': date, 'action': 'SELL', 'code': code,
                    'name': TARGET_ETFS.get(str(code).zfill(6), ''),
                    'shares': num_shares, 'price': sell_price,
                    'amount': sell_revenue, 'fee': sell_fee + sell_tax,
                    'balance': self.balance,
                    'portfolio_value': self.balance,
                    'sell_return_pct': sell_return,
                })
        else:
            self.num_hold += 1

        # ── 포트폴리오 가치 갱신 ──
        self.portfolio_value = self.balance
        self.peak_portfolio_value = max(self.peak_portfolio_value, self.portfolio_value)
        self.pv_history.append(self.portfolio_value)

        # ── 일중 수익률 기록 ──
        day_return = (self.portfolio_value - prev_pv) / (prev_pv + 1e-8)
        self.daily_returns.append(day_return)

        if day_return > 0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        elif day_return < 0:
            self.consecutive_losses += 1
            self.consecutive_wins = 0

        # ── 보상 설계 ──
        reward = 0.0

        if traded:
            # 1. 수익률 보상 (핵심: 일중 수익률 × 포지션 크기)
            net_return = realized_pnl / (invest_amount + 1e-8)
            reward += net_return * self.reward_scale

            # 2. 수수료 패널티 (과도한 소규모 거래 억제)
            fee_ratio = fee_paid / (prev_pv + 1e-8)
            reward -= fee_ratio * self.fee_penalty_scale
        else:
            # 미매수 시: 대칭 보상 (양의 시장=기회비용 패널티, 음의 시장=손실회피 보상)
            if self.inaction_penalty > 0:
                reward -= intraday_return * self.inaction_penalty

        # ── 틱 전진 ──
        self.tick += 1
        done = self.tick >= self.total_ticks

        # ── 터미널 보상 ──
        if done:
            episode_return = (self.portfolio_value - self.initial_balance) / self.initial_balance
            # Sharpe 기반 터미널 보상 (단순 수익률 대비 위험 조정)
            if len(self.daily_returns) > 5:
                dr = np.array(self.daily_returns)
                sharpe_sign = dr.mean() / (dr.std() + 1e-8)
                terminal_bonus = float(np.clip(
                    (episode_return + sharpe_sign * 0.1) * self.reward_terminal_scale,
                    -3.0, 3.0
                ))
            else:
                terminal_bonus = float(np.clip(
                    episode_return * self.reward_terminal_scale, -3.0, 3.0
                ))
            reward += terminal_bonus

        reward = float(np.clip(reward, -self.reward_clip, self.reward_clip))

        # ── 히스토리 ──
        self.history.append({
            'step': self.tick - 1,
            'tick': self.tick - 1,
            'position_size': position_size,
            'traded': traded,
            'open_price': open_price,
            'close_price': close_price,
            'intraday_return': intraday_return,
            'realized_pnl': realized_pnl,
            'fee_paid': fee_paid,
            'balance': self.balance,
            'portfolio_value': self.portfolio_value,
            'prev_portfolio_value': prev_pv,
            'reward': reward,
        })

        next_state = self._get_state() if not done else np.zeros(self.num_features, dtype=np.float32)

        info = {
            'portfolio_value': self.portfolio_value,
            'balance': self.balance,
            'position_size': position_size,
            'traded': traded,
            'intraday_return': intraday_return,
        }
        return next_state, reward, done, info

    # ──────────────────── 통계 ────────────────────

    def get_stats(self) -> Dict:
        profit = self.portfolio_value - self.initial_balance
        profit_rate = (profit / self.initial_balance) * 100

        # Buy & Hold 벤치마크 (매일 시초가 매수→종가 매도 반복)
        opens = self.env_data['open'].values
        closes = self.env_data['close'].values
        intraday_returns = (closes - opens) / (opens + 1e-8)
        bnh_cumulative = float(np.prod(1 + intraday_returns) - 1) * 100
        excess_bnh = profit_rate - bnh_cumulative

        # Max Drawdown
        pv_arr = np.array(self.pv_history)
        peak_arr = np.maximum.accumulate(pv_arr)
        drawdowns = (pv_arr - peak_arr) / (peak_arr + 1e-8)
        max_drawdown = float(drawdowns.min()) * 100

        # Max Draw Up (MDU): 최저점→최고점 최대 상승률, 최저점 t < 최고점 t
        trough_arr = np.minimum.accumulate(pv_arr)
        draw_ups = (pv_arr - trough_arr) / (trough_arr + 1e-8)
        max_draw_up = float(draw_ups.max()) * 100

        # 승률
        if self.daily_returns:
            traded_returns = [r for r in self.daily_returns if abs(r) > 1e-10]
            win_rate = sum(1 for r in traded_returns if r > 0) / max(len(traded_returns), 1) * 100
            avg_win = np.mean([r for r in traded_returns if r > 0]) * 100 if any(r > 0 for r in traded_returns) else 0
            avg_loss = np.mean([r for r in traded_returns if r < 0]) * 100 if any(r < 0 for r in traded_returns) else 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0

        # Sharpe Ratio (일간 기준 연율화)
        if len(self.daily_returns) > 1:
            dr = np.array(self.daily_returns)
            sharpe = float(dr.mean() / (dr.std() + 1e-8) * np.sqrt(252))
        else:
            sharpe = 0.0

        return {
            'num_buy': self.num_buy,
            'num_sell': self.num_sell,
            'num_hold': self.num_hold,
            'initial_balance': self.initial_balance,
            'portfolio_value': self.portfolio_value,
            'profit': profit,
            'profit_rate': profit_rate,
            'total_trade_amount': self.total_trade_amount,
            'total_fee_paid': self.total_fee_paid,
            'bnh_return': bnh_cumulative,
            'excess_bnh': excess_bnh,
            'max_drawdown': max_drawdown,
            'max_draw_up': max_draw_up,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'sharpe_ratio': sharpe,
        }


class SwingTradingEnvironment:
    """Swing Trading 환경

    행동: target_ratio ∈ [0.0, 1.0]
        0.0 = 전액 현금, 1.0 = 전액 주식
    관찰: 전일 종가 후 데이터 (look-ahead 없음)
    거래: 당일 시가 기준 포지션 비율 조정 (매수/매도/유지)
    평가: 당일 종가 기준 포트폴리오 가치
    포지션: 여러 날에 걸쳐 유지 가능
    """

    PORTFOLIO_FEATURE_NUM = 6  # 5 + current_position_ratio

    def __init__(
        self,
        env_data: pd.DataFrame,
        training_data: np.ndarray,
        initial_balance: float = 10_000_000.0,
        trading_fee: float = 0.00015,
        trading_tax: float = 0.0,
        slippage: float = 0.0003,
        min_trading_price: float = 10000.0,
        action_scale: float = 1.0,
        reward_clip: float = 5.0,
        reward_scale: float = 30.0,
        fee_penalty_scale: float = 15.0,
        reward_terminal_scale: float = 30.0,
        inaction_penalty: float = 0.0,      # swing에선 기본 비활성
        hold_threshold: float = 0.05,       # 비율 변화 5% 이하 → 리밸런싱 안 함
        # ── 리스크 인식 보상 파라미터 ──
        drawdown_penalty_scale: float = 0.0,  # 낙폭 패널티 강도 (0=비활성)
        drawdown_penalty_threshold: float = 0.15,  # 패널티 시작 MDD 수준
        rolling_sharpe_window: int = 20,  # 롤링 Sharpe 윈도우
        rolling_sharpe_scale: float = 0.0,  # 롤링 Sharpe 보너스 강도 (0=비활성)
        loss_aversion: float = 1.0,  # 손실 비대칭 배율 (>1: 손실 패널티 강화)
    ):
        self.env_data = env_data.reset_index(drop=True)
        self.training_data = training_data
        self.initial_balance = initial_balance
        self.trading_fee = trading_fee
        self.trading_tax = trading_tax
        self.slippage = slippage
        self.min_trading_price = min_trading_price
        self.action_scale = action_scale
        self.reward_clip = reward_clip
        self.reward_scale = reward_scale
        self.fee_penalty_scale = fee_penalty_scale
        self.reward_terminal_scale = reward_terminal_scale
        self.inaction_penalty = inaction_penalty
        self.hold_threshold = hold_threshold
        self.drawdown_penalty_scale = drawdown_penalty_scale
        self.drawdown_penalty_threshold = drawdown_penalty_threshold
        self.rolling_sharpe_window = rolling_sharpe_window
        self.rolling_sharpe_scale = rolling_sharpe_scale
        self.loss_aversion = loss_aversion

        assert len(self.env_data) == len(self.training_data), \
            "환경 데이터와 학습 데이터의 길이가 일치하지 않습니다."

        self.total_ticks = len(self.env_data)
        self.num_steps = self.total_ticks - 1  # 첫날은 피처 확보용 (전일 데이터 기반 판단)
        self.num_features = training_data.shape[1] + self.PORTFOLIO_FEATURE_NUM
        # ETF 전환 경계 사전 계산 (다중 ETF 연결 데이터용)
        self._etf_boundaries = set()
        if 'etf_code' in self.env_data.columns:
            codes = self.env_data['etf_code'].values
            for i in range(1, len(codes)):
                if codes[i] != codes[i - 1]:
                    self._etf_boundaries.add(i)
        self.reset()
        self.history = []

    # ──────────────────── 리셋 ────────────────────

    def reset(self) -> np.ndarray:
        self.tick = 1  # 0일차 피처를 보고 1일차부터 거래 (전일 데이터 기반)
        self.balance = self.initial_balance
        self.num_shares = 0              # 보유 주식 수 (정수)
        self.avg_buy_price = 0.0       # 평균 매수가
        self.portfolio_value = self.initial_balance
        self.peak_portfolio_value = self.initial_balance
        self.accumulated_profit = 0.0  # ETF 전환 시 누적 수익

        # 통계
        self.num_buy = 0
        self.num_sell = 0
        self.num_hold = 0
        self.total_trade_amount = 0.0
        self.total_fee_paid = 0.0

        # 히스토리
        self.pv_history = [self.initial_balance]
        self.history = []
        self.trade_log = []  # 개별 매매 로그

        # 연속 승/패 추적
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.daily_returns = []

        return self._get_state()

    # ──────────────────── 상태 ────────────────────

    def _get_position_ratio(self, price: float) -> float:
        """현재 주식 비율 계산"""
        stock_value = self.num_shares * price
        pv = self.balance + stock_value
        return stock_value / (pv + 1e-8) if pv > 0 else 0.0

    def _get_state(self) -> np.ndarray:
        """포트폴리오 상태 피처 (6개):
          1. 누적 수익률
          2. 최대 낙폭
          3. 최근 5일 승률
          4. 연속 승/패 수
          5. 변동성
          6. 현재 포지션 비율
        """
        features = self.training_data[self.tick - 1].copy()  # 전일 피처 (look-ahead 방지)

        cumulative_return = (self.portfolio_value - self.initial_balance) / self.initial_balance
        cumulative_return_scaled = np.clip(cumulative_return * 20.0, -3.0, 3.0)

        drawdown = (self.portfolio_value - self.peak_portfolio_value) / (self.peak_portfolio_value + 1e-8)
        drawdown_scaled = np.clip(drawdown * 50.0, -3.0, 0.0)

        if len(self.daily_returns) >= 5:
            recent_wins = sum(1 for r in self.daily_returns[-5:] if r > 0) / 5.0
        elif len(self.daily_returns) > 0:
            recent_wins = sum(1 for r in self.daily_returns if r > 0) / len(self.daily_returns)
        else:
            recent_wins = 0.5
        win_rate_scaled = (recent_wins - 0.5) * 4.0

        streak = self.consecutive_wins - self.consecutive_losses
        streak_scaled = np.clip(streak / 5.0, -2.0, 2.0)

        if len(self.daily_returns) >= 5:
            vol = np.std(self.daily_returns[-20:]) if len(self.daily_returns) >= 20 else np.std(self.daily_returns[-5:])
        else:
            vol = 0.01
        vol_scaled = np.clip(vol * 100.0, 0.0, 3.0)

        # 현재 포지션 비율 (전일 종가 기준)
        close_price = self.env_data.loc[self.tick - 1, 'close']
        pos_ratio = self._get_position_ratio(close_price)
        pos_scaled = pos_ratio * 2.0 - 1.0  # [-1, 1]

        state = np.concatenate([
            features,
            [cumulative_return_scaled, drawdown_scaled, win_rate_scaled,
             streak_scaled, vol_scaled, pos_scaled]
        ])
        state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
        return state.astype(np.float32)

    # ──────────────────── 스텝 ────────────────────

    def step(self, target_ratio: float) -> Tuple[np.ndarray, float, bool, Dict]:
        """한 거래일 실행: 시가 기준 리밸런싱 → 종가 기준 평가

        전일 데이터 기반 의사결정 → 당일 시가에 매매 실행
        포트폴리오 가치는 종가 기준으로 평가

        Args:
            target_ratio: 목표 주식 비율 ∈ [0, 1]

        Returns:
            next_state, reward, done, info
        """
        target_ratio = float(np.clip(target_ratio, 0.0, 1.0))

        if self.action_scale != 1.0:
            target_ratio = 0.5 + (target_ratio - 0.5) * self.action_scale
            target_ratio = float(np.clip(target_ratio, 0.0, 1.0))

        # ETF 전환 시 청산 + 잔고 리셋 (ETF 간 복리 누적 방지)
        if self.tick in self._etf_boundaries:
            if self.num_shares > 0:
                prev_close = self.env_data.loc[self.tick - 1, 'close']
                sell_price = prev_close * (1 - self.slippage)
                sell_revenue = self.num_shares * sell_price
                sell_fee = sell_revenue * self.trading_fee
                sell_tax = sell_revenue * self.trading_tax
                sell_return = (sell_price - self.avg_buy_price) / (self.avg_buy_price + 1e-8) * 100 if self.avg_buy_price > 0 else 0.0
                date = self.env_data.loc[self.tick - 1, 'date'] if 'date' in self.env_data.columns else self.tick - 1
                code = self.env_data.loc[self.tick - 1, 'etf_code'] if 'etf_code' in self.env_data.columns else ''
                self.trade_log.append({
                    'date': date, 'action': 'SELL', 'code': code,
                    'name': TARGET_ETFS.get(str(code).zfill(6), ''),
                    'shares': self.num_shares, 'price': sell_price,
                    'amount': sell_revenue, 'fee': sell_fee + sell_tax,
                    'balance': self.balance + sell_revenue - sell_fee - sell_tax,
                    'portfolio_value': self.balance + sell_revenue - sell_fee - sell_tax,
                    'sell_return_pct': sell_return,
                })
                self.balance += sell_revenue - sell_fee - sell_tax
                self.total_fee_paid += sell_fee + sell_tax
                self.num_shares = 0.0
                self.avg_buy_price = 0.0
            # 수익 누적 후 잔고 리셋
            self.accumulated_profit += self.balance - self.initial_balance
            self.balance = self.initial_balance
            self.portfolio_value = self.initial_balance
            self.peak_portfolio_value = max(self.peak_portfolio_value, self.portfolio_value)

        open_price = self.env_data.loc[self.tick, 'open']
        close_price = self.env_data.loc[self.tick, 'close']

        # 시가 기준 포트폴리오 (리밸런싱 전)
        stock_value_at_open = self.num_shares * open_price
        prev_pv = self.balance + stock_value_at_open

        # 시가 기준 현재 포지션 비율
        current_ratio = stock_value_at_open / (prev_pv + 1e-8) if prev_pv > 0 else 0.0

        # 비율 차이 계산
        ratio_diff = target_ratio - current_ratio

        # ── 시가 기준 리밸런싱 ──
        traded = False
        fee_paid = 0.0
        trade_amount = 0.0

        if abs(ratio_diff) > self.hold_threshold:
            target_stock_value = prev_pv * target_ratio

            if ratio_diff > 0:
                # 매수: 추가 주식 필요
                buy_value = target_stock_value - stock_value_at_open
                if buy_value >= self.min_trading_price and buy_value <= self.balance:
                    buy_price = open_price * (1 + self.slippage)
                    buy_fee = buy_value * self.trading_fee
                    actual_buy = buy_value - buy_fee
                    new_shares = int(actual_buy / buy_price)
                    if new_shares > 0:
                        # 정수 주 기준으로 실제 투자금 재계산
                        actual_buy = new_shares * buy_price
                        buy_value = actual_buy + actual_buy * self.trading_fee
                        buy_fee = buy_value - actual_buy

                        # 평균 매수가 갱신
                        total_cost = self.avg_buy_price * self.num_shares + buy_value
                        self.num_shares += new_shares
                        self.avg_buy_price = total_cost / (self.num_shares + 1e-8)

                        self.balance -= buy_value
                        fee_paid = buy_fee
                        trade_amount = buy_value
                        self.num_buy += 1
                        traded = True
                        date = self.env_data.loc[self.tick, 'date'] if 'date' in self.env_data.columns else self.tick
                        code = self.env_data.loc[self.tick, 'etf_code'] if 'etf_code' in self.env_data.columns else ''
                        self.trade_log.append({
                            'date': date, 'action': 'BUY', 'code': code,
                            'name': TARGET_ETFS.get(str(code).zfill(6), ''),
                            'shares': new_shares, 'price': buy_price,
                            'amount': buy_value, 'fee': buy_fee,
                            'balance': self.balance,
                            'portfolio_value': self.balance + self.num_shares * open_price,
                            'sell_return_pct': 0.0,
                        })
            else:
                # 매도: 주식 감소
                sell_stock_value = stock_value_at_open - target_stock_value
                sell_shares = int(sell_stock_value / (open_price + 1e-8))
                sell_shares = min(sell_shares, self.num_shares)

                if sell_shares * open_price >= self.min_trading_price:
                    sell_price = open_price * (1 - self.slippage)
                    sell_revenue = sell_shares * sell_price
                    sell_fee = sell_revenue * self.trading_fee
                    sell_tax = sell_revenue * self.trading_tax

                    net_revenue = sell_revenue - sell_fee - sell_tax
                    self.num_shares -= sell_shares
                    self.balance += net_revenue
                    fee_paid = sell_fee + sell_tax
                    trade_amount = sell_revenue
                    self.num_sell += 1
                    traded = True
                    sell_return = (sell_price - self.avg_buy_price) / (self.avg_buy_price + 1e-8) * 100 if self.avg_buy_price > 0 else 0.0
                    date = self.env_data.loc[self.tick, 'date'] if 'date' in self.env_data.columns else self.tick
                    code = self.env_data.loc[self.tick, 'etf_code'] if 'etf_code' in self.env_data.columns else ''
                    self.trade_log.append({
                        'date': date, 'action': 'SELL', 'code': code,
                        'name': TARGET_ETFS.get(str(code).zfill(6), ''),
                        'shares': sell_shares, 'price': sell_price,
                        'amount': sell_revenue, 'fee': sell_fee + sell_tax,
                        'balance': self.balance,
                        'portfolio_value': self.balance + self.num_shares * open_price,
                        'sell_return_pct': sell_return,
                    })

            if traded:
                self.total_trade_amount += trade_amount
                self.total_fee_paid += fee_paid
        else:
            self.num_hold += 1

        # ── 종가 기준 포트폴리오 가치 갱신 ──
        stock_value = self.num_shares * close_price
        self.portfolio_value = self.balance + stock_value
        self.peak_portfolio_value = max(self.peak_portfolio_value, self.portfolio_value)
        self.pv_history.append(self.portfolio_value)

        # 일간 수익률 (시가→종가)
        intraday_return = (close_price - open_price) / (open_price + 1e-8)

        # ── 일간 수익률 기록 ──
        day_return = (self.portfolio_value - prev_pv) / (prev_pv + 1e-8)
        self.daily_returns.append(day_return)

        if day_return > 0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        elif day_return < 0:
            self.consecutive_losses += 1
            self.consecutive_wins = 0

        # ── 보상 설계 ──
        reward = 0.0

        # 포트폴리오 수익률 기반 보상 (손실 비대칭: loss_aversion > 1 이면 손실 패널티 강화)
        if day_return < 0 and self.loss_aversion > 1.0:
            reward += day_return * self.reward_scale * self.loss_aversion
        else:
            reward += day_return * self.reward_scale

        # 수수료 패널티
        if fee_paid > 0:
            fee_ratio = fee_paid / (prev_pv + 1e-8)
            reward -= fee_ratio * self.fee_penalty_scale

        # ── 포지션 인식 낙폭 패널티: 주식 비중이 높을수록 하락 패널티 증폭 ──
        if self.drawdown_penalty_scale > 0:
            current_dd = (self.portfolio_value - self.peak_portfolio_value) / (self.peak_portfolio_value + 1e-8)
            if current_dd < -self.drawdown_penalty_threshold:
                excess_dd = abs(current_dd) - self.drawdown_penalty_threshold
                # 포지션 비율에 비례한 패널티: 현금이면 1×, 풀투자면 2×
                pos_ratio = self._get_position_ratio(close_price)
                position_multiplier = 1.0 + pos_ratio
                reward -= excess_dd * self.drawdown_penalty_scale * position_multiplier

        # ── 롤링 Sharpe 보너스: 안정적 수익에 보상 ──
        if self.rolling_sharpe_scale > 0 and len(self.daily_returns) >= self.rolling_sharpe_window:
            recent = self.daily_returns[-self.rolling_sharpe_window:]
            r_mean = np.mean(recent)
            r_std = np.std(recent) + 1e-8
            rolling_sharpe = r_mean / r_std
            reward += float(np.clip(rolling_sharpe * self.rolling_sharpe_scale, -1.0, 1.0))

        # ── 틱 전진 ──
        self.tick += 1
        done = self.tick >= self.total_ticks

        # ── 터미널 보상 ──
        if done:
            # 마지막 날 잔여 포지션 청산 (종가 기준)
            if self.num_shares > 0:
                final_price = close_price * (1 - self.slippage)
                sell_revenue = self.num_shares * final_price
                sell_fee = sell_revenue * self.trading_fee
                sell_tax = sell_revenue * self.trading_tax
                sell_return = (final_price - self.avg_buy_price) / (self.avg_buy_price + 1e-8) * 100 if self.avg_buy_price > 0 else 0.0
                date = self.env_data.loc[self.tick - 1, 'date'] if 'date' in self.env_data.columns else self.tick - 1
                code = self.env_data.loc[self.tick - 1, 'etf_code'] if 'etf_code' in self.env_data.columns else ''
                self.trade_log.append({
                    'date': date, 'action': 'SELL', 'code': code,
                    'name': TARGET_ETFS.get(str(code).zfill(6), ''),
                    'shares': self.num_shares, 'price': final_price,
                    'amount': sell_revenue, 'fee': sell_fee + sell_tax,
                    'balance': self.balance + sell_revenue - sell_fee - sell_tax,
                    'portfolio_value': self.balance + sell_revenue - sell_fee - sell_tax,
                    'sell_return_pct': sell_return,
                })
                self.balance += sell_revenue - sell_fee - sell_tax
                self.total_fee_paid += sell_fee + sell_tax
                self.num_shares = 0.0
                self.portfolio_value = self.balance

            episode_profit = self.accumulated_profit + (self.portfolio_value - self.initial_balance)
            episode_return = episode_profit / self.initial_balance

            # CAGR 기반 연환산 수익률: 기간이 길수록 같은 수익에 낮은 보상 → 과적합 억제
            total_days = max(self.total_ticks, 1)
            cagr = float((1.0 + episode_return) ** (252.0 / total_days) - 1.0)

            if len(self.daily_returns) > 5:
                dr = np.array(self.daily_returns)
                sharpe_sign = dr.mean() / (dr.std() + 1e-8)
                sharpe_weight = 0.3 if self.drawdown_penalty_scale > 0 else 0.1

                # 정확한 최대 낙폭 계산 (running peak 기준)
                pv = np.array(self.pv_history)
                peak = np.maximum.accumulate(pv)
                dd_series = (pv - peak) / (peak + 1e-8)
                max_dd = float(abs(dd_series.min()))

                # Calmar 보너스: 가중치 0.05 → 0.3 (MDD 억제 핵심 신호)
                calmar_bonus = 0.0
                if self.drawdown_penalty_scale > 0 and max_dd > 1e-6:
                    calmar = cagr / (max_dd + 1e-8)
                    calmar_bonus = float(np.clip(calmar * 0.3, -1.5, 1.5))

                # PLR 보너스: 평균 수익 > 평균 손실 → PLR 개선 유도
                plr_bonus = 0.0
                wins = [r for r in dr if r > 0]
                losses = [r for r in dr if r < 0]
                if wins and losses:
                    avg_win = float(np.mean(wins))
                    avg_loss = float(abs(np.mean(losses)))
                    plr = avg_win / (avg_loss + 1e-8)
                    plr_bonus = float(np.clip((plr - 1.0) * 0.5, -0.5, 0.5))

                # 터미널 클립 ±3.0 → ±6.0: 고품질 에피소드 구분 공간 확장
                terminal_bonus = float(np.clip(
                    (cagr + sharpe_sign * sharpe_weight + calmar_bonus + plr_bonus) * self.reward_terminal_scale,
                    -6.0, 6.0
                ))
            else:
                terminal_bonus = float(np.clip(
                    cagr * self.reward_terminal_scale, -6.0, 6.0
                ))
            reward += terminal_bonus

        reward = float(np.clip(reward, -self.reward_clip, self.reward_clip))

        # ── 히스토리 ──
        final_ratio = self._get_position_ratio(close_price)
        self.history.append({
            'step': self.tick - 1,
            'tick': self.tick - 1,
            'position_size': target_ratio,
            'traded': traded,
            'open_price': open_price,
            'close_price': close_price,
            'intraday_return': intraday_return,
            'realized_pnl': 0.0,
            'fee_paid': fee_paid,
            'balance': self.balance,
            'portfolio_value': self.portfolio_value,
            'prev_portfolio_value': prev_pv,
            'reward': reward,
            'position_ratio': final_ratio,
            'num_shares': self.num_shares,
        })

        next_state = self._get_state() if not done else np.zeros(self.num_features, dtype=np.float32)

        info = {
            'portfolio_value': self.portfolio_value,
            'balance': self.balance,
            'position_size': target_ratio,
            'traded': traded,
            'intraday_return': intraday_return,
            'position_ratio': final_ratio,
        }
        return next_state, reward, done, info

    # ──────────────────── 통계 ────────────────────

    def get_stats(self) -> Dict:
        profit = self.accumulated_profit + (self.portfolio_value - self.initial_balance)
        num_etf_segments = len(self._etf_boundaries) + 1
        # 평균 per-ETF 수익률 (ETF별 동일 초기자본으로 시작하므로 단순 평균)
        profit_rate = (profit / self.initial_balance) * 100 / num_etf_segments

        # Buy & Hold: ETF별 B&H 수익률 평균
        if 'etf_code' in self.env_data.columns:
            bnh_total = 0.0
            codes = self.env_data['etf_code'].values
            closes = self.env_data['close'].values
            start_idx = 0
            for i in range(1, len(codes)):
                if codes[i] != codes[i - 1]:
                    bnh_total += (closes[i - 1] / closes[start_idx] - 1) * 100
                    start_idx = i
            bnh_total += (closes[-1] / closes[start_idx] - 1) * 100
            bnh_cumulative = float(bnh_total) / num_etf_segments
        else:
            closes = self.env_data['close'].values
            bnh_cumulative = float((closes[-1] / closes[0] - 1) * 100)
        excess_bnh = profit_rate - bnh_cumulative

        # Max Drawdown
        pv_arr = np.array(self.pv_history)
        peak_arr = np.maximum.accumulate(pv_arr)
        drawdowns = (pv_arr - peak_arr) / (peak_arr + 1e-8)
        max_drawdown = float(drawdowns.min()) * 100

        # Max Draw Up
        trough_arr = np.minimum.accumulate(pv_arr)
        draw_ups = (pv_arr - trough_arr) / (trough_arr + 1e-8)
        max_draw_up = float(draw_ups.max()) * 100

        # 승률
        if self.daily_returns:
            traded_returns = [r for r in self.daily_returns if abs(r) > 1e-10]
            win_rate = sum(1 for r in traded_returns if r > 0) / max(len(traded_returns), 1) * 100
            avg_win = np.mean([r for r in traded_returns if r > 0]) * 100 if any(r > 0 for r in traded_returns) else 0
            avg_loss = np.mean([r for r in traded_returns if r < 0]) * 100 if any(r < 0 for r in traded_returns) else 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0

        # Sharpe Ratio (일간 기준 연율화)
        if len(self.daily_returns) > 1:
            dr = np.array(self.daily_returns)
            sharpe = float(dr.mean() / (dr.std() + 1e-8) * np.sqrt(252))
        else:
            sharpe = 0.0

        return {
            'num_buy': self.num_buy,
            'num_sell': self.num_sell,
            'num_hold': self.num_hold,
            'initial_balance': self.initial_balance,
            'portfolio_value': self.portfolio_value,
            'profit': profit,
            'profit_rate': profit_rate,
            'total_trade_amount': self.total_trade_amount,
            'total_fee_paid': self.total_fee_paid,
            'bnh_return': bnh_cumulative,
            'excess_bnh': excess_bnh,
            'max_drawdown': max_drawdown,
            'max_draw_up': max_draw_up,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'sharpe_ratio': sharpe,
        }
