"""
시각화 모듈: ETF Swing Trading 학습/백테스트 시각화

패널 구성 (5개):
  1. 종가 차트 (매수/미매수 배경)
  2. 포지션 크기 (투입 비율)
  3. 정책 분포 (Beta mean + concentration)
  4. 가치 추정
  5. 포트폴리오 가치
"""
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict


class TradingVisualizer:
    """ETF Swing Trading 시각화"""

    def __init__(self, save_dir: str = 'visualizations'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def plot_episode(
        self,
        episode: int,
        env_data: pd.DataFrame,
        history: List[Dict],
        policy_outputs: List[np.ndarray],
        values: List[float],
        initial_balance: float,
        filename: str = None,
        chunk_info: dict = None,
    ):
        """에피소드 시각화"""
        history_df = pd.DataFrame(history)
        policy_array = np.array(policy_outputs)  # [N, 2] → alpha, beta
        ticks = history_df['tick'].values

        num_rows = 5
        row_heights = [0.2, 0.2, 0.2, 0.2, 0.2]

        fig = make_subplots(
            rows=num_rows, cols=1,
            shared_xaxes=True,
            row_heights=row_heights,
            vertical_spacing=0.02,
        )

        # 1. 종가 차트 + 매수/미매수 배경
        self._add_price_chart(fig, env_data, history_df, row=1)
        # 2. 포지션 크기
        self._add_position_chart(fig, history_df, ticks, row=2)
        # 3. 정책 분포
        self._add_policy_chart(fig, policy_array, ticks, row=3)
        # 4. 가치 추정
        self._add_value_chart(fig, values, ticks, row=4)
        # 5. 포트폴리오 가치
        self._add_pv_chart(fig, history_df, initial_balance, ticks, row=5)

        # 기간 표시
        if 'date' in env_data.columns:
            start_date = str(env_data['date'].iloc[0])
            end_date = str(env_data['date'].iloc[-1])
            # YYYYMMDD → YYYY-MM-DD
            if len(start_date) == 8:
                start_date = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
                end_date = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"
            period = f" ({start_date} ~ {end_date})"
        else:
            period = ""
        # 제목 구성
        etf_code = chunk_info.get('etf_code', '') if chunk_info else ''
        etf_name = chunk_info.get('etf_name', '') if chunk_info else ''
        etf_label = f" [{etf_code} {etf_name}]".rstrip('] ') + ']' if etf_code else ''
        title = f"ETF Swing Trading{etf_label}{period} - Episode: {episode}"
        fig.update_layout(
            title_text=title,
            height=900,
            showlegend=False,
            hovermode='x unified',
            margin=dict(t=50, b=30, l=30, r=10),
        )
        fig.update_xaxes(title_text="Date", row=num_rows, col=1)

        if filename is None:
            filename = f'episode_{episode}.html'
        save_path = os.path.join(self.save_dir, filename)
        fig.write_html(save_path, include_plotlyjs='cdn')
        return save_path

    # ──────────────────── 헬퍼 ────────────────────

    @staticmethod
    def _vlines_trace(ticks, y0, y1, color):
        """복수 수직선 trace"""
        xs, ys = [], []
        for t in ticks:
            xs += [t, t, None]
            ys += [y0, y1, None]
        return go.Scatter(
            x=xs, y=ys,
            mode='lines',
            line=dict(color=color, width=1.5),
            hoverinfo='skip',
        )

    def _add_price_chart(self, fig, env_data: pd.DataFrame,
                         history_df: pd.DataFrame, row: int):
        """종가 차트 + 매수일 배경"""
        fig.add_trace(
            go.Scatter(
                x=list(range(len(env_data))),
                y=env_data['close'],
                name='종가',
                line=dict(color='black', width=1),
                hovertemplate='종가: %{y:,.0f}<extra></extra>',
            ),
            row=row, col=1,
        )

        # 시가 (반투명 라인)
        if 'open' in env_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(env_data))),
                    y=env_data['open'],
                    name='시가',
                    line=dict(color='gray', width=0.5, dash='dot'),
                    hovertemplate='시가: %{y:,.0f}<extra></extra>',
                ),
                row=row, col=1,
            )

        fig.update_yaxes(title_text="Price", row=row, col=1)

    def _add_position_chart(self, fig, history_df: pd.DataFrame,
                            ticks: np.ndarray, row: int):
        """포지션 크기 (보유 비율 + 목표 비율)"""
        target_ratio = history_df['position_size'].values

        if 'position_ratio' in history_df.columns:
            position_ratio = history_df['position_ratio'].values
            fig.add_trace(
                go.Scatter(
                    x=ticks, y=position_ratio,
                    name='보유 비율',
                    line=dict(color='black', width=1.5),
                    hovertemplate='보유: %{y:.2%}<extra></extra>',
                ),
                row=row, col=1,
            )
        fig.add_trace(
            go.Scatter(
                x=ticks, y=target_ratio,
                name='목표 비율',
                line=dict(color='rgba(255,80,80,0.8)', width=1),
                hovertemplate='목표: %{y:.2%}<extra></extra>',
            ),
            row=row, col=1,
        )
        fig.add_hline(y=0.5, line=dict(color='gray', width=1, dash='dash'),
                      row=row, col=1)
        fig.update_yaxes(title_text="Position", range=[0, 1], row=row, col=1)

    def _add_policy_chart(self, fig, policy_array: np.ndarray,
                          ticks: np.ndarray, row: int):
        """정책 분포 시각화 (Beta mean + concentration)"""
        alpha = policy_array[:, 0]
        beta = policy_array[:, 1]
        mean = alpha / (alpha + beta)
        concentration = alpha + beta

        fig.add_trace(
            go.Scatter(
                x=ticks, y=mean,
                name='Policy Mean',
                line=dict(color='black', width=2),
                hovertemplate='Mean: %{y:.3f}<extra></extra>',
            ),
            row=row, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=ticks, y=concentration / concentration.max() if concentration.max() > 0 else concentration,
                name='Concentration (norm)',
                line=dict(color='gray', width=1),
                hovertemplate='Conc: %{y:.3f}<extra></extra>',
            ),
            row=row, col=1,
        )
        fig.add_hline(y=0.5, line=dict(color='gray', width=1, dash='dash'),
                      row=row, col=1)
        fig.update_yaxes(title_text="Policy", row=row, col=1)

    def _add_value_chart(self, fig, values: List[float],
                         ticks: np.ndarray, row: int):
        fig.add_trace(
            go.Scatter(
                x=ticks, y=values,
                name='State Value',
                line=dict(color='black', width=2),
                hovertemplate='Value: %{y:.4f}<extra></extra>',
            ),
            row=row, col=1,
        )
        fig.add_hline(y=0, line=dict(color='gray', width=1, dash='dash'),
                      row=row, col=1)
        fig.update_yaxes(title_text="Value", row=row, col=1)

    def _add_pv_chart(self, fig, history_df: pd.DataFrame,
                      initial_balance: float, ticks: np.ndarray, row: int):
        pv = history_df['portfolio_value'].values

        fig.add_trace(
            go.Scatter(
                x=ticks, y=pv,
                name='Portfolio Value',
                line=dict(color='black', width=2),
                hovertemplate='PV: %{y:,.0f}<extra></extra>',
            ),
            row=row, col=1,
        )
        fig.add_hline(y=initial_balance,
                      line=dict(color='gray', width=1, dash='dash'),
                      row=row, col=1)
        fig.update_yaxes(title_text="PV", row=row, col=1)
