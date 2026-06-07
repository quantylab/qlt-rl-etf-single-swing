"""
트레이너 모듈: ETF Swing Trading PPO 학습 루프

시초매수→마감매도 전략에 맞춘 트레이너:
  - 멀티 에피소드 버퍼 누적 (update_interval 기반)
  - 승률, Sharpe, 일중 수익률 로깅
  - Entropy 적응 스케줄링
"""
import os
import json
import shutil
import logging
import tempfile
import threading
from datetime import datetime
from typing import Dict, List
import numpy as np
import pandas as pd
from tqdm import tqdm

from environment import DayTradingEnvironment
from agent import TradingAgent
from visualizer import TradingVisualizer
from train_monitor import generate_dashboard


class PPOTrainer:
    """PPO 학습기 (ETF Swing Trading)"""

    def __init__(
        self,
        env: DayTradingEnvironment,
        agent: TradingAgent,
        num_episodes: int = 500,
        max_steps_per_episode: int = None,
        update_interval: int = 128,
        log_dir: str = 'logs',
        output_dir: str = 'output',
        visualize: bool = True,
        viz_interval: int = 10,
        entropy_coef_start: float = 0.05,
        entropy_coef_end: float = 0.01,
        entropy_decay_episodes: int = 300,
        target_bias_low: float = 0.10,
        target_bias_high: float = 0.90,
        trade_rate_threshold: float = 0.15,
        entropy_boost_factor: float = 1.15,
        low_policy_std_threshold: float = 0.08,
        action_mix_start: float = 0.05,
        action_mix_end: float = 0.01,
        val_min_trades: int = 1,
        validation_interval: int = 5,
        early_stop_patience: int = 35,
        early_stop_min_delta: float = 0.1,
        early_stop_warmup_episodes: int = 120,
        val_env: DayTradingEnvironment = None,
        chunk_info: dict = None,
    ):
        self.env = env
        self.agent = agent
        self.val_env = val_env
        self.num_episodes = num_episodes
        self.max_steps_per_episode = max_steps_per_episode or env.num_steps
        self.update_interval = update_interval

        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Entropy 스케줄링
        self.entropy_coef_start = entropy_coef_start
        self.entropy_coef_end = entropy_coef_end
        self.entropy_decay_episodes = entropy_decay_episodes
        self.target_bias_low = target_bias_low
        self.target_bias_high = target_bias_high
        self.trade_rate_threshold = trade_rate_threshold
        self.entropy_boost_factor = entropy_boost_factor
        self.low_policy_std_threshold = low_policy_std_threshold
        self.action_mix_start = action_mix_start
        self.action_mix_end = action_mix_end
        self.val_min_trades = val_min_trades
        self.validation_interval = max(1, validation_interval)
        self.early_stop_patience = max(0, early_stop_patience)
        self.early_stop_min_delta = max(0.0, early_stop_min_delta)
        self.early_stop_warmup_episodes = max(1, early_stop_warmup_episodes)
        self.agent.entropy_coef = entropy_coef_start
        self.agent.action_mix_prob = action_mix_start

        # 시각화
        self.visualize = visualize
        self.viz_interval = viz_interval
        self.visualizer = TradingVisualizer(save_dir=log_dir) if visualize else None
        self._viz_thread: threading.Thread = None
        self._monitor_thread: threading.Thread = None

        self.chunk_info = chunk_info or {}

        self._setup_logger()

        # 에피소드 기록
        self.episode_rewards = []
        self.episode_profits = []
        self.episode_profit_rates = []
        self.episode_num_buys = []
        self.episode_num_holds = []
        self.episode_fees = []
        self.episode_mean_targets = []

        # Best 추적
        self.best_profit_rate = -float('inf')
        self.best_sharpe = -float('inf')
        self.best_val_profit_rate = -float('inf')
        self.best_val_score = -float('inf')
        self.no_improve_count = 0
        self.early_stopped = False
        self.best_episode = 0

        # 멀티 에피소드 버퍼
        self.states_buffer = []
        self.next_states_buffer = []
        self.actions_buffer = []
        self.log_probs_buffer = []
        self.rewards_buffer = []
        self.dones_buffer = []
        self.global_step = 0

        # 에피소드 loss
        self._episode_policy_losses = []
        self._episode_value_losses = []
        self._episode_kl_divs = []

    def _setup_logger(self):
        log_file = os.path.join(self.log_dir, 'episodes.jsonl')
        self.logger = logging.getLogger(f'PPOTrainer_{id(self)}')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []

        fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(fh)

        self.chunk_id = self.chunk_info.get('start_step', 0)

        meta = {
            'type': 'meta',
            'trading_method': 'swing',
            'action_type': 'continuous',
            'num_episodes': self.num_episodes,
            'update_interval': self.update_interval,
            'total_days': self.env.total_ticks,
            'initial_balance': self.env.initial_balance,
        }
        if self.chunk_info:
            meta.update(self.chunk_info)
        self.logger.info(json.dumps(meta))

    def _log_json(self, data: dict):
        data['timestamp'] = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        data['chunk_id'] = self.chunk_id
        self.logger.info(json.dumps(data, ensure_ascii=False))

    def train(self):
        estimated_updates = (self.num_episodes * self.env.num_steps) // self.update_interval + 1
        self.agent.setup_lr_scheduler(estimated_updates)

        etf_code = self.chunk_info.get('etf_code', '')
        etf_name = self.chunk_info.get('etf_name', '')
        desc = f"{etf_code} {etf_name}".strip() or "Training"
        pbar = tqdm(
            range(1, self.num_episodes + 1),
            desc=desc,
            ncols=150,
        )

        for episode in pbar:
            state = self.env.reset()
            episode_reward = 0.0
            done = False
            step = 0

            self.agent.reset_stats()
            self._episode_policy_losses = []
            self._episode_value_losses = []
            self._episode_kl_divs = []

            policy_outputs_list = []
            values_list = []

            while not done and step < self.max_steps_per_episode:
                action, log_prob, policy_output = self.agent.get_action(
                    state, training=True
                )
                value = self.agent.get_value(state)

                policy_outputs_list.append(policy_output)
                values_list.append(value)

                next_state, reward, done, info = self.env.step(action)

                self.states_buffer.append(state)
                self.next_states_buffer.append(next_state)
                self.actions_buffer.append(action)
                self.log_probs_buffer.append(log_prob)
                self.rewards_buffer.append(reward)
                self.dones_buffer.append(done)

                state = next_state
                episode_reward += reward
                step += 1
                self.global_step += 1

                if len(self.states_buffer) >= self.update_interval:
                    self._update_networks()

            # 에피소드 경계에서 done 마킹
            if self.dones_buffer and not self.dones_buffer[-1]:
                self.dones_buffer[-1] = True

            env_stats = self.env.get_stats()
            agent_stats = self.agent.get_policy_stats()

            self.episode_rewards.append(episode_reward)
            self.episode_profits.append(env_stats['profit'])
            self.episode_profit_rates.append(env_stats['profit_rate'])
            self.episode_num_buys.append(env_stats['num_buy'])
            self.episode_num_holds.append(env_stats['num_hold'])
            self.episode_fees.append(env_stats['total_fee_paid'])
            self.episode_mean_targets.append(agent_stats['mean_target_ratio'])

            # 진행률 표시
            prf = env_stats['profit_rate']
            win_rate = env_stats.get('win_rate', 0)
            sharpe = env_stats.get('sharpe_ratio', 0)
            excess = env_stats.get('excess_bnh', 0)
            prf_color = '\033[91m' if prf >= 0 else '\033[94m'
            exc_color = '\033[91m' if excess >= 0 else '\033[94m'
            rst = '\033[0m'

            pbar.set_postfix_str(
                f"B={env_stats['num_buy']} S={env_stats['num_sell']} H={env_stats['num_hold']} "
                f"Pos={agent_stats['mean_target_ratio']:.2f} "
                f"WR={win_rate:.0f}% "
                f"SR={sharpe:.2f} "
                f"{exc_color}Exc={excess:+.1f}%{rst} "
                f"{prf_color}Prf={prf:+.2f}%{rst}"
            )

            # Best: Sharpe ratio 기준
            sharpe = env_stats.get('sharpe_ratio', 0)
            is_best_train = sharpe > self.best_sharpe and env_stats['num_buy'] >= 5
            if is_best_train:
                self.best_sharpe = sharpe
                self.best_profit_rate = env_stats['profit_rate']
                self.best_episode = episode

            # Validation
            val_stats = None
            val_profit_rate = None
            val_score = None
            is_best = is_best_train if self.val_env is None else False
            should_validate = (
                self.val_env is not None
                and (episode == 1 or is_best_train or episode % self.validation_interval == 0)
            )
            if should_validate:
                val_stats = self._validate()
                if val_stats is not None:
                    val_profit_rate = val_stats['profit_rate']
                    val_trade_count = val_stats.get('num_buy', 0)
                    val_stats['trade_count'] = val_trade_count
                    val_stats['is_valid_for_best'] = val_trade_count >= self.val_min_trades
                    if val_stats['is_valid_for_best']:
                        val_score = self._compute_val_score(val_stats)
                        val_stats['score'] = val_score
                        is_best = val_score > (self.best_val_score + self.early_stop_min_delta)
                        if is_best:
                            self.best_val_profit_rate = val_profit_rate
                            self.best_val_score = val_score
                            self.no_improve_count = 0
                        elif episode >= self.early_stop_warmup_episodes:
                            self.no_improve_count += 1
                    else:
                        is_best = False

            self._log_episode(episode, env_stats, agent_stats, episode_reward,
                              is_best, val_profit_rate, val_score, is_best_train, val_stats)

            if is_best:
                self._save_model('best')

            if self.visualize:
                do_viz = episode % self.viz_interval == 0 or episode == self.num_episodes or is_best_train
                if do_viz:
                    self._visualize_episode(episode, policy_outputs_list, values_list,
                                            'last.html', copy_best=is_best_train,
                                            best_filename='best.html')

            if episode % self.viz_interval == 0 or episode == self.num_episodes or is_best_train:
                self._update_monitor()

            self._update_entropy(episode, agent_stats, env_stats, step)

            if (
                self.val_env is not None
                and self.early_stop_patience > 0
                and episode >= self.early_stop_warmup_episodes
                and self.no_improve_count >= self.early_stop_patience
            ):
                self.early_stopped = True
                pbar.write(
                    f"Early stopping: no val-score improvement for {self.no_improve_count} evals "
                    f"(best={self.best_val_score:.3f})"
                )
                break


        pbar.close()

        if len(self.states_buffer) > 0:
            self._update_networks()

        self._save_model('final')
        self._save_training_summary()

    def _update_networks(self):
        if not self.states_buffer:
            return

        states = np.array(self.states_buffer)
        next_states = np.array(self.next_states_buffer)
        actions = np.array(self.actions_buffer, dtype=np.float32)
        probs = np.array(self.log_probs_buffer)
        rewards = np.array(self.rewards_buffer)
        dones = np.array(self.dones_buffer)

        policy_loss, value_loss, kl_div = self.agent.update(
            states, next_states, actions, probs, rewards, dones
        )

        self._episode_policy_losses.append(policy_loss)
        self._episode_value_losses.append(value_loss)
        self._episode_kl_divs.append(kl_div)

        self.states_buffer.clear()
        self.next_states_buffer.clear()
        self.actions_buffer.clear()
        self.log_probs_buffer.clear()
        self.rewards_buffer.clear()
        self.dones_buffer.clear()

    def _validate(self):
        if self.val_env is None:
            return None

        self.agent.policy_net.eval()
        self.agent.value_net.eval()

        saved_ph = self.agent.policy_hidden
        saved_vh = self.agent.value_hidden
        self.agent.policy_hidden = None
        self.agent.value_hidden = None

        state = self.val_env.reset()
        done = False
        step = 0
        max_steps = self.val_env.num_steps

        while not done and step < max_steps:
            action, _, _ = self.agent.get_action(state, training=False)
            state, _, done, _ = self.val_env.step(action)
            step += 1

        val_stats = self.val_env.get_stats()

        self.agent.policy_hidden = saved_ph
        self.agent.value_hidden = saved_vh
        self.agent.policy_net.train()
        self.agent.value_net.train()

        return val_stats

    def _update_entropy(self, episode, agent_stats, env_stats, step):
        """Entropy coefficient 동적 조정"""
        if episode <= self.entropy_decay_episodes:
            progress = episode / self.entropy_decay_episodes
            base_entropy_coef = self.entropy_coef_start + progress * (
                self.entropy_coef_end - self.entropy_coef_start
            )
        else:
            base_entropy_coef = self.entropy_coef_end

        mean_target = agent_stats['mean_target_ratio']
        std_target = agent_stats['std_target_ratio']
        trade_count = env_stats['num_buy']
        trade_rate = trade_count / max(1, step)

        biased = mean_target < self.target_bias_low or mean_target > self.target_bias_high
        over_trading = trade_rate > self.trade_rate_threshold
        under_reactive = std_target < self.low_policy_std_threshold
        collapsed = mean_target < 0.05

        if collapsed:
            self.agent.entropy_coef = self.entropy_coef_start
        elif biased or over_trading or under_reactive:
            self.agent.entropy_coef = min(
                base_entropy_coef * self.entropy_boost_factor,
                self.entropy_coef_start,
            )
        else:
            self.agent.entropy_coef = base_entropy_coef

        # Action mixing 스케줄
        if episode <= self.entropy_decay_episodes:
            mix_progress = episode / self.entropy_decay_episodes
            base_mix = self.action_mix_start + mix_progress * (
                self.action_mix_end - self.action_mix_start
            )
        else:
            base_mix = self.action_mix_end

        if collapsed:
            self.agent.action_mix_prob = self.action_mix_start
        elif under_reactive:
            self.agent.action_mix_prob = min(
                max(base_mix * 1.5, self.action_mix_end),
                self.action_mix_start,
            )
        else:
            self.agent.action_mix_prob = base_mix

    def _compute_val_score(self, val_stats: dict) -> float:
        profit_rate = float(val_stats.get('profit_rate', 0.0))
        excess_bnh = float(val_stats.get('excess_bnh', 0.0))
        sharpe = float(val_stats.get('sharpe_ratio', 0.0))
        max_drawdown = float(val_stats.get('max_drawdown', 0.0))
        drawdown_penalty = max(0.0, -max_drawdown) * 0.2
        return profit_rate + (0.5 * excess_bnh) + (2.0 * sharpe) - drawdown_penalty

    def _log_episode(
        self, episode, env_stats, agent_stats, episode_reward,
        is_best=False, val_profit_rate=None, val_score=None, is_best_train=False, val_stats=None,
    ):
        avg_pl = round(np.mean(self._episode_policy_losses), 6) if self._episode_policy_losses else 0.0
        avg_vl = round(np.mean(self._episode_value_losses), 6) if self._episode_value_losses else 0.0
        avg_kl = round(np.mean(self._episode_kl_divs), 6) if self._episode_kl_divs else 0.0

        lr_p = self.agent.optimizer_policy.param_groups[0]['lr']
        lr_v = self.agent.optimizer_value.param_groups[0]['lr']

        log_data = {
            'type': 'episode',
            'episode': episode,
            'is_best': bool(is_best),
            'is_best_train': bool(is_best_train),
            'entropy_coef': round(self.agent.entropy_coef, 4),
            'lr_policy': round(lr_p, 6),
            'lr_value': round(lr_v, 6),
            'loss': {'policy': avg_pl, 'value': avg_vl, 'kl_div': avg_kl},
            'policy': {
                'mean_target_ratio': round(agent_stats['mean_target_ratio'], 4),
                'std_target_ratio': round(agent_stats['std_target_ratio'], 4),
            },
            'train': {
                'buy': env_stats['num_buy'],
                'sell': env_stats['num_sell'],
                'hold': env_stats['num_hold'],
                'total_fee_paid': round(env_stats['total_fee_paid'], 2),
                'initial_balance': env_stats['initial_balance'],
                'portfolio_value': round(env_stats['portfolio_value'], 2),
                'profit': round(env_stats['profit'], 2),
                'profit_rate': round(env_stats['profit_rate'], 4),
                'bnh_return': round(env_stats.get('bnh_return', 0), 4),
                'excess_bnh': round(env_stats.get('excess_bnh', 0), 4),
                'max_drawdown': round(env_stats.get('max_drawdown', 0), 4),
                'max_draw_up': round(env_stats.get('max_draw_up', 0), 4),
                'win_rate': round(env_stats.get('win_rate', 0), 2),
                'sharpe_ratio': round(env_stats.get('sharpe_ratio', 0), 4),
            },
            'reward': round(episode_reward, 6),
        }
        if val_stats is not None:
            log_data['val'] = {
                'buy': val_stats.get('num_buy', 0),
                'hold': val_stats.get('num_hold', 0),
                'profit_rate': round(val_stats.get('profit_rate', 0.0), 4),
                'bnh_return': round(val_stats.get('bnh_return', 0.0), 4),
                'excess_bnh': round(val_stats.get('excess_bnh', 0.0), 4),
                'max_drawdown': round(val_stats.get('max_drawdown', 0.0), 4),
                'max_draw_up': round(val_stats.get('max_draw_up', 0.0), 4),
                'win_rate': round(val_stats.get('win_rate', 0.0), 2),
                'trade_count': int(val_stats.get('trade_count', 0)),
                'is_valid_for_best': bool(val_stats.get('is_valid_for_best', False)),
                'score': round(val_score, 4) if val_score is not None else None,
            }
        self._log_json(log_data)

    def _visualize_episode(self, episode, policy_outputs, values, filename,
                            copy_best: bool = False, best_filename: str = 'best.html'):
        history = self.env.history
        if not history:
            return

        history_snap = list(history)
        env_data_snap = self.env.env_data
        initial_balance = self.env.initial_balance
        policy_outputs_snap = list(policy_outputs)
        values_snap = list(values)
        log_dir = self.log_dir
        visualizer = self.visualizer

        chunk_info_snap = dict(self.chunk_info)

        def _run():
            visualizer.plot_episode(
                episode=episode,
                env_data=env_data_snap,
                history=history_snap,
                policy_outputs=policy_outputs_snap,
                values=values_snap,
                initial_balance=initial_balance,
                filename=filename,
                chunk_info=chunk_info_snap,
            )
            if copy_best:
                src_path = os.path.join(log_dir, filename)
                best_path = os.path.join(log_dir, best_filename)
                if os.path.exists(src_path):
                    shutil.copy2(src_path, best_path)

        if self._viz_thread is not None and self._viz_thread.is_alive():
            self._viz_thread.join()

        self._viz_thread = threading.Thread(target=_run, daemon=True)
        self._viz_thread.start()

    def _update_monitor(self):
        """train_monitor.html을 백그라운드 스레드에서 갱신"""
        log_dir = self.log_dir

        def _run():
            try:
                generate_dashboard(log_dir)
            except Exception:
                pass

        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            return  # 이전 갱신이 아직 실행 중이면 건너뜀

        self._monitor_thread = threading.Thread(target=_run, daemon=True)
        self._monitor_thread.start()

    def _save_model(self, name: str):
        policy_path = os.path.join(self.output_dir, f'policy_{name}.pt')
        value_path = os.path.join(self.output_dir, f'value_{name}.pt')
        self.agent.save(policy_path, value_path)

    def _save_training_summary(self):
        df = pd.DataFrame({
            'episode': range(1, len(self.episode_rewards) + 1),
            'reward': self.episode_rewards,
            'profit': self.episode_profits,
            'profit_rate': self.episode_profit_rates,
            'num_buy': self.episode_num_buys,
            'num_hold': self.episode_num_holds,
            'total_fee': self.episode_fees,
            'mean_target': self.episode_mean_targets,
        })

        summary = {
            'type': 'summary',
            'mean_profit_rate': round(df['profit_rate'].mean(), 4),
            'max_profit_rate': round(df['profit_rate'].max(), 4),
            'min_profit_rate': round(df['profit_rate'].min(), 4),
            'std_profit_rate': round(df['profit_rate'].std(), 4),
            'best_profit_rate': round(self.best_profit_rate, 4),
            'best_val_profit_rate': round(self.best_val_profit_rate, 4) if self.best_val_profit_rate > -float('inf') else None,
            'best_val_score': round(self.best_val_score, 4) if self.best_val_score > -float('inf') else None,
            'early_stopped': self.early_stopped,
            'no_improve_count': self.no_improve_count,
        }
        self._log_json(summary)
        self._append_summary_log(df)

    def _append_summary_log(self, df: pd.DataFrame):
        """chunk 학습 결과를 logs/chunks.jsonl에 한 줄 추가"""
        summary_log_path = os.path.join(self.log_dir, 'chunks.jsonl')
        os.makedirs(os.path.dirname(summary_log_path), exist_ok=True)

        val_prf = self.best_val_profit_rate if self.best_val_profit_rate > -float('inf') else None

        chunk_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
            'type': 'chunk',
            'iteration': self.chunk_info.get('iteration_name', os.path.basename(self.log_dir)),
            'start_date': self.chunk_info.get('start_date'),
            'end_date': self.chunk_info.get('end_date'),
            'etf_code': self.chunk_info.get('etf_code'),
            'total_days': self.env.total_ticks,
            'episodes': len(df),
            'best_episode': self.best_episode,
            'best_profit_rate': round(self.best_profit_rate, 4),
            'best_sharpe': round(self.best_sharpe, 4),
            'mean_profit_rate': round(float(df['profit_rate'].mean()), 4),
            'last_profit_rate': round(float(df['profit_rate'].iloc[-1]), 4) if len(df) > 0 else 0,
            'avg_buys': round(float(df['num_buy'].mean()), 1),
            'best_val_profit_rate': round(val_prf, 4) if val_prf is not None else None,
        }

        with open(summary_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(chunk_data, ensure_ascii=False) + '\n')
