"""
에이전트 모듈: 연속 행동 트레이딩 에이전트 (PPO + Beta 분포)

v3 개선:
  - 보상 정규화: 간소화 (running std만 사용, std clip 범위 조정)
  - GAE: 올바른 bootstrapping (마지막 상태 가치 사용)
  - 코사인 LR 스케줄링 지원
  - 코드 구조 정리
"""
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple


class RunningStd:
    """보상 정규화를 위한 온라인 표준편차 (Welford's algorithm)"""

    def __init__(self):
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4

    def update(self, x: np.ndarray):
        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = len(x)
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        self.mean += delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        self.var = M2 / tot_count
        self.count = tot_count

    @property
    def std(self) -> float:
        return max(np.sqrt(self.var), 0.1)  # 최소 0.1 보장


class TradingAgent:
    """암호화폐 트레이딩 에이전트 (연속 행동 PPO)

    행동: Beta(alpha, beta)에서 샘플링한 목표 포지션 비율 ∈ [0, 1]
    """

    def __init__(
        self,
        policy_network,
        value_network,
        lr_policy: float = 0.0002,
        lr_value: float = 0.0005,
        gamma: float = 0.995,
        epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        reward_clip: float = 5.0,
        action_mix_prob: float = 0.05,
        concentration_target: float = 4.0,
        concentration_penalty_coef: float = 0.01,
        policy_weight_decay: float = 1e-5,
        value_weight_decay: float = 1e-5,
        device: str = 'cpu',
        use_lstm: bool = False,
    ):
        self.device = device
        self.policy_net = policy_network.to(device)
        self.value_net = value_network.to(device)
        self.use_lstm = use_lstm

        self.policy_hidden = None
        self.value_hidden = None

        self.optimizer_policy = torch.optim.Adam(
            self.policy_net.parameters(), lr=lr_policy,
            weight_decay=policy_weight_decay, eps=1e-5,
        )
        self.optimizer_value = torch.optim.Adam(
            self.value_net.parameters(), lr=lr_value,
            weight_decay=value_weight_decay, eps=1e-5,
        )

        # 코사인 LR 스케줄러 (외부에서 total_steps 설정 후 활성화)
        self.scheduler_policy = None
        self.scheduler_value = None

        self.gamma = gamma
        self.epsilon = epsilon
        self.entropy_coef = entropy_coef
        self.reward_clip = reward_clip
        self.action_mix_prob = action_mix_prob
        self.concentration_target = concentration_target
        self.concentration_penalty_coef = concentration_penalty_coef

        self.reward_normalizer = RunningStd()

        self.reset_stats()

    def setup_lr_scheduler(self, total_updates: int):
        """코사인 LR 스케줄러 설정"""
        self.scheduler_policy = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_policy, T_max=max(total_updates, 1), eta_min=2e-5
        )
        self.scheduler_value = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_value, T_max=max(total_updates, 1), eta_min=5e-5
        )

    # ──────────────────── 통계 ────────────────────

    def reset_stats(self):
        self.target_ratios = []
        self.large_policy_update_count = 0
        if self.use_lstm:
            self.policy_hidden = None
            self.value_hidden = None

    def get_policy_stats(self) -> dict:
        ratios = np.array(self.target_ratios) if self.target_ratios else np.array([0.5])
        return {
            'mean_target_ratio': float(ratios.mean()),
            'std_target_ratio': float(ratios.std()),
            'min_target_ratio': float(ratios.min()),
            'max_target_ratio': float(ratios.max()),
            'large_policy_update_count': self.large_policy_update_count,
        }

    # ──────────────────── 행동 선택 ────────────────────

    def get_action(
        self,
        state: np.ndarray,
        training: bool = True,
    ) -> Tuple[float, float, np.ndarray]:
        """Beta 분포에서 행동 샘플링

        Returns:
            action: 목표 포지션 비율 ∈ [0, 1]
            log_prob: 로그 확률 (학습 시)
            policy_output: [alpha, beta]
        """
        state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
        state_tensor = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            if self.use_lstm:
                if self.policy_hidden is not None:
                    self.policy_hidden = tuple(h.detach() for h in self.policy_hidden)
                alpha, beta_param, self.policy_hidden = self.policy_net(
                    state_tensor, self.policy_hidden
                )
            else:
                alpha, beta_param = self.policy_net(state_tensor)

            dist = torch.distributions.Beta(alpha, beta_param)

            if training:
                action = dist.sample()
                if np.random.rand() < self.action_mix_prob:
                    action = torch.rand_like(action)
                action = torch.clamp(action, 1e-6, 1 - 1e-6)
                log_prob = dist.log_prob(action).item()
            else:
                action = alpha / (alpha + beta_param)  # mean
                log_prob = None

        action_val = float(action.item())
        policy_output = np.array([alpha.item(), beta_param.item()], dtype=np.float32)
        self.target_ratios.append(action_val)

        return action_val, log_prob, policy_output

    def get_value(self, state: np.ndarray) -> float:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.use_lstm:
                if self.value_hidden is not None:
                    self.value_hidden = tuple(h.detach() for h in self.value_hidden)
                value, self.value_hidden = self.value_net(state_tensor, self.value_hidden)
            else:
                value = self.value_net(state_tensor)
        return value.cpu().item()

    # ──────────────────── 업데이트 ────────────────────

    def update(
        self,
        states: np.ndarray,
        next_states: np.ndarray,
        actions: np.ndarray,
        old_log_probs: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        num_epochs: int = 4,
        batch_size: int = 32,
        target_kl: float = 0.015,
        value_loss_coef: float = 0.5,
    ):
        states_tensor = torch.FloatTensor(states).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        actions_tensor = torch.FloatTensor(actions).to(self.device)
        actions_tensor = torch.clamp(actions_tensor, 1e-6, 1 - 1e-6)
        old_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device)

        # 보상 정규화 (std 기반 스케일링, 방향성 보존)
        self.reward_normalizer.update(rewards)
        scaled_rewards = rewards / self.reward_normalizer.std
        scaled_rewards = np.clip(scaled_rewards, -self.reward_clip, self.reward_clip)

        # GAE: transition의 next_state 값을 사용해 rollout 경계 편향 방지
        advantages, returns = self._compute_gae(
            states_tensor, next_states_tensor, scaled_rewards, dones
        )

        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)

        with torch.no_grad():
            if self.use_lstm:
                old_values = self._get_values_sequence(states_tensor)
            else:
                old_values = self.value_net(states_tensor).squeeze(-1)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_kl_div = 0.0
        num_updates = 0

        for epoch in range(num_epochs):
            if self.use_lstm:
                policy_loss, kl_div = self._update_policy_sequence(
                    states_tensor, actions_tensor, old_probs_tensor, advantages_tensor
                )
                value_loss = self._update_value_sequence(
                    states_tensor, returns_tensor, old_values
                )
                total_policy_loss += policy_loss
                total_value_loss += value_loss
                total_kl_div += kl_div
                num_updates += 1
                if kl_div > target_kl:
                    break
            else:
                indices = np.arange(len(states))
                np.random.shuffle(indices)
                epoch_kl = 0.0
                epoch_updates = 0

                for start in range(0, len(states), batch_size):
                    batch_idx = indices[start:start + batch_size]
                    policy_loss, kl_div = self._update_policy(
                        states_tensor[batch_idx],
                        actions_tensor[batch_idx],
                        old_probs_tensor[batch_idx],
                        advantages_tensor[batch_idx],
                    )
                    value_loss = self._update_value(
                        states_tensor[batch_idx],
                        returns_tensor[batch_idx],
                        old_values[batch_idx],
                    )
                    total_policy_loss += policy_loss
                    total_value_loss += value_loss
                    total_kl_div += kl_div
                    num_updates += 1
                    epoch_kl += kl_div
                    epoch_updates += 1

                if epoch_updates > 0 and epoch_kl / epoch_updates > target_kl:
                    break

        if num_updates > 0 and total_kl_div / num_updates > 0.05:
            self.large_policy_update_count += 1

        # KL 기반 학습률 적응 (스케줄러 없을 때만)
        if self.scheduler_policy is None:
            avg_kl = total_kl_div / max(num_updates, 1)
            if avg_kl > target_kl * 2.0:
                for pg in self.optimizer_policy.param_groups:
                    pg['lr'] = max(pg['lr'] * 0.5, 1e-5)
            elif avg_kl < target_kl * 0.5:
                for pg in self.optimizer_policy.param_groups:
                    pg['lr'] = min(pg['lr'] * 1.5, 1e-3)

        # 스케줄러 step
        if self.scheduler_policy is not None:
            self.scheduler_policy.step()
        if self.scheduler_value is not None:
            self.scheduler_value.step()

        return (
            total_policy_loss / max(num_updates, 1),
            total_value_loss / max(num_updates, 1),
            total_kl_div / max(num_updates, 1),
        )

    # ──────────────────── GAE ────────────────────

    def _compute_gae(
        self,
        states_tensor: torch.Tensor,
        next_states_tensor: torch.Tensor,
        rewards: np.ndarray,
        dones: np.ndarray,
        gae_lambda: float = 0.95,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generalized Advantage Estimation (올바른 bootstrapping)"""
        with torch.no_grad():
            if self.use_lstm:
                values = self._get_values_sequence(states_tensor).cpu().numpy()
                next_values = self._get_values_sequence(next_states_tensor).cpu().numpy()
            else:
                values = self.value_net(states_tensor).cpu().numpy().flatten()
                next_values = self.value_net(next_states_tensor).cpu().numpy().flatten()

        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(T)):
            next_value = 0.0 if dones[t] else next_values[t]

            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * gae_lambda * (0.0 if dones[t] else 1.0) * gae
            advantages[t] = gae

        returns = advantages + values

        # Advantage normalization
        if T > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    # ──────────────────── LSTM 시퀀스 처리 ────────────────────

    def _get_values_sequence(self, states_tensor):
        x = states_tensor.unsqueeze(0)
        lstm_out, _ = self.value_net.lstm(x, None)
        vals = self.value_net.fc_layers(lstm_out.squeeze(0))
        return torch.clamp(vals.squeeze(-1), -100.0, 100.0)

    def _update_policy_sequence(self, states, actions, old_log_probs, advantages):
        x = states.unsqueeze(0)
        lstm_out, _ = self.policy_net.lstm(x, None)
        feat = self.policy_net.fc(lstm_out.squeeze(0))
        min_conc = float(getattr(self.policy_net, 'min_concentration', 1.2))
        alpha = (F.softplus(self.policy_net.alpha_head(feat)) + min_conc).squeeze(-1)
        beta_param = (F.softplus(self.policy_net.beta_head(feat)) + min_conc).squeeze(-1)

        dist = torch.distributions.Beta(alpha, beta_param)
        new_log_probs = dist.log_prob(actions)

        log_ratio = torch.clamp(new_log_probs - old_log_probs, -5.0, 5.0)
        ratio = torch.clamp(torch.exp(log_ratio), 0.0, 10.0)

        with torch.no_grad():
            kl_div = (ratio - 1.0 - log_ratio).mean().item()

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        entropy = dist.entropy().mean()
        concentration = alpha + beta_param
        concentration_penalty = torch.relu(
            concentration - self.concentration_target
        ).mean()

        loss = (
            policy_loss
            - self.entropy_coef * entropy
            + self.concentration_penalty_coef * concentration_penalty
        )

        self.optimizer_policy.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=0.5)
        self.optimizer_policy.step()

        return policy_loss.item(), kl_div

    def _update_value_sequence(self, states, returns, old_values=None):
        x = states.unsqueeze(0)
        lstm_out, _ = self.value_net.lstm(x, None)
        vals = self.value_net.fc_layers(lstm_out.squeeze(0))
        values = torch.clamp(vals.squeeze(-1), -100.0, 100.0)

        value_loss = F.smooth_l1_loss(values, returns)

        self.optimizer_value.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=0.5)
        self.optimizer_value.step()

        return value_loss.item()

    # ──────────────────── MLP 미니배치 처리 ────────────────────

    def _update_policy(self, states, actions, old_log_probs, advantages):
        if self.use_lstm:
            alpha, beta_param, _ = self.policy_net(states, None)
        else:
            alpha, beta_param = self.policy_net(states)

        dist = torch.distributions.Beta(alpha, beta_param)
        new_log_probs = dist.log_prob(actions)

        log_ratio = torch.clamp(new_log_probs - old_log_probs, -5.0, 5.0)
        ratio = torch.clamp(torch.exp(log_ratio), 0.0, 10.0)

        with torch.no_grad():
            kl_div = (ratio - 1.0 - log_ratio).mean().item()

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        entropy = dist.entropy().mean()
        concentration = alpha + beta_param
        concentration_penalty = torch.relu(
            concentration - self.concentration_target
        ).mean()

        loss = (
            policy_loss
            - self.entropy_coef * entropy
            + self.concentration_penalty_coef * concentration_penalty
        )

        self.optimizer_policy.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=0.5)
        self.optimizer_policy.step()

        return policy_loss.item(), kl_div

    def _update_value(self, states, returns, old_values=None):
        if self.use_lstm:
            values, _ = self.value_net(states, None)
        else:
            values = self.value_net(states)
        values = values.squeeze(-1)

        value_loss = F.smooth_l1_loss(values, returns)

        self.optimizer_value.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=0.5)
        self.optimizer_value.step()

        return value_loss.item()

    # ──────────────────── 저장/로드 ────────────────────

    def save(self, policy_path: str, value_path: str):
        if hasattr(self.policy_net, 'lstm'):
            input_dim = self.policy_net.lstm.input_size
        elif hasattr(self.policy_net, 'input_proj'):
            input_dim = self.policy_net.input_proj[0].in_features
        elif hasattr(self.policy_net, 'feature_layers'):
            input_dim = self.policy_net.feature_layers[0].in_features
        else:
            input_dim = None

        torch.save({
            'state_dict': self.policy_net.state_dict(),
            'input_dim': input_dim,
            'network_type': 'lstm' if self.use_lstm else 'standard',
        }, policy_path)

        torch.save({
            'state_dict': self.value_net.state_dict(),
            'input_dim': input_dim,
            'network_type': 'lstm' if self.use_lstm else 'standard',
        }, value_path)

    def load(self, policy_path: str, value_path: str, strict: bool = True):
        policy_checkpoint = torch.load(policy_path, map_location=self.device, weights_only=False)
        value_checkpoint = torch.load(value_path, map_location=self.device, weights_only=False)

        if isinstance(policy_checkpoint, dict) and 'state_dict' in policy_checkpoint:
            policy_state = policy_checkpoint['state_dict']
            saved_input_dim = policy_checkpoint.get('input_dim')
            saved_network_type = policy_checkpoint.get('network_type')
        else:
            policy_state = policy_checkpoint
            saved_input_dim = None
            saved_network_type = None

        if isinstance(value_checkpoint, dict) and 'state_dict' in value_checkpoint:
            value_state = value_checkpoint['state_dict']
        else:
            value_state = value_checkpoint

        # 입력 차원 검증
        if saved_input_dim is None and 'lstm.weight_ih_l0' in policy_state:
            saved_input_dim = policy_state['lstm.weight_ih_l0'].shape[1]

        if hasattr(self.policy_net, 'lstm'):
            current_input_dim = self.policy_net.lstm.input_size
        elif hasattr(self.policy_net, 'input_proj'):
            current_input_dim = self.policy_net.input_proj[0].in_features
        elif hasattr(self.policy_net, 'feature_layers'):
            current_input_dim = self.policy_net.feature_layers[0].in_features
        else:
            current_input_dim = None

        if saved_input_dim and current_input_dim and saved_input_dim != current_input_dim:
            raise ValueError(
                f"\n입력 차원 불일치: 저장={saved_input_dim}, 현재={current_input_dim}\n"
                f"동일한 환경 설정을 사용하거나 새로 학습하세요."
            )

        if saved_network_type:
            current_type = 'lstm' if self.use_lstm else 'standard'
            if saved_network_type != current_type:
                raise ValueError(
                    f"\n네트워크 타입 불일치: 저장={saved_network_type}, 현재={current_type}"
                )

        self.policy_net.load_state_dict(policy_state, strict=strict)
        self.value_net.load_state_dict(value_state, strict=strict)
