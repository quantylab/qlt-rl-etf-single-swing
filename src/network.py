"""
신경망 모듈: 연속 행동 정책 신경망 (Beta 분포)과 가치 신경망

v5 개선:
  - Mamba (Selective State Space Model) 추가 — ICLR 2024
  - 피처를 토큰 시퀀스로 처리, Bi-Mamba로 양방향 피처 상호작용 포착
  - MLP: ResidualBlock 기반 깊은 네트워크 (256dim, 3 blocks)
  - 입력 프로젝션 → Residual Blocks → Head FC 구조
  - LSTM 유지 (하위 호환)
  - 가치 네트워크 출력 범위 확대 (-200, 200)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual FC Block: LayerNorm + ReLU + Linear + Skip"""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return x + self.net(x) * 0.1  # residual scaling


# ═══════════════════════════════════════════════════════
#  연속 행동 정책 신경망 (MLP + ResidualBlock)
# ═══════════════════════════════════════════════════════

class ContinuousPolicyNetwork(nn.Module):
    """연속 행동 정책 신경망 (Input Projection → ResidualBlocks → Head → Beta 분포)

    기존 [128,64,32] 구조 대비:
      - 더 넓은 hidden dim (256) + ResidualBlock으로 표현력 강화
      - Skip connection으로 그래디언트 흐름 개선
      - 미니배치 셔플 가능 → PPO 업데이트 안정화
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256,
                 num_blocks: int = 3, dropout: float = 0.1,
                 min_concentration: float = 1.5):
        super().__init__()
        self.min_concentration = min_concentration

        # 입력 프로젝션: input_dim → hidden_dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Residual Blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )

        # Head: hidden_dim → 64 → alpha/beta
        self.head_fc = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
        )
        self.alpha_head = nn.Linear(64, 1)
        self.beta_head = nn.Linear(64, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        # Input projection
        for m in self.input_proj:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # Head FC
        for m in self.head_fc:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # alpha/beta heads
        for head in [self.alpha_head, self.beta_head]:
            nn.init.orthogonal_(head.weight, gain=0.5)
            nn.init.constant_(head.bias, 0.54)

    def forward(self, x: torch.Tensor):
        h = self.input_proj(x)
        h = self.res_blocks(h)
        feat = self.head_fc(h)
        alpha = F.softplus(self.alpha_head(feat)) + self.min_concentration
        beta = F.softplus(self.beta_head(feat)) + self.min_concentration
        return alpha.squeeze(-1), beta.squeeze(-1)


# ═══════════════════════════════════════════════════════
#  연속 행동 정책 신경망 (LSTM)
# ═══════════════════════════════════════════════════════

class LSTMContinuousPolicyNetwork(nn.Module):
    """LSTM 기반 연속 행동 정책 신경망 (+ Residual FC)"""

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_layers: int = 2, dropout: float = 0.2,
                 min_concentration: float = 1.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.min_concentration = min_concentration

        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        self.alpha_head = nn.Linear(32, 1)
        self.beta_head = nn.Linear(32, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        for name, p in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(p)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(p)
            elif 'bias' in name:
                nn.init.constant_(p, 0)
                n = p.size(0)
                p.data[n // 4:n // 2].fill_(1.)  # forget gate bias

        for m in self.fc:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for head in [self.alpha_head, self.beta_head]:
            nn.init.orthogonal_(head.weight, gain=0.5)
            nn.init.constant_(head.bias, 0.54)

    def forward(self, x: torch.Tensor, hidden=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        lstm_out, hidden = self.lstm(x, hidden)
        last = lstm_out[:, -1, :]
        feat = self.fc(last)

        alpha = F.softplus(self.alpha_head(feat)) + self.min_concentration
        beta = F.softplus(self.beta_head(feat)) + self.min_concentration
        return alpha.squeeze(-1), beta.squeeze(-1), hidden


# ═══════════════════════════════════════════════════════
#  가치 신경망 (MLP + ResidualBlock)
# ═══════════════════════════════════════════════════════

class ValueNetwork(nn.Module):
    """가치 신경망 (Critic, MLP + ResidualBlocks)"""

    def __init__(self, input_dim: int, hidden_dim: int = 256,
                 num_blocks: int = 3, dropout: float = 0.1):
        super().__init__()

        # 입력 프로젝션
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Residual Blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )

        # Head
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.input_proj:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # Head의 마지막 Linear만 작은 gain
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.res_blocks(h)
        return torch.clamp(self.head(h), -50.0, 50.0)


# ═══════════════════════════════════════════════════════
#  가치 신경망 (LSTM)
# ═══════════════════════════════════════════════════════

class LSTMValueNetwork(nn.Module):
    """LSTM 기반 가치 신경망"""

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for name, p in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(p)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(p)
            elif 'bias' in name:
                nn.init.constant_(p, 0)
                n = p.size(0)
                p.data[n // 4:n // 2].fill_(1.)

        for m in self.fc_layers:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, hidden=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        lstm_out, hidden = self.lstm(x, hidden)
        last = lstm_out[:, -1, :]
        value = self.fc_layers(last)
        return torch.clamp(value, -50.0, 50.0), hidden


# ═══════════════════════════════════════════════════════
#  GRN (Gated Residual Network) — from TFT
# ═══════════════════════════════════════════════════════

class GRNBlock(nn.Module):
    """Gated Residual Network block
    GRN(x) = LayerNorm(x + σ(gate) ⊙ value)
    where [gate, value] = FC₂(ELU(FC₁(x)))
    """

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        inner = max(dim // 2, 32)
        self.fc1 = nn.Linear(dim, inner)
        self.fc2 = nn.Linear(inner, dim * 2)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = F.elu(self.fc1(x))
        h = self.dropout(self.fc2(h))
        gate, value = h.chunk(2, dim=-1)
        return self.norm(x + torch.sigmoid(gate) * value)


class GRNPolicyNetwork(nn.Module):
    """GRN 기반 정책 신경망 (피처 게이팅으로 노이즈 억제)"""

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_blocks: int = 2, dropout: float = 0.1,
                 min_concentration: float = 1.5):
        super().__init__()
        self.min_concentration = min_concentration

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.grn_blocks = nn.Sequential(
            *[GRNBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )
        self.head_fc = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 64),
            nn.ELU(),
        )
        self.alpha_head = nn.Linear(64, 1)
        self.beta_head = nn.Linear(64, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for head in [self.alpha_head, self.beta_head]:
            nn.init.orthogonal_(head.weight, gain=0.5)
            nn.init.constant_(head.bias, 0.54)

    def forward(self, x: torch.Tensor):
        h = self.input_proj(x)
        h = self.grn_blocks(h)
        feat = self.head_fc(h)
        alpha = F.softplus(self.alpha_head(feat)) + self.min_concentration
        beta = F.softplus(self.beta_head(feat)) + self.min_concentration
        return alpha.squeeze(-1), beta.squeeze(-1)


class GRNValueNetwork(nn.Module):
    """GRN 기반 가치 신경망"""

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_blocks: int = 2, dropout: float = 0.1):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.grn_blocks = nn.Sequential(
            *[GRNBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 64),
            nn.ELU(),
            nn.Linear(64, 1),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.grn_blocks(h)
        return torch.clamp(self.head(h), -50.0, 50.0)


# ═══════════════════════════════════════════════════════
#  FT-Transformer (Feature Tokenizer + Transformer)
# ═══════════════════════════════════════════════════════

class FTTransformerPolicyNetwork(nn.Module):
    """FT-Transformer 정책 신경망: 피처별 토큰화 + Self-Attention"""

    def __init__(self, input_dim: int, d_token: int = 64,
                 n_heads: int = 4, n_layers: int = 2,
                 dropout: float = 0.1, min_concentration: float = 1.5):
        super().__init__()
        self.min_concentration = min_concentration
        self.input_dim = input_dim

        # 피처 토크나이저: 각 스칼라 피처 → d_token 차원
        self.feature_weights = nn.Parameter(torch.empty(input_dim, d_token))
        self.feature_biases = nn.Parameter(torch.zeros(input_dim, d_token))

        # [CLS] 토큰
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))

        # Transformer (Pre-LN)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads,
            dim_feedforward=d_token * 4,
            dropout=dropout, batch_first=True,
            activation='gelu', norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(d_token)

        self.head_fc = nn.Sequential(
            nn.Linear(d_token, 64),
            nn.ReLU(),
        )
        self.alpha_head = nn.Linear(64, 1)
        self.beta_head = nn.Linear(64, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_normal_(self.feature_weights)
        nn.init.zeros_(self.feature_biases)
        nn.init.normal_(self.cls_token, std=0.02)
        for head in [self.alpha_head, self.beta_head]:
            nn.init.orthogonal_(head.weight, gain=0.5)
            nn.init.constant_(head.bias, 0.54)

    def forward(self, x: torch.Tensor):
        B = x.size(0)
        tokens = x.unsqueeze(-1) * self.feature_weights + self.feature_biases
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        h = self.transformer(tokens)
        cls_out = self.norm(h[:, 0])
        feat = self.head_fc(cls_out)
        alpha = F.softplus(self.alpha_head(feat)) + self.min_concentration
        beta = F.softplus(self.beta_head(feat)) + self.min_concentration
        return alpha.squeeze(-1), beta.squeeze(-1)


class FTTransformerValueNetwork(nn.Module):
    """FT-Transformer 가치 신경망"""

    def __init__(self, input_dim: int, d_token: int = 64,
                 n_heads: int = 4, n_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim

        self.feature_weights = nn.Parameter(torch.empty(input_dim, d_token))
        self.feature_biases = nn.Parameter(torch.zeros(input_dim, d_token))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads,
            dim_feedforward=d_token * 4,
            dropout=dropout, batch_first=True,
            activation='gelu', norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(d_token)

        self.head = nn.Sequential(
            nn.Linear(d_token, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_normal_(self.feature_weights)
        nn.init.zeros_(self.feature_biases)
        nn.init.normal_(self.cls_token, std=0.02)
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        tokens = x.unsqueeze(-1) * self.feature_weights + self.feature_biases
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        h = self.transformer(tokens)
        cls_out = self.norm(h[:, 0])
        return torch.clamp(self.head(cls_out), -50.0, 50.0)


# ═══════════════════════════════════════════════════════
#  gMLP (Gated MLP with Spatial Gating)
# ═══════════════════════════════════════════════════════

class SpatialGatingUnit(nn.Module):
    """Spatial Gating Unit: Attention 대체 메커니즘"""

    def __init__(self, d_ffn: int, seq_len: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_ffn // 2)
        self.spatial_proj = nn.Linear(seq_len, seq_len)
        nn.init.constant_(self.spatial_proj.weight, 0.0)
        nn.init.constant_(self.spatial_proj.bias, 1.0)

    def forward(self, x):
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        v = self.spatial_proj(v.transpose(1, 2)).transpose(1, 2)
        return u * v


class gMLPBlock(nn.Module):
    """gMLP block: FFN + Spatial Gating"""

    def __init__(self, dim: int, d_ffn: int, seq_len: int,
                 dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj_in = nn.Linear(dim, d_ffn)
        self.sgu = SpatialGatingUnit(d_ffn, seq_len)
        self.proj_out = nn.Linear(d_ffn // 2, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm(x)
        h = F.gelu(self.proj_in(h))
        h = self.sgu(h)
        h = self.dropout(self.proj_out(h))
        return x + h


class gMLPPolicyNetwork(nn.Module):
    """gMLP 정책 신경망: Attention-free 피처 혼합"""

    def __init__(self, input_dim: int, d_model: int = 64,
                 d_ffn: int = 128, n_blocks: int = 2,
                 dropout: float = 0.1, min_concentration: float = 1.5):
        super().__init__()
        self.min_concentration = min_concentration

        self.feature_weights = nn.Parameter(torch.empty(input_dim, d_model))
        self.feature_biases = nn.Parameter(torch.zeros(input_dim, d_model))

        self.blocks = nn.Sequential(
            *[gMLPBlock(d_model, d_ffn, input_dim, dropout)
              for _ in range(n_blocks)]
        )
        self.norm = nn.LayerNorm(d_model)

        self.head_fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
        )
        self.alpha_head = nn.Linear(64, 1)
        self.beta_head = nn.Linear(64, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_normal_(self.feature_weights)
        nn.init.zeros_(self.feature_biases)
        for head in [self.alpha_head, self.beta_head]:
            nn.init.orthogonal_(head.weight, gain=0.5)
            nn.init.constant_(head.bias, 0.54)

    def forward(self, x: torch.Tensor):
        tokens = x.unsqueeze(-1) * self.feature_weights + self.feature_biases
        h = self.blocks(tokens)
        h = self.norm(h).mean(dim=1)
        feat = self.head_fc(h)
        alpha = F.softplus(self.alpha_head(feat)) + self.min_concentration
        beta = F.softplus(self.beta_head(feat)) + self.min_concentration
        return alpha.squeeze(-1), beta.squeeze(-1)


class gMLPValueNetwork(nn.Module):
    """gMLP 가치 신경망"""

    def __init__(self, input_dim: int, d_model: int = 64,
                 d_ffn: int = 128, n_blocks: int = 2,
                 dropout: float = 0.1):
        super().__init__()

        self.feature_weights = nn.Parameter(torch.empty(input_dim, d_model))
        self.feature_biases = nn.Parameter(torch.zeros(input_dim, d_model))

        self.blocks = nn.Sequential(
            *[gMLPBlock(d_model, d_ffn, input_dim, dropout)
              for _ in range(n_blocks)]
        )
        self.norm = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_normal_(self.feature_weights)
        nn.init.zeros_(self.feature_biases)
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.unsqueeze(-1) * self.feature_weights + self.feature_biases
        h = self.blocks(tokens)
        h = self.norm(h).mean(dim=1)
        return torch.clamp(self.head(h), -50.0, 50.0)


# ═══════════════════════════════════════════════════════
#  Mamba-inspired SSM Network — CPU-optimized
#  선택적 상태공간 게이팅을 MLP 구조에 결합
#  피처 토큰화 대신 flat 처리로 CPU 효율 극대화
# ═══════════════════════════════════════════════════════

class SSMGate(nn.Module):
    """Selective State Space Gate — Mamba의 핵심 아이디어를 1D에 적용

    Hidden dim을 "시퀀스"로 보고 diagonal SSM을 적용:
      - 입력 의존적 decay (selectivity)
      - 채널 간 정보 전파 (state space)
      - 게이트 메커니즘

    (B, D) → (B, D): 시퀀스 차원 없이 직접 연산
    """

    def __init__(self, d_model: int):
        super().__init__()
        # 입력 의존적 decay rate
        self.proj_dt = nn.Linear(d_model, d_model, bias=True)
        self.A_log = nn.Parameter(torch.zeros(d_model))
        self.D = nn.Parameter(torch.ones(d_model))

        # dt bias 초기화
        dt_init_min, dt_init_max = 0.001, 0.1
        dt_bias = torch.exp(
            torch.rand(d_model) * (math.log(dt_init_max) - math.log(dt_init_min))
            + math.log(dt_init_min)
        )
        self.proj_dt.bias.data = torch.log(torch.exp(dt_bias) - 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, D)
        Returns:
            (B, D) — SSM 게이트 적용된 출력

        채널 방향으로 cumsum 기반 SSM:
          log_a = softplus(dt) * A  (입력 의존적 decay)
          h = exp(cumsum_a) * cumsum(exp(-cumsum_a) * x)
        """
        dt = F.softplus(self.proj_dt(x))         # (B, D)
        A = -torch.exp(self.A_log)               # (D,)

        log_a = dt * A.unsqueeze(0)              # (B, D)
        log_cumA = torch.cumsum(log_a, dim=-1)   # (B, D) — 채널 방향 scan
        log_cumA = log_cumA.clamp(min=-20.0)     # float32 overflow 방지
        weighted_x = torch.exp(-log_cumA) * x
        h = torch.exp(log_cumA) * torch.cumsum(weighted_x, dim=-1)

        return h + x * self.D.unsqueeze(0)


class MambaMLPBlock(nn.Module):
    """SSM-Gated MLP Block

    GRN의 게이팅 + Mamba의 선택적 상태공간을 결합:
      x → LayerNorm → Linear → SiLU → SSMGate → * gate → Linear → + residual

    GRN 대비 장점: 입력 의존적 decay로 피처 중요도 동적 학습
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.up_proj = nn.Linear(d_model, d_model * 2, bias=False)
        self.ssm_gate = SSMGate(d_model)
        self.down_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm(x)
        xz = self.up_proj(h)
        x_branch, z = xz.chunk(2, dim=-1)
        x_branch = self.ssm_gate(F.silu(x_branch))
        x_branch = x_branch * F.silu(z)
        return residual + self.dropout(self.down_proj(x_branch))


class MambaPolicyNetwork(nn.Module):
    """Mamba-inspired 정책 신경망 (CPU-optimized)

    Mamba의 선택적 상태공간 게이팅을 flat MLP 구조에 적용.
    피처 토큰화 없이 (B, F) → (B, D)로 직접 처리하여 CPU 효율적.

    장점:
      - 선택적 게이팅: 입력에 따라 피처 중요도 동적 학습 (Mamba 핵심)
      - GRN 수준의 속도: 시퀀스 차원 없이 flat 연산
      - 채널 SSM: hidden dim 방향 cumsum으로 피처 간 상호작용
    """

    def __init__(self, input_dim: int, d_model: int = 64,
                 d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, n_blocks: int = 2,
                 dropout: float = 0.1, min_concentration: float = 1.5):
        super().__init__()
        self.min_concentration = min_concentration

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(
            *[MambaMLPBlock(d_model, dropout) for _ in range(n_blocks)]
        )

        # Policy Head
        head_dim = max(64, d_model // 2)
        self.head_fc = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, head_dim),
            nn.ReLU(),
        )
        self.alpha_head = nn.Linear(head_dim, 1)
        self.beta_head = nn.Linear(head_dim, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        ssm_linears = {id(m.proj_dt) for m in self.modules() if isinstance(m, SSMGate)}
        for m in self.modules():
            if isinstance(m, nn.Linear) and id(m) not in ssm_linears:
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for head in [self.alpha_head, self.beta_head]:
            nn.init.orthogonal_(head.weight, gain=0.5)
            nn.init.constant_(head.bias, 0.54)

    def forward(self, x: torch.Tensor):
        h = self.input_proj(x)         # (B, d_model)
        h = self.blocks(h)             # (B, d_model)
        feat = self.head_fc(h)         # (B, 64)
        alpha = F.softplus(self.alpha_head(feat)) + self.min_concentration
        beta = F.softplus(self.beta_head(feat)) + self.min_concentration
        return alpha.squeeze(-1), beta.squeeze(-1)


class MambaValueNetwork(nn.Module):
    """Mamba-inspired 가치 신경망 (CPU-optimized)"""

    def __init__(self, input_dim: int, d_model: int = 64,
                 d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, n_blocks: int = 2,
                 dropout: float = 0.1):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(
            *[MambaMLPBlock(d_model, dropout) for _ in range(n_blocks)]
        )
        head_dim = max(64, d_model // 2)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, 1),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        ssm_linears = {id(m.proj_dt) for m in self.modules() if isinstance(m, SSMGate)}
        for m in self.modules():
            if isinstance(m, nn.Linear) and id(m) not in ssm_linears:
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)         # (B, d_model)
        h = self.blocks(h)             # (B, d_model)
        return torch.clamp(self.head(h), -50.0, 50.0)


# ═══════════════════════════════════════════════════════
#  회귀 신경망 (지도학습 Day Trading — 일중 수익률 예측)
# ═══════════════════════════════════════════════════════

class MambaRegressionNetwork(nn.Module):
    """Mamba-inspired 회귀 신경망 — 일중 수익률 예측

    MambaValueNetwork 와 동일한 백본이지만 출력 clamp 없음.
    입력: 시장 피처만 사용 (포트폴리오 피처 불필요 — 매일 독립 결정)
    출력: 예측 일중 수익률 (close - open) / open
    """

    def __init__(self, input_dim: int, d_model: int = 64,
                 d_state: int = 16, n_blocks: int = 2,
                 dropout: float = 0.1):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(
            *[MambaMLPBlock(d_model, dropout) for _ in range(n_blocks)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        ssm_linears = {id(m.proj_dt) for m in self.modules() if isinstance(m, SSMGate)}
        for m in self.modules():
            if isinstance(m, nn.Linear) and id(m) not in ssm_linears:
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)         # (B, d_model)
        h = self.blocks(h)             # (B, d_model)
        return self.head(h).squeeze(-1)  # (B,) 예측 일중 수익률
