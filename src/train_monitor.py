"""
학습 상태 모니터링 대시보드 생성기

output/train/episodes.jsonl + chunks.jsonl 을 읽어
Plotly 기반 HTML 대시보드(train_monitor.html)를 생성합니다.

사용법:
  # 1회 생성
  python -m train_monitor --output-dir output/train

  # 파일 변경 감지 후 자동 재생성 (30초 간격)
  python -m train_monitor --output-dir output/train --watch
"""
import argparse
import json
import os
from datetime import datetime


# ──────────────────────────────────────────────
# 데이터 로딩
# ──────────────────────────────────────────────

def load_episodes(filepath: str) -> tuple:
    """episodes.jsonl 파싱 → (meta_list, episodes)
    meta_list: chunk별 meta 딕셔너리 목록
    episodes:  episode 딕셔너리 목록 (chunk_id 포함)
    """
    meta_list = []
    episodes = []
    current_meta = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    if d.get('type') == 'meta':
                        current_meta = d
                        meta_list.append(d)
                    elif d.get('type') == 'episode':
                        d['_meta'] = current_meta
                        episodes.append(d)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        pass
    return meta_list, episodes


def load_chunks(filepath: str) -> list:
    """chunks.jsonl 파싱"""
    chunks = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    chunks.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        pass
    return chunks


# ──────────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────────

def smooth(values: list, window: int = 20) -> list:
    """단순 이동평균 스무딩"""
    if not values:
        return []
    result = []
    for i, v in enumerate(values):
        start = max(0, i - window + 1)
        result.append(sum(values[start:i + 1]) / (i - start + 1))
    return result


def _safe(lst, idx, default=0):
    try:
        return lst[idx]
    except IndexError:
        return default


# ──────────────────────────────────────────────
# 대시보드 생성
# ──────────────────────────────────────────────

def build_dashboard(meta_list: list, episodes: list, chunks: list, output_path: str):
    """Plotly HTML 대시보드 생성"""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("[Monitor] plotly 미설치. pip install plotly")
        return

    if not episodes:
        print("[Monitor] 에피소드 데이터 없음")
        return

    # ── 현재 chunk meta ──
    meta = meta_list[-1] if meta_list else {}
    iteration     = meta.get('iteration_name', 'N/A')
    etf_code      = meta.get('etf_code', '')
    etf_name      = meta.get('etf_name', '')
    chunk_idx     = meta.get('chunk_idx', 0)
    total_chunks  = meta.get('total_chunks', 0)
    start_step    = meta.get('start_step', 0)
    end_step      = meta.get('end_step', start_step)
    start_date    = meta.get('start_date', '')
    end_date      = meta.get('end_date', '')
    dataset       = meta.get('dataset', '')
    num_total_eps = meta.get('num_episodes', 500)

    # ── 현재 chunk 에피소드만 분리 ──
    cur_chunk_id = meta.get('start_step', -1)
    cur_eps  = [e for e in episodes if e.get('chunk_id') == cur_chunk_id]
    if not cur_eps:
        cur_eps = episodes  # fallback: 전체

    eps     = [e['episode'] for e in cur_eps]
    current_episode = eps[-1] if eps else 0

    # ── 시계열 데이터 추출 ──
    policy_losses = [e.get('loss', {}).get('policy', 0) for e in cur_eps]
    value_losses  = [e.get('loss', {}).get('value', 0)  for e in cur_eps]
    kl_divs       = [e.get('loss', {}).get('kl_div', 0) for e in cur_eps]

    profit_rates  = [e.get('train', {}).get('profit_rate', 0) for e in cur_eps]
    excess_bnhs   = [e.get('train', {}).get('excess_bnh', 0)  for e in cur_eps]
    sharpes       = [e.get('train', {}).get('sharpe_ratio', 0) for e in cur_eps]
    win_rates     = [e.get('train', {}).get('win_rate', 0)    for e in cur_eps]
    drawdowns     = [e.get('train', {}).get('max_drawdown', 0) for e in cur_eps]
    rewards       = [e.get('reward', 0) for e in cur_eps]

    val_eps_data  = [(e['episode'], e['val']['profit_rate']) for e in cur_eps if e.get('val')]
    has_val       = bool(val_eps_data)

    lr_policy     = [e.get('lr_policy', 0)    for e in cur_eps]
    lr_value      = [e.get('lr_value', 0)     for e in cur_eps]
    entropy_coefs = [e.get('entropy_coef', 0) for e in cur_eps]

    buys          = [e.get('train', {}).get('buy', 0)  for e in cur_eps]
    sells         = [e.get('train', {}).get('sell', 0) for e in cur_eps]
    holds         = [e.get('train', {}).get('hold', 0) for e in cur_eps]

    mean_targets  = [e.get('policy', {}).get('mean_target_ratio', 0) for e in cur_eps]
    std_targets   = [e.get('policy', {}).get('std_target_ratio', 0)  for e in cur_eps]

    best_train_eps = [e['episode'] for e in cur_eps if e.get('is_best_train')]
    best_eps       = [e['episode'] for e in cur_eps if e.get('is_best')]

    # ── 진행률 ──
    overall_pct = round((chunk_idx / max(total_chunks, 1)) * 100, 1)
    chunk_pct   = round((current_episode / max(num_total_eps, 1)) * 100, 1)

    # ── 요약 통계 ──
    best_profit  = max(profit_rates) if profit_rates else 0
    latest_profit = profit_rates[-1] if profit_rates else 0
    latest_sharpe = sharpes[-1] if sharpes else 0
    latest_wr     = win_rates[-1] if win_rates else 0
    latest_mdd    = drawdowns[-1] if drawdowns else 0

    # ──────────────────────────────────────────
    # Subplot 구성: 5 rows × 2 cols
    # ──────────────────────────────────────────
    fig = make_subplots(
        rows=5, cols=2,
        subplot_titles=[
            'Policy Loss',          'Value Loss',
            'KL Divergence',        'Reward',
            'Profit Rate (%)',      'Sharpe Ratio',
            'Win Rate (%)  &  MDD', 'Trade Counts (stacked)',
            'Learning Rate & Entropy (scaled)', 'Policy Output Stats',
        ],
        vertical_spacing=0.07,
        horizontal_spacing=0.07,
    )

    W = 20  # 이동평균 window

    # ── Row 1: Policy Loss / Value Loss ──
    fig.add_trace(go.Scatter(
        x=eps, y=policy_losses, mode='lines', name='Policy Loss',
        line=dict(color='rgba(255,100,100,0.25)', width=1), legendgroup='pl', showlegend=True,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=eps, y=smooth(policy_losses, W), mode='lines', name='Policy Loss MA',
        line=dict(color='tomato', width=2), legendgroup='pl', showlegend=True,
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=eps, y=value_losses, mode='lines', name='Value Loss',
        line=dict(color='rgba(100,150,255,0.25)', width=1), legendgroup='vl', showlegend=True,
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=eps, y=smooth(value_losses, W), mode='lines', name='Value Loss MA',
        line=dict(color='cornflowerblue', width=2), legendgroup='vl', showlegend=True,
    ), row=1, col=2)

    # ── Row 2: KL Divergence / Reward ──
    fig.add_trace(go.Scatter(
        x=eps, y=kl_divs, mode='lines', name='KL Div',
        line=dict(color='rgba(255,180,0,0.25)', width=1), legendgroup='kl', showlegend=True,
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=eps, y=smooth(kl_divs, W), mode='lines', name='KL Div MA',
        line=dict(color='gold', width=2), legendgroup='kl', showlegend=True,
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=eps, y=rewards, mode='lines', name='Reward',
        line=dict(color='rgba(100,220,100,0.25)', width=1), legendgroup='rw', showlegend=True,
    ), row=2, col=2)
    fig.add_trace(go.Scatter(
        x=eps, y=smooth(rewards, W), mode='lines', name='Reward MA',
        line=dict(color='limegreen', width=2), legendgroup='rw', showlegend=True,
    ), row=2, col=2)

    # ── Row 3: Profit Rate / Sharpe ──
    fig.add_trace(go.Scatter(
        x=eps, y=profit_rates, mode='lines', name='Profit%',
        line=dict(color='rgba(0,200,100,0.25)', width=1), legendgroup='pr',
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=eps, y=smooth(profit_rates, W), mode='lines', name='Profit% MA',
        line=dict(color='mediumseagreen', width=2), legendgroup='pr',
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=eps, y=excess_bnhs, mode='lines', name='Excess BnH',
        line=dict(color='rgba(0,200,200,0.4)', width=1, dash='dot'),
    ), row=3, col=1)

    if has_val:
        vx, vy = zip(*val_eps_data)
        fig.add_trace(go.Scatter(
            x=list(vx), y=list(vy), mode='markers+lines', name='Val Profit%',
            marker=dict(color='violet', size=5),
            line=dict(color='violet', width=1, dash='dot'),
        ), row=3, col=1)

    if best_train_eps:
        bt_y = [profit_rates[i - 1] for i in best_train_eps if 0 < i <= len(profit_rates)]
        fig.add_trace(go.Scatter(
            x=best_train_eps[:len(bt_y)], y=bt_y, mode='markers', name='Best Train ★',
            marker=dict(symbol='star', color='gold', size=10, line=dict(color='orange', width=1)),
        ), row=3, col=1)

    if best_eps:
        bx = [e for e in best_eps if 0 < e <= len(profit_rates)]
        by = [profit_rates[e - 1] for e in bx]
        fig.add_trace(go.Scatter(
            x=bx, y=by, mode='markers', name='Best Val ★',
            marker=dict(symbol='star', color='cyan', size=10, line=dict(color='teal', width=1)),
        ), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=eps, y=sharpes, mode='lines', name='Sharpe',
        line=dict(color='rgba(50,130,255,0.25)', width=1), legendgroup='sh',
    ), row=3, col=2)
    fig.add_trace(go.Scatter(
        x=eps, y=smooth(sharpes, W), mode='lines', name='Sharpe MA',
        line=dict(color='royalblue', width=2), legendgroup='sh',
    ), row=3, col=2)
    fig.add_hline(y=0, line_dash='dash', line_color='rgba(200,200,200,0.3)', row=3, col=2)
    fig.add_hline(y=1, line_dash='dot',  line_color='rgba(100,255,100,0.3)', row=3, col=2)

    # ── Row 4: Win Rate & MDD / Trade Counts ──
    fig.add_trace(go.Scatter(
        x=eps, y=win_rates, mode='lines', name='Win Rate',
        line=dict(color='rgba(200,100,255,0.3)', width=1), legendgroup='wr',
    ), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=eps, y=smooth(win_rates, W), mode='lines', name='Win Rate MA',
        line=dict(color='mediumpurple', width=2), legendgroup='wr',
    ), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=eps, y=drawdowns, mode='lines', name='MDD',
        line=dict(color='rgba(255,80,80,0.4)', width=1, dash='dot'),
    ), row=4, col=1)
    fig.add_hline(y=50, line_dash='dash', line_color='rgba(200,200,200,0.3)', row=4, col=1)

    fig.add_trace(go.Bar(
        x=eps, y=buys, name='Buy',
        marker_color='rgba(0,180,80,0.6)', marker_line_width=0,
    ), row=4, col=2)
    fig.add_trace(go.Bar(
        x=eps, y=sells, name='Sell',
        marker_color='rgba(220,60,60,0.6)', marker_line_width=0,
    ), row=4, col=2)
    fig.add_trace(go.Bar(
        x=eps, y=holds, name='Hold',
        marker_color='rgba(120,120,120,0.4)', marker_line_width=0,
    ), row=4, col=2)

    # ── Row 5: LR & Entropy / Policy Stats ──
    fig.add_trace(go.Scatter(
        x=eps, y=lr_policy, mode='lines', name='LR Policy',
        line=dict(color='tomato', width=1.5),
    ), row=5, col=1)
    fig.add_trace(go.Scatter(
        x=eps, y=lr_value, mode='lines', name='LR Value',
        line=dict(color='cornflowerblue', width=1.5),
    ), row=5, col=1)
    # entropy를 lr 스케일로 환산해서 overlay
    max_lr = max(max(lr_policy, default=1e-5), max(lr_value, default=1e-5))
    max_ent = max(entropy_coefs) if entropy_coefs else 1
    ent_scaled = [e * (max_lr / max(max_ent, 1e-8)) for e in entropy_coefs]
    fig.add_trace(go.Scatter(
        x=eps, y=ent_scaled, mode='lines', name='Entropy (scaled)',
        line=dict(color='gold', width=1.5, dash='dot'),
    ), row=5, col=1)

    fig.add_trace(go.Scatter(
        x=eps, y=mean_targets, mode='lines', name='Mean Target',
        line=dict(color='mediumturquoise', width=1.5),
    ), row=5, col=2)
    fig.add_trace(go.Scatter(
        x=eps, y=std_targets, mode='lines', name='Std Target',
        line=dict(color='lightcyan', width=1.5),
    ), row=5, col=2)
    fig.add_hline(y=0.5, line_dash='dash', line_color='rgba(200,200,200,0.2)', row=5, col=2)

    # ── Layout ──
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    fig.update_layout(
        title=None,
        height=1700,
        template='plotly_dark',
        paper_bgcolor='#12121f',
        plot_bgcolor='#1a1a2e',
        showlegend=True,
        legend=dict(
            orientation='h', yanchor='bottom', y=1.01,
            xanchor='right', x=1, font=dict(size=10),
        ),
        barmode='stack',
        margin=dict(t=60, b=40, l=60, r=60),
    )

    # 축 레이블 색상 통일
    for i in range(1, 6):
        for j in range(1, 3):
            fig.update_xaxes(showgrid=True, gridcolor='#2a2a3e', row=i, col=j)
            fig.update_yaxes(showgrid=True, gridcolor='#2a2a3e', row=i, col=j)

    # ── 스탯 배너 ──
    bnh_latest = cur_eps[-1].get('train', {}).get('bnh_return', 0) if cur_eps else 0
    latest_reward = rewards[-1] if rewards else 0

    def _stat(label, value, highlight=False):
        color = "#7fff7f" if highlight else "#e0e0ff"
        return (
            f'<div class="stat-item">'
            f'<span class="stat-label">{label}</span>'
            f'<span class="stat-value" style="color:{color}">{value}</span>'
            f'</div>'
        )

    stats_items = "".join([
        _stat("Iteration",  iteration),
        _stat("Dataset",    dataset),
        _stat("ETF",        f"{etf_code} {etf_name}"),
        _stat("기간",        f"{start_date} → {end_date}"),
        _stat("Chunk",      f"{chunk_idx} / {total_chunks}"),
        _stat("Episode",    f"{current_episode} / {num_total_eps}"),
        _stat("Best Profit",  f"{best_profit:.2f}%",   highlight=best_profit > 0),
        _stat("최근 Profit",  f"{latest_profit:.2f}%", highlight=latest_profit > 0),
        _stat("BnH Return", f"{bnh_latest:.2f}%"),
        _stat("최근 Sharpe", f"{latest_sharpe:.3f}",   highlight=latest_sharpe > 0),
        _stat("Win Rate",   f"{latest_wr:.1f}%",       highlight=latest_wr > 50),
        _stat("MDD",        f"{latest_mdd:.2f}%"),
        _stat("Reward",     f"{latest_reward:.3f}"),
        _stat("갱신시각",    now),
    ])

    # ── Chunk 요약 테이블 ──
    chunk_table_html = ""
    if chunks:
        rows_html = []
        for c in chunks[-30:]:
            bv = c.get('best_val_profit_rate')
            bv_str = f"{bv:.1f}%" if bv is not None else "-"
            bp = c.get('best_profit_rate', 0)
            color = "color:#7fff7f" if bp > 0 else "color:#ff7f7f"
            rows_html.append(
                f"<tr>"
                f"<td>{c.get('etf_code','')}</td>"
                f"<td>{c.get('etf_name', c.get('etf_code',''))[:14]}</td>"
                f"<td>{c.get('start_date','')}~{c.get('end_date','')}</td>"
                f"<td style='{color}'>{bp:.1f}%</td>"
                f"<td>{c.get('best_sharpe', 0):.2f}</td>"
                f"<td>{c.get('mean_profit_rate', 0):.1f}%</td>"
                f"<td>{c.get('last_profit_rate', 0):.1f}%</td>"
                f"<td>{bv_str}</td>"
                f"<td>{c.get('best_episode','-')}/{c.get('episodes','-')}</td>"
                f"</tr>"
            )
        chunk_table_html = f"""
        <h3 style="color:#aaa;margin-top:24px;font-size:13px">
          📋 완료된 Chunk 요약 (최근 {len(rows_html)}개 / 전체 {len(chunks)}개)
        </h3>
        <table style="border-collapse:collapse;font-size:11px;color:#bbb;width:100%">
          <thead>
            <tr style="background:#1e1e3a;color:#ddd">
              <th style="padding:4px 8px;border:1px solid #333">Code</th>
              <th style="border:1px solid #333">Name</th>
              <th style="border:1px solid #333">기간</th>
              <th style="border:1px solid #333">Best Profit</th>
              <th style="border:1px solid #333">Best Sharpe</th>
              <th style="border:1px solid #333">Mean Profit</th>
              <th style="border:1px solid #333">Last Profit</th>
              <th style="border:1px solid #333">Val Best</th>
              <th style="border:1px solid #333">Best Ep</th>
            </tr>
          </thead>
          <tbody>{''.join(rows_html)}</tbody>
        </table>
        """

    # ── 전체 HTML 조립 ──
    chart_html = fig.to_html(include_plotlyjs='cdn', full_html=False,
                             config={'displayModeBar': False})
    now_full = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    full_html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="30">
  <title>학습 모니터 | {iteration}</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      background: #12121f;
      color: #ccc;
      font-family: 'Segoe UI', Arial, sans-serif;
      padding: 12px 16px;
      margin: 0;
    }}
    h1.page-title {{
      font-size: 15px;
      color: #aac;
      margin: 0 0 10px;
      padding: 0;
      border-bottom: 1px solid #2a2a4a;
      padding-bottom: 6px;
    }}
    .stat-grid {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-bottom: 12px;
    }}
    .stat-item {{
      background: #1a1a3a;
      border: 1px solid #2a2a4a;
      border-radius: 5px;
      padding: 5px 10px;
      min-width: 100px;
    }}
    .stat-label {{
      display: block;
      font-size: 10px;
      color: #778;
      margin-bottom: 2px;
    }}
    .stat-value {{
      display: block;
      font-size: 13px;
      font-weight: bold;
      color: #e0e0ff;
      white-space: nowrap;
    }}
    .progress-row {{
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 6px;
    }}
    .progress-label {{
      font-size: 11px;
      color: #778;
      white-space: nowrap;
      width: 220px;
      flex-shrink: 0;
    }}
    .progress-bar-wrap {{
      flex: 1;
      background: #2a2a3e;
      border-radius: 4px;
      height: 8px;
      overflow: hidden;
    }}
    .progress-bar {{
      height: 100%;
      background: linear-gradient(90deg, #3366ff, #66aaff);
      border-radius: 4px;
    }}
    table tr:nth-child(even) {{ background: #1e1e30; }}
    table td, table th {{ padding: 3px 8px; border: 1px solid #2a2a3e; white-space: nowrap; }}
    .footer {{ color: #444; font-size: 10px; text-align: right; margin-top: 12px; }}
  </style>
</head>
<body>
  <h1 class="page-title">🔍 학습 모니터</h1>
  <div class="stat-grid">
    {stats_items}
  </div>
  <div class="progress-row">
    <span class="progress-label">전체 진행 (Chunk {chunk_idx}/{total_chunks}, {overall_pct}%)</span>
    <div class="progress-bar-wrap">
      <div class="progress-bar" style="width:{overall_pct}%"></div>
    </div>
  </div>
  <div class="progress-row">
    <span class="progress-label">현재 Chunk (Episode {current_episode}/{num_total_eps}, {chunk_pct}%)</span>
    <div class="progress-bar-wrap">
      <div class="progress-bar" style="width:{chunk_pct}%;background:linear-gradient(90deg,#33aa66,#66ffaa)"></div>
    </div>
  </div>

  {chart_html}
  {chunk_table_html}

  <div class="footer">
    Generated: {now_full} &nbsp;|&nbsp; 30초마다 자동 새로고침 &nbsp;|&nbsp;
    episodes={len(cur_eps)} &nbsp;|&nbsp; chunks_done={len(chunks)}
  </div>
</body>
</html>"""

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_html)

    print(
        f"[Monitor] 대시보드 저장: {output_path} "
        f"(ep={len(cur_eps)}/{len(episodes)}, chunks={len(chunks)})"
    )


def generate_dashboard(output_dir: str, output_filename: str = 'train_monitor.html'):
    """트레이너에서 호출하는 진입점"""
    episodes_path = os.path.join(output_dir, 'episodes.jsonl')
    chunks_path   = os.path.join(output_dir, 'chunks.jsonl')
    output_path   = os.path.join(output_dir, output_filename)

    meta_list, episodes = load_episodes(episodes_path)
    chunks = load_chunks(chunks_path)
    build_dashboard(meta_list, episodes, chunks, output_path)


# ──────────────────────────────────────────────
# 포트폴리오 대시보드
# ──────────────────────────────────────────────

def load_portfolio_log(filepath: str) -> list:
    """train_log.jsonl 파싱"""
    records = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        pass
    return records


def build_portfolio_dashboard(records: list, config: dict, output_path: str):
    """포트폴리오 학습 Plotly HTML 대시보드 생성"""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("[Monitor] plotly 미설치. pip install plotly")
        return

    if not records:
        print("[Monitor] 포트폴리오 로그 데이터 없음")
        return

    eps           = [r['episode'] for r in records]
    current_ep    = eps[-1]
    num_total_eps = config.get('episodes', 500)
    model_name    = config.get('model_name', 'N/A')
    dataset       = config.get('dataset', 'N/A')
    n_assets      = config.get('n_assets', '?')
    ep_pct        = round(current_ep / max(num_total_eps - 1, 1) * 100, 1)

    # ── 시계열 ──
    policy_losses = [r.get('policy_loss', 0) for r in records]
    value_losses  = [r.get('value_loss', 0)  for r in records]
    entropies     = [r.get('entropy', 0)      for r in records]
    entropy_coefs = [r.get('entropy_coef', 0) for r in records]
    rewards       = [r.get('episode_reward', 0) for r in records]
    cagrs         = [r.get('cagr', 0) * 100    for r in records]
    returns       = [r.get('total_return', 0) * 100 for r in records]
    sharpes       = [r.get('sharpe', 0)        for r in records]
    mdds          = [r.get('mdd', 0) * 100     for r in records]

    W = 20  # MA window

    best_cagr   = max(cagrs) if cagrs else 0
    latest_cagr = cagrs[-1] if cagrs else 0
    latest_sharpe = sharpes[-1] if sharpes else 0
    latest_mdd    = mdds[-1] if mdds else 0
    latest_reward = rewards[-1] if rewards else 0

    # ── Subplot 구성: 4 rows × 2 cols ──
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=[
            'Policy Loss',    'Value Loss',
            'Entropy',        'Episode Reward',
            'CAGR (%)',       'Sharpe Ratio',
            'Total Return (%)', 'MDD (%)',
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.07,
    )

    def _add(row, col, y, name, color_raw, color_ma):
        fig.add_trace(go.Scatter(
            x=eps, y=y, mode='lines', name=name,
            line=dict(color=color_raw, width=1), showlegend=True,
        ), row=row, col=col)
        fig.add_trace(go.Scatter(
            x=eps, y=smooth(y, W), mode='lines', name=f'{name} MA',
            line=dict(color=color_ma, width=2), showlegend=True,
        ), row=row, col=col)

    _add(1, 1, policy_losses, 'Policy Loss', 'rgba(255,100,100,0.25)', 'tomato')
    _add(1, 2, value_losses,  'Value Loss',  'rgba(100,150,255,0.25)', 'cornflowerblue')
    _add(2, 1, entropies,     'Entropy',     'rgba(255,200,0,0.25)',   'gold')

    # Entropy coef 오버레이 (scaled)
    max_ent_val = max(entropies) if entropies else 1
    max_ent_coef = max(entropy_coefs) if entropy_coefs else 1
    ec_scaled = [e * (max_ent_val / max(max_ent_coef, 1e-8)) for e in entropy_coefs]
    fig.add_trace(go.Scatter(
        x=eps, y=ec_scaled, mode='lines', name='Entropy Coef (scaled)',
        line=dict(color='orange', width=1.5, dash='dot'),
    ), row=2, col=1)

    _add(2, 2, rewards, 'Reward', 'rgba(100,220,100,0.25)', 'limegreen')
    fig.add_hline(y=0, line_dash='dash', line_color='rgba(200,200,200,0.3)', row=2, col=2)

    _add(3, 1, cagrs,   'CAGR%',    'rgba(0,200,100,0.25)', 'mediumseagreen')
    fig.add_hline(y=0, line_dash='dash', line_color='rgba(200,200,200,0.3)', row=3, col=1)

    _add(3, 2, sharpes, 'Sharpe',   'rgba(50,130,255,0.25)', 'royalblue')
    fig.add_hline(y=0, line_dash='dash', line_color='rgba(200,200,200,0.3)', row=3, col=2)
    fig.add_hline(y=1, line_dash='dot',  line_color='rgba(100,255,100,0.25)', row=3, col=2)

    _add(4, 1, returns, 'Return%',  'rgba(0,180,180,0.25)', 'mediumturquoise')
    _add(4, 2, mdds,    'MDD%',     'rgba(255,80,80,0.25)', 'tomato')
    fig.add_hline(y=0, line_dash='dash', line_color='rgba(200,200,200,0.3)', row=4, col=2)

    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    fig.update_layout(
        title=None,
        height=1300,
        template='plotly_dark',
        paper_bgcolor='#12121f',
        plot_bgcolor='#1a1a2e',
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.01,
                    xanchor='right', x=1, font=dict(size=10)),
        margin=dict(t=60, b=40, l=60, r=60),
    )
    for i in range(1, 5):
        for j in range(1, 3):
            fig.update_xaxes(showgrid=True, gridcolor='#2a2a3e', row=i, col=j)
            fig.update_yaxes(showgrid=True, gridcolor='#2a2a3e', row=i, col=j)

    # ── 스탯 배너 ──
    def _stat(label, value, highlight=False):
        color = "#7fff7f" if highlight else "#e0e0ff"
        return (
            f'<div class="stat-item">'
            f'<span class="stat-label">{label}</span>'
            f'<span class="stat-value" style="color:{color}">{value}</span>'
            f'</div>'
        )

    stats_items = "".join([
        _stat("Model",       model_name),
        _stat("Dataset",     dataset),
        _stat("N Assets",    str(n_assets)),
        _stat("Episode",     f"{current_ep} / {num_total_eps - 1}"),
        _stat("Best CAGR",   f"{best_cagr:.2f}%",    highlight=best_cagr > 0),
        _stat("최근 CAGR",   f"{latest_cagr:.2f}%",  highlight=latest_cagr > 0),
        _stat("최근 Sharpe", f"{latest_sharpe:.3f}",  highlight=latest_sharpe > 0),
        _stat("최근 MDD",    f"{latest_mdd:.2f}%"),
        _stat("최근 Reward", f"{latest_reward:.2f}"),
        _stat("갱신시각",    now),
    ])

    chart_html = fig.to_html(include_plotlyjs='cdn', full_html=False,
                             config={'displayModeBar': False})
    full_html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="30">
  <title>Portfolio 학습 모니터 | {model_name}</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ background:#12121f; color:#ccc; font-family:'Segoe UI',Arial,sans-serif; padding:12px 16px; margin:0; }}
    h1.page-title {{ font-size:15px; color:#aac; margin:0 0 10px; padding-bottom:6px; border-bottom:1px solid #2a2a4a; }}
    .stat-grid {{ display:flex; flex-wrap:wrap; gap:6px; margin-bottom:12px; }}
    .stat-item {{ background:#1a1a3a; border:1px solid #2a2a4a; border-radius:5px; padding:5px 10px; min-width:100px; }}
    .stat-label {{ display:block; font-size:10px; color:#778; margin-bottom:2px; }}
    .stat-value {{ display:block; font-size:13px; font-weight:bold; color:#e0e0ff; white-space:nowrap; }}
    .progress-row {{ display:flex; align-items:center; gap:10px; margin-bottom:6px; }}
    .progress-label {{ font-size:11px; color:#778; white-space:nowrap; width:220px; flex-shrink:0; }}
    .progress-bar-wrap {{ flex:1; background:#2a2a3e; border-radius:4px; height:8px; overflow:hidden; }}
    .progress-bar {{ height:100%; background:linear-gradient(90deg,#3366ff,#66aaff); border-radius:4px; }}
    .footer {{ color:#444; font-size:10px; text-align:right; margin-top:12px; }}
  </style>
</head>
<body>
  <h1 class="page-title">📊 Portfolio 학습 모니터</h1>
  <div class="stat-grid">{stats_items}</div>
  <div class="progress-row">
    <span class="progress-label">학습 진행 (Episode {current_ep}/{num_total_eps - 1}, {ep_pct}%)</span>
    <div class="progress-bar-wrap">
      <div class="progress-bar" style="width:{ep_pct}%;background:linear-gradient(90deg,#33aa66,#66ffaa)"></div>
    </div>
  </div>
  {chart_html}
  <div class="footer">Generated: {now} &nbsp;|&nbsp; 30초마다 자동 새로고침 &nbsp;|&nbsp; episodes={len(records)}</div>
</body>
</html>"""

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_html)
    print(f"[Monitor] Portfolio 대시보드 저장: {output_path} (ep={current_ep})")


def generate_portfolio_dashboard(log_dir: str, output_filename: str = 'train_monitor.html'):
    """포트폴리오 트레이너에서 호출하는 진입점"""
    log_path    = os.path.join(log_dir, 'train_log.jsonl')
    config_path = os.path.join(log_dir, 'train_config.json')
    output_path = os.path.join(log_dir, output_filename)

    records = load_portfolio_log(log_path)
    config = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except Exception:
            pass
    build_portfolio_dashboard(records, config, output_path)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='학습 상태 모니터링 대시보드 생성')
    parser.add_argument('--output-dir', default='output/train',
                        help='episodes.jsonl이 있는 디렉토리 (기본: output/train)')
    parser.add_argument('--watch', action='store_true',
                        help='파일 변경 감지 시 자동 재생성')
    parser.add_argument('--interval', type=int, default=30,
                        help='--watch 모드 갱신 간격 (초, 기본: 30)')
    args = parser.parse_args()

    episodes_path = os.path.join(args.output_dir, 'episodes.jsonl')
    chunks_path   = os.path.join(args.output_dir, 'chunks.jsonl')
    output_path   = os.path.join(args.output_dir, 'train_monitor.html')

    if args.watch:
        import time
        print(f"[Watch] {episodes_path} 모니터링 중 (간격={args.interval}s) ... Ctrl+C로 종료")
        last_ep_mtime    = 0
        last_chunk_mtime = 0
        while True:
            try:
                ep_mtime = os.path.getmtime(episodes_path) if os.path.exists(episodes_path) else 0
                ck_mtime = os.path.getmtime(chunks_path)   if os.path.exists(chunks_path)   else 0
                if ep_mtime != last_ep_mtime or ck_mtime != last_chunk_mtime:
                    meta_list, episodes = load_episodes(episodes_path)
                    chunks = load_chunks(chunks_path)
                    build_dashboard(meta_list, episodes, chunks, output_path)
                    last_ep_mtime    = ep_mtime
                    last_chunk_mtime = ck_mtime
                time.sleep(args.interval)
            except KeyboardInterrupt:
                print("\n[Watch] 종료")
                break
            except Exception as e:
                print(f"[Watch] 오류: {e}")
                time.sleep(args.interval)
    else:
        meta_list, episodes = load_episodes(episodes_path)
        chunks = load_chunks(chunks_path)
        build_dashboard(meta_list, episodes, chunks, output_path)


if __name__ == '__main__':
    main()
