"""
ETF 예측/랭킹 모듈

공유 프로젝트 버전에서는 quantylab 내부 모듈 없이 동작하도록
기존 데이터셋 CSV 또는 Quantylab REST API 기반 최신 스냅샷만 사용합니다.
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

from dataset_builder import fetch_latest_snapshot_from_api
from api_client import QuantylabRestClient
from network import (
    ContinuousPolicyNetwork,
    GRNPolicyNetwork,
    GRNValueNetwork,
    MambaPolicyNetwork,
    MambaRegressionNetwork,
    MambaValueNetwork,
    ValueNetwork,
)
from target_etfs import TARGET_ETFS


DEFAULT_PORTFOLIO_FEATURES = np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def _detect_network_type(state_dict_keys):
    keys = set(state_dict_keys)
    if any("ssm_gate" in key for key in keys):
        return "mamba"
    if any("grn" in key.lower() or "gating" in key.lower() for key in keys):
        return "grn"
    if any("res_blocks" in key for key in keys):
        return "standard"
    return "mamba"


def load_model(model_dir: str, device: str = "cpu"):
    regression_path = os.path.join(model_dir, "model_best.pt")
    if os.path.exists(regression_path):
        ckpt = torch.load(regression_path, map_location=device, weights_only=False)
        input_dim = ckpt.get("input_dim", 121)
        model = MambaRegressionNetwork(input_dim, d_model=64, d_state=16, n_blocks=2)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return {"type": "regression", "model": model, "input_dim": input_dim}

    policy_path = os.path.join(model_dir, "policy_best.pt")
    value_path = os.path.join(model_dir, "value_best.pt")
    policy_ckpt = torch.load(policy_path, map_location=device, weights_only=False)
    value_ckpt = torch.load(value_path, map_location=device, weights_only=False)
    input_dim = policy_ckpt.get("input_dim", 126)
    network_type = _detect_network_type(policy_ckpt["state_dict"].keys())

    if network_type == "mamba":
        policy_net = MambaPolicyNetwork(input_dim, d_model=64, d_state=16, n_blocks=2)
        value_net = MambaValueNetwork(input_dim, d_model=64, d_state=16, n_blocks=2)
    elif network_type == "grn":
        policy_net = GRNPolicyNetwork(input_dim, hidden_dim=128, num_blocks=2)
        value_net = GRNValueNetwork(input_dim, hidden_dim=128, num_blocks=2)
    else:
        policy_net = ContinuousPolicyNetwork(input_dim, hidden_dim=256, num_blocks=3)
        value_net = ValueNetwork(input_dim, hidden_dim=256, num_blocks=3)

    policy_net.load_state_dict(policy_ckpt["state_dict"])
    value_net.load_state_dict(value_ckpt["state_dict"])
    policy_net.eval()
    value_net.eval()
    return {
        "type": "rl",
        "policy_net": policy_net,
        "value_net": value_net,
        "input_dim": input_dim,
    }


def _resolve_model_dir(base_path: str, model_arg: str) -> str:
    if os.path.isabs(model_arg):
        return model_arg
    if os.path.isdir(model_arg):
        return os.path.abspath(model_arg)
    return os.path.join(base_path, "models", model_arg)


def load_dataset(data_dir: str):
    env_data = pd.read_csv(os.path.join(data_dir, "environment.csv"), dtype={"etf_code": str})
    training_data = pd.read_csv(os.path.join(data_dir, "training_scaled.csv")).values
    etf_codes = pd.read_csv(os.path.join(data_dir, "etf_codes.csv"), dtype={"etf_code": str})["etf_code"].values
    return env_data, training_data, etf_codes


def load_live_snapshot(api_token: str, end_date: str = None):
    client = QuantylabRestClient(token=api_token)
    result = fetch_latest_snapshot_from_api(client, list(TARGET_ETFS.keys()), end_date=end_date)
    snapshot = result["snapshot"]
    if snapshot.empty:
        raise ValueError("No live snapshot rows were returned from the API.")
    feature_cols = result["feature_columns"]
    env_data = snapshot[["date", "open", "high", "low", "close", "volume", "etf_code"]].copy()
    training_data = snapshot[feature_cols].values
    etf_codes = snapshot["etf_code"].astype(str).values
    return env_data, training_data, etf_codes


def score_etfs(model_info, training_data, env_data, etf_codes, lookback=1, device="cpu"):
    if model_info["type"] == "regression":
        return _score_etfs_regression(model_info["model"], training_data, env_data, etf_codes, lookback, device)
    return _score_etfs_rl(
        model_info["policy_net"],
        model_info["value_net"],
        training_data,
        env_data,
        etf_codes,
        lookback,
        device,
        model_input_dim=model_info.get("input_dim"),
    )


def _score_etfs_regression(model, training_data, env_data, etf_codes, lookback, device):
    results = []
    for code in np.unique(etf_codes):
        indices = np.where(etf_codes == code)[0]
        if len(indices) < lookback:
            continue
        target_indices = indices[-lookback:]
        features_tensor = torch.FloatTensor(training_data[target_indices]).to(device)
        with torch.no_grad():
            predicted_returns = model(features_tensor).cpu().numpy()
        last_env = env_data.iloc[indices[-1]]
        mean_pred = float(predicted_returns.mean())
        results.append(
            {
                "etf_code": str(code).zfill(6),
                "etf_name": TARGET_ETFS.get(str(code).zfill(6), ""),
                "date": int(last_env["date"]),
                "close": float(last_env["close"]),
                "predicted_return": mean_pred,
                "position_score": max(0.0, mean_pred * 100.0),
                "confidence": abs(mean_pred) * 1000.0,
                "value_score": mean_pred * 100.0,
            }
        )

    df = pd.DataFrame(results)
    if df.empty:
        return df
    df["composite_score"] = df["predicted_return"]
    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    df.index.name = "rank"
    return df


def _score_etfs_rl(policy_net, value_net, training_data, env_data, etf_codes, lookback, device, model_input_dim=None):
    results = []
    for code in np.unique(etf_codes):
        indices = np.where(etf_codes == code)[0]
        if len(indices) < lookback:
            continue
        states = []
        for idx in indices[-lookback:]:
            state = np.concatenate([training_data[idx], DEFAULT_PORTFOLIO_FEATURES])
            if model_input_dim is not None:
                if len(state) < model_input_dim:
                    state = np.concatenate([state, np.zeros(model_input_dim - len(state), dtype=np.float32)])
                elif len(state) > model_input_dim:
                    state = state[:model_input_dim]
            states.append(state)
        states_tensor = torch.FloatTensor(np.array(states)).to(device)
        with torch.no_grad():
            alpha, beta = policy_net(states_tensor)
            values = value_net(states_tensor)
        mean_position = (alpha / (alpha + beta)).cpu().numpy()
        concentration = (alpha + beta).cpu().numpy()
        value_scores = values.squeeze(-1).cpu().numpy()
        last_env = env_data.iloc[indices[-1]]
        results.append(
            {
                "etf_code": str(code).zfill(6),
                "etf_name": TARGET_ETFS.get(str(code).zfill(6), ""),
                "date": int(last_env["date"]),
                "close": float(last_env["close"]),
                "position_score": float(mean_position.mean()),
                "confidence": float(concentration.mean()),
                "value_score": float(value_scores.mean()),
            }
        )

    df = pd.DataFrame(results)
    if df.empty:
        return df
    v_min = df["value_score"].min()
    v_max = df["value_score"].max()
    if abs(v_max - v_min) > 1e-8:
        norm_value = (df["value_score"] - v_min) / (v_max - v_min)
    else:
        norm_value = 0.5
    df["composite_score"] = df["position_score"] * 0.6 + norm_value * 0.4
    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    df.index.name = "rank"
    return df


def select_investments(scores_df, top_n=10, min_position_score=0.5, min_composite_score=0.4):
    selected = scores_df[
        (scores_df["position_score"] >= min_position_score)
        & (scores_df["composite_score"] >= min_composite_score)
    ].head(top_n).copy()
    if len(selected) > 0:
        selected["weight"] = selected["position_score"] / selected["position_score"].sum()
    else:
        selected["weight"] = []
    summary = {
        "total_etfs": len(scores_df),
        "selected_etfs": len(selected),
        "avg_position_score": float(scores_df["position_score"].mean()),
        "avg_value_score": float(scores_df["value_score"].mean()),
        "market_signal": (
            "bullish"
            if scores_df["position_score"].mean() > 0.55
            else "bearish"
            if scores_df["position_score"].mean() < 0.45
            else "neutral"
        ),
    }
    return selected, summary


def main():
    parser = argparse.ArgumentParser(description="ETF single swing prediction")
    parser.add_argument(
        "--base-path",
        type=str,
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
    )
    parser.add_argument("--model", required=True, help="Model directory name under models/ or absolute path")
    parser.add_argument("--dataset", default=None, help="Dataset directory name under data/")
    parser.add_argument("--api-token", default=os.environ.get("QUANTYLAB_API_KEY"))
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--lookback", type=int, default=1)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--min-position", type=float, default=0.5)
    parser.add_argument("--min-composite", type=float, default=0.4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    model_dir = _resolve_model_dir(args.base_path, args.model)
    model_info = load_model(model_dir, args.device)

    if args.dataset:
        data_dir = os.path.join(args.base_path, "data", args.dataset)
        env_data, training_data, etf_codes = load_dataset(data_dir)
    else:
        if not args.api_token:
            parser.error("--dataset or --api-token/QUANTYLAB_API_KEY is required")
        env_data, training_data, etf_codes = load_live_snapshot(
            api_token=args.api_token,
            end_date=args.end_date,
        )

    scores = score_etfs(model_info, training_data, env_data, etf_codes, args.lookback, args.device)
    selected, summary = select_investments(scores, args.top_n, args.min_position, args.min_composite)

    print("=" * 70)
    print("ETF Ranking")
    print("=" * 70)
    for _, row in scores.head(20).iterrows():
        print(
            f"{row.name:>3d}. {row['etf_code']} {row['etf_name']:<20s} "
            f"Pos={row['position_score']:.3f} Val={row['value_score']:+.3f} "
            f"Comp={row['composite_score']:.3f} Close={row['close']:,.0f}"
        )

    print("\nSelected")
    print("=" * 70)
    print(f"market_signal={summary['market_signal']} selected={summary['selected_etfs']}/{summary['total_etfs']}")
    for _, row in selected.iterrows():
        print(
            f"{row['etf_code']} {row['etf_name']:<20s} "
            f"weight={row['weight']:.1%} pos={row['position_score']:.3f} comp={row['composite_score']:.3f}"
        )

    if args.output:
        output_path = args.output if os.path.isabs(args.output) else os.path.join(args.base_path, args.output)
        scores.to_csv(output_path)
        print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
