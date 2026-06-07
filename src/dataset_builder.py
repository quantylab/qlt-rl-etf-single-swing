import argparse
import json
import os
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from api_client import QuantylabRestClient
from target_etfs import TARGET_ETFS


def _normalize_date(value: object) -> str:
    text = str(value).strip()
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) < 8:
        raise ValueError(f"Invalid date value: {value}")
    return digits[:8]


def _find_column(columns: Iterable[str], *candidates: str) -> Optional[str]:
    normalized = {col.lower().replace("_", ""): col for col in columns}
    for candidate in candidates:
        key = candidate.lower().replace("_", "")
        if key in normalized:
            return normalized[key]
    return None


def _normalize_candle_frame(records: List[Dict[str, object]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

    frame = pd.DataFrame(records)
    date_col = _find_column(frame.columns, "date", "x", "bas_dt", "stck_bsop_date")
    open_col = _find_column(frame.columns, "open", "stck_oprc", "시가")
    high_col = _find_column(frame.columns, "high", "stck_hgpr", "고가")
    low_col = _find_column(frame.columns, "low", "stck_lwpr", "저가")
    close_col = _find_column(frame.columns, "close", "stck_clpr", "현재가", "종가")
    volume_col = _find_column(frame.columns, "volume", "acml_vol", "누적거래량", "거래량")

    required = [date_col, open_col, high_col, low_col, close_col]
    if any(col is None for col in required):
        raise ValueError(f"Unexpected candle schema: {list(frame.columns)}")

    result = pd.DataFrame(
        {
            "date": frame[date_col].map(_normalize_date),
            "open": pd.to_numeric(frame[open_col], errors="coerce"),
            "high": pd.to_numeric(frame[high_col], errors="coerce"),
            "low": pd.to_numeric(frame[low_col], errors="coerce"),
            "close": pd.to_numeric(frame[close_col], errors="coerce"),
            "volume": pd.to_numeric(frame[volume_col], errors="coerce") if volume_col else 0.0,
        }
    )
    result = result.dropna(subset=["date", "open", "high", "low", "close"]).copy()
    result["date"] = result["date"].astype(str)
    result["volume"] = result["volume"].fillna(0.0)
    return result.sort_values("date").reset_index(drop=True)


def _feature_frame_from_records(records: List[Dict[str, object]]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    feature_names: Optional[Sequence[str]] = None

    for record in records:
        meta = record.get("meta") or {}
        names = meta.get("features") or feature_names
        values = record.get("y") or []
        if names is None:
            names = [f"f_{idx:03d}" for idx in range(len(values))]
        if len(names) != len(values):
            raise ValueError("Feature name and value length mismatch.")
        feature_names = names
        row = {"date": _normalize_date(record.get("x"))}
        row.update(dict(zip(feature_names, values)))
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    feature_cols = [col for col in frame.columns if col != "date"]
    frame[feature_cols] = frame[feature_cols].apply(pd.to_numeric, errors="coerce")
    frame[feature_cols] = frame[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return frame


def _compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    return 100.0 - (100.0 / (1.0 + rs))


def build_features_from_candles(candles: pd.DataFrame) -> pd.DataFrame:
    frame = candles.copy().sort_values("date").reset_index(drop=True)
    close = frame["close"].astype(float)
    open_ = frame["open"].astype(float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    volume = frame["volume"].astype(float)
    prev_close = close.shift(1)

    feature_df = pd.DataFrame({"date": frame["date"]})
    feature_df["ret_1"] = close.pct_change()
    feature_df["ret_3"] = close.pct_change(3)
    feature_df["ret_5"] = close.pct_change(5)
    feature_df["ret_10"] = close.pct_change(10)
    feature_df["ret_20"] = close.pct_change(20)
    feature_df["gap_open"] = open_ / prev_close - 1.0
    feature_df["intraday_return"] = close / open_ - 1.0
    feature_df["range_pct"] = (high - low) / (close + 1e-8)
    feature_df["upper_shadow"] = high / np.maximum(open_, close) - 1.0
    feature_df["lower_shadow"] = np.minimum(open_, close) / (low + 1e-8) - 1.0
    feature_df["volume_change_1"] = volume.pct_change()
    feature_df["volume_change_5"] = volume / (volume.rolling(5).mean() + 1e-8) - 1.0
    feature_df["volume_change_20"] = volume / (volume.rolling(20).mean() + 1e-8) - 1.0

    for window in [5, 10, 20, 60]:
        ma = close.rolling(window).mean()
        feature_df[f"ma_ratio_{window}"] = close / (ma + 1e-8) - 1.0
        feature_df[f"volatility_{window}"] = feature_df["ret_1"].rolling(window).std()
        feature_df[f"momentum_{window}"] = close / close.shift(window) - 1.0

    feature_df["rsi_14"] = _compute_rsi(close, 14) / 100.0
    feature_df["price_z_20"] = (close - close.rolling(20).mean()) / (close.rolling(20).std() + 1e-8)
    feature_df["price_z_60"] = (close - close.rolling(60).mean()) / (close.rolling(60).std() + 1e-8)

    feature_cols = [col for col in feature_df.columns if col != "date"]
    feature_df[feature_cols] = feature_df[feature_cols].replace([np.inf, -np.inf], np.nan)
    feature_df = feature_df.dropna().reset_index(drop=True)
    feature_df[feature_cols] = feature_df[feature_cols].fillna(0.0)
    return feature_df


def build_feature_vectors_from_candles(candles: pd.DataFrame) -> pd.DataFrame:
    """사용자 OHLCV 데이터에서 feature vector를 생성합니다.

    입력 컬럼:
    - date
    - open
    - high
    - low
    - close
    - volume (선택, 없으면 0으로 처리)

    반환값:
    - date + feature columns DataFrame
    """
    candle_frame = _normalize_candle_frame(candles.to_dict("records"))
    return build_features_from_candles(candle_frame)


def _prepare_dataset_frames(candles: pd.DataFrame):
    features = build_features_from_candles(candles)
    if features.empty:
        return None
    env = candles.merge(features[["date"]], on="date", how="inner").reset_index(drop=True)
    env = env[["date", "open", "high", "low", "close", "volume"]].copy()
    training = features.drop(columns=["date"]).reset_index(drop=True)
    return env, training


def fetch_latest_snapshot_from_api(
    client: QuantylabRestClient,
    codes: Sequence[str],
    end_date: Optional[str] = None,
) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    start_date = None
    if end_date:
        end = pd.Timestamp(_normalize_date(end_date))
        start_date = (end - pd.Timedelta(days=180)).strftime("%Y%m%d")

    for code in codes:
        feature_records = client.get_feature_vectors(code=code, start_date=start_date, end_date=end_date, n=1)
        candle_records = client.get_stock_candles(code=code, start_date=start_date, end_date=end_date)
        feature_frame = _feature_frame_from_records(feature_records)
        candle_frame = _normalize_candle_frame(candle_records)
        if feature_frame.empty or candle_frame.empty:
            continue
        merged = candle_frame.merge(feature_frame, on="date", how="inner")
        if merged.empty:
            continue
        latest = merged.iloc[-1].to_dict()
        latest["etf_code"] = code
        latest["etf_name"] = TARGET_ETFS.get(code, "")
        rows.append(latest)

    if not rows:
        return {"snapshot": pd.DataFrame(), "feature_columns": []}

    snapshot = pd.DataFrame(rows).sort_values(["date", "etf_code"]).reset_index(drop=True)
    feature_columns = [
        col for col in snapshot.columns
        if col not in {"date", "open", "high", "low", "close", "volume", "etf_code", "etf_name"}
    ]
    return {"snapshot": snapshot, "feature_columns": feature_columns}


def build_dataset_from_api(
    output_dir: str,
    token: str,
    start_date: str,
    end_date: Optional[str] = None,
    codes: Optional[Sequence[str]] = None,
    feature_builder=None,
) -> Dict[str, object]:
    os.makedirs(output_dir, exist_ok=True)
    client = QuantylabRestClient(token=token)
    codes = list(codes or TARGET_ETFS.keys())

    all_env: List[pd.DataFrame] = []
    all_features: List[pd.DataFrame] = []
    all_codes: List[str] = []
    feature_columns: Optional[List[str]] = None
    skipped: List[str] = []

    for code in codes:
        candle_records = client.get_stock_candles(code=code, start_date=start_date, end_date=end_date)
        candle_frame = _normalize_candle_frame(candle_records)
        if candle_frame.empty:
            skipped.append(code)
            continue
        if feature_builder is None:
            feature_records = client.get_feature_vectors(code=code, start_date=start_date, end_date=end_date)
            feature_frame = _feature_frame_from_records(feature_records)
        else:
            feature_frame = feature_builder(candle_frame)
        if feature_frame.empty:
            skipped.append(code)
            continue
        merged = candle_frame.merge(feature_frame, on="date", how="inner")
        if merged.empty:
            skipped.append(code)
            continue
        env_frame = merged[["date", "open", "high", "low", "close", "volume"]].copy()
        feature_frame = merged.drop(columns=["date", "open", "high", "low", "close", "volume"]).reset_index(drop=True)
        env_frame["etf_code"] = code
        current_feature_columns = list(feature_frame.columns)

        if feature_columns is None:
            feature_columns = current_feature_columns
        else:
            missing_new = [col for col in current_feature_columns if col not in feature_columns]
            if missing_new:
                for col in missing_new:
                    for prev in all_features:
                        prev[col] = 0.0
                feature_columns.extend(missing_new)

        for col in feature_columns:
            if col not in feature_frame.columns:
                feature_frame[col] = 0.0

        all_env.append(env_frame.reset_index(drop=True))
        all_features.append(feature_frame[feature_columns].reset_index(drop=True))
        all_codes.extend([code] * len(env_frame))

    if not all_env or feature_columns is None:
        raise ValueError("No dataset rows were built from the API.")

    env_data = pd.concat(all_env, ignore_index=True).sort_values(["date", "etf_code"]).reset_index(drop=True)
    training_selected = pd.concat(all_features, ignore_index=True).reset_index(drop=True)
    etf_codes = pd.DataFrame({"etf_code": all_codes})
    training_scaled = training_selected.copy()

    env_path = os.path.join(output_dir, "environment.csv")
    selected_path = os.path.join(output_dir, "training_selected.csv")
    train_path = os.path.join(output_dir, "training_scaled.csv")
    code_path = os.path.join(output_dir, "etf_codes.csv")
    meta_path = os.path.join(output_dir, "dataset_meta.json")

    env_data.to_csv(env_path, index=False)
    training_selected.to_csv(selected_path, index=False)
    training_scaled.to_csv(train_path, index=False)
    etf_codes.to_csv(code_path, index=False)

    with open(meta_path, "w", encoding="utf-8") as fp:
        json.dump(
            {
                "start_date": start_date,
                "end_date": end_date,
                "codes": codes,
                "feature_columns": feature_columns,
                "rows": len(env_data),
                "skipped_codes": skipped,
                "feature_source": "local-candles" if feature_builder is not None else "feature-vectors",
            },
            fp,
            ensure_ascii=False,
            indent=2,
        )

    return {
        "environment_path": env_path,
        "selected_path": selected_path,
        "training_path": train_path,
        "etf_codes_path": code_path,
        "meta_path": meta_path,
        "rows": len(env_data),
        "feature_count": len(feature_columns),
        "codes": len(set(all_codes)),
        "skipped_codes": skipped,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ETF single swing dataset from Quantylab REST API.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--token", default=os.environ.get("QUANTYLAB_API_KEY"))
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--codes", nargs="*", default=None)
    args = parser.parse_args()

    if not args.token:
        parser.error("--token or QUANTYLAB_API_KEY is required")

    result = build_dataset_from_api(
        output_dir=args.output_dir,
        token=args.token,
        start_date=args.start_date,
        end_date=args.end_date,
        codes=args.codes,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
