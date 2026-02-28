from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
import json

import numpy as np
import pandas as pd


def rolling_z_score(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    if min_periods is None:
        min_periods = max(window // 2, 1)

    rolling_mean = series.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = series.rolling(window=window, min_periods=min_periods).std(ddof=0)
    rolling_std = rolling_std.replace(0, np.nan)
    return (series - rolling_mean) / rolling_std


def _symbol_universe_from_train(train_data_dir: Path) -> set[str]:
    if not train_data_dir.exists():
        return set()
    return {p.stem for p in train_data_dir.glob("*.parquet")}


def _flag_indicator(
    indicator_df: pd.DataFrame,
    symbols: set[str],
    window: int,
    z_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = [col for col in indicator_df.columns if col in symbols]
    if not cols:
        return pd.DataFrame(index=indicator_df.index), pd.DataFrame(index=indicator_df.index)

    numeric_df = indicator_df[cols].apply(pd.to_numeric, errors="coerce")
    z_scores = numeric_df.apply(lambda col: rolling_z_score(col, window=window))
    flags = z_scores.abs() >= z_threshold
    return z_scores, flags


def build_anomaly_report(
    data_cache_dir: str | Path,
    train_data_dir: str | Path,
    z_threshold: float = 3.0,
    window: int = 96,
    write_parquet: bool = True,
    max_indicators: int | None = None,
) -> dict:
    data_cache_dir = Path(data_cache_dir)
    train_data_dir = Path(train_data_dir)

    symbols = _symbol_universe_from_train(train_data_dir)
    indicator_files = sorted(data_cache_dir.glob("df_*.parquet"))
    if max_indicators is not None:
        indicator_files = indicator_files[:max_indicators]

    per_indicator = []
    latest_by_asset = defaultdict(list)

    for indicator_file in indicator_files:
        indicator_name = indicator_file.stem
        try:
            indicator_df = pd.read_parquet(indicator_file)
        except Exception:
            continue

        z_scores, flags = _flag_indicator(indicator_df, symbols=symbols, window=window, z_threshold=z_threshold)
        if flags.empty:
            continue

        if write_parquet:
            out_file = data_cache_dir / f"anomaly_flags_{indicator_name}.parquet"
            flags.to_parquet(out_file)

        last_idx = flags.index[-1]
        latest_flags = flags.iloc[-1]
        latest_scores = z_scores.iloc[-1]

        anomalous_assets = latest_flags[latest_flags].index.tolist()
        anomalous_scores = {
            asset: round(float(latest_scores[asset]), 4)
            for asset in anomalous_assets
            if pd.notna(latest_scores[asset])
        }

        for asset in anomalous_assets:
            latest_by_asset[asset].append(indicator_name)

        per_indicator.append(
            {
                "indicator": indicator_name,
                "timestamp": str(last_idx),
                "anomaly_count": len(anomalous_assets),
                "anomalous_assets": anomalous_assets,
                "scores": anomalous_scores,
            }
        )

    total_flags = sum(item["anomaly_count"] for item in per_indicator)
    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "z_threshold": z_threshold,
        "rolling_window": window,
        "symbols_in_train": len(symbols),
        "indicators_scanned": len(indicator_files),
        "indicators_with_flags": len(per_indicator),
        "total_current_flags": total_flags,
        "latest_by_asset": dict(latest_by_asset),
        "indicator_results": per_indicator,
    }
    return report


def save_report_json(report: dict, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
