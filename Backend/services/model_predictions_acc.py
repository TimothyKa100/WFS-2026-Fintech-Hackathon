import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

from core.config import config

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_PREDICTION_DATE = "2026-02-28"

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from json_inference_common import (  # noqa: E402
    booster_objective,
    build_prediction_matrix,
    factor_dfs_from_load_panel_data,
    predict_ensemble,
)


def _canonical_symbol(symbol: str) -> str:
    if symbol is None:
        return ""
    return "".join(ch for ch in str(symbol).upper() if ch.isalnum())


def _asset_label_map() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for asset in config.ASSET_UNIVERSE:
        name = asset["name"]
        mapping[_canonical_symbol(name)] = name
        mapping[_canonical_symbol(asset.get("yf", ""))] = name
        mapping[_canonical_symbol(asset.get("binance", ""))] = name

    alias_pairs = {
        "BTC": "Bitcoin",
        "ETH": "Ethereum",
        "VXX": "VIX ETF",
        "SPY": "S&P 500 ETF",
        "QQQ": "Nasdaq ETF",
        "DIA": "Dow ETF",
        "IWM": "Russell 2000 ETF",
        "AAPL": "Apple",
        "MSFT": "Microsoft",
        "NVDA": "Nvidia",
        "TSLA": "Tesla",
        "GLD": "Gold ETF",
        "SLV": "Silver ETF",
        "USO": "Oil ETF",
        "TLT": "20Y Treasury ETF",
    }
    for key, value in alias_pairs.items():
        mapping[_canonical_symbol(key)] = value

    return mapping


def _load_boosters_from_paths(model_paths: List[Path]) -> List[xgb.Booster]:
    boosters: List[xgb.Booster] = []
    for model_path in model_paths:
        if not model_path.exists():
            continue
        booster = xgb.Booster()
        booster.load_model(str(model_path))
        boosters.append(booster)
    return boosters


def _select_latest_rows_for_date(x: pd.DataFrame, date_str: str) -> Tuple[pd.DataFrame, str]:
    idx_dates = x.index.get_level_values(0)
    idx_dates_only = idx_dates.date
    target_date = pd.Timestamp(date_str).date()

    same_day_mask = idx_dates_only == target_date
    if same_day_mask.any():
        day_x = x[same_day_mask]
        used_date = str(target_date)
    else:
        available_dates = pd.Series(idx_dates_only).dropna().unique().tolist()
        prior_dates = sorted([d for d in available_dates if d <= target_date])
        if not prior_dates:
            prior_dates = sorted(available_dates)
        if not prior_dates:
            raise RuntimeError("No available feature rows for inference")
        best_date = prior_dates[-1]
        day_x = x[idx_dates_only == best_date]
        used_date = str(best_date)

    latest_dt_by_symbol = (
        day_x.reset_index()
        .groupby("symbol", as_index=False)["datetime"]
        .max()
        .set_index(["datetime", "symbol"])  # type: ignore[arg-type]
        .index
    )
    return day_x.reindex(latest_dt_by_symbol).dropna(how="all"), used_date


@lru_cache(maxsize=4)
def _stock7d_predictions_for_date(date_str: str) -> Tuple[pd.DataFrame, str]:
    model_paths = [ROOT_DIR / "7 days ahead stock.json"]
    boosters = _load_boosters_from_paths(model_paths)
    if not boosters:
        raise FileNotFoundError("Missing stock classification model: 7 days ahead stock.json")

    expected_features = boosters[0].feature_names or []
    if not expected_features:
        raise RuntimeError("Stock classification model is missing feature names")

    factor_dfs = factor_dfs_from_load_panel_data(
        module_path=str((ROOT_DIR / "7 days ahead stock.py").resolve()),
        class_name="Stock7DayClassificationModel",
        module_name="stock7d_cls_module_backend",
        factor_position=1,
    )

    x = build_prediction_matrix(
        factor_dfs=factor_dfs,
        expected_features=expected_features,
        scope="all",
        sample_submission_path=str((ROOT_DIR / "sample_submission.csv").resolve()),
    )

    x_day, used_date = _select_latest_rows_for_date(x, date_str)
    objective = booster_objective(boosters[0])
    pred = predict_ensemble(boosters, x_day, objective=objective)
    if pred.ndim != 2 or pred.shape[1] != 3:
        raise RuntimeError("Expected 3-class probability output for stock 7d model")

    out = pd.DataFrame(index=x_day.index)
    out["prob_down"] = pred[:, 0]
    out["prob_neutral"] = pred[:, 1]
    out["prob_up"] = pred[:, 2]
    out["predict_class"] = np.argmax(pred, axis=1).astype(np.int32)
    out["confidence"] = np.max(pred, axis=1)
    out["signal_score"] = out["prob_up"] - out["prob_down"]
    return out, used_date


@lru_cache(maxsize=4)
def _stock_vol_predictions_for_date(date_str: str) -> Tuple[pd.DataFrame, str]:
    model_paths = [ROOT_DIR / "stock_vol.json"]
    boosters = _load_boosters_from_paths(model_paths)
    if not boosters:
        raise FileNotFoundError("Missing stock volatility model: stock_vol.json")

    expected_features = boosters[0].feature_names or []
    if not expected_features:
        raise RuntimeError("Stock volatility model is missing feature names")

    factor_dfs = factor_dfs_from_load_panel_data(
        module_path=str((ROOT_DIR / "stock volatility.py").resolve()),
        class_name="StockVolatilityModel",
        module_name="stock_vol_module_backend",
        factor_position=1,
    )

    x = build_prediction_matrix(
        factor_dfs=factor_dfs,
        expected_features=expected_features,
        scope="all",
        sample_submission_path=str((ROOT_DIR / "sample_submission.csv").resolve()),
    )

    x_day, used_date = _select_latest_rows_for_date(x, date_str)
    objective = booster_objective(boosters[0])
    pred = predict_ensemble(boosters, x_day, objective=objective)
    if pred.ndim != 1:
        raise RuntimeError("Expected regression output for stock volatility model")

    out = pd.DataFrame(index=x_day.index)
    out["vol_pred"] = pred.astype(np.float32)
    return out, used_date


def _to_symbol_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.reset_index()
    out = out.sort_values(["symbol", "datetime"]).drop_duplicates(subset=["symbol"], keep="last")
    out = out.set_index("symbol")
    return out


def build_model_forecast_for_date(date_str: str = DEFAULT_PREDICTION_DATE):
    label_map = _asset_label_map()

    stock_cls, cls_date = _stock7d_predictions_for_date(date_str)
    stock_vol, vol_date = _stock_vol_predictions_for_date(date_str)

    cls_sym = _to_symbol_frame(stock_cls)
    vol_sym = _to_symbol_frame(stock_vol)

    symbols = sorted(set(cls_sym.index.tolist()) | set(vol_sym.index.tolist()))

    merged_rows = []
    for symbol in symbols:
        cls_row = cls_sym.loc[symbol] if symbol in cls_sym.index else None
        vol_row = vol_sym.loc[symbol] if symbol in vol_sym.index else None

        canonical = _canonical_symbol(symbol)
        label = label_map.get(canonical, str(symbol))

        predict_class = int(cls_row["predict_class"]) if cls_row is not None else 1
        class_label = {0: "down", 1: "neutral", 2: "up"}.get(predict_class, "neutral")

        merged_rows.append(
            {
                "symbol": str(symbol),
                "id": label,
                "label": label,
                "predict_class": predict_class,
                "class_label": class_label,
                "prob_down": float(cls_row["prob_down"]) if cls_row is not None else 0.0,
                "prob_neutral": float(cls_row["prob_neutral"]) if cls_row is not None else 1.0,
                "prob_up": float(cls_row["prob_up"]) if cls_row is not None else 0.0,
                "confidence": float(cls_row["confidence"]) if cls_row is not None else 0.0,
                "signal_score": float(cls_row["signal_score"]) if cls_row is not None else 0.0,
                "vol": float(vol_row["vol_pred"]) if vol_row is not None else 0.0,
            }
        )

    wanted_labels = {asset["name"] for asset in config.ASSET_UNIVERSE}
    filtered_rows = [r for r in merged_rows if r["label"] in wanted_labels]

    if not filtered_rows:
        filtered_rows = merged_rows

    trend_up = sorted(filtered_rows, key=lambda r: r["prob_up"], reverse=True)[:8]
    trend_down = sorted(filtered_rows, key=lambda r: r["prob_down"], reverse=True)[:8]
    vol_forecast = sorted(filtered_rows, key=lambda r: r["vol"], reverse=True)

    for row in trend_up:
        row["prob"] = row["prob_up"]
    for row in trend_down:
        row["prob"] = row["prob_down"]

    forecast = {
        "as_of_date_requested": date_str,
        "as_of_date_used": min(cls_date, vol_date),
        "asset_predictions": filtered_rows,
        "trend_up": trend_up,
        "trend_down": trend_down,
        "vol_forecast": [
            {"id": r["id"], "label": r["label"], "vol": r["vol"], "symbol": r["symbol"]}
            for r in vol_forecast
        ],
    }

    cache_dir = ROOT_DIR / "data_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"model_forecast_{date_str}.json"
    pd.DataFrame(filtered_rows).to_json(out_path, orient="records", indent=2)

    return forecast
