import argparse
import importlib.util
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb


def load_python_module(module_path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_boosters(model_glob: str) -> Tuple[List[xgb.Booster], List[str]]:
    model_paths = sorted(str(p) for p in Path().glob(model_glob))
    if not model_paths:
        raise FileNotFoundError(f"No model files found for pattern: {model_glob}")

    boosters: List[xgb.Booster] = []
    for model_path in model_paths:
        booster = xgb.Booster()
        booster.load_model(model_path)
        boosters.append(booster)

    return boosters, model_paths


def booster_objective(booster: xgb.Booster) -> str:
    cfg = booster.save_config()
    if "multi:softprob" in cfg:
        return "multi:softprob"
    return "regression"


def _feature_panel_to_long(factor_dfs: Dict[str, pd.DataFrame], mode: str) -> pd.DataFrame:
    if not factor_dfs:
        raise RuntimeError("No factor DataFrames provided")

    first_key = next(iter(factor_dfs.keys()))
    first_df = factor_dfs[first_key]

    if mode == "latest":
        if len(first_df.index) == 0:
            raise RuntimeError("Factor DataFrames are empty")
        latest_ts = first_df.index.max()
        out = pd.DataFrame(index=first_df.columns)
        out.index.name = "symbol"
        for feature_name, df in factor_dfs.items():
            out[feature_name] = df.loc[latest_ts].astype(np.float32)
        out["datetime"] = latest_ts
        out = out.set_index("datetime", append=True)
        out = out.reorder_levels([1, 0]).sort_index()
        return out

    long_columns = []
    for feature_name, df in factor_dfs.items():
        s = df.replace([np.inf, -np.inf], np.nan).stack(dropna=False)
        s.name = feature_name
        long_columns.append(s)

    out = pd.concat(long_columns, axis=1)
    out.index.names = ["datetime", "symbol"]
    return out


def _parse_sample_submission_ids(sample_submission_path: str) -> pd.MultiIndex:
    sample = pd.read_csv(sample_submission_path)
    if "id" not in sample.columns:
        raise ValueError(f"Missing 'id' column in {sample_submission_path}")

    datetimes: List[pd.Timestamp] = []
    symbols: List[str] = []
    for row_id in sample["id"].astype(str):
        if "_" not in row_id:
            continue
        dt_str, symbol = row_id.split("_", 1)
        datetimes.append(pd.to_datetime(dt_str))
        symbols.append(symbol)

    if not datetimes:
        raise RuntimeError("No valid id rows parsed from sample submission")

    return pd.MultiIndex.from_arrays([datetimes, symbols], names=["datetime", "symbol"])


def build_prediction_matrix(
    factor_dfs: Dict[str, pd.DataFrame],
    expected_features: Sequence[str],
    scope: str,
    sample_submission_path: Optional[str] = None,
) -> pd.DataFrame:
    if scope not in {"all", "latest", "sample"}:
        raise ValueError("scope must be one of: all, latest, sample")

    mode = "latest" if scope == "latest" else "all"
    feature_long = _feature_panel_to_long(factor_dfs, mode=mode)

    missing_features = [f for f in expected_features if f not in feature_long.columns]
    if missing_features:
        raise RuntimeError(
            "Missing required model features: " + ", ".join(missing_features[:12])
        )

    x = feature_long.loc[:, list(expected_features)].astype(np.float32)

    if scope == "sample":
        if sample_submission_path is None:
            raise ValueError("sample_submission_path is required when scope='sample'")
        target_index = _parse_sample_submission_ids(sample_submission_path)
        x = x.reindex(target_index).fillna(0.0)
    else:
        x = x.fillna(0.0)

    return x


def predict_ensemble(
    boosters: Sequence[xgb.Booster],
    x: pd.DataFrame,
    objective: str,
) -> np.ndarray:
    dmatrix = xgb.DMatrix(x, feature_names=list(x.columns))

    if objective == "multi:softprob":
        sample_pred = boosters[0].predict(dmatrix)
        if sample_pred.ndim == 1:
            num_class = int(round(float(sample_pred.shape[0]) ** 0.5))
            sample_pred = sample_pred.reshape(-1, num_class)
        pred = np.zeros_like(sample_pred, dtype=np.float32)
        for booster in boosters:
            model_pred = booster.predict(dmatrix)
            if model_pred.ndim == 1:
                model_pred = model_pred.reshape(sample_pred.shape)
            pred += model_pred.astype(np.float32)
        pred /= max(1, len(boosters))
        pred = np.clip(pred, 1e-8, 1.0)
        pred = pred / pred.sum(axis=1, keepdims=True)
        return pred

    pred = np.zeros(len(x), dtype=np.float32)
    for booster in boosters:
        pred += booster.predict(dmatrix).astype(np.float32)
    pred /= max(1, len(boosters))
    return pred


def write_regression_output(pred: np.ndarray, x_index: pd.MultiIndex, out_csv: str):
    out = pd.DataFrame(index=x_index)
    out["predict_return"] = pred.astype(np.float32)
    out = out.reset_index()
    out["id"] = out["datetime"].astype(str) + "_" + out["symbol"].astype(str)
    out[["id", "predict_return"]].to_csv(out_csv, index=False)


def write_3class_output(prob: np.ndarray, x_index: pd.MultiIndex, out_csv: str):
    if prob.shape[1] != 3:
        raise RuntimeError(f"Expected 3 classes for output, got {prob.shape[1]}")

    out = pd.DataFrame(index=x_index)
    out["prob_down"] = prob[:, 0]
    out["prob_neutral"] = prob[:, 1]
    out["prob_up"] = prob[:, 2]
    out["predict_class"] = np.argmax(prob, axis=1).astype(np.int32)
    out["confidence"] = np.max(prob, axis=1)
    out["signal_score"] = out["prob_up"] - out["prob_down"]

    out = out.reset_index()
    out["id"] = out["datetime"].astype(str) + "_" + out["symbol"].astype(str)
    out[
        [
            "id",
            "predict_class",
            "prob_down",
            "prob_neutral",
            "prob_up",
            "confidence",
            "signal_score",
        ]
    ].to_csv(out_csv, index=False)


def write_multiclass_time_output(
    prob: np.ndarray,
    x_index: pd.MultiIndex,
    class_names: Sequence[str],
    out_csv: str,
):
    out = pd.DataFrame(index=x_index)
    pred_ids = np.argmax(prob, axis=1).astype(np.int32)
    out["regime_pred_id"] = pred_ids
    out["regime_pred_name"] = [class_names[i] for i in pred_ids]
    for class_idx, class_name in enumerate(class_names):
        out[f"prob_{class_name}"] = prob[:, class_idx]

    out = out.reset_index()
    out.to_csv(out_csv, index=False)


def capture_factor_dfs_via_run(
    module_path: str,
    class_name: str,
    module_name: str,
    train_method_name: str = "train",
    arg_index_factor_dfs: int = 1,
) -> Dict[str, pd.DataFrame]:
    module = load_python_module(module_path, module_name)
    model_cls = getattr(module, class_name)
    original_train = getattr(model_cls, train_method_name)

    def patched_train(self, *args, **kwargs):
        setattr(self, "_captured_train_args", args)
        setattr(self, "_captured_train_kwargs", kwargs)
        print("Captured features in patched train; skipping model training.")
        return None

    setattr(model_cls, train_method_name, patched_train)
    try:
        model = model_cls()
        model.run()
    finally:
        setattr(model_cls, train_method_name, original_train)

    args = getattr(model, "_captured_train_args", None)
    if args is None or len(args) <= arg_index_factor_dfs:
        raise RuntimeError("Failed to capture factor_dfs from run()")

    factor_dfs = args[arg_index_factor_dfs]
    if not isinstance(factor_dfs, dict):
        raise RuntimeError("Captured factor_dfs is not a dict")
    return factor_dfs


def factor_dfs_from_load_panel_data(
    module_path: str,
    class_name: str,
    module_name: str,
    factor_position: int,
) -> Dict[str, pd.DataFrame]:
    module = load_python_module(module_path, module_name)
    model_cls = getattr(module, class_name)
    model = model_cls()
    payload = model.load_panel_data()
    if not isinstance(payload, tuple) or len(payload) <= factor_position:
        raise RuntimeError("Unexpected return payload from load_panel_data()")

    factor_dfs = payload[factor_position]
    if not isinstance(factor_dfs, dict):
        raise RuntimeError("factor_dfs payload is not dict")
    return factor_dfs


def add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument("--model-glob", required=True, help="Glob for model JSON files")
    parser.add_argument(
        "--scope",
        choices=["sample", "latest", "all"],
        default="sample",
        help="Which rows to predict: sample ids, latest timestamp only, or all rows",
    )
    parser.add_argument(
        "--sample-submission",
        default="sample_submission.csv",
        help="Path to sample_submission.csv (used when scope=sample)",
    )
    parser.add_argument("--out", required=True, help="Output CSV path")


def print_inference_header(model_paths: Sequence[str], x: pd.DataFrame):
    print(f"Loaded {len(model_paths)} model files")
    print(f"Prediction rows: {len(x):,}")
    print(f"Prediction features: {x.shape[1]}")
