import datetime
import glob
import os
import time

import numpy as np
import pandas as pd  # type: ignore
import xgboost as xgb  # type: ignore
from sklearn.model_selection import TimeSeriesSplit  # type: ignore

BASE_DIR = os.getcwd()
DATA_DIR_CANDIDATES = [
    os.path.join(BASE_DIR, "SCRAPPED_DATA"),
    os.path.join(BASE_DIR, "SCRAPED_DATA"),
]
SAMPLE_SUBMISSION_PATH = os.path.join(BASE_DIR, "sample_submission.csv")
SUBMISSION_PATH = os.path.join(BASE_DIR, "submit.csv")


class StockVolatilityModel:
    def __init__(self):
        self.start_datetime = datetime.datetime(2021, 3, 1)
        self.forecast_horizon_days = int(os.getenv("VOL_HORIZON_DAYS", "7"))
        self.base_vol_window = int(os.getenv("VOL_BASE_WINDOW", "20"))
        self.data_dir = self._resolve_data_dir()
        self.device = os.getenv("VOL_DEVICE", "cuda")
        self.train_seeds = [42, 2026, 3407]
        self.n_splits = int(os.getenv("VOL_N_SPLITS", "6"))
        self.market_neutral_strength = float(os.getenv("VOL_MKT_NEUTRAL", "0.35"))
        self.feature_prune_ratio = float(os.getenv("VOL_FEATURE_PRUNE_RATIO", "0.65"))
        self.holdout_days = int(os.getenv("VOL_HOLDOUT_DAYS", "252"))
        self.liquid_top_n = int(os.getenv("VOL_LIQUID_TOP_N", "220"))
        self.output_tag = f"stock_vol_{self.forecast_horizon_days}d"

    def _resolve_data_dir(self):
        for candidate in DATA_DIR_CANDIDATES:
            if os.path.isdir(candidate):
                print(f"Using stock data directory: {candidate}")
                return candidate
        raise FileNotFoundError(
            "Cannot find SCRAPPED_DATA or SCRAPED_DATA directory in workspace"
        )

    @staticmethod
    def _time_series_zscore(df, window=60, min_periods=20):
        roll_mean = df.rolling(window=window, min_periods=min_periods).mean()
        roll_std = df.rolling(window=window, min_periods=min_periods).std().replace(0, np.nan)
        return (df - roll_mean) / roll_std

    @staticmethod
    def _cross_sectional_zscore(df):
        cs_mean = df.mean(axis=1)
        cs_std = df.std(axis=1).replace(0, np.nan)
        return df.sub(cs_mean, axis=0).div(cs_std, axis=0)

    @staticmethod
    def _safe_spearman(x, y):
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 200:
            return np.nan
        x_series = pd.Series(x[valid])
        y_series = pd.Series(y[valid])
        corr = x_series.corr(y_series, method="spearman")
        return float(corr) if np.isfinite(corr) else np.nan

    @staticmethod
    def weighted_spearmanr(y_true, y_pred):
        y_true = pd.Series(y_true)
        y_pred = pd.Series(y_pred, index=y_true.index)
        n = len(y_true)
        if n < 3:
            return np.nan

        r_true = y_true.rank(ascending=False, method="average")
        r_pred = y_pred.rank(ascending=False, method="average")
        x = 2 * (r_true - 1) / (n - 1) - 1
        w = x**2
        w_sum = w.sum()
        if w_sum == 0:
            return np.nan

        mu_true = (w * r_true).sum() / w_sum
        mu_pred = (w * r_pred).sum() / w_sum
        cov = (w * (r_true - mu_true) * (r_pred - mu_pred)).sum()
        var_true = (w * (r_true - mu_true) ** 2).sum()
        var_pred = (w * (r_pred - mu_pred) ** 2).sum()
        denom = np.sqrt(var_true * var_pred)
        return float(cov / denom) if denom > 0 else np.nan

    @staticmethod
    def regression_metrics(y_true, y_pred):
        y_true_values = y_true.values if isinstance(y_true, pd.Series) else np.asarray(y_true)
        y_pred_values = np.asarray(y_pred)
        err = y_true_values - y_pred_values
        mae = float(np.nanmean(np.abs(err)))
        rmse = float(np.sqrt(np.nanmean(err**2)))
        return mae, rmse

    @staticmethod
    def _standardize_ohlcv(df):
        rename_map = {
            "timestamp": "timestamp",
            "Timestamp": "timestamp",
            "date": "timestamp",
            "Date": "timestamp",
            "open": "open_price",
            "Open": "open_price",
            "high": "high_price",
            "High": "high_price",
            "low": "low_price",
            "Low": "low_price",
            "close": "close_price",
            "Close": "close_price",
            "adj close": "adj_close",
            "Adj Close": "adj_close",
            "volume": "volume",
            "Volume": "volume",
        }

        df = df.rename(columns={col: rename_map.get(col, col) for col in df.columns})
        required = [
            "timestamp",
            "open_price",
            "high_price",
            "low_price",
            "close_price",
            "volume",
        ]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required OHLCV columns: {missing}")

        if "adj_close" in df.columns:
            adj_ratio = df["adj_close"] / df["close_price"].replace(0, np.nan)
            for col in ["open_price", "high_price", "low_price", "close_price"]:
                df[col] = df[col] * adj_ratio

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
        return df.set_index("timestamp")

    def _compute_symbol_features(self, df):
        close = df["close_price"]
        high = df["high_price"]
        low = df["low_price"]
        open_price = df["open_price"]
        volume = df["volume"].replace(0, np.nan)

        ret_1d = close.pct_change()
        log_ret_1d = np.log(close / close.shift(1).replace(0, np.nan))
        abs_ret_1d = log_ret_1d.abs()

        rv_5 = (log_ret_1d**2).rolling(5, min_periods=3).mean()
        rv_10 = (log_ret_1d**2).rolling(10, min_periods=5).mean()
        rv_20 = (log_ret_1d**2).rolling(20, min_periods=10).mean()
        rv_60 = (log_ret_1d**2).rolling(60, min_periods=20).mean()

        ew_vol_10 = log_ret_1d.ewm(span=10, adjust=False, min_periods=5).std()
        ew_vol_20 = log_ret_1d.ewm(span=20, adjust=False, min_periods=10).std()

        downside_vol_20 = log_ret_1d.clip(upper=0).rolling(20, min_periods=10).std()
        upside_vol_20 = log_ret_1d.clip(lower=0).rolling(20, min_periods=10).std()

        prev_close = close.shift(1)
        tr = pd.concat(
            [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        atr_pct_14 = tr.rolling(14, min_periods=7).mean() / close.replace(0, np.nan)

        hl_log = np.log(high / low.replace(0, np.nan))
        co_log = np.log(close / open_price.replace(0, np.nan))
        parkinson_var = (hl_log**2) / (4 * np.log(2))
        gk_var = 0.5 * (hl_log**2) - (2 * np.log(2) - 1) * (co_log**2)
        gk_var = gk_var.clip(lower=0)

        park_vol_10 = np.sqrt(parkinson_var.rolling(10, min_periods=5).mean())
        park_vol_20 = np.sqrt(parkinson_var.rolling(20, min_periods=10).mean())
        gk_vol_10 = np.sqrt(gk_var.rolling(10, min_periods=5).mean())
        gk_vol_20 = np.sqrt(gk_var.rolling(20, min_periods=10).mean())

        sma20 = close.rolling(20, min_periods=10).mean()
        std20 = close.rolling(20, min_periods=10).std()
        bb_upper = sma20 + 2 * std20
        bb_lower = sma20 - 2 * std20

        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta.where(delta < 0, 0.0)).astype(float)
        avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi14 = 100 - (100 / (1 + rs))

        log_vol = np.log1p(volume)
        vol_z20 = (log_vol - log_vol.rolling(20, min_periods=10).mean()) / log_vol.rolling(
            20, min_periods=10
        ).std().replace(0, np.nan)

        rv_base = rv_20
        vol_of_vol_20 = rv_base.pct_change().rolling(20, min_periods=10).std()

        feature_map = {
            "ret_1d": ret_1d,
            "ret_5d": close.pct_change(5),
            "ret_20d": close.pct_change(20),
            "abs_ret_1d": abs_ret_1d,
            "rv_5": rv_5,
            "rv_10": rv_10,
            "rv_20": rv_20,
            "rv_60": rv_60,
            "rv_ratio_5_20": rv_5 / rv_20.replace(0, np.nan) - 1,
            "rv_ratio_10_60": rv_10 / rv_60.replace(0, np.nan) - 1,
            "ew_vol_10": ew_vol_10,
            "ew_vol_20": ew_vol_20,
            "down_up_vol_ratio": downside_vol_20 / upside_vol_20.replace(0, np.nan) - 1,
            "atr_pct_14": atr_pct_14,
            "range_pct": (high - low) / close.replace(0, np.nan),
            "gap_open": (open_price - close.shift(1)) / close.shift(1).replace(0, np.nan),
            "close_open_pct": (close - open_price) / open_price.replace(0, np.nan),
            "park_vol_10": park_vol_10,
            "park_vol_20": park_vol_20,
            "gk_vol_10": gk_vol_10,
            "gk_vol_20": gk_vol_20,
            "vol_of_vol_20": vol_of_vol_20,
            "bb_width_20": (bb_upper - bb_lower) / sma20.replace(0, np.nan),
            "bb_pos_20": (close - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan),
            "rsi14": rsi14,
            "vol_z20": vol_z20,
            "vol_mom_5_20": log_vol.rolling(5, min_periods=3).mean()
            - log_vol.rolling(20, min_periods=10).mean(),
            "rv_volume_interact": rv_10 * vol_z20,
            "ret_skew_20": log_ret_1d.rolling(20, min_periods=10).skew(),
            "ret_kurt_20": log_ret_1d.rolling(20, min_periods=10).kurt(),
        }

        features = pd.DataFrame(feature_map, index=df.index)
        features = features.replace([np.inf, -np.inf], np.nan)

        h = max(2, self.forecast_horizon_days)
        eps = 1e-8
        current_var = (log_ret_1d**2).rolling(self.base_vol_window, min_periods=self.base_vol_window // 2).mean()
        future_var = (
            (log_ret_1d**2)
            .iloc[::-1]
            .rolling(h, min_periods=max(2, h // 2))
            .mean()
            .iloc[::-1]
            .shift(-1)
        )
        raw_target = np.log((future_var + eps) / (current_var + eps))
        raw_target = raw_target.replace([np.inf, -np.inf], np.nan)

        return features, raw_target

    def load_panel_data(self):
        t0 = time.monotonic()
        parquet_files = sorted(glob.glob(os.path.join(self.data_dir, "*.parquet")))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {self.data_dir}")

        feature_per_symbol = {}
        target_per_symbol = {}
        liquidity_score = {}

        for file_path in parquet_files:
            symbol = os.path.splitext(os.path.basename(file_path))[0]
            try:
                raw_df = pd.read_parquet(file_path)
                std_df = self._standardize_ohlcv(raw_df)
                min_rows = max(500, self.base_vol_window * 10)
                if len(std_df) < min_rows:
                    continue

                dollar_volume = (std_df["close_price"] * std_df["volume"]).replace(
                    [np.inf, -np.inf], np.nan
                )
                liquidity_score[symbol] = float(dollar_volume.median(skipna=True))

                feat_df, tgt = self._compute_symbol_features(std_df)
                feature_per_symbol[symbol] = feat_df
                target_per_symbol[symbol] = tgt
            except Exception as exc:
                print(f"Skipping {symbol}: {exc}")

        if not feature_per_symbol:
            raise RuntimeError("No valid symbols loaded from stock data")

        if self.liquid_top_n > 0 and len(feature_per_symbol) > self.liquid_top_n:
            sorted_syms = sorted(
                feature_per_symbol.keys(),
                key=lambda s: liquidity_score.get(s, -np.inf),
                reverse=True,
            )
            keep_syms = set(sorted_syms[: self.liquid_top_n])
            feature_per_symbol = {
                s: df for s, df in feature_per_symbol.items() if s in keep_syms
            }
            target_per_symbol = {
                s: ser for s, ser in target_per_symbol.items() if s in keep_syms
            }
            print(f"Liquidity universe: kept top {len(keep_syms)} symbols by median dollar volume")

        all_symbols = sorted(feature_per_symbol.keys())
        all_dates = sorted(set().union(*[set(df.index) for df in feature_per_symbol.values()]))
        common_index = pd.DatetimeIndex(all_dates)

        feature_names = list(next(iter(feature_per_symbol.values())).columns)
        factor_dfs = {}
        for feature_name in feature_names:
            wide = pd.DataFrame(
                {
                    symbol: feature_per_symbol[symbol][feature_name].reindex(common_index)
                    for symbol in all_symbols
                },
                index=common_index,
            )
            factor_dfs[feature_name] = wide

        raw_target = pd.DataFrame(
            {symbol: target_per_symbol[symbol].reindex(common_index) for symbol in all_symbols},
            index=common_index,
        )

        keep_slice = slice(self.start_datetime, None)
        raw_target = raw_target.loc[keep_slice].astype(np.float32)

        market_component = raw_target.mean(axis=1)
        df_target = raw_target.sub(self.market_neutral_strength * market_component, axis=0)
        print(
            f"Target mode: market-neutralized forward volatility log-ratio "
            f"(strength={self.market_neutral_strength})"
        )

        normalized_factors = {}
        for name, df in factor_dfs.items():
            clipped = df.loc[keep_slice].replace([np.inf, -np.inf], np.nan)
            clipped = clipped.clip(
                lower=clipped.quantile(0.01, axis=1),
                upper=clipped.quantile(0.99, axis=1),
                axis=0,
            )
            ts_norm = self._time_series_zscore(clipped)
            cs_norm = self._cross_sectional_zscore(clipped)
            normalized_factors[f"{name}_ts"] = ts_norm.astype(np.float32)
            normalized_factors[f"{name}_cs"] = cs_norm.astype(np.float32)

        print(
            f"Loaded symbols={len(all_symbols)}, features={len(normalized_factors)}, "
            f"rows={len(df_target)} in {time.monotonic() - t0:.1f}s"
        )
        return all_symbols, normalized_factors, df_target

    def _sample_weights(self, y_train):
        abs_y = y_train.abs()
        q80 = abs_y.quantile(0.80)
        q95 = abs_y.quantile(0.95)

        tail_weight = np.where(abs_y >= q95, 2.5, np.where(abs_y >= q80, 1.6, 1.0))

        dates = pd.Series(y_train.index.get_level_values(0))
        recency = dates.rank(method="average", pct=True).to_numpy(dtype=np.float32)
        recency_weight = 0.7 + 0.6 * recency

        return (tail_weight * recency_weight).astype(np.float32)

    def train_and_predict(self, factor_dfs, df_target):
        target_long = df_target.stack()
        target_long.name = "target"
        common_index = target_long.index

        factor_long = []
        for feature_name, df in factor_dfs.items():
            series = df.replace([np.inf, -np.inf], np.nan).stack().reindex(common_index)
            series.name = feature_name
            factor_long.append(series)

        data = pd.concat([*factor_long, target_long], axis=1)
        data = data.dropna(subset=["target"])

        feature_names = list(factor_dfs.keys())
        X = data[feature_names]
        y = data["target"]

        min_feature_count = max(10, int(0.7 * len(feature_names)))
        valid_mask = X.notna().sum(axis=1) >= min_feature_count
        X = X[valid_mask].fillna(0.0).astype(np.float32)
        y = y[valid_mask].astype(np.float32)
        data = data.loc[X.index]

        print(f"Training rows after filtering: {len(X):,}")

        times = X.index.get_level_values(0).to_numpy()
        uniq_times = np.unique(times)

        holdout_active = self.holdout_days > 0 and len(uniq_times) > (self.holdout_days + 350)
        if holdout_active:
            holdout_times = uniq_times[-self.holdout_days :]
            cv_times = uniq_times[: -self.holdout_days]
            holdout_mask_all = np.isin(times, holdout_times)
            cv_mask_all = np.isin(times, cv_times)
            print(
                f"Time holdout enabled: last {self.holdout_days} days "
                f"({len(holdout_times)} timestamps)"
            )
        else:
            holdout_mask_all = np.zeros(len(X), dtype=bool)
            cv_mask_all = np.ones(len(X), dtype=bool)
            if self.holdout_days > 0:
                print("Time holdout disabled due to insufficient history length")

        X_cv = X[cv_mask_all]
        y_cv = y[cv_mask_all]
        times_cv = X_cv.index.get_level_values(0).to_numpy()
        uniq_times_cv = np.unique(times_cv)

        feature_names = list(X_cv.columns)
        feature_select_cut = max(160, int(len(uniq_times_cv) * 0.60))
        fs_times = uniq_times_cv[:feature_select_cut]
        fs_mask = np.isin(times_cv, fs_times)
        X_fs = X_cv[fs_mask]
        y_fs = y_cv[fs_mask]

        feature_scores = []
        y_fs_values = y_fs.values.astype(np.float64)
        for name in feature_names:
            x_vals = X_fs[name].values.astype(np.float64)
            score = self._safe_spearman(x_vals, y_fs_values)
            feature_scores.append((name, abs(score) if np.isfinite(score) else 0.0))

        feature_scores = sorted(feature_scores, key=lambda t: t[1], reverse=True)
        keep_n_features = max(24, int(len(feature_names) * self.feature_prune_ratio))
        selected_features = [name for name, _ in feature_scores[:keep_n_features]]
        X = X[selected_features]
        X_cv = X_cv[selected_features]
        print(
            f"Feature pruning kept {len(selected_features)}/{len(feature_names)} "
            f"(ratio={self.feature_prune_ratio:.2f})"
        )

        purge_gap = max(self.forecast_horizon_days + 2, 7)
        test_size = min(252 * 2, max(120, len(uniq_times_cv) // (self.n_splits + 1)))
        tscv = TimeSeriesSplit(
            n_splits=self.n_splits,
            gap=purge_gap,
            max_train_size=252 * 5,
            test_size=test_size,
        )

        base_params = {
            "objective": "reg:pseudohubererror",
            "learning_rate": 0.02,
            "max_depth": 5,
            "min_child_weight": 12,
            "subsample": 0.75,
            "colsample_bytree": 0.65,
            "reg_alpha": 1.2,
            "reg_lambda": 10.0,
            "gamma": 0.25,
            "tree_method": "hist",
            "device": self.device,
            "max_bin": 256,
            "eval_metric": ["rmse", "mae"],
        }

        fold_model_groups = []
        fold_scores = []
        fold_maes = []
        oof_pred = np.full(len(X_cv), np.nan, dtype=np.float32)

        if holdout_active and holdout_mask_all.any():
            X_holdout = X[holdout_mask_all]
            y_holdout = y[holdout_mask_all]
            d_holdout = xgb.DMatrix(X_holdout)
        else:
            X_holdout = None
            y_holdout = None
            d_holdout = None

        for fold, (train_t_idx, val_t_idx) in enumerate(tscv.split(uniq_times_cv), start=1):
            train_times = uniq_times_cv[train_t_idx]
            val_times = uniq_times_cv[val_t_idx]

            train_mask = np.isin(times_cv, train_times)
            val_mask = np.isin(times_cv, val_times)

            X_train, X_val = X_cv[train_mask], X_cv[val_mask]
            y_train, y_val = y_cv[train_mask], y_cv[val_mask]
            w_train = self._sample_weights(y_train)

            d_val_predict = xgb.DMatrix(X_val)
            seed_models = []
            val_pred = np.zeros(len(X_val), dtype=np.float32)

            for seed in self.train_seeds:
                params = dict(base_params)
                params["random_state"] = seed

                d_train = xgb.QuantileDMatrix(X_train, y_train, weight=w_train, max_bin=256)
                d_val = xgb.QuantileDMatrix(X_val, y_val, max_bin=256, ref=d_train)

                model = xgb.train(
                    params=params,
                    dtrain=d_train,
                    evals=[(d_train, "train"), (d_val, "val")],
                    num_boost_round=1600,
                    early_stopping_rounds=140,
                    verbose_eval=False,
                )

                model_pred = model.predict(
                    d_val_predict,
                    iteration_range=(0, model.best_iteration + 1),
                ).astype(np.float32)
                val_pred += model_pred
                seed_models.append(model)

            val_pred /= len(seed_models)
            oof_pred[val_mask] = val_pred
            fold_rho = self.weighted_spearmanr(y_val, val_pred)
            fold_mae, fold_rmse = self.regression_metrics(y_val, val_pred)

            print(
                f"Fold {fold} - Spearman {fold_rho:.4f}, MAE {fold_mae:.6f}, RMSE {fold_rmse:.6f}"
            )
            fold_model_groups.append(seed_models)
            fold_scores.append(fold_rho)
            fold_maes.append(fold_mae)

            if d_holdout is not None and y_holdout is not None:
                holdout_pred = np.zeros(len(X_holdout), dtype=np.float32)
                for model in seed_models:
                    holdout_pred += model.predict(
                        d_holdout,
                        iteration_range=(0, model.best_iteration + 1),
                    ).astype(np.float32)
                holdout_pred /= len(seed_models)
                holdout_rho = self.weighted_spearmanr(y_holdout, holdout_pred)
                holdout_mae, holdout_rmse = self.regression_metrics(y_holdout, holdout_pred)
                print(
                    f"Fold {fold} - Holdout Spearman {holdout_rho:.4f}, "
                    f"MAE {holdout_mae:.6f}, RMSE {holdout_rmse:.6f}"
                )

        if not fold_model_groups:
            raise RuntimeError("No fold model was trained")

        keep_n = max(1, len(fold_model_groups) - 1)
        best_idx = np.argsort(np.array(fold_maes))[:keep_n]
        selected_maes = np.array([fold_maes[i] for i in best_idx], dtype=np.float64)
        fold_weights = 1.0 / np.clip(selected_maes, 1e-8, None)
        fold_weights = fold_weights / fold_weights.sum()

        model_weight_pairs = []
        for local_rank, fold_idx in enumerate(best_idx):
            seed_models = fold_model_groups[fold_idx]
            per_model_weight = fold_weights[local_rank] / max(1, len(seed_models))
            for model in seed_models:
                model_weight_pairs.append((model, per_model_weight))

        print(f"Selected folds: {list(best_idx)}")
        print(f"Fold MAE weights: {fold_weights.tolist()}")

        oof_mask = np.isfinite(oof_pred)
        if oof_mask.any():
            oof_y = y_cv.iloc[oof_mask]
            oof_p = oof_pred[oof_mask]
            oof_rho = self.weighted_spearmanr(oof_y, oof_p)
            oof_mae, oof_rmse = self.regression_metrics(oof_y, oof_p)
            print(
                f"OOF   - Spearman {oof_rho:.4f}, MAE {oof_mae:.6f}, RMSE {oof_rmse:.6f}, "
                f"Coverage {oof_mask.mean():.2%}"
            )

        d_all = xgb.DMatrix(X)
        y_pred = np.zeros(len(X), dtype=np.float32)
        for model, weight in model_weight_pairs:
            y_pred += (
                model.predict(d_all, iteration_range=(0, model.best_iteration + 1)).astype(np.float32)
                * weight
            )

        pred_df = pd.DataFrame(index=X.index)
        pred_df["target"] = y.values
        pred_df["predict_return"] = y_pred

        pred_df["predict_return"] = pred_df["predict_return"].replace([np.inf, -np.inf], np.nan).fillna(0)
        pred_df["predict_return"] = pred_df.groupby(level=0)["predict_return"].transform(
            lambda s: s.clip(s.quantile(0.01), s.quantile(0.99))
        )
        pred_df["predict_return"] = pred_df.groupby(level=1)["predict_return"].transform(
            lambda s: s.ewm(span=4, adjust=False).mean()
        )

        lo, hi = pred_df["target"].quantile([0.005, 0.995])
        pred_df["predict_return"] = pred_df["predict_return"].clip(lo, hi)

        pred_abs = pred_df["predict_return"].abs()
        startup_n = min(1000, len(pred_abs))
        startup_abs_mean = float(pred_abs.iloc[:startup_n].mean()) if startup_n > 0 else np.nan
        later_abs_mean = float(pred_abs.iloc[startup_n:].mean()) if len(pred_abs) > startup_n else np.nan
        print(
            f"Startup prediction |abs| mean (first {startup_n} rows): {startup_abs_mean:.6f}; "
            f"later rows |abs| mean: {later_abs_mean:.6f}"
        )

        if holdout_active and holdout_mask_all.any():
            holdout_df = pred_df.iloc[holdout_mask_all]
            holdout_rho = self.weighted_spearmanr(
                holdout_df["target"], holdout_df["predict_return"]
            )
            holdout_mae, holdout_rmse = self.regression_metrics(
                holdout_df["target"], holdout_df["predict_return"]
            )
            print(
                f"Holdout - Spearman {holdout_rho:.4f}, MAE {holdout_mae:.6f}, "
                f"RMSE {holdout_rmse:.6f}, Rows {len(holdout_df):,}"
            )

            holdout_out = holdout_df.reset_index()
            holdout_out.columns = ["datetime", "symbol", "target", "predict_return"]
            holdout_out["id"] = holdout_out["datetime"].astype(str) + "_" + holdout_out["symbol"].astype(str)
            holdout_out[["id", "target", "predict_return"]].to_csv(
                f"{self.output_tag}_holdout_predictions.csv", index=False
            )
            print(f"Saved {self.output_tag}_holdout_predictions.csv")

        overall_rho = self.weighted_spearmanr(pred_df["target"], pred_df["predict_return"])
        overall_mae, overall_rmse = self.regression_metrics(pred_df["target"], pred_df["predict_return"])
        print(
            f"Overall - Spearman {overall_rho:.4f}, MAE {overall_mae:.6f}, RMSE {overall_rmse:.6f}"
        )

        cv_metrics_df = pd.DataFrame(
            {
                "fold": list(range(1, len(fold_scores) + 1)),
                "spearman": fold_scores,
                "mae": fold_maes,
            }
        )
        cv_metrics_path = f"{self.output_tag}_cv_metrics.csv"
        cv_metrics_df.to_csv(cv_metrics_path, index=False)
        print(f"Saved {cv_metrics_path}")

        return pred_df

    def save_outputs(self, pred_df):
        out = pred_df.reset_index()
        out.columns = ["datetime", "symbol", "true_return", "predict_return"]
        out = out[out["datetime"] >= self.start_datetime]
        out["id"] = out["datetime"].astype(str) + "_" + out["symbol"].astype(str)

        full_pred = out[["id", "predict_return"]].drop_duplicates(subset=["id"], keep="last")
        full_check = out[["id", "true_return"]].drop_duplicates(subset=["id"], keep="last")

        pred_path = f"{self.output_tag}_predictions.csv"
        check_path = f"{self.output_tag}_check.csv"

        full_pred.to_csv(pred_path, index=False)
        full_check.to_csv(check_path, index=False)
        print(f"Saved {pred_path} and {check_path}")

        if os.path.isfile(SAMPLE_SUBMISSION_PATH):
            try:
                sample = pd.read_csv(SAMPLE_SUBMISSION_PATH)
                if "id" in sample.columns:
                    sample_ids = sample["id"].astype(str)
                    matched = full_pred[full_pred["id"].isin(sample_ids)]
                    if len(matched) > 0:
                        missing_ids = list(set(sample_ids) - set(matched["id"]))
                        fill_rows = pd.DataFrame(
                            {"id": missing_ids, "predict_return": [0.0] * len(missing_ids)}
                        )
                        submit_df = pd.concat([matched, fill_rows], ignore_index=True)
                        submit_df = submit_df.drop_duplicates(subset=["id"], keep="last")
                        submit_df.to_csv(SUBMISSION_PATH, index=False)
                        print(f"Saved competition submit to {SUBMISSION_PATH}, rows={len(submit_df)}")
                        return
            except Exception as exc:
                print(f"Sample submission alignment skipped: {exc}")

        full_pred.to_csv(SUBMISSION_PATH, index=False)
        print(f"Saved fallback submit to {SUBMISSION_PATH}, rows={len(full_pred)}")

    def run(self):
        all_symbols, factor_dfs, df_target = self.load_panel_data()
        print(
            f"Training on {len(all_symbols)} symbols for {self.forecast_horizon_days}-day volatility horizon"
        )
        pred_df = self.train_and_predict(factor_dfs, df_target)
        self.save_outputs(pred_df)


if __name__ == "__main__":
    model = StockVolatilityModel()
    model.run()
