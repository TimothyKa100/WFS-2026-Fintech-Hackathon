from operator import index
import numpy as np
import pandas as pd  # type: ignore
import datetime, os, time
import torch
import torch.multiprocessing as mp
import xgboost as xgb  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.model_selection import TimeSeriesSplit  # type: ignore
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import shap  # type: ignore

PROCESSING_DEVICE = "cpu"
TRAIN_DEVICE = "cuda"
PROCESSES = mp.cpu_count() // 2  # use half of the available CPU cores
BASE_DIR = os.getcwd()
TRAIN_DATA_DIR = os.path.join(BASE_DIR, "kline_data", "train_data")
TRAIN_CACHE_DIR = os.path.join(BASE_DIR, "data_cache")
SAMPLE_SUBMISSION_PATH = os.path.join(BASE_DIR, "sample_submission.csv")
SUBMISSION_PATH = os.path.join(BASE_DIR, "submit.csv")


def compute_factors_torch(df, device):
    #convert to tensor
    close = torch.tensor(df["close_price"].values, dtype=torch.float32, device=device)
    volume = torch.tensor(df["volume"].values, dtype=torch.float32, device=device)
    amount = torch.tensor(df["amount"].values, dtype=torch.float32, device=device)
    high = torch.tensor(df["high_price"].values, dtype=torch.float32, device=device)
    low = torch.tensor(df["low_price"].values, dtype=torch.float32, device=device)
    buy_volume = torch.tensor(
        df["buy_volume"].values, dtype=torch.float32, device=device
    )

    def apply_pool(tensor, window):
        max_pad = window // 2
        input_length = len(tensor)
        padded = torch.nn.functional.avg_pool1d(
            tensor.unsqueeze(0),
            kernel_size=window,
            stride=1,
            padding=max_pad,
        ).squeeze()
        pad_size = input_length - len(padded)
        if pad_size > 0:
            padded = torch.nn.functional.pad(
                padded,
                (0, pad_size),
                mode="replicate",
            )
        elif pad_size < 0:
            padded = padded[:input_length]
        return padded[:input_length]

    # VWAP - Volume-Weighted Average Price: https://www.investopedia.com/terms/v/vwap.asp
    ## Average price of the security has traded at throughout the day weighted by volume
    vwap = torch.where(volume > 0, amount / volume, close)
    vwap = torch.where(torch.isfinite(vwap), vwap, close)

    # RSI - Relative Strength Index (Full Implementation, 14 day window): https://www.investopedia.com/terms/r/rsi.asp
    ## Momentum indicator to measure how fast & how much a stock moved recently
    ## RSI > 70 -> Asset is overbought, possible trend reversal
    ## RSI < 30 -> Asset maybe oversold, possible reversal outward
    delta = torch.diff(
        close, prepend=close[:1]
    )  # calculate delta between consecutive elements
    gain = torch.where(delta > 0, delta, torch.tensor(0.0, device=device))
    loss = torch.where(delta < 0, -delta, torch.tensor(0.0, device=device))

    init_gain = gain[:14].mean() if len(gain) > 14 else torch.tensor(0.0, device=device)
    init_loss = loss[:14].mean() if len(gain) > 14 else torch.tensor(0.0, device=device)

    avg_gain = torch.zeros_like(close, device=device)
    avg_loss = torch.zeros_like(close, device=device)
    avg_gain[:14] = init_gain
    avg_loss[:14] = init_loss
    for i in range(
        14, len(close)
    ):  # Wilder's smoothing method -> less reactive than standard ema, more stable
        avg_gain[i] = (avg_gain[i - 1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i - 1] * 13 + loss[i]) / 14

    rs = torch.where(
        avg_loss > 0, avg_gain / avg_loss, torch.tensor(0.0, device=device)
    )
    rsi = 100 - 100 / (1 + rs)
    rsi = torch.where(
        torch.isnan(rsi), torch.tensor(50.0, device=device), rsi
    )  # set to neutral if nan

    # ATR - Average True Range: https://www.investopedia.com/terms/a/atr.asp
    ## Measure of Volatility
    ## Price moves a lot -> High ATR, Price mostly flat -> Low ATR
    tr = torch.max(
        high - low, torch.max(torch.abs(high - close), torch.abs(low - close))
    )
    atr = torch.zeros_like(tr, device=device)
    for i in range(14, len(tr)):  # Wilder's
        atr[i] = (atr[i - 1] * 13 + tr[i]) / 14
    atr = torch.where(
        torch.isnan(atr), torch.tensor(0.0, device=device), atr
    )  # set to neutral if nan

    # MACD - Moving Average Convergence Divergence: https://www.investopedia.com/terms/m/macd.asp
    ## Momentum + trend-following indicator widely used to spot trade direction, strength, as well as buy/sell signals
    ema12 = torch.zeros_like(close, device=device)
    ema26 = torch.zeros_like(close, device=device)
    alpha12 = 2 / (12 + 1)
    alpha26 = 2 / (26 + 1)
    ema12[:12] = close[:12].mean()
    ema26[:26] = close[:26].mean()
    for i in range(12, len(close)):  # short term ema
        ema12[i] = alpha12 * close[i] + (1 - alpha12) * ema12[i - 1]
    for i in range(26, len(close)):  # long term ema
        ema26[i] = alpha26 * close[i] + (1 - alpha26) * ema26[i - 1]
    macd = ema12 - ema26
    macd = torch.where(torch.isnan(macd), torch.tensor(0.0, device=device), macd)

    # Keltner Channels
    ## Volatility-based bands around a moving average
    ## Upper Band = EMA + (ATR * Multiplier)
    ## Lower Band = EMA - (ATR * Multiplier)
    ## Commonly used multiplier is 2
    keltner_multiplier = 2
    keltner_upper = ema12 + (atr * keltner_multiplier)
    keltner_lower = ema12 - (atr * keltner_multiplier)
    keltner_upper = torch.where(torch.isfinite(keltner_upper), keltner_upper, close)
    keltner_lower = torch.where(torch.isfinite(keltner_lower), keltner_lower, close)
    # Remove manual padding since tensors should already have correct length
    # keltner_upper = torch.nn.functional.pad(keltner_upper, (19, 0), mode='constant', value=0)
    # keltner_lower = torch.nn.functional.pad(keltner_lower, (19, 0), mode='constant', value=0)

    # Buy Ratio
    ## Estimate of how much total trading volume is made up of buying pressure vs. selling
    ## Buy Volume: trades that happened at or near ask price -> if relatively high, could signal upward momentum
    ## Sell Volume: trades that happed at or near bid price -> if relatively high, could signal downward momentum
    buy_ratio = torch.where(
        volume > 0, buy_volume / volume, torch.tensor(0.5, device=device)
    )

    # Bollinger Bands - https://www.investopedia.com/terms/b/bollingerbands.asp
    ## Volatility bands around a moving average
    rolling_mean = apply_pool(close, 20)
    # Calculate squared differences and apply the same pooling
    squared_diff = (close - rolling_mean) ** 2
    rolling_var = apply_pool(squared_diff, 20)
    rolling_std = torch.sqrt(rolling_var)
    bb_upper = rolling_mean + 2 * rolling_std
    bb_lower = rolling_mean - 2 * rolling_std
    bb_mid = (bb_upper + bb_lower) / 2

    bb_width = bb_upper - bb_lower
    bb_dev = (close - bb_mid) / bb_width

    # Remove the manual padding since apply_pool already handles it
    # bb_upper = torch.nn.functional.pad(bb_upper, (19, 0), mode='constant', value=0)
    # bb_lower = torch.nn.functional.pad(bb_lower, (19, 0), mode='constant', value=0)

    # Stochastic Oscillator
    ## Momentum indicator comparing closing price to a range of prices over a period
    ## %K = (Current Close - Lowest Low) / (Highest High - Lowest Low) * 100
    input_length = len(close)
    highest_high = torch.nn.functional.max_pool1d(
        close.unsqueeze(0), kernel_size=14, stride=1, padding=7
    ).squeeze()
    lowest_low = -torch.nn.functional.max_pool1d(
        -close.unsqueeze(0), kernel_size=14, stride=1, padding=7
    ).squeeze()
    pad_size = input_length - len(highest_high)
    if pad_size > 0:
        highest_high = torch.nn.functional.pad(
            highest_high, (0, pad_size), mode="replicate"
        )
        lowest_low = torch.nn.functional.pad(
            lowest_low, (0, pad_size), mode="replicate"
        )
    elif pad_size < 0:
        highest_high = highest_high[:input_length]
        lowest_low = lowest_low[:input_length]
    stochastic_k = (close - lowest_low) / (highest_high - lowest_low + 1e-8) * 100
    stochastic_k = torch.where(
        torch.isfinite(stochastic_k), stochastic_k, torch.tensor(50.0, device=device)
    )
    stochastic_k = torch.clamp(stochastic_k, 0, 100)
    stochastic_d = apply_pool(stochastic_k, 3)
    stochastic_d = torch.clamp(stochastic_d, 0, 100)

    # CCI - Commodity Channel Index: https://www.investopedia.com/terms/c/cci.asp
    ## Measures deviation of price from its average price over a period
    ## CCI = (Typical Price - SMA(Typical Price)) / (0.015 * Mean Deviation)
    typical_price = (high + low + close) / 3
    sma_typical_price = apply_pool(typical_price, 20)
    mean_deviation = apply_pool(torch.abs(typical_price - sma_typical_price), 20)
    cci = (typical_price - sma_typical_price) / (0.015 * mean_deviation + 1e-8)
    cci = torch.where(torch.isfinite(cci), cci, torch.tensor(0.0, device=device))
    cci = torch.clamp(cci, -100, 100)

    # Chaikin Money Flow Index (MFI): https://www.investopedia.com/terms/m/mfi.asp
    ## Measures buying and selling pressure over a period
    ## MFI = (Sum of Positive Money Flow - Sum of Negative Money Flow) / (Sum of Positive Money Flow + Sum of Negative Money Flow)
    money_flow = close * volume
    positive_money_flow = torch.where(
        close[1:] > close[:-1], money_flow[1:], torch.tensor(0.0, device=device)
    )
    negative_money_flow = torch.where(
        close[1:] < close[:-1], money_flow[1:], torch.tensor(0.0, device=device)
    )
    sum_positive_flow = apply_pool(
        torch.cat([torch.tensor([0.0], device=device), positive_money_flow]), 20
    )
    sum_negative_flow = apply_pool(
        torch.cat([torch.tensor([0.0], device=device), negative_money_flow]), 20
    )
    mfi = (sum_positive_flow - sum_negative_flow) / (
        sum_positive_flow + sum_negative_flow + 1e-8
    )
    mfi = torch.where(torch.isfinite(mfi), mfi, torch.tensor(0.0, device=device))
    mfi = torch.clamp(mfi, -1, 1)

    # obv - On-Balance Volume: https://www.investopedia.com/terms/o/onbalancevolume.asp
    ## Volume-based indicator that uses volume flow to predict price changes
    obv = torch.zeros_like(close, device=device)
    for i in range(1, len(close)):
        obv[i] = obv[i - 1] + torch.where(
            close[i] > close[i - 1],
            volume[i],
            torch.where(
                close[i] < close[i - 1], -volume[i], torch.tensor(0.0, device=device)
            ),
        )
    obv = torch.where(torch.isfinite(obv), obv, torch.tensor(0.0, device=device))
    obv = torch.clamp(obv, -1e6, 1e6)

    # VWAP Deviation
    ## Measure of how far current price is from VWAP
    ## Price > VWAP - Buyers pushing up price -> potentially overbought
    ## Price < VWAP - Sellers pushing down price -> potentially oversold
    ## Large deviations can signal mean reversion
    vwap_deviation = (close - vwap) / torch.where(
        vwap != 0, vwap, torch.tensor(1.0, device=device)
    )
    vwap_deviation = torch.where(
        torch.isfinite(vwap_deviation), vwap_deviation, torch.tensor(0.0, device=device)
    )

    # Convert torch tensors to dataframes
    df["vwap"] = vwap.cpu().numpy()
    df["rsi"] = rsi.cpu().numpy()
    df["macd"] = macd.cpu().numpy()
    df["atr"] = atr.cpu().numpy()
    df["buy_ratio"] = buy_ratio.cpu().numpy()
    df["vwap_deviation"] = vwap_deviation.cpu().numpy()
    df["keltner_upper"] = keltner_upper.cpu().numpy()
    df["keltner_lower"] = keltner_lower.cpu().numpy()
    df["stochastic_d"] = stochastic_d.cpu().numpy()
    df["cci"] = cci.cpu().numpy()
    df["mfi"] = mfi.cpu().numpy()
    df["obv"] = obv.cpu().numpy()

    df["bb_upper"] = bb_upper.cpu().numpy()
    df["bb_lower"] = bb_lower.cpu().numpy()
    df["bb_width"] = bb_width.cpu().numpy()
    df["bb_dev"] = bb_dev.cpu().numpy()
    return df


def get_single_symbol_kline_data(symbol, train_data_path, device):
    try:
        df = pd.read_parquet(f"{train_data_path}/{symbol}.parquet")
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.set_index("timestamp")
        df = df.astype(np.float64)
        required_cols = [
            "open_price",
            "high_price",
            "low_price",
            "close_price",
            "volume",
            "amount",
            "buy_volume",
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"{symbol} missing columns: {missing_cols}")
            return pd.DataFrame(
                columns=required_cols
                + ["vwap", "rsi", "macd", "buy_ratio", "vwap_deviation", "atr"]
            )

        # Calculate the indicators
        df = compute_factors_torch(df, device)
        print(
            f"Loaded data for {symbol}, shape: {df.shape}, vwap NaNs: {df['vwap'].isna().sum()}"
        )
        return df
    except Exception as e:
        print(f"Error loading {symbol}: {e}")
        return pd.DataFrame(
            columns=[
                "open_price",
                "high_price",
                "low_price",
                "close_price",
                "volume",
                "amount",
                "buy_volume",
                "vwap",
                "rsi",
                "macd",
                "buy_ratio",
                "vwap_deviation",
                "atr",
            ]
        )


def get_all_symbols(train_data_path):
    try:
        parquet_name_list = os.listdir(train_data_path)
        symbol_list = [parquet_name.split(".")[0] for parquet_name in parquet_name_list]
        return symbol_list
    except Exception as e:
        print(f"Error in get_all_symbols: {e}")
        return []


class OptimizedModel:
    REGIME_NAME_MAP = {
        0: "bearish_calm",
        1: "bearish_volatile",
        2: "neutral",
        3: "bullish_calm",
        4: "bullish_volatile",
    }

    RAW_INDICATORS = [
        "open_price",
        "high_price",
        "low_price",
        "close_price",
        "vwap",
        "amount",
        "atr",
        "macd",
        "buy_volume",
        "volume",
        "vwap_deviation",
        "rsi",
        "bb_upper",
        "bb_lower",
        "bb_width",
        "bb_dev",
        "keltner_upper",
        "keltner_lower",
        "stochastic_d",
        "cci",
        "mfi",
        "obv",
    ]

    DERIVED_INDICATORS = [
        "1h_momentum",
        "4h_momentum",
        "7d_momentum",
        "squeeze_ratio",
        "ult_osc",
        # "log_return_1h",
        # "log_return_4h",
        # "log_return_7d",
        "atr_pct",
        "amount_7d_surge",
        "amount_sum_7d",
        "vol_norm_mom",
        "vol_momentum",
        "skew_24h_c",
        "kurt_24h_c",
        "squeeze_ratio",
        "buy_ratio",
        "gk_vol",
        "24hour_rtn",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
    ]

    def __init__(self):
        self.train_data_path = TRAIN_DATA_DIR
        self.sample_submission_path = SAMPLE_SUBMISSION_PATH
        self.submission_path = SUBMISSION_PATH
        self.start_datetime = datetime.datetime(2021, 3, 1, 0, 0, 0)
        self.scaler = StandardScaler()
        self.processing_device = PROCESSING_DEVICE
        self.training_device = TRAIN_DEVICE
        self.cache_dir = TRAIN_CACHE_DIR
        print(f"Using device: {self.processing_device}")

    def get_all_symbol_list(self):
        try:
            parquet_name_list = os.listdir(self.train_data_path)
            symbol_list = [
                parquet_name.split(".")[0] for parquet_name in parquet_name_list
            ]
            return symbol_list
        except Exception as e:
            print(f"Error in get_all_symbol_list: {e}")
            return []

    def get_all_symbol_kline(self):
        t0 = time.monotonic()

        try:
            pool = mp.Pool(processes=PROCESSES)  # use multi core

            all_symbol_list = self.get_all_symbol_list()
            if not all_symbol_list:
                print("No symbols found, exiting.")
                pool.close()
                return [], [], [], [], [], [], [], [], [], []

            promise_list = [
                pool.apply_async(
                    get_single_symbol_kline_data,
                    (symbol, self.train_data_path, self.processing_device),
                )
                for symbol in all_symbol_list
            ]

            loaded_symbols = []
            df_results = []
            for async_result, symbol in zip(promise_list, all_symbol_list):
                df = async_result.get()
                df_results.append(df)

                if not df.empty and "vwap" in df.columns:
                    loaded_symbols.append(symbol)
                else:
                    print(f"{symbol} failed: empty or missing 'vwap'")
            failed_symbols = [s for s in all_symbol_list if s not in loaded_symbols]
            print(f"Failed symbols: {failed_symbols}")

        except KeyboardInterrupt:
            pool.terminate()
            raise

        finally:
            pool.close()
            pool.join()

        # Clean data
        time_index = pd.date_range(
            start="2021-01-01 00:00:00", end="2024-12-31 23:45:00", freq="15min"
        )  # 15 min time index
        df_open_price = pd.concat(
            [
                result["open_price"]
                for result in df_results
                if not result.empty and "open_price" in result.columns
            ],
            axis=1,
        ).sort_index(
            ascending=True
        )  # get open price dfs for each symbol
        print(
            f"df_open_price index dtype: {df_open_price.index.dtype}, shape: {df_open_price.shape}"
        )
        df_open_price.columns = loaded_symbols
        df_open_price = df_open_price.reindex(
            columns=all_symbol_list, fill_value=0
        ).reindex(
            time_index, method="ffill"
        )  # unify time steps
        time_arr = pd.to_datetime(df_open_price.index, unit="ms").values

        def align_df(key):
            valid_dfs = [
                df[key]
                for df, s in zip(df_results, all_symbol_list)
                if not df.empty and key in df.columns and s in loaded_symbols
            ]
            if not valid_dfs:
                print(f"No valid data for {key}, filling with zeros")
                return np.zeros((len(time_index), len(all_symbol_list)))
            df = pd.concat(valid_dfs, axis=1).sort_index(ascending=True)
            df.columns = loaded_symbols
            return (
                df.reindex(columns=all_symbol_list, fill_value=0)
                .reindex(time_index, method="ffill")
                .values
            )

        raw_ind_arrays = {ind: align_df(ind) for ind in self.RAW_INDICATORS}

        print(f"Finished get all symbols kline, time elapsed: {time.monotonic() - t0}")
        return (all_symbol_list, time_arr, raw_ind_arrays)

    def make_market_regime_target(self, raw_dfs, keep_slice, windows_1h):
        windows_6h = windows_1h * 6
        market_symbols = [
            s
            for s in ["BTCUSDT", "ETHUSDT"]
            if s in raw_dfs["vwap"].columns
        ]
        if not market_symbols:
            market_proxy = raw_dfs["vwap"].median(axis=1)
            print("BTC/ETH not both available for regime target, using median market proxy.")
        else:
            market_proxy = raw_dfs["vwap"][market_symbols].mean(axis=1)

        market_ret_1h = np.log(
            market_proxy / market_proxy.shift(windows_1h).replace(0, np.nan)
        )
        fwd_ret_6h = np.log(
            market_proxy.shift(-windows_6h) / market_proxy.replace(0, np.nan)
        )
        realized_vol_6h = market_ret_1h.rolling(windows_6h, min_periods=windows_1h).std().shift(-windows_6h)

        regime_df = pd.DataFrame(
            {
                "fwd_ret_6h": fwd_ret_6h,
                "realized_vol_6h": realized_vol_6h,
            }
        ).loc[keep_slice]

        ret_low = regime_df["fwd_ret_6h"].quantile(0.30)
        ret_high = regime_df["fwd_ret_6h"].quantile(0.70)
        vol_high = regime_df["realized_vol_6h"].quantile(0.70)

        regime_id = pd.Series(2, index=regime_df.index, dtype=np.int8)
        bearish_mask = regime_df["fwd_ret_6h"] <= ret_low
        bullish_mask = regime_df["fwd_ret_6h"] >= ret_high
        high_vol_mask = regime_df["realized_vol_6h"] >= vol_high

        regime_id[bearish_mask & ~high_vol_mask] = 0
        regime_id[bearish_mask & high_vol_mask] = 1
        regime_id[bullish_mask & ~high_vol_mask] = 3
        regime_id[bullish_mask & high_vol_mask] = 4

        regime_df["regime_id"] = regime_id
        regime_df["regime_name"] = regime_df["regime_id"].map(self.REGIME_NAME_MAP)

        print("Regime class distribution:")
        print(regime_df["regime_name"].value_counts(dropna=False).to_string())
        return regime_df

    def classification_metrics(self, y_true, y_pred, y_prob=None):
        acc = float(accuracy_score(y_true, y_pred))
        f1_macro = float(f1_score(y_true, y_pred, average="macro"))
        if y_prob is not None:
            try:
                mlogloss = float(log_loss(y_true, y_prob, labels=np.arange(len(self.REGIME_NAME_MAP))))
            except Exception:
                mlogloss = np.nan
        else:
            mlogloss = np.nan
        return acc, f1_macro, mlogloss

    def build_market_level_features(self, factor_dfs):
        market_feature_map = {}
        core_market_symbols = ["BTCUSDT", "ETHUSDT"]

        for feature_name, df in factor_dfs.items():
            aligned_df = df.astype(np.float32).replace([np.inf, -np.inf], np.nan)

            market_feature_map[f"{feature_name}_mean"] = aligned_df.mean(axis=1)
            market_feature_map[f"{feature_name}_std"] = aligned_df.std(axis=1)
            market_feature_map[f"{feature_name}_q25"] = aligned_df.quantile(0.25, axis=1)
            market_feature_map[f"{feature_name}_q75"] = aligned_df.quantile(0.75, axis=1)

            for symbol in core_market_symbols:
                if symbol in aligned_df.columns:
                    market_feature_map[f"{feature_name}_{symbol}"] = aligned_df[symbol]

        market_features = pd.DataFrame(market_feature_map).sort_index(ascending=True)
        market_features = market_features.replace([np.inf, -np.inf], np.nan)
        return market_features

    def train(self, regime_df, factor_dfs):
        print("Preparing data for training...")
        t0 = time.monotonic()

        market_X = self.build_market_level_features(factor_dfs)
        target_y = regime_df["regime_id"].astype("float32")

        data = pd.concat([market_X, target_y.rename("target")], axis=1)

        elapsed_time = time.monotonic() - t0
        print(
            f"Data processing completed in {int(elapsed_time // 60)} min and {elapsed_time % 60:.1f} sec"
        )

        print(
            f"Data memory usage: {data.memory_usage(deep=True).sum() / 1024**3:.2f} GB"
        )
        print(f"Data shape before dropna: {data.shape}")
        data = data.dropna(subset=["target"])
        print(f"Data shape after dropna: {data.shape}")
        print(
            f"Data memory usage: {data.memory_usage(deep=True).sum() / 1024**3:.2f} GB"
        )
        print(
            f"Index memory usage: {data.index.memory_usage(deep=True) / 1024**3:.2f} GB"
        )

        feature_names = [col for col in data.columns if col != "target"]
        X = data[feature_names].astype(np.float32)
        y = data["target"].astype("int32")

        class_counts = y.value_counts().sort_index()
        class_weight_map = {
            cls: (len(y) / max(1, len(class_counts) * cnt))
            for cls, cnt in class_counts.items()
        }
        print(f"Class weights: {class_weight_map}")

        print("Begin training model...")
        t0 = time.monotonic()

        MAX_BINS = 128
        GAP = 4 * 24 * 3
        TEST_SIZE = 4 * 24 * 365  # max training of 1 year data
        NUM_BOOST_ROUNDS = 1200
        EARLY_STOP_ROUNDS = 120
        N_SPLITS = 5
        NUM_CLASSES = len(self.REGIME_NAME_MAP)

        XGB_PARAMS = {
            "objective": "multi:softprob",
            "num_class": NUM_CLASSES,
            "learning_rate": 0.03,
            "max_depth": 6,
            "subsample": 0.85,
            "min_child_weight": 5,
            "gamma": 0.2,
            "reg_lambda": 3.0,
            "reg_alpha": 0.1,
            "colsample_bytree": 0.85,
            "tree_method": "hist",
            "device": self.training_device,
            "max_bin": MAX_BINS,
            "random_state": 42,
            "eval_metric": ["mlogloss", "merror"],
        }

        times = X.index.to_numpy()
        uniq_times = np.unique(times)
        tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=GAP, max_train_size=TEST_SIZE)

        models = []
        fold_f1s = []
        fold_losses = []

        for fold, (train_t_idx, val_t_idx) in enumerate(tscv.split(uniq_times), start=1):  # type: ignore
            print(f"Training fold {fold}...")

            train_times = uniq_times[train_t_idx]
            val_times = uniq_times[val_t_idx]

            # map timestamp folds back to row masks
            train_mask = np.isin(times, train_times)
            val_mask = np.isin(times, val_times)

            X_train, X_val = X[train_mask], X[val_mask]
            y_train, y_val = y[train_mask], y[val_mask]
            w_train = y_train.map(class_weight_map).astype(np.float32).values

            d_train = xgb.QuantileDMatrix(
                X_train, y_train, weight=w_train, max_bin=MAX_BINS
            )
            d_val = xgb.QuantileDMatrix(X_val, y_val, max_bin=MAX_BINS, ref=d_train)

            model = xgb.train(
                params=XGB_PARAMS,
                dtrain=d_train,
                evals=[(d_train, "train"), (d_val, "val")],
                num_boost_round=NUM_BOOST_ROUNDS,
                early_stopping_rounds=EARLY_STOP_ROUNDS,
                verbose_eval=20,
            )

            y_prob_val = model.predict(
                d_val,
                iteration_range=(0, model.best_iteration + 1),
            )
            y_pred_val = np.argmax(y_prob_val, axis=1)

            acc, f1_macro, mlogloss = self.classification_metrics(y_val, y_pred_val, y_prob_val)
            models.append(model)
            fold_f1s.append(f1_macro)
            fold_losses.append(mlogloss)
            print(
                f"Fold {fold} - Accuracy {acc:.4f}, "
                f"F1-macro {f1_macro:.4f}, LogLoss {mlogloss:.6f}"
            )

        elapsed_time = time.monotonic() - t0
        print(
            f"Training completed in {int(elapsed_time // 60)} min and {elapsed_time % 60:.1f} sec"
        )

        print("Ensembling models and evaluating...")
        # Keep all but worst 2 folds by macro-F1 (higher is better)
        keep_n = max(1, len(models) - 2)
        best_models_idx = np.argsort(np.array(fold_f1s))[::-1][:keep_n]

        best_models = [models[i] for i in best_models_idx]
        selected_f1 = np.array([fold_f1s[i] for i in best_models_idx], dtype=np.float64)
        model_weights = np.clip(selected_f1, 1e-6, None)
        model_weights = model_weights / model_weights.sum()
        print(f"Selected fold indices: {list(best_models_idx)}")
        print(f"Ensemble model weights (F1): {model_weights.tolist()}")

        d_all = xgb.DMatrix(X)
        y_prob = np.zeros((len(X), NUM_CLASSES), dtype=np.float32)
        for model, model_weight in zip(best_models, model_weights):
            y_prob += model.predict(
                d_all,
                iteration_range=(0, model.best_iteration + 1),
            ).astype(np.float32) * model_weight

        y_prob = y_prob / np.clip(y_prob.sum(axis=1, keepdims=True), 1e-8, None)
        y_pred = np.argmax(y_prob, axis=1)

        overall_acc, overall_f1, overall_loss = self.classification_metrics(y, y_pred, y_prob)
        print(
            f"Overall Accuracy {overall_acc:.4f}, "
            f"F1-macro {overall_f1:.4f}, LogLoss {overall_loss:.6f}"
        )

        cm = confusion_matrix(y, y_pred, labels=np.arange(NUM_CLASSES))
        cm_df = pd.DataFrame(
            cm,
            index=[f"true_{self.REGIME_NAME_MAP[i]}" for i in range(NUM_CLASSES)],
            columns=[f"pred_{self.REGIME_NAME_MAP[i]}" for i in range(NUM_CLASSES)],
        )
        print("Confusion matrix:")
        print(cm_df.to_string())

        pred_df = pd.DataFrame(index=data.index)
        pred_df["regime_true_id"] = y.values
        pred_df["regime_pred_id"] = y_pred
        pred_df["regime_true_name"] = pred_df["regime_true_id"].map(self.REGIME_NAME_MAP)
        pred_df["regime_pred_name"] = pred_df["regime_pred_id"].map(self.REGIME_NAME_MAP)
        for i in range(NUM_CLASSES):
            pred_df[f"prob_{self.REGIME_NAME_MAP[i]}"] = y_prob[:, i]

        pred_df.to_csv("market_regime_predictions.csv", index_label="datetime")
        print("Saved regime predictions to market_regime_predictions.csv")

        fold_metric_df = pd.DataFrame(
            {
                "fold": list(range(1, len(fold_f1s) + 1)),
                "f1_macro": fold_f1s,
                "logloss": fold_losses,
            }
        )
        fold_metric_df.to_csv("market_regime_cv_metrics.csv", index=False)
        print("Saved CV metrics to market_regime_cv_metrics.csv")

        SAVE_MODEL = True
        MODEL_DIR = "models"

        if SAVE_MODEL:
            try:
                os.makedirs(MODEL_DIR, exist_ok=True)  # keep saved models in a folder
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

                print(f"Saving best models to {MODEL_DIR}...")
                for i, model in enumerate(best_models, start=1):
                    save_path = os.path.join(
                        MODEL_DIR, f"{timestamp}_market_regime_fold{i}.json"
                    )
                    model.save_model(save_path)
                print("Model saved successfully.")
            except Exception as e:
                print(f"Error saving models: {e}")

        MAX_NUM_FEATURES = 30

        print("Plotting feature importance...")
        try:
            xgb.plot_importance(
                best_models[0],
                importance_type="gain",
                max_num_features=MAX_NUM_FEATURES,
            )
            plt.title("XGBoost Market Regime Feature Importance (Gain)")
            plt.tight_layout()
            plt.show(block=False)

        except KeyboardInterrupt:
            print("KeyboardInterrupt, closing plots.")
            plt.close("all")
            return

    def run(self):
        print("Train data directory contents:", os.listdir(self.train_data_path))

        os.makedirs(self.cache_dir, exist_ok=True)  # ensure cache directory exists
        print("Cache directory contents:", os.listdir(self.cache_dir))

        # Separate cache files for raw and derived indicators
        RAW_CACHE_FILES = {
            key: os.path.join(self.cache_dir, f"df_{key}.parquet")
            for key in self.RAW_INDICATORS
        }

        DERIVED_CACHE_FILES = {
            key: os.path.join(self.cache_dir, f"df_{key}.parquet")
            for key in self.DERIVED_INDICATORS
        }

        all_symbol_list = None

        raw_indicator_cache_exists = all(
            os.path.isfile(path) for path in RAW_CACHE_FILES.values()
        )
        derived_indicator_cache_exists = all(
            os.path.isfile(path) for path in DERIVED_CACHE_FILES.values()
        )

        use_raw_cache = raw_indicator_cache_exists
        use_derived_cache = (
            derived_indicator_cache_exists and raw_indicator_cache_exists
        )

        def winsorize(df, lower_quantile=0.01, upper_quantile=0.99):
            """Winsorize a DataFrame to limit extreme values."""
            lower_bound = df.quantile(lower_quantile)
            upper_bound = df.quantile(upper_quantile)
            return df.clip(lower=lower_bound, upper=upper_bound, axis=1)

        # ================
        # Raw indicators
        # ================

        raw_dfs = {}
        if use_raw_cache:
            print(f'Found raw indicator cache "{self.cache_dir}", using cached data.')

            for feature in self.RAW_INDICATORS:
                try:
                    file = RAW_CACHE_FILES[feature]
                    raw_dfs[feature] = pd.read_parquet(file)
                    print(
                        f"Loaded {feature} from cache, shape: {raw_dfs[feature].shape}"
                    )
                except Exception as e:
                    print(f"Error loading {feature} from cache: {e}")
                    use_raw_cache = False
                    break

        if not use_raw_cache:
            print(
                f'Cannot find all raw indicator cache files in "{self.cache_dir}", recalculating raw indicators.'
            )

            (all_symbol_list, time_arr, raw_ind_arrays) = self.get_all_symbol_kline()

            if not all_symbol_list:
                print("No data loaded, exiting.")
                return

            for feature in self.RAW_INDICATORS:
                file = RAW_CACHE_FILES[feature]
                print(f"Calculating {feature} indicators, saving to {file}")
                arr = raw_ind_arrays[feature]

                if arr is None or len(arr) == 0:
                    print(f"No data for {feature}, skipping.")
                    continue

                raw_dfs[feature] = pd.DataFrame(
                    arr, columns=all_symbol_list, index=time_arr
                )
                raw_dfs[feature].to_parquet(file)
                print(f"Saved {feature} to cache, shape: {raw_dfs[feature].shape}")

        winsorize_raw = ["atr", "macd", "bb_width", "bb_dev", "cci", "obv"]
        for feature in winsorize_raw:
            if feature in raw_dfs:
                print(f"Winsorizing {feature} indicator")
                raw_dfs[feature] = winsorize(raw_dfs[feature])

        windows_60d = 4 * 24 * 60
        windows_7d = 4 * 24 * 7
        windows_1d = 4 * 24 * 1
        windows_4h = 4 * 4
        windows_1h = 4

        if all_symbol_list is None:
            all_symbol_list = self.get_all_symbol_list()
            if not all_symbol_list:
                print("No symbols found, exiting.")
                return

        # ================
        # Derived indicators
        # ================

        derived_dfs = {}

        time_index = raw_dfs["open_price"].index

        if use_derived_cache:
            print(
                f'Found all derived indicators in cache "{self.cache_dir}", using cached data.'
            )

            for feature in self.DERIVED_INDICATORS:
                try:
                    file = DERIVED_CACHE_FILES[feature]
                    derived_dfs[feature] = pd.read_parquet(file)
                    print(
                        f"Loaded {feature} from cache, shape: {derived_dfs[feature].shape}"
                    )
                except Exception as e:
                    print(f"Error loading {feature} from cache: {e}")
                    use_raw_cache = False
                    break

        if not use_derived_cache:
            print(
                f'Cannot find derived indicator cache files in "{self.cache_dir}", recalculating derived indicators.'
            )
            # 1h_momentum
            derived_dfs["1h_momentum"] = winsorize(
                (raw_dfs["vwap"] / raw_dfs["vwap"].shift(windows_1h).replace(0, np.nan))
                - 1.0
            )
            # derived_dfs["1h_momentum"].fillna(0, inplace=True)

            # 4h_momentum
            derived_dfs["4h_momentum"] = winsorize(
                (raw_dfs["vwap"] / raw_dfs["vwap"].shift(windows_4h).replace(0, np.nan))
                - 1.0
            )
            # derived_dfs["4h_momentum"].fillna(0, inplace=True)

            # 7d_momentum
            derived_dfs["7d_momentum"] = winsorize(
                raw_dfs["vwap"] / (raw_dfs["vwap"].shift(windows_7d).replace(0, np.nan))
                - 1.0
            )
            # derived_dfs["7d_momentum"].fillna(0, inplace=True)

            # # lOG returns
            # derived_dfs["log_return_1h"] = np.log(
            #     raw_dfs["close_price"] / raw_dfs["close_price"].shift(windows_1h)
            # )
            # derived_dfs["log_return_1h"].replace(
            #     [np.inf, -np.inf], np.nan, inplace=True
            # )
            # derived_dfs["log_return_1h"].fillna(0, inplace=True)

            # derived_dfs["log_return_4h"] = np.log(
            #     raw_dfs["close_price"] / raw_dfs["close_price"].shift(windows_4h)
            # )
            # derived_dfs["log_return_4h"].replace(
            #     [np.inf, -np.inf], np.nan, inplace=True
            # )
            # derived_dfs["log_return_4h"].fillna(0, inplace=True)

            # derived_dfs["log_return_7d"] = np.log(
            #     raw_dfs["close_price"] / raw_dfs["close_price"].shift(windows_7d)
            # )
            # derived_dfs["log_return_7d"].replace(
            #     [np.inf, -np.inf], np.nan, inplace=True
            # )
            # derived_dfs["log_return_7d"].fillna(0, inplace=True)

            # ultimate oscillator
            prev_close = raw_dfs["close_price"].shift(1)

            buy_pressure = raw_dfs["close_price"] - np.minimum(
                raw_dfs["low_price"], prev_close
            )
            true_range = np.maximum(raw_dfs["high_price"], prev_close) - np.minimum(
                raw_dfs["low_price"], prev_close
            )

            true_range.replace(0, np.nan, inplace=True)  # prevent div by 0

            short_win = 7 * 4
            med_win = 14 * 4
            long_win = 28 * 4

            bp1 = buy_pressure.rolling(short_win, min_periods=short_win).sum()
            tr1 = true_range.rolling(short_win, min_periods=short_win).sum()
            bp2 = buy_pressure.rolling(med_win, min_periods=med_win).sum()
            tr2 = true_range.rolling(med_win, min_periods=med_win).sum()
            bp3 = buy_pressure.rolling(long_win, min_periods=long_win).sum()
            tr3 = true_range.rolling(long_win, min_periods=long_win).sum()

            avg1 = bp1 / tr1
            avg2 = bp2 / tr2
            avg3 = bp3 / tr3

            derived_dfs["ult_osc"] = 100.0 * (4 * avg1 + 2 * avg2 + 1 * avg3) / 7.0

            # # percentage change
            # derived_dfs["pct_change"] = raw_dfs["close_price"].pct_change()
            # derived_dfs["pct_change"].replace([np.inf, -np.inf], np.nan, inplace=True)
            # # derived_dfs["pct_change"].fillna(0, inplace=True)

            # volume factor
            derived_dfs["amount_sum_7d"] = winsorize(
                raw_dfs["amount"].rolling(windows_7d).sum()
            )

            rolling_median = derived_dfs["amount_sum_7d"].rolling(windows_60d).median()
            derived_dfs["amount_7d_surge"] = np.log(
                derived_dfs["amount_sum_7d"] / rolling_median.replace(0, np.nan)
            )

            # derived_dfs["amount_sum"].fillna(0, inplace=True)

            k = windows_1d  # 6 hr lookback
            returns = raw_dfs["vwap"].pct_change()  # or absolute change
            vol = returns.rolling(k).std()

            derived_dfs["vol_norm_mom"] = winsorize(
                raw_dfs["vwap"] - raw_dfs["vwap"].shift(k)
            ) / vol.replace(0, np.nan)

            # volume momentum
            derived_dfs["vol_momentum"] = winsorize(
                (raw_dfs["amount"] / raw_dfs["amount"].shift(windows_1d)) - 1
            )
            # derived_dfs["vol_momentum"].fillna(0, inplace=True)

            # buy ratio
            derived_dfs["buy_ratio"] = raw_dfs["buy_volume"] / raw_dfs[
                "volume"
            ].replace(0, np.nan)
            # derived_dfs["buy_ratio"].fillna(0, inplace=True)

            # 24 hour return
            derived_dfs["24hour_rtn"] = winsorize(
                (raw_dfs["vwap"] / raw_dfs["vwap"].shift(windows_1d)) - 1
            )

            # squeeze ratio
            derived_dfs["squeeze_ratio"] = raw_dfs["bb_width"] / (
                raw_dfs["keltner_upper"] - raw_dfs["keltner_lower"]
            ).replace(0, np.nan)

            # skew and kurt
            derived_dfs["skew_24h_c"] = (
                returns.rolling(windows_1d).skew().clip(-3, 3)
            )  # 96 = 24h of 15-min bars
            derived_dfs["kurt_24h_c"] = (
                returns.rolling(windows_1d).kurt().clip(-1, 10)
            )  # 96 = 24h of 15-min bars

            # atr_pct
            derived_dfs["atr_pct"] = raw_dfs["atr"] / raw_dfs["close_price"].replace(
                0, np.nan
            )

            # garman-klass volatility
            GK_WINDOW = windows_1d
            log_hl = np.log(
                raw_dfs["high_price"] / raw_dfs["low_price"].replace(0, np.nan)
            )  # log range
            log_co = np.log(
                raw_dfs["close_price"] / raw_dfs["open_price"].replace(0, np.nan)
            )  # log close-open
            rs = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
            gk_var_sum = rs.rolling(GK_WINDOW).sum().clip(lower=0)
            gk_std_24h = np.sqrt(gk_var_sum)

            derived_dfs["gk_vol"] = winsorize(gk_std_24h)

            # temporal features
            hourly = time_index.hour / 4  # put in 4-hour bins
            dow = time_index.dayofweek

            hour_sin = np.sin(2 * np.pi * hourly / 24)
            hour_cos = np.cos(2 * np.pi * hourly / 24)
            dow_sin = np.sin(2 * np.pi * dow / 7)
            dow_cos = np.cos(2 * np.pi * dow / 7)

            derived_dfs["hour_sin"] = pd.DataFrame(
                {symbol: hour_sin for symbol in all_symbol_list}, index=time_index
            )
            derived_dfs["hour_cos"] = pd.DataFrame(
                {symbol: hour_cos for symbol in all_symbol_list}, index=time_index
            )
            derived_dfs["dow_sin"] = pd.DataFrame(
                {symbol: dow_sin for symbol in all_symbol_list}, index=time_index
            )
            derived_dfs["dow_cos"] = pd.DataFrame(
                {symbol: dow_cos for symbol in all_symbol_list}, index=time_index
            )

            for feature, df in derived_dfs.items():
                file = DERIVED_CACHE_FILES[feature]
                print(f"Saving {feature} to cache, shape: {df.shape}")
                df.to_parquet(file)
                print(f"Saved {feature} to cache at {file}")

        # fast vol/flow features
        FEATURES_TO_FAST_Z = [
            "atr_pct",
            "vwap_deviation",
            "vol_momentum",
            "gk_vol",
        ]

        # trend/momentum
        FEATURES_TO_MED_Z = [
            "close_price",
            # "1h_momentum",
            "4h_momentum",
            "7d_momentum",
            "macd",
            "cci",
            "obv",
            "rsi",
            "bb_width",
            "bb_dev",
        ]

        # regime shape
        FEATURES_TO_SLOW_Z = [
            "skew_24h_c",
            "kurt_24h_c",
            "24hour_rtn",
        ]

        def rolling_z_scaling(df, window, min_periods):
            """Apply rolling z-score scaling to a DataFrame."""
            mean = df.rolling(window=window, min_periods=min_periods).mean()
            std = df.rolling(window=window, min_periods=min_periods).std(ddof=0)
            return (df - mean) / std

        FAST_Z_WIN = 4 * 24 * 5  # 5 days
        MEDIUM_Z_WIN = 4 * 24 * 20  # 20 days
        SLOW_Z_WIN = 4 * 24 * 60  # 80 days

        for features, z_window in zip(
            (FEATURES_TO_FAST_Z, FEATURES_TO_MED_Z, FEATURES_TO_SLOW_Z),
            (FAST_Z_WIN, MEDIUM_Z_WIN, SLOW_Z_WIN),
        ):
            for feature in features:
                if feature in raw_dfs:
                    df = raw_dfs[feature]
                elif feature in derived_dfs:
                    df = derived_dfs[feature]
                else:
                    print(f"Warning: {feature} not found in raw or derived data.")
                    continue

                print(f"Applying rolling z-score scaling to {feature}...")
                derived_dfs[f"{feature}_rz"] = rolling_z_scaling(
                    df, z_window, min_periods=z_window // 2
                )

        raw_factors = [
            # "vwap",
            # "atr",
            "mfi",
            "rsi",
            # "bb_upper",
            # "bb_lower",
            "bb_width",
            "bb_dev",
            # "keltner_upper",
            # "keltner_lower",
            # "stochastic_d",
        ]

        derived_factors = [
            "macd_rz",
            "cci_rz",
            "obv_rz",
            "vwap_deviation_rz",
            "1h_momentum",
            "4h_momentum_rz",
            "7d_momentum_rz",
            # "log_return_1h",
            # "log_return_4h",
            # "log_return_7d",
            # "pct_change",
            "ult_osc",
            # "amount_sum_7d",
            # "amount_7d_surge",
            # "vol_norm_mom",
            "vol_momentum",
            "squeeze_ratio",
            "gk_vol_rz",
            "skew_24h_c",
            "kurt_24h_c",
            "atr_pct_rz",
            "24hour_rtn_rz",
            "buy_ratio",
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
        ]

        features_dfs_to_use = {}
        for key in raw_factors:
            if key not in raw_dfs:
                print(f"Warning: {key} not found in raw_dfs, skipping.")
                continue

            features_dfs_to_use[key] = raw_dfs[key]

        for key in derived_factors:
            if key not in derived_dfs:
                print(f"Warning: {key} not found in derived_dfs, skipping.")
                continue

            features_dfs_to_use[key] = derived_dfs[key]

        # ================
        # External indicators
        # ================

        # Add BTCUSDT and ETHUSDT for macro features
        print("Adding BTCUSDT and ETHUSDT macro features...")

        market_syms = [
            "BTCUSDT",
            "ETHUSDT",
        ]
        # market_inds = ["vwap", "rsi", "4h_momentum", "7d_momentum"]
        market_inds = [
            "vwap_deviation_rz",
            "ult_osc",
            "rsi",
            # "vol_norm_mom",
            "1h_momentum",
            "4h_momentum_rz",
            "7d_momentum_rz",
        ]

        for factor in market_inds:
            if factor not in features_dfs_to_use:
                print(f"Missing raw factor {factor}, cannot proceed.")
                return

            for market_symbol in market_syms:
                combined_col = None

                if market_symbol not in features_dfs_to_use[factor].columns:
                    print(
                        f"Missing symbol {market_symbol} in factor {factor}, cannot proceed."
                    )
                    return
                if combined_col is None:
                    combined_col = features_dfs_to_use[factor][market_symbol]
                else:
                    combined_col += features_dfs_to_use[factor][market_symbol]

            features_dfs_to_use[f"btc_eth_agg_{factor}"] = pd.DataFrame(
                {
                    symbol: combined_col if symbol not in market_syms else np.nan
                    for symbol in all_symbol_list
                },
                index=features_dfs_to_use[factor].index,
            )

        # Calculate Beta
        BETA_LOOKBACK_WINDOW = MEDIUM_Z_WIN  # bars in lockback
        BETA_Z_WINDOW = SLOW_Z_WIN
        BENCH_SYMBOL = "BTCUSDT"

        rets = raw_dfs["close_price"].pct_change()
        bench_ret = rets[BENCH_SYMBOL]

        rolling_var = bench_ret.rolling(
            BETA_LOOKBACK_WINDOW, min_periods=BETA_LOOKBACK_WINDOW // 2
        ).var()
        rolling_cov = rets.rolling(
            BETA_LOOKBACK_WINDOW, min_periods=BETA_LOOKBACK_WINDOW // 2
        ).cov(bench_ret)

        beta_df = rolling_cov.div(rolling_var, axis=0)

        beta_z_df = rolling_z_scaling(
            beta_df, BETA_Z_WINDOW, min_periods=BETA_Z_WINDOW // 2
        )
        beta_z_df = beta_z_df.drop(columns=[BENCH_SYMBOL], errors="ignore")
        features_dfs_to_use["btc_beta_z"] = beta_z_df.ffill()

        # # Add USDT features
        # print("Adding USDT to USD macro features...")
        # EXTERNAL_DATA_DIR = os.path.join(BASE_DIR, "external_data")

        # try:
        #     usdt_usd_df = (
        #         get_single_symbol_kline_data(
        #             "USDT_USD", EXTERNAL_DATA_DIR, self.processing_device
        #         )
        #         .reindex(time_index)
        #         .ffill()
        #     )

        #     print(f"Shape of USDT_USD data: after reindex {usdt_usd_df.shape}")

        #     # USDT is a 1:1 peg to USD
        #     usdt_usd_df["depeg"] = usdt_usd_df["close_price"] - 1.0

        #     usdt_usd_df["depeg"].replace([np.inf, -np.inf], np.nan, inplace=True)
        #     # usdt_usd_df["depeg"].fillna(0, inplace=True)

        #     # USDT is a 1:1 peg to USD
        #     # usdt_usd_df["depeg_low"] = usdt_usd_df["low_price"] - 1.0
        #     # usdt_usd_df["depeg_low"].replace([np.inf, -np.inf], np.nan, inplace=True)

        #     # usdt_usd_df["depeg_high"] = usdt_usd_df["high_price"] - 1.0
        #     # usdt_usd_df["depeg_high"].replace([np.inf, -np.inf], np.nan, inplace=True)

        #     usdt_usd_df["buy_ratio"] = usdt_usd_df["buy_volume"] / usdt_usd_df["volume"]
        #     usdt_usd_df["buy_ratio"].replace([np.inf, -np.inf], np.nan, inplace=True)
        #     # usdt_usd_df["buy_ratio"].fillna(0, inplace=True)

        #     usdt_usd_df["trade_intensity"] = (
        #         (usdt_usd_df["count"] / (usdt_usd_df["volume"].replace(0, np.nan)))
        #         # .fillna(0)
        #         .astype("float32")
        #     )

        #     # depeg_rolling_mean = usdt_usd_df["depeg"].rolling(Z_SCORE_WINDOW).mean()
        #     # depeg_rolling_std = usdt_usd_df["depeg"].rolling(Z_SCORE_WINDOW).std()

        #     # # Step 3: Z-score (avoid div/0)
        #     # usdt_usd_df["depeg_z"] = (usdt_usd_df["depeg"] - depeg_rolling_mean) / (
        #     #     depeg_rolling_std.replace(0, np.nan)
        #     # )

        #     USDT_FEATURES = [
        #         # "vwap",
        #         "depeg",
        #         # "depeg_z",
        #         # "depeg_low",
        #         # "depeg_high",
        #         "buy_ratio",
        #         "trade_intensity",
        #     ]

        #     for feature in USDT_FEATURES:
        #         features_dfs_to_use[f"usdt_{feature}"] = pd.DataFrame(
        #             {symbol: usdt_usd_df[feature] for symbol in all_symbol_list},
        #             index=usdt_usd_df[feature].index,
        #         )
        # except Exception as e:
        #     print(f"Error processing USDT_USD data: {e}, skipping")
        #     raise

        keep_slice = slice(self.start_datetime, None)

        for key, df in features_dfs_to_use.items():
            features_dfs_to_use[key] = df.astype(np.float32).loc[keep_slice].ffill()

        feature_keys = list(features_dfs_to_use.keys())

        print("Finished loading factors")
        print(f"Number of symbols: {len(all_symbol_list)}")
        print(f"Number of features: {len(feature_keys)}")
        print(f"Features: {feature_keys}")

        regime_df = self.make_market_regime_target(
            raw_dfs=raw_dfs,
            keep_slice=keep_slice,
            windows_1h=windows_1h,
        )

        self.train(regime_df, features_dfs_to_use)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    model = OptimizedModel()
    model.run()