# Backend/services/pipeline.py

import pandas as pd
import numpy as np
from datetime import datetime

from core.config import config
from state.store import store
from ingestion.datascrape import get_yfinance_data


def run_pipeline(days: int = 5):

    dfs = []

    for asset in config.ASSET_UNIVERSE:
        ticker = asset["yf"]

        df = get_yfinance_data(
            token=ticker,
            interval=config.INTERVAL,
            days_from_today=days
        )

        if df is None or df.empty:
            continue

        df = df[["Timestamp", "Close"]]
        df = df.rename(columns={"Close": asset["name"]})
        df = df.set_index("Timestamp")

        dfs.append(df)

    if not dfs:
        store.update("error", "No market data available.")
        return

    price_df = pd.concat(dfs, axis=1).dropna()

    if price_df.empty:
        store.update("error", "Aligned dataset empty.")
        return

    returns = np.log(price_df / price_df.shift(1)).dropna()
    corr_matrix = returns.corr()

    store.update("prices", price_df.tail(100).to_dict())
    store.update("returns", returns.tail(100).to_dict())
    store.update("correlation_matrix", corr_matrix.to_dict())
    store.update("last_update", datetime.utcnow().isoformat())
    store.update("window_days", days)


