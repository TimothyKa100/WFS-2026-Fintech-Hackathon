from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import pandas as pd
import numpy as np

from core.config import config
from api.routes import router
from state.store import store
from ingestion.datascrape import get_yfinance_data


app = FastAPI(title="Financial Contagion Network API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


# =========================
# Feature Engineering
# =========================

def compute_log_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    log_returns = np.log(price_df / price_df.shift(1))
    return log_returns.dropna()


def compute_correlation(returns_df: pd.DataFrame) -> pd.DataFrame:
    return returns_df.corr()


def detect_stress(corr_matrix: pd.DataFrame) -> list:
    """
    Returns list of asset pairs exceeding stress threshold
    """
    stressed_pairs = []
    threshold = config.STRESS_THRESHOLD

    for i in corr_matrix.columns:
        for j in corr_matrix.columns:
            if i != j:
                value = corr_matrix.loc[i, j]
                if abs(value) >= threshold:
                    stressed_pairs.append({
                        "asset_1": i,
                        "asset_2": j,
                        "correlation": round(float(value), 4)
                    })

    return stressed_pairs


# =========================
# Data Pipeline
# =========================

def load_prices():
    dfs = []

    for asset in config.ASSET_UNIVERSE:
        ticker = asset["yf"]

        df = get_yfinance_data(
            token=ticker,
            interval=config.INTERVAL,
            days_from_today=3
        )

        if df is None or df.empty:
            continue

        df = df[["Timestamp", "Close"]]
        df = df.rename(columns={"Close": asset["name"]})
        df = df.set_index("Timestamp")

        dfs.append(df)

    price_df = pd.concat(dfs, axis=1)
    price_df = price_df.dropna()

    return price_df


def run_pipeline():
    price_df = load_prices()

    if price_df.empty:
        return

    returns_df = compute_log_returns(price_df)
    corr_matrix = compute_correlation(returns_df)
    stressed_pairs = detect_stress(corr_matrix)

    store.update("prices", price_df.tail(100).to_dict())
    store.update("returns", returns_df.tail(100).to_dict())
    store.update("correlation_matrix", corr_matrix.to_dict())
    store.update("stressed_pairs", stressed_pairs)
    store.update("last_update", datetime.utcnow().isoformat())


@app.on_event("startup")
def startup_event():
    run_pipeline()
