import numpy as np
import pandas as pd


def compute_log_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    price_df:
        index = Timestamp
        columns = assets
        values = Close prices
    """
    log_returns = np.log(price_df / price_df.shift(1))
    return log_returns.dropna()


def compute_rolling_correlation(
    returns_df: pd.DataFrame,
    window: int
) -> pd.DataFrame:
    """
    Returns the latest rolling correlation matrix
    """
    rolling = returns_df.rolling(window).corr()

    # Get last window correlation matrix
    last_timestamp = rolling.index.get_level_values(0).max()
    corr_matrix = rolling.loc[last_timestamp]

    return corr_matrix
