import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Global display settings for terminal clarity
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


def get_binance_data(token="BTCUSDT", interval="15m", days_from_today=1):
    """Fetches 15m data from Binance and formats to standard OHLCV."""
    # 1. Calculate Yesterday's range (Unix Milliseconds)
    yesterday = datetime.now() - timedelta(days_from_today)
    start_ts = int(yesterday.replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)
    end_ts = int(yesterday.replace(hour=23, minute=59, second=59, microsecond=0).timestamp() * 1000)

    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": token, "interval": interval, "startTime": start_ts, "endTime": end_ts, "limit": 1000}

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        # 2. Map Binance columns to standard names
        # Binance returns: [OpenTime, Open, High, Low, Close, Volume, CloseTime, ...]
        df = pd.DataFrame(data)
        df = df[[0, 1, 2, 3, 4, 5]]  # Keep only the first 6 columns
        df.columns = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']

        # 3. Format Data
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
        # Ensure prices are floats (Binance returns them as strings)
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = df[col].astype(float)

        # 4. Save and Return
        filename = f"BINANCE_{token}_15m.csv"
        df.to_csv(filename, index=False)
        print(f"SUCCESS: {token} (Binance) saved to {filename} ({len(df)} rows)")
        return df

    except Exception as e:
        print(f"BINANCE ERROR for {token}: {e}")
        return None


def get_yfinance_data(token="BTC-USD", interval="15m", days_from_today=1):
    """Fetches 15m data from Yahoo Finance and formats to standard OHLCV."""
    # 1. Calculate Yesterday's range (Dates)
    today = datetime.now().date()
    yesterday = today - timedelta(days_from_today)

    try:
        # 2. Fetch Data
        df = yf.download(tickers=token, start=yesterday, end=today, interval=interval, progress=False)

        if df.empty:
            print(f"YFINANCE ERROR: No data found for {token}.")
            return None

        # 3. Reset index and reorder to match Binance style
        df = df.reset_index()
        # yFinance columns: [Datetime, Open, High, Low, Close, Adj Close, Volume]
        # We rename Datetime to Timestamp and drop 'Adj Close'
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]  # Flatten MultiIndex if present
        df = df.rename(columns={df.columns[0]: 'Timestamp'})

        # Force column order to match: Timestamp, Open, High, Low, Close, Volume
        df = df[['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']]

        # 4. Save and Return
        clean_name = token.replace("^", "").replace("=", "")
        filename = f"YF_{clean_name}_15m.csv"
        df.to_csv(filename, index=False)
        print(f"SUCCESS: {token} (yFinance) saved to {filename} ({len(df)} rows)")
        return df

    except Exception as e:
        print(f"YFINANCE ERROR for {token}: {e}")
        return None


def get_yfinance_data_for_date(token: str, date_str: str, interval="15m"):
    """
    Pull intraday data for a specific date (YYYY-MM-DD)
    """
    import yfinance as yf
    from datetime import datetime, timedelta

    start_date = datetime.strptime(date_str, "%Y-%m-%d")
    end_date = start_date + timedelta(days=1)

    df = yf.download(
        tickers=token,
        start=start_date,
        end=end_date,
        interval=interval,
        progress=False
    )

    if df.empty:
        return None

    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: "Timestamp"})
    df = df[["Timestamp", "Open", "High", "Low", "Close", "Volume"]]

    return df


if __name__ == "__main__":
    # Pulling the S&P 500 Index
    get_yfinance_data("^GSPC", "15m", 3)

    # Pulling Bitcoin
    get_binance_data("BTCUSDT", "15m", 3)
