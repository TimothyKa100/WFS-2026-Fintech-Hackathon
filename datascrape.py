import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path

# Global display settings for terminal clarity
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

OUTPUT_DIR = Path("SCRAPED_DATA")


def save_ohlcv_parquet(df: pd.DataFrame, source: str, token: str, interval: str) -> Path:
    clean_name = token.replace("^", "").replace("=", "")
    filename = OUTPUT_DIR / f"{source}_{clean_name}_{interval}.parquet"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(filename, index=False)
    return filename


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
        filename = save_ohlcv_parquet(df, "BINANCE", token, interval)
        print(f"SUCCESS: {token} (Binance) saved to {filename} ({len(df)} rows)")
        return df

    except Exception as e:
        print(f"BINANCE ERROR for {token}: {e}")
        return None

FALLBACK_STOCK_TOKENS = [
    'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'TSLA', 'AMD', 'NFLX', 'AVGO',
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'SCHW', 'BLK', 'AXP', 'PYPL',
    'V', 'MA', 'ADBE', 'CRM', 'ORCL', 'INTC', 'QCOM', 'MU', 'IBM', 'CSCO',
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'VLO', 'OXY', 'PSX', 'KMI',
    'JNJ', 'PFE', 'MRK', 'UNH', 'LLY', 'ABBV', 'TMO', 'DHR', 'ABT', 'BMY',
    'WMT', 'COST', 'HD', 'LOW', 'NKE', 'SBUX', 'MCD', 'DIS', 'KO', 'PEP',
]


def get_top_1000_stock_tokens() -> list[str]:
    """Gets top ~1000 US stocks by market value proxy via Russell 1000 (IWB holdings)."""
    url = "https://www.ishares.com/us/products/239707/ishares-russell-1000-etf/1467271812596.ajax?fileType=csv&fileName=IWB_holdings&dataType=fund"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # iShares export has metadata rows before the holdings header.
        start_idx = response.text.find("Ticker,Name,Sector,Asset Class,Market Value,Weight (%)")
        if start_idx == -1:
            raise ValueError("Could not find holdings header in IWB CSV")

        csv_text = response.text[start_idx:]
        df = pd.read_csv(StringIO(csv_text))

        if "Ticker" not in df.columns:
            raise ValueError("Ticker column missing in IWB holdings data")

        tickers = []
        for raw_ticker in df["Ticker"].dropna().astype(str).tolist():
            ticker = raw_ticker.strip().upper()
            if not ticker or ticker in {"-", "CASH", "USD", "N/A"}:
                continue
            # Yahoo Finance expects dots for class shares (e.g., BRK.B)
            ticker = ticker.replace(" ", "").replace("/", ".")
            tickers.append(ticker)

        unique_tickers = list(dict.fromkeys(tickers))
        top_1000 = unique_tickers[:1000]

        if len(top_1000) < 900:
            raise ValueError(f"Unexpectedly low ticker count from IWB holdings: {len(top_1000)}")

        print(f"Loaded {len(top_1000)} top-value stock tokens (Russell 1000 proxy).")
        return top_1000

    except Exception as e:
        print(f"TOP1000 TOKEN FETCH ERROR: {e}")
        print("Falling back to built-in stock token list.")
        return FALLBACK_STOCK_TOKENS

ETF_TOKENS = [
    'SPY', 'VOO', 'IVV', 'QQQ', 'VTI', 'IWM', 'DIA',
    'XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB', 'XLRE', 'XLC',
    'ARKK', 'SMH', 'SOXX', 'VGT', 'IWF', 'IWD', 'MTUM', 'QUAL',
    'TLT', 'IEF', 'SHY', 'LQD', 'HYG', 'BND',
    'GLD', 'SLV', 'USO', 'UNG', 'DBC',
    'EEM', 'EFA', 'VEA', 'VWO',
    'VNQ', 'SCHD', 'VYM', 'DGRO', 'JEPI', 'JEPQ',
    'VXX', 'UVXY',
]

STOCK_TOKENS = get_top_1000_stock_tokens()
tokens = list(dict.fromkeys(STOCK_TOKENS + ETF_TOKENS))

def get_yfinance_data(token = "BTC-USD", interval="1d", days_from_today=30):
    """Fetches 15m data from Yahoo Finance and formats to standard OHLCV."""
    # 1. Calculate Yesterday's range (Dates)
    # today = datetime.now().date()
    # yesterday = today - timedelta(days_from_today)

    try:
        # 2. Fetch Data
        df = yf.download(tickers=token, period="max", interval=interval, progress=False, auto_adjust=True)

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
        filename = save_ohlcv_parquet(df, "YF", token, interval)
        print(f"SUCCESS: {token} (yFinance) saved to {filename} ({len(df)} rows)")
        return df

    except Exception as e:
        print(f"YFINANCE ERROR for {token}: {e}")
        return None


if __name__ == "__main__":
    for token in tokens:
        get_yfinance_data(token, "1d")

    # get_binance_data("BTCUSDT", "1d")