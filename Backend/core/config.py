import os


class Config:
    # Assets to track for the Contagion Network
    # Format: (DisplayName, yFinanceTicker, BinanceTicker)
    ASSET_UNIVERSE = [
        {"name": "Bitcoin", "yf": "BTC-USD", "binance": "BTCUSDT"},
        {"name": "Ethereum", "yf": "ETH-USD", "binance": "ETHUSDT"},
        {"name": "S&P 500", "yf": "^GSPC", "binance": None},
        {"name": "Gold", "yf": "GC=F", "binance": None},
        {"name": "Volatility Index", "yf": "^VIX", "binance": None}
    ]

    INTERVAL = "15m"
    DATABASE_NAME = "market_state.db"

    # Threshold for "Contagion" (Correlation coefficient)
    STRESS_THRESHOLD = 0.75


config = Config()
