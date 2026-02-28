import os


class Config:
    # Assets to track for the Contagion Network
    # Format: (DisplayName, yFinanceTicker, BinanceTicker)
    ASSET_UNIVERSE = [
        {"name": "Bitcoin", "yf": "BTC-USD", "binance": "BTCUSDT"},
        {"name": "Ethereum", "yf": "ETH-USD", "binance": "ETHUSDT"},

        # US Equity ETFs
        {"name": "S&P 500 ETF", "yf": "SPY", "binance": None},
        {"name": "Nasdaq ETF", "yf": "QQQ", "binance": None},
        {"name": "Dow ETF", "yf": "DIA", "binance": None},
        {"name": "Russell 2000 ETF", "yf": "IWM", "binance": None},

        # Big Tech Stocks
        {"name": "Apple", "yf": "AAPL", "binance": None},
        {"name": "Microsoft", "yf": "MSFT", "binance": None},
        {"name": "Nvidia", "yf": "NVDA", "binance": None},
        {"name": "Tesla", "yf": "TSLA", "binance": None},

        # Commodities ETFs
        {"name": "Gold ETF", "yf": "GLD", "binance": None},
        {"name": "Silver ETF", "yf": "SLV", "binance": None},
        {"name": "Oil ETF", "yf": "USO", "binance": None},

        # Bonds ETF
        {"name": "20Y Treasury ETF", "yf": "TLT", "binance": None},

        # Volatility ETF proxy
        {"name": "VIX ETF", "yf": "VXX", "binance": None},
    ]

    INTERVAL = "15m"
    DATABASE_NAME = "market_state.db"

    # Threshold for "Contagion" (Correlation coefficient)
    STRESS_THRESHOLD = 0.75

    # Rolling z-score anomaly detection settings
    ANOMALY_Z_WINDOW = 96
    ANOMALY_Z_THRESHOLD = 3.0


config = Config()
