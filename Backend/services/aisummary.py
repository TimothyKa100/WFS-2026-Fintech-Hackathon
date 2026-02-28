ASSET_OVERVIEWS = {
    "Bitcoin": """
Bitcoin remains the dominant crypto asset, often acting as a proxy for
overall digital asset risk sentiment. Intraday movements typically reflect
macro liquidity conditions and speculative flows.
""",

    "Ethereum": """
Ethereum continues to represent the core smart contract ecosystem.
Price action often reflects shifts in DeFi, staking, and broader crypto
risk appetite.
""",

    "S&P 500 ETF": """
The S&P 500 ETF tracks broad U.S. large-cap equities and serves as a
benchmark for systemic equity risk. Movements typically reflect macro
economic sentiment and earnings expectations.
""",

    "Nasdaq ETF": """
The Nasdaq ETF is heavily technology-weighted and tends to exhibit
higher sensitivity to interest rates and growth expectations.
""",

    "Dow ETF": """
The Dow ETF represents blue-chip industrial exposure and often shows
relative stability compared to tech-heavy indices.
""",

    "Russell 2000 ETF": """
The Russell 2000 ETF reflects small-cap equity performance and is
often viewed as a proxy for domestic economic strength.
""",

    "Apple": """
Apple remains a defensive mega-cap technology stock with strong
cash flow fundamentals. Intraday volatility is typically lower than
high-growth peers.
""",

    "Microsoft": """
Microsoft benefits from enterprise cloud exposure and recurring
software revenue, often demonstrating resilience during market stress.
""",

    "Nvidia": """
Nvidia represents high-beta semiconductor exposure and is often
associated with AI-driven growth narratives.
""",

    "Tesla": """
Tesla exhibits elevated volatility due to its growth profile and
strong retail investor participation.
""",

    "Gold ETF": """
Gold ETF serves as a defensive hedge asset, typically gaining during
risk-off or inflationary environments.
""",

    "Silver ETF": """
Silver ETF provides exposure to both industrial demand and precious
metal hedge dynamics.
""",

    "Oil ETF": """
Oil ETF reflects global energy demand and geopolitical supply risks,
often reacting sharply to macro developments.
""",

    "20Y Treasury ETF": """
The 20Y Treasury ETF captures long-duration interest rate exposure and
typically moves inversely to equity risk sentiment.
""",

    "VIX ETF": """
The VIX ETF tracks short-term volatility expectations and spikes
during market stress periods.
"""
}


def get_asset_overview(asset_name: str) -> str:
    return ASSET_OVERVIEWS.get(
        asset_name,
        "No overview available for this asset."
    )