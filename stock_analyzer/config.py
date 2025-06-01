"""
Configuration settings for the StockLyzer application
"""

from typing import Dict, List

# Available symbols for analysis
SYMBOLS: List[str] = [
    'AAPL', 'ADBE', 'ADA-USD', 'AMD', 'ARB11841-USD', 'AVAX-USD', 'BNB-USD',
    'BTC-USD', 'COIN', 'DOGE-USD', 'ETH-USD', 'GOOGL', 'ICP-USD', 'INTC',
    'KSM-USD', 'LINK-USD', 'MATIC-USD', 'MSTR', 'MSFT', 'NEAR-USD', 'NVDA',
    'SEI-USD', 'SOL-USD', 'TSLA', 'TSM', 'Custom symbol...'
]

# Available intervals for data
INTERVALS: Dict[str, str] = {
    'Daily': '1d',
    'Hourly': '1h'
}

# Technical indicator configurations
CHART_CONFIGS: List[Dict] = [
    {'indicators': ['Close', 'EMA_50', 'SMA_20'], 'title': 'Price Trend'},
    {'indicators': ['OBV'], 'title': 'On-Balance Volume (OBV) Indicator'},
    {'indicators': ['MACD_12_26_9', 'MACDh_12_26_9'], 'title': 'MACD Indicator'},
    {'indicators': ['RSI_14'], 'title': 'RSI Indicator', 
     'hlines': [(70, 'red', '--'), (30, 'green', '--')]},
    {'indicators': ['BBU_5_2.0', 'BBM_5_2.0', 'BBL_5_2.0', 'Close'], 
     'title': 'Bollinger Bands'},
    {'indicators': ['STOCHk_14_3_3', 'STOCHd_14_3_3'], 
     'title': 'Stochastic Oscillator'},
    {'indicators': ['WILLR_14'], 'title': 'Williams %R'},
    {'indicators': ['ADX_14'], 'title': 'Average Directional Index (ADX)'},
    {'indicators': ['CMF_20'], 'title': 'Chaikin Money Flow (CMF)'}
]

# AI Analysis settings
AI_MODEL = "llama-3.3-70b-versatile"
AI_MAX_TOKENS = 1000

# Data fetching settings
DAILY_DAYS = 120
HOURLY_DAYS = 60
CACHE_TTL = 3600  # Cache time-to-live in seconds

# News search settings
DEFAULT_NEWS_SEARCH_ENABLED = True  # Can be disabled for faster processing
NEWS_SEARCH_TIMEOUT = 10  # Seconds 