import yfinance as yf
from datetime import datetime, timedelta
import pandas_ta as ta

def fetch_and_calculate_indicators(symbol, interval):
    # Adjust the start date based on the interval
    end_date = datetime.today()
    if interval == '1d':
        start_date = end_date - timedelta(days=120)  # 4 months for daily data
    else:
        # Adjust for hourly and 15-minute data to ensure within the 60-day limit
        start_date = end_date - timedelta(days=60)  # Adjusted for yfinance limitation
    
    try:
        # Fetch stock data using yfinance with the specified interval
        stock_data = yf.download(symbol, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), interval=interval)
        
        if stock_data.empty:
            raise ValueError(f"No data returned for {symbol} using interval {interval}.")
        
        # Calculate technical indicators using pandas-ta
        stock_data.ta.macd(append=True)
        stock_data.ta.rsi(append=True)
        stock_data.ta.bbands(append=True)
        stock_data.ta.obv(append=True)
        stock_data.ta.sma(length=20, append=True)
        stock_data.ta.ema(length=50, append=True)
        stock_data.ta.stoch(append=True)
        stock_data.ta.adx(append=True)
        stock_data.ta.willr(append=True)
        stock_data.ta.cmf(append=True)
        stock_data.ta.psar(append=True)
        
        # Convert OBV to million and handle MACD histogram naming
        stock_data['OBV_in_million'] = stock_data['OBV'] / 1e6
        stock_data['MACD_histogram_12_26_9'] = stock_data['MACDh_12_26_9']
        
    except Exception as e:
        print(f"Error fetching data for {symbol} at interval {interval}: {str(e)}")
        stock_data = None  # Ensure we return None to indicate failure

    return stock_data

def load_stock_data(symbol, intervals=['1d', '1h']):
    summaries = {}
    for interval in intervals:
        stock_data = fetch_and_calculate_indicators(symbol, interval)
        if stock_data is not None and not stock_data.empty:
            # Generate a summary for the last available data point
            last_summary = stock_data.iloc[-1].to_dict()
            summary_formatted = {f"{key}_{interval}": value for key, value in last_summary.items()}
            summaries[interval] = summary_formatted
        else:
            summaries[interval] = {'error': f'No data available for {interval} interval.'}
    return summaries

# Example usage
summaries = load_stock_data('AAPL', intervals=['1d', '1h'])
print(summaries)
