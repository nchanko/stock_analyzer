"""
Core stock analysis functionality
"""

import streamlit as st
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from datetime import datetime, timedelta
import plotly.graph_objects as go
from typing import Dict, List, Optional, Union, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockAnalyzer:
    """Main class for stock analysis functionality"""
    
    def __init__(self):
        """Initialize the StockAnalyzer with required components"""
        pass

    @staticmethod
    @st.cache_data(ttl=3600)
    def fetch_stock_data(symbol: str, interval: str) -> Optional[pd.DataFrame]:
        """
        Fetch stock data from Yahoo Finance
        
        Args:
            symbol: Stock or cryptocurrency symbol
            interval: Data interval ('1d' for daily, '1h' for hourly)
            
        Returns:
            DataFrame containing stock data or None if fetch fails
        """
        try:
            logger.info(f"Fetching stock data for symbol: {symbol}, interval: {interval}")
            end_date = datetime.today()
            start_date = end_date - timedelta(days=120 if interval == '1d' else 60)
            
            stock_data = yf.download(
                symbol,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval=interval,
                progress=False,
                auto_adjust=True  # Explicitly set to True to avoid warning
            )
            
            if stock_data.empty:
                st.error(f"No data returned for {symbol} using interval {interval}.")
                return None
            
            logger.info(f"Successfully fetched {len(stock_data)} rows of data for {symbol}")
            
            # Handle MultiIndex columns - flatten them if they exist
            if isinstance(stock_data.columns, pd.MultiIndex):
                # For single symbol, use the first level (price types: Open, High, Low, Close, Volume)
                # and ignore the second level (symbol name)
                stock_data.columns = stock_data.columns.get_level_values(0)
            
            # Ensure we have a single-level index
            if isinstance(stock_data.index, pd.MultiIndex):
                stock_data = stock_data.reset_index()
                stock_data.set_index('Date', inplace=True)
            
            stock_data.index = pd.to_datetime(stock_data.index)
            return stock_data.sort_index()
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            st.error(f"Error fetching data: {str(e)}")
            return None

    @staticmethod
    def calculate_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for the stock data
        
        Args:
            data: DataFrame containing stock data
            
        Returns:
            DataFrame with added technical indicators
        """
        if data is None or data.empty:
            return None
            
        try:
            # Create a copy to avoid modifying the original
            df = data.copy()
            
            # Handle MultiIndex columns - flatten them if they exist
            if isinstance(df.columns, pd.MultiIndex):
                # For single symbol, use the first level (price types: Open, High, Low, Close, Volume)
                df.columns = df.columns.get_level_values(0)
            
            # Ensure we have a single-level index and it's datetime
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index()
                if 'Date' in df.columns:
                    df.set_index('Date', inplace=True)
                else:
                    # If no Date column, use the first level of the MultiIndex
                    df.set_index(df.index.get_level_values(0), inplace=True)
            
            # Ensure index is datetime
            df.index = pd.to_datetime(df.index)
            
            # Sort by index to ensure proper calculation
            df = df.sort_index()
            
            # Ensure column names are strings (pandas_ta requirement)
            df.columns = df.columns.astype(str)
            
            # Map yfinance column names to pandas_ta expected names (lowercase)
            column_mapping = {
                'Open': 'open',
                'High': 'high', 
                'Low': 'low',
                'Close': 'close',
                'Adj Close': 'close',  # Use Adj Close as close if available
                'Volume': 'volume'
            }
            
            # Apply column mapping for pandas_ta compatibility
            for old_name, new_name in column_mapping.items():
                if old_name in df.columns:
                    df[new_name] = df[old_name]
            
            # Ensure we have the required columns for pandas_ta
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.warning(f"Missing required columns for technical analysis: {missing_columns}")
                logger.info(f"Available columns: {list(df.columns)}")
                return data
            
            # Calculate indicators one by one and merge results properly
            try:
                # Trend indicators
                macd_result = df.ta.macd()
                if macd_result is not None:
                    df = pd.concat([df, macd_result], axis=1)
                
                sma_20 = df.ta.sma(length=20)
                if sma_20 is not None:
                    df['SMA_20'] = sma_20
                    
                ema_50 = df.ta.ema(length=50)
                if ema_50 is not None:
                    df['EMA_50'] = ema_50
                    
                psar = df.ta.psar()
                if psar is not None:
                    df = pd.concat([df, psar], axis=1)
                    
            except Exception as e:
                logger.warning(f"Failed to calculate trend indicators: {str(e)}")
            
            try:
                # Momentum indicators
                rsi = df.ta.rsi()
                if rsi is not None:
                    df['RSI_14'] = rsi
                    
                stoch = df.ta.stoch()
                if stoch is not None:
                    df = pd.concat([df, stoch], axis=1)
                    
                adx = df.ta.adx()
                if adx is not None:
                    df = pd.concat([df, adx], axis=1)
                    
                willr = df.ta.willr()
                if willr is not None:
                    df['WILLR_14'] = willr
                    
            except Exception as e:
                logger.warning(f"Failed to calculate momentum indicators: {str(e)}")
            
            try:
                # Volatility indicators
                bbands = df.ta.bbands()
                if bbands is not None:
                    df = pd.concat([df, bbands], axis=1)
                    
            except Exception as e:
                logger.warning(f"Failed to calculate volatility indicators: {str(e)}")
            
            try:
                # Volume indicators
                obv = df.ta.obv()
                if obv is not None:
                    df['OBV'] = obv
                    
                cmf = df.ta.cmf()
                if cmf is not None:
                    df['CMF_20'] = cmf
                    
            except Exception as e:
                logger.warning(f"Failed to calculate volume indicators: {str(e)}")
            
            # Post-processing
            if 'OBV' in df.columns:
                df['OBV_in_million'] = df['OBV'] / 1e6
            if 'MACDh_12_26_9' in df.columns:
                df['MACD_histogram_12_26_9'] = df['MACDh_12_26_9']
            
            # Clean up NaN values using newer methods
            df = df.ffill().bfill()
            return df
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            st.warning("Some technical indicators could not be calculated")
            return data

    @staticmethod
    def create_technical_chart(
        data: pd.DataFrame,
        indicators: List[str],
        title: str,
        hlines: Optional[List[Tuple[float, str, str]]] = None
    ) -> go.Figure:
        """
        Create a Plotly chart for technical indicators
        
        Args:
            data: DataFrame containing stock data and indicators
            indicators: List of indicator column names to plot
            title: Chart title
            hlines: Optional list of horizontal lines to add
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        for indicator in indicators:
            if indicator in data.columns:
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data[indicator],
                    mode='lines',
                    name=indicator
                ))
        
        if hlines:
            for y, color, style in hlines:
                fig.add_shape(
                    type="line",
                    x0=data.index.min(),
                    y0=y,
                    x1=data.index.max(),
                    y1=y,
                    line=dict(color=color, dash='dash' if style == '--' else style)
                )
                fig.add_annotation(
                    x=data.index.mean(),
                    y=y,
                    text=f"{y}",
                    showarrow=False
                )
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Value",
            legend_title="Indicator",
            height=400
        )
        fig.update_xaxes(tickangle=-45)
        return fig 