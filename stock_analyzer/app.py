"""
Streamlit application for stock analysis
"""

import streamlit as st
import pandas as pd
from typing import Dict, List
import logging
from .analyzer import StockAnalyzer
from .ai_analyzer import AIAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockLyzerApp:
    """Main Streamlit application class"""
    
    def __init__(self):
        """Initialize the application"""
        self.analyzer = StockAnalyzer()
        self.ai_analyzer = AIAnalyzer()
        self.setup_page()
        
    def setup_page(self):
        """Configure the Streamlit page"""
        st.set_page_config(
            layout="wide",
            page_title="StockLyzer - Stock Analysis Tool",
            page_icon="📈",
            initial_sidebar_state="expanded"
        )
        self.display_header()
        
    def display_header(self):
        """Display the application header"""
        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            st.image('stocklyzer.png', width=150)
        with col2:
            st.title("StockLyzer 📈")
            st.markdown("""
                <div style='text-align: center; color: #666;'>
                    Advanced stock analysis and prediction tool powered by AI
                </div>
            """, unsafe_allow_html=True)
        with col3:
            st.image("qr_code.png", width=100)
            
        st.markdown("""
            <div style='background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0;'>
                <p style='margin: 0; color: #666;'>
                    <strong>⚠️ Disclaimer:</strong> Some information on this page is AI-generated. 
                    This app is developed for educational purposes only and is not advisable to rely on it for financial decision-making.
                </p>
            </div>
        """, unsafe_allow_html=True)

    def run(self):
        """Run the main application"""
        # Initialize session state for AI providers
        if 'ai_key' not in st.session_state or st.session_state.ai_key is None:
            try:
                st.session_state.ai_key = st.secrets["GROQ_API_KEY"]
                logger.info("Groq API key loaded from secrets")
            except Exception as e:
                logger.error(f"Failed to load Groq API key: {str(e)}")
                st.session_state.ai_key = None
        
        # Initialize Gemini API key
        if 'gemini_key' not in st.session_state or st.session_state.gemini_key is None:
            try:
                st.session_state.gemini_key = st.secrets["GEMINI_API_KEY"]
                logger.info("Gemini API key loaded from secrets")
            except Exception as e:
                logger.error(f"Failed to load Gemini API key: {str(e)}")
                st.session_state.gemini_key = None
        
        # Load AI configuration from secrets
        ai_config = {
            'groq': {
                'name': st.secrets.get("GROQ_PROVIDER_NAME", "Groq"),
                'model': st.secrets.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
                'available': st.session_state.ai_key is not None
            },
            'gemini': {
                'name': st.secrets.get("GEMINI_PROVIDER_NAME", "Gemini"),
                'model': st.secrets.get("GEMINI_MODEL", "gemini-2.5-flash-preview-04-17"),
                'available': st.session_state.gemini_key is not None
            }
        }
        
        # Symbol selection with improved UI
        with st.sidebar:
            st.markdown("### 📊 Analysis Settings")
            
            # News Search Toggle
            st.markdown("**News Search:**")
            enable_news = st.toggle(
                "Include latest news in analysis",
                value=True,
                help="Disable this for faster analysis if news search is slow or problematic"
            )
            
            # Store in session state for use in other parts of the app
            st.session_state.enable_news = enable_news
            
            # Update the AI analyzer setting
            self.ai_analyzer.set_news_search_enabled(enable_news)
            
            st.markdown("---")
            
            # AI Provider Selection
            st.markdown("**AI Provider:**")
            available_providers = []
            provider_mapping = {}
            
            for provider_key, config in ai_config.items():
                if config['available']:
                    available_providers.append(config['name'])
                    provider_mapping[config['name']] = {
                        'key': provider_key,
                        'model': config['model']
                    }
            
            if available_providers:
                selected_ai_provider = st.pills(
                    "ai_provider_selection",
                    available_providers,
                    default=available_providers[0],
                    help="Choose AI provider for analysis",
                    label_visibility="collapsed"
                )
                
                # Initialize the appropriate AI client
                provider_info = provider_mapping[selected_ai_provider]
                
                if provider_info['key'] == 'groq' and st.session_state.ai_key:
                    if not self.ai_analyzer.groq_client or self.ai_analyzer.groq_model_name != provider_info['model']:
                        try:
                            self.ai_analyzer.initialize_groq_client(
                                st.session_state.ai_key, 
                                provider_info['model']
                            )
                        except Exception as e:
                            st.error(f"Failed to initialize Groq client: {str(e)}")
                            
                elif provider_info['key'] == 'gemini' and st.session_state.gemini_key:
                    if not self.ai_analyzer.gemini_model or self.ai_analyzer.gemini_model_name != provider_info['model']:
                        try:
                            self.ai_analyzer.initialize_gemini_client(
                                st.session_state.gemini_key,
                                provider_info['model']
                            )
                        except Exception as e:
                            st.error(f"Failed to initialize Gemini client: {str(e)}")
            else:
                st.warning("⚠️ No AI API keys found. AI analysis will be unavailable.")
                st.info("Add GROQ_API_KEY or GEMINI_API_KEY to your secrets configuration.")
            
            st.markdown("---")
            
            # Symbol selection with improved UX
            st.markdown("**Symbol Selection:**")
            
            # Popular symbols as quick-select chips
            st.markdown("**Quick Select:**")
            
            # Create pill selection for popular symbols in sidebar-friendly layout
            popular_symbols = [
                ('📱 AAPL', 'AAPL', 'Apple Inc.'),
                ('💻 MSFT', 'MSFT', 'Microsoft Corp.'),
                ('🔍 GOOGL', 'GOOGL', 'Alphabet Inc.'),
                ('🚗 TSLA', 'TSLA', 'Tesla Inc.'),
                ('🎮 NVDA', 'NVDA', 'NVIDIA Corp.'),
                ('₿ BTC-USD', 'BTC-USD', 'Bitcoin'),
                ('⟠ ETH-USD', 'ETH-USD', 'Ethereum')
            ]
            
            selected_popular = None
            for display_name, symbol, description in popular_symbols:
                if st.button(display_name, use_container_width=True, help=description, key=f"popular_{symbol}"):
                    selected_popular = symbol
                    break
            
            # Symbol search with autocomplete
            st.markdown("**Or Search/Enter Symbol:**")
            
            # Extended symbol list for autocomplete
            extended_symbols = [
                # Popular Stocks
                'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'ADBE',
                'CRM', 'ORCL', 'INTC', 'AMD', 'AVGO', 'TXN', 'QCOM', 'AMAT', 'MRVL', 'KLAC',
                # Financial
                'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BLK', 'SCHW', 'USB',
                # Healthcare & Pharma
                'JNJ', 'PFE', 'ABBV', 'MRK', 'BMY', 'GILD', 'AMGN', 'BIIB', 'REGN', 'VRTX',
                # Consumer
                'KO', 'PEP', 'WMT', 'COST', 'TGT', 'HD', 'LOW', 'MCD', 'SBUX', 'NKE',
                # Energy
                'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'OXY', 'MPC', 'VLO', 'PSX', 'BKR',
                # ETFs
                'SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'VEA', 'VWO', 'GLD', 'SLV', 'TLT',
                # Crypto
                'BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'SOL-USD', 'DOGE-USD', 'MATIC-USD',
                'AVAX-USD', 'LINK-USD', 'UNI-USD', 'LTC-USD', 'BCH-USD', 'XLM-USD', 'VET-USD',
                # International
                'TSM', 'ASML', 'SAP', 'TM', 'NVO', 'NESN.SW', 'MC.PA', 'OR.PA', 'SAN.PA'
            ]
            
            # Initialize session state for symbol input
            if 'symbol_input' not in st.session_state:
                st.session_state.symbol_input = 'AAPL'
            
            # Update symbol input if popular symbol was selected
            if selected_popular:
                st.session_state.symbol_input = selected_popular
                st.rerun()
            
            # Text input with autocomplete suggestions
            symbol_input = st.text_input(
                "Enter Stock/Crypto Symbol",
                value=st.session_state.symbol_input,
                help="Enter any stock ticker (AAPL, TSLA) or crypto symbol (BTC-USD, ETH-USD)",
                placeholder="e.g., AAPL, BTC-USD, GOOGL"
            )
            
            # Auto-suggest matching symbols as user types (show in sidebar)
            if symbol_input and len(symbol_input) >= 1:
                matching_symbols = [s for s in extended_symbols if symbol_input.upper() in s.upper()][:8]
                if matching_symbols and symbol_input.upper() not in [s.upper() for s in matching_symbols]:
                    st.markdown("**💡 Suggestions:**")
                    for suggestion in matching_symbols:
                        if st.button(f"📊 {suggestion}", key=f"suggest_{suggestion}", use_container_width=True):
                            st.session_state.symbol_input = suggestion
                            st.rerun()
            
            # Update the selected symbol
            if symbol_input and symbol_input != st.session_state.symbol_input:
                st.session_state.symbol_input = symbol_input
            
            selected_symbol = st.session_state.symbol_input.upper() if st.session_state.symbol_input else 'AAPL'
            
            # Symbol validation and info
            if selected_symbol:
                st.markdown(f"**Selected:** `{selected_symbol}`")
                
                # Add helpful context for different symbol types
                if '-USD' in selected_symbol:
                    st.info("🪙 Cryptocurrency symbol detected")
                elif selected_symbol in ['SPY', 'QQQ', 'IWM', 'VTI', 'VOO']:
                    st.info("📊 ETF symbol detected")
                elif len(selected_symbol) <= 4 and selected_symbol.isalpha():
                    st.info("📈 Stock symbol detected")
                else:
                    st.warning("⚠️ Custom symbol - make sure it's valid")
        
            # Interval selection with clickable pills
            st.markdown("**Choose Interval:**")
            interval_options = {
                'Daily': '1d',
                'Hourly': '1h'
            }
            
            selected_interval = st.pills(
                "interval_selection",
                list(interval_options.keys()),
                default="Daily",
                help="Select the time interval for analysis",
                label_visibility="collapsed"
            )
            interval_value = interval_options[selected_interval]
            
            # Analysis button with spinner
            analyze_button = st.button(
                'Analyze 📊',
                type='primary',
                use_container_width=True,
                help="Click to analyze the selected symbol"
            )
            
            # Add cache clearing button for debugging
            if st.button(
                'Clear Cache 🗑️',
                use_container_width=True,
                help="Clear cached data if you experience issues with wrong symbol data"
            ):
                st.cache_data.clear()
                st.success("Cache cleared! Try analyzing again.")
                st.rerun()
            
            # Add app information
            st.markdown("---")
            st.markdown("### ℹ️ About")
            st.markdown("""
                StockLyzer is an advanced stock analysis tool that combines:
                - Technical analysis
                - Configurable AI providers (Groq & Gemini)
                - Optional news integration (can be disabled for speed)
                - Real-time market data
                - Interactive visualizations
                
                **AI Providers:**
                - **Groq**: Fast Llama models
                - **Gemini**: Google's latest AI models
                
                **Performance Tips:**
                - Disable news search for faster analysis
                - Clear cache if you experience issues
                
                Configure providers and models in your secrets.toml file.
            """)
        
        if analyze_button:
            with st.spinner(f'Analyzing {selected_symbol}...'):
                self.process_and_display_data(selected_symbol, interval_value)
        
        self.display_footer()
    def process_and_display_data(self, symbol: str, interval: str):
        """Process and display stock data and analysis"""
        logger.info(f"Processing data for symbol: {symbol}, interval: {interval}")
        
        # Fetch and process data
        stock_data = self.analyzer.fetch_stock_data(symbol, interval)
        if stock_data is not None:
            stock_data = self.analyzer.calculate_technical_indicators(stock_data)
            
            if stock_data is not None:
                # Display metrics in a container
                with st.container():
                    st.markdown("### 📈 Key Metrics")
                    self.display_metrics(stock_data)
                
                # Display analysis and charts in tabs
                tab1, tab2 = st.tabs(["📊 Technical Analysis", "🤖 AI Insights"])
                
                with tab1:
                    st.markdown(f"## Technical Analysis for {symbol}")
                    self.display_charts(stock_data)
                    
                with tab2:
                    try:
                        # Check if any AI provider is available
                        has_groq = st.session_state.ai_key is not None and self.ai_analyzer.groq_client is not None
                        has_gemini = st.session_state.gemini_key is not None and self.ai_analyzer.gemini_model is not None
                        has_any_ai = has_groq or has_gemini
                        
                        if has_any_ai:
                            # Show which AI provider is being used
                            if self.ai_analyzer.current_provider == "groq":
                                provider_display = f"Groq ({self.ai_analyzer.groq_model_name})"
                            else:
                                provider_display = f"Gemini ({self.ai_analyzer.gemini_model_name})"
                            
                            # Show AI provider and news status
                            news_status = "with news" if st.session_state.get('enable_news', True) else "technical analysis only"
                            st.info(f"🤖 Analysis powered by {provider_display} ({news_status})")
                            
                            summaries = self.get_data_summaries(stock_data, [interval])
                            ai_analysis = self.ai_analyzer.generate_analysis(
                                symbol,
                                interval,
                                summaries
                            )
                            formatted_analysis = self.ai_analyzer.format_analysis(ai_analysis)
                            st.markdown(formatted_analysis, unsafe_allow_html=True)
                            st.markdown("""
                                <div style='background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0;'>
                                    <p style='margin: 0; color: #666;'>
                                        <strong>Note:</strong> This analysis has been generated using AI and is intended solely for educational purposes.
                                    </p>
                                </div>
                            """, unsafe_allow_html=True)
                        else:
                            # Debug output
                            st.warning("⚠️ AI Analysis unavailable: No AI provider initialized")
                            st.info("AI analysis requires either GROQ_API_KEY or GEMINI_API_KEY in the secrets configuration.")
                            
                            # Add debug information
                            with st.expander("Debug Information"):
                                st.write(f"Has Groq Key: {st.session_state.ai_key is not None}")
                                st.write(f"Has Gemini Key: {st.session_state.gemini_key is not None}")
                                st.write(f"Groq Client: {self.ai_analyzer.groq_client is not None}")
                                st.write(f"Gemini Client: {self.ai_analyzer.gemini_model is not None}")
                                st.write(f"Current Provider: {self.ai_analyzer.current_provider}")
                                
                    except Exception as e:
                        st.error(f"Error in AI analysis: {str(e)}")
                        logger.error(f"AI analysis error: {str(e)}")

    def display_metrics(self, stock_data: pd.DataFrame) -> None:
        """
        Display key stock metrics
        
        Args:
            stock_data: DataFrame containing stock data and indicators
        """
        if stock_data is None or stock_data.empty:
            return
            
        try:
            # Get the latest data point
            current = stock_data.iloc[-1]
            previous = stock_data.iloc[-2]
            
            # Determine which price column to use - try multiple options
            price_column = None
            for col in ['Adj Close', 'Close', 'close']:
                if col in stock_data.columns:
                    price_column = col
                    break
            
            if price_column is None:
                st.warning("No price column found in the data")
                return
            
            # Convert Series to scalar values safely
            def safe_scalar(series, default=None):
                if isinstance(series, pd.Series):
                    return float(series.iloc[0]) if not series.empty else default
                return float(series) if series is not None else default
            
            # Get scalar values
            current_price = safe_scalar(current[price_column])
            prev_price = safe_scalar(previous[price_column])
            rsi_value = safe_scalar(current.get('RSI_14'))
            
            # Try different volume column names
            volume_value = None
            for vol_col in ['Volume', 'volume']:
                if vol_col in stock_data.columns:
                    volume_value = safe_scalar(current.get(vol_col))
                    break
            
            if current_price is None or prev_price is None:
                st.warning("Could not calculate price metrics")
                return
                
            # Calculate price change
            price_change = current_price - prev_price
            price_change_pct = (price_change / prev_price) * 100
            
            # Format metrics
            metrics = {
                'Current Price': {
                    'value': f"${current_price:.2f}",
                    'delta': f"{price_change_pct:.2f}%"
                },
                'RSI': {
                    'value': f"{rsi_value:.1f}" if rsi_value is not None else "N/A",
                    'delta': None
                },
                'Volume': {
                    'value': f"{volume_value:,.0f}" if volume_value is not None else "N/A",
                    'delta': None
                }
            }
            
            # Display metrics in columns
            cols = st.columns(len(metrics))
            for col, (metric_name, metric_data) in zip(cols, metrics.items()):
                with col:
                    st.metric(
                        label=metric_name,
                        value=metric_data['value'],
                        delta=metric_data['delta']
                    )
                    
        except Exception as e:
            logger.error(f"Error displaying metrics: {str(e)}")
            st.warning("Could not display some metrics")

    def display_charts(self, data: pd.DataFrame):
        """Display technical analysis charts"""
        # Determine which price column to use - try multiple options
        price_column = None
        for col in ['Adj Close', 'Close', 'close']:
            if col in data.columns:
                price_column = col
                break
        
        if price_column is None:
            st.warning("No price column found for charting")
            return
        
        chart_configs = [
            {'indicators': [price_column, 'EMA_50', 'SMA_20'], 'title': 'Price Trend'},
            {'indicators': ['OBV'], 'title': 'On-Balance Volume (OBV) Indicator'},
            {'indicators': ['MACD_12_26_9', 'MACDh_12_26_9'], 'title': 'MACD Indicator'},
            {'indicators': ['RSI_14'], 'title': 'RSI Indicator', 
             'hlines': [(70, 'red', 'dash'), (30, 'green', 'dash')]},
            {'indicators': ['BBU_5_2.0', 'BBM_5_2.0', 'BBL_5_2.0', price_column], 
             'title': 'Bollinger Bands'},
            {'indicators': ['STOCHk_14_3_3', 'STOCHd_14_3_3'], 
             'title': 'Stochastic Oscillator'},
            {'indicators': ['WILLR_14'], 'title': 'Williams %R'},
            {'indicators': ['ADX_14'], 'title': 'Average Directional Index (ADX)'},
            {'indicators': ['CMF_20'], 'title': 'Chaikin Money Flow (CMF)'}
        ]
        
        for config in chart_configs:
            # Filter out indicators that don't exist in the data
            available_indicators = [ind for ind in config['indicators'] if ind in data.columns]
            if available_indicators:  # Only create chart if there are available indicators
                fig = self.analyzer.create_technical_chart(
                    data,
                    available_indicators,
                    config['title'],
                    config.get('hlines')
                )
                st.plotly_chart(fig, use_container_width=True)
            
    @staticmethod
    def format_number(number: float) -> str:
        """Format numbers for display"""
        if abs(number) >= 1_000_000:
            return f"{number/1_000_000:.2f}M"
        elif abs(number) >= 1_000:
            return f"{number/1_000:.2f}K"
        return f"{number:.2f}"
        
    @staticmethod
    def get_data_summaries(data: pd.DataFrame, intervals: List[str]) -> Dict:
        """Get summaries of stock data for different intervals"""
        summaries = {}
        for interval in intervals:
            if data is not None and not data.empty:
                last_summary = data.iloc[-1].to_dict()
                summaries[interval] = {
                    f"{key}_{interval}": value 
                    for key, value in last_summary.items()
                }
            else:
                summaries[interval] = {'error': f'No data available for {interval} interval.'}
        return summaries
        
    def display_footer(self):
        """Display the application footer"""
        st.markdown("---")
        
        # Footer in columns
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
                ### 💡 About This App
                StockLyzer demonstrates advanced stock analysis capabilities using:
                - Real-time market data from Yahoo Finance
                - Technical indicators and charting
                - AI-powered analysis and predictions
                - Interactive visualizations
                
                Created by [Nyein Chan Ko Ko](https://github.com/nchanko)
            """)
            
        with col2:
            st.markdown("""
                ### ☕ Support the Project
                If you find this app helpful, consider buying me a coffee!
                
                <a href="https://www.buymeacoffee.com/nyeinchankoko" target="_blank">
                    <img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" 
                         alt="Buy Me A Coffee" 
                         style="height: 60px !important; width: 217px !important;">
                </a>
            """, unsafe_allow_html=True) 