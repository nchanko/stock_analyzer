"""
AI analysis component for StockLyzer
"""

import streamlit as st
from typing import Dict, Optional
import logging
from .search_engine import AISearch
from .base_provider import AIProvider
from .groq_provider import GroqProvider
from .gemini_provider import GeminiProvider

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIAnalyzer:
    """Main AI analyzer class that manages multiple providers"""
    
    def __init__(self, enable_news_search: bool = True):
        """Initialize the AI analyzer with required components"""
        self.aisearch = AISearch()
        self.groq_provider = GroqProvider()
        self.gemini_provider = GeminiProvider()
        self.current_provider = None
        self.enable_news_search = enable_news_search  # Option to disable news search
        
    def initialize_groq_client(self, api_key: str, model_name: str = "llama-3.3-70b-versatile") -> None:
        """Initialize the Groq provider"""
        self.groq_provider.initialize(api_key, model_name)
        self.current_provider = "groq"
    
    def initialize_gemini_client(self, api_key: str, model_name: str = "gemini-2.5-flash-preview-04-17") -> None:
        """Initialize the Gemini provider"""
        self.gemini_provider.initialize(api_key, model_name)
        self.current_provider = "gemini"
    
    def set_news_search_enabled(self, enabled: bool) -> None:
        """Enable or disable news search"""
        self.enable_news_search = enabled
        logger.info(f"News search {'enabled' if enabled else 'disabled'}")
    
    @property
    def client(self):
        """Backward compatibility property"""
        if self.current_provider == "groq":
            return self.groq_provider.client
        elif self.current_provider == "gemini":
            return self.gemini_provider.model
        return None
    
    @property
    def groq_client(self):
        """Access to Groq client"""
        return self.groq_provider.client
    
    @property
    def gemini_model(self):
        """Access to Gemini model"""
        return self.gemini_provider.model
    
    @property
    def groq_model_name(self):
        """Get Groq model name"""
        return self.groq_provider.model_name
    
    @property
    def gemini_model_name(self):
        """Get Gemini model name"""
        return self.gemini_provider.model_name
    
    @staticmethod
    @st.cache_data(ttl=3600)  # Cache news for 1 hour
    def get_cached_news(symbol: str) -> str:
        """
        Get cached news for a symbol to prevent rate limiting
        
        Args:
            symbol: Stock symbol
            
        Returns:
            News context string
        """
        try:
            # Quick check if we can import and create AISearch
            aisearch = AISearch()
            
            # Try to get news with a simple fallback
            news = aisearch.serch_prompt_generate(symbol, search_mode=True)
            
            if news and news.strip():
                logger.info(f"Successfully retrieved and cached news for {symbol}")
                return news
            else:
                logger.info(f"No news found for {symbol}")
                return "No recent news available. Focus on technical analysis only."
                
        except ImportError as e:
            logger.warning(f"AISearch import failed for {symbol}: {str(e)}")
            return "News search unavailable. Focus on technical analysis only."
        except Exception as e:
            logger.warning(f"Failed to retrieve news for {symbol}: {str(e)}")
            return "No recent news available due to search limitations. Focus on technical analysis only."
        
    def generate_analysis(
        self,
        symbol: str,
        timeframe: str,
        data_summary: Dict
    ) -> str:
        """
        Generate AI analysis using either Groq or Gemini with automatic fallback
        
        Args:
            symbol: Stock symbol
            timeframe: Analysis timeframe
            data_summary: Summary of technical indicators
            
        Returns:
            AI-generated analysis text
        """
        if not self._has_any_provider():
            st.error("No AI provider available")
            return "Error: AI analysis unavailable"
            
        try:
            # Handle news retrieval based on settings
            if self.enable_news_search:
                # Try to get cached news, but don't let it block the analysis
                try:
                    latest_news = self.get_cached_news(symbol)
                    logger.info(f"News search enabled - retrieved news for {symbol}")
                except Exception as e:
                    logger.warning(f"News retrieval failed for {symbol}: {str(e)}")
                    latest_news = "No recent news available. Focus on technical analysis only."
            else:
                # Skip news entirely for faster processing
                latest_news = "News search disabled. Focus on technical analysis only."
                logger.info(f"News search disabled - proceeding with technical analysis only for {symbol}")
            
            # Prepare prompts
            system_prompt = self._prepare_system_prompt(latest_news)
            user_message = self._prepare_user_message(symbol, data_summary)
            
            # Try current provider first
            result = self._try_current_provider(system_prompt, user_message)
            if result:
                return result
            
            # If current provider failed, try fallback
            fallback_result = self._try_fallback_provider(system_prompt, user_message)
            if fallback_result:
                return fallback_result
            
            return "Error: Both AI providers are temporarily unavailable. Please try again later."
            
        except Exception as e:
            logger.error(f"Error generating AI analysis: {str(e)}")
            return f"Error generating analysis: {str(e)}"
    
    def _has_any_provider(self) -> bool:
        """Check if any provider is available"""
        return self.groq_provider.is_available() or self.gemini_provider.is_available()
    
    def _get_current_provider(self) -> Optional[AIProvider]:
        """Get the current provider instance"""
        if self.current_provider == "groq":
            return self.groq_provider
        elif self.current_provider == "gemini":
            return self.gemini_provider
        return None
    
    def _get_fallback_provider(self) -> Optional[AIProvider]:
        """Get the fallback provider instance"""
        if self.current_provider == "groq" and self.gemini_provider.is_available():
            return self.gemini_provider
        elif self.current_provider == "gemini" and self.groq_provider.is_available():
            return self.groq_provider
        return None
    
    def _try_current_provider(self, system_prompt: str, user_message: str) -> Optional[str]:
        """Try to generate analysis with the current provider"""
        provider = self._get_current_provider()
        if not provider or not provider.is_available():
            return None
        
        return provider.generate_response(system_prompt, user_message)
    
    def _try_fallback_provider(self, system_prompt: str, user_message: str) -> Optional[str]:
        """Try to generate analysis with the fallback provider"""
        fallback_provider = self._get_fallback_provider()
        if not fallback_provider:
            return None
        
        logger.info(f"Falling back to {fallback_provider.provider_name} after {self.current_provider} failure")
        result = fallback_provider.generate_response(system_prompt, user_message)
        
        if result:
            return f"🔄 Analysis generated using {fallback_provider.provider_name} (fallback)\n\n{result}"
        
        return None
    
    def _prepare_system_prompt(self, latest_news: str) -> str:
        """
        Prepare the system prompt for AI analysis
        
        Args:
            latest_news: Latest news about the stock
            
        Returns:
            Formatted system prompt
        """
        return f"""
        You are an experienced trading analyst providing actionable trading advice. Focus on what traders need to know to make profitable decisions.

        CONFIDENCE LEVEL CALCULATION - BE PRECISE:
        Calculate confidence based on REAL indicator alignment:
        
        **HIGH CONFIDENCE (75-95%):**
        - 4+ indicators align in same direction
        - Strong volume confirmation (>average)
        - Clear trend with momentum
        - Support/resistance levels confirmed
        - MACD + RSI + Volume + Price action all agree
        
        **MEDIUM CONFIDENCE (45-75%):**
        - 2-3 indicators align
        - Mixed signals but trend still visible
        - Moderate volume
        - Some conflicting indicators
        
        **LOW CONFIDENCE (15-45%):**
        - Indicators conflict with each other
        - Low volume/weak momentum
        - Choppy/sideways price action
        - No clear trend direction
        
        TRADING GUIDELINES:
        - Be specific about entry/exit points and price levels
        - Always include risk management (stop losses, position sizing)
        - Differentiate between day trading and investment strategies
        - Use clear buy/sell/hold/short recommendations
        - Base confidence on actual technical indicator alignment
        
        Latest Market News: {latest_news}
        
        Format your response as follows:

        ## 🚀 Stock Overview
        Company/crypto brief (1-2 sentences) + current market cap/price context.

        ## ⚡ IMMEDIATE ACTION REQUIRED
        **Right Now:** BUY / SELL / HOLD / SHORT / WAIT
        **Confidence Level:** High/Medium/Low (X%) - CALCULATE this based on indicator alignment
        **Technical Reasoning:** Explain which specific indicators support this confidence level
        **Why:** One clear reason for this recommendation

        ## 📊 Technical Setup
        **Current Trend:** Bullish/Bearish/Neutral with strength (1-10)
        **Indicator Alignment Score:** X/10 (how many indicators agree)
        **Key Levels:** 
        - Support: $X.XX (buy zone)
        - Resistance: $X.XX (sell/short zone)
        - Stop Loss: $X.XX (risk management)

        ## 💰 Trading Strategies

        ### Day Trading (Next 1-5 Days)
        - **Entry:** Specific price level and conditions
        - **Target:** Profit taking levels
        - **Stop Loss:** Maximum risk level
        - **Position Size:** Suggested % of portfolio
        - **Confidence:** X% based on short-term indicators

        ### Swing Trading (1-4 Weeks)
        - **Setup:** What pattern/signal to wait for
        - **Entry Zone:** Price range to buy/short
        - **Targets:** Multiple profit levels
        - **Risk Management:** Stop loss strategy
        - **Confidence:** X% based on medium-term indicators

        ### Long-Term Investment (3+ Months)
        - **Investment Thesis:** Why hold long-term
        - **Entry Strategy:** Dollar cost average vs lump sum
        - **Price Targets:** 6-12 month projections
        - **Risk Factors:** What could go wrong
        - **Confidence:** X% based on fundamental + technical

        ## 📈 Market Momentum
        **Institutional Activity:** Are big players buying/selling?
        **Retail Sentiment:** What retail traders are doing
        **Volume Analysis:** Is there conviction behind moves?
        **Momentum Score:** X/10 (based on actual volume + price action)

        ## 🔥 High-Priority Alerts
        - **Watch for:** Specific events/levels that change everything
        - **If price hits $X:** Then do this immediately
        - **Risk Warning:** Biggest threat to current position

        ## 📰 News Impact Score
        **Impact:** High/Medium/Low on stock price
        **Timeline:** How long will news affect price
        **Trading Opportunity:** How to profit from news

        ## 🎯 Confidence Breakdown
        **Overall Confidence:** X% 
        **Calculation:**
        - RSI alignment: +/- X points
        - MACD confirmation: +/- X points  
        - Volume support: +/- X points
        - Trend strength: +/- X points
        - Support/Resistance: +/- X points
        **Total Score:** X/10 = X% confidence

        IMPORTANT: Always calculate confidence percentages based on actual technical indicator data provided. Never use generic percentages like 60%. Be specific about which indicators support your confidence level.

        Keep analysis under 1200 words. Be direct and actionable - traders need decisions, not theory.
        """
    
    def _prepare_user_message(self, symbol: str, data_summary: Dict) -> str:
        """Prepare the user message for AI analysis"""
        return f"""
        Please analyze this stock/crypto: {symbol}
        
        Here are the latest technical indicators and market data:
        {data_summary}
        
        Please provide a clear, simple analysis following the format specified in your instructions.
        """
    
    def format_analysis(self, analysis: str) -> str:
        """
        Format the AI analysis for display
        
        Args:
            analysis: Raw AI analysis text
            
        Returns:
            Clean analysis text without HTML formatting
        """
        try:
            # Clean up any markdown artifacts and return plain text
            cleaned_analysis = analysis.strip()
            
            # Remove any markdown code block indicators that might leak through
            if cleaned_analysis.startswith('```markdown'):
                cleaned_analysis = cleaned_analysis.replace('```markdown', '').strip()
            if cleaned_analysis.endswith('```'):
                cleaned_analysis = cleaned_analysis.replace('```', '').strip()
            
            return cleaned_analysis
            
        except Exception as e:
            logger.error(f"Error formatting analysis: {str(e)}")
            return analysis 