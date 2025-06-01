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
        You are a friendly stock analyst. Provide clear analysis that anyone can understand.

        GUIDELINES:
        - Use simple language, avoid jargon
        - Keep sections concise but informative
        - Use bullet points for clarity
        - Focus on practical insights
        
        Latest news: {latest_news}
        
        Format your response as follows:

        ## 📈 What This Stock Is
        Brief explanation of the company/crypto (2-3 sentences max).

        ## 🎯 Quick Summary
        - Current trend (up/down/sideways)
        - What might happen next (in simple terms)
        - Key thing to watch

        ## 📊 Technical Analysis
        - What the charts show
        - Important price levels
        - Momentum strength

        ## 💡 Investment Ideas
        **Long-term:** Brief recommendation with reasoning
        **Short-term:** Trading opportunities if any

        ## 📰 News Impact
        How recent news affects the stock (2-3 sentences).

        Keep total response under 1500 words. Be concise but helpful.
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
            Formatted analysis with proper styling
        """
        try:
            # Add styling to the analysis with improved readability
            formatted_analysis = f"""
            <div style='background-color: #f8f9fa; padding: 2rem; border-radius: 0.75rem; margin: 1rem 0; border-left: 4px solid #007acc; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                <div style='font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; line-height: 1.6; color: #333;'>
                    {analysis}
                </div>
            </div>
            """
            return formatted_analysis
        except Exception as e:
            logger.error(f"Error formatting analysis: {str(e)}")
            return analysis 