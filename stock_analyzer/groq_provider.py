"""
Groq AI Provider for StockLyzer
"""

from groq import Groq
from typing import Optional
import logging
import os
from .base_provider import AIProvider

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GroqProvider(AIProvider):
    """Groq AI provider implementation"""
    
    def __init__(self):
        self.client = None
        self._model_name = None
    
    def initialize(self, api_key: str, model_name: str) -> None:
        """Initialize the Groq client"""
        try:
            os.environ['GROQ_API_KEY'] = api_key
            self.client = Groq()
            self._model_name = model_name
            logger.info(f"Groq client initialized successfully with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {str(e)}")
            raise
    
    def generate_response(self, system_prompt: str, user_message: str) -> Optional[str]:
        """Generate response using Groq"""
        if not self.is_available():
            return None
            
        try:
            response = self.client.chat.completions.create(
                model=self._model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_tokens=5000,
                top_p=1,
                stream=False,
                stop=None
            )
            
            # Check for truncation
            choice = response.choices[0]
            if hasattr(choice, 'finish_reason') and choice.finish_reason == 'length':
                logger.warning("Groq response was truncated due to token limit")
                return choice.message.content + "\n\n*[Response was truncated due to length limits]*"
            
            return choice.message.content
            
        except Exception as e:
            logger.warning(f"Groq provider failed: {str(e)}")
            return None
    
    def is_available(self) -> bool:
        """Check if Groq is available"""
        return self.client is not None and self._model_name is not None
    
    @property
    def provider_name(self) -> str:
        return "Groq"
    
    @property
    def model_name(self) -> str:
        return self._model_name or "Unknown" 