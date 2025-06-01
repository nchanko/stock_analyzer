"""
Google Gemini AI Provider for StockLyzer
"""

import google.generativeai as genai
from typing import Optional
import logging
from .base_provider import AIProvider

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiProvider(AIProvider):
    """Google Gemini AI provider implementation"""
    
    def __init__(self):
        self.model = None
        self._model_name = None
    
    def initialize(self, api_key: str, model_name: str) -> None:
        """Initialize the Gemini client"""
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)
            self._model_name = model_name
            logger.info(f"Gemini client initialized successfully with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {str(e)}")
            raise
    
    def generate_response(self, system_prompt: str, user_message: str) -> Optional[str]:
        """Generate response using Gemini"""
        if not self.is_available():
            return None
            
        try:
            # Combine system prompt and user message for Gemini
            full_prompt = f"{system_prompt}\n\n{user_message}"
            
            response = self.model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=5000,
                    top_p=1,
                ),
                safety_settings=[
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH", 
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_NONE"
                    }
                ]
            )
            
            # Check if response was blocked or empty
            if not response.candidates or not response.candidates[0].content:
                logger.warning(f"Gemini response blocked or empty. Finish reason: {response.candidates[0].finish_reason if response.candidates else 'No candidates'}")
                return None
            
            # Check if response has parts
            if not response.candidates[0].content.parts:
                logger.warning("Gemini response has no content parts")
                return None
            
            # Log response details for debugging
            candidate = response.candidates[0]
            logger.info(f"Gemini response - Finish reason: {candidate.finish_reason}, Parts count: {len(candidate.content.parts)}")
            
            # Check for truncation (finish_reason should be STOP for complete responses)
            if hasattr(candidate, 'finish_reason') and candidate.finish_reason != 1:  # 1 = STOP
                logger.warning(f"Gemini response may be truncated. Finish reason: {candidate.finish_reason}")
                if candidate.finish_reason == 3:  # MAX_TOKENS
                    return response.text + "\n\n*[Response was truncated due to length limits]*"
            
            return response.text
            
        except Exception as e:
            logger.warning(f"Gemini provider failed: {str(e)}")
            return None
    
    def is_available(self) -> bool:
        """Check if Gemini is available"""
        return self.model is not None and self._model_name is not None
    
    @property
    def provider_name(self) -> str:
        return "Gemini"
    
    @property
    def model_name(self) -> str:
        return self._model_name or "Unknown" 