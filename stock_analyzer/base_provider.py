"""
Base AI Provider abstract class for StockLyzer
"""

from abc import ABC, abstractmethod
from typing import Optional

class AIProvider(ABC):
    """Abstract base class for AI providers"""
    
    @abstractmethod
    def initialize(self, api_key: str, model_name: str) -> None:
        """Initialize the AI provider"""
        pass
    
    @abstractmethod
    def generate_response(self, system_prompt: str, user_message: str) -> Optional[str]:
        """Generate a response using the AI provider"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available"""
        pass
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Get the provider name"""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model name"""
        pass 