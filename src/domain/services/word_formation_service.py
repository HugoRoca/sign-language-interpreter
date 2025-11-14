"""Word formation service interface."""
from abc import ABC, abstractmethod
from typing import List, Deque
from collections import deque


class WordFormationService(ABC):
    """Abstract interface for forming words from detected letters."""
    
    @abstractmethod
    def add_letter(self, letter: str, confidence: float) -> None:
        """Add a detected letter to the current word.
        
        Args:
            letter: Detected letter.
            confidence: Confidence of the detection.
        """
        pass
    
    @abstractmethod
    def get_current_word(self) -> str:
        """Get the current word being formed.
        
        Returns:
            Current word as string.
        """
        pass
    
    @abstractmethod
    def clear_word(self) -> None:
        """Clear the current word."""
        pass
    
    @abstractmethod
    def get_word_history(self) -> List[str]:
        """Get history of formed words.
        
        Returns:
            List of completed words.
        """
        pass

