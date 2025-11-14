"""Word formation service implementation."""
from typing import List, Deque, Optional
from collections import deque
import time

from src.domain.services.word_formation_service import WordFormationService
from config.settings import ASL_CLASSES


class WordFormationServiceImpl(WordFormationService):
    """Concrete implementation of WordFormationService."""
    
    def __init__(
        self,
        min_confidence: float = 0.7,
        stability_threshold: int = 10,
        space_delay: float = 2.0
    ):
        """Initialize the word formation service.
        
        Args:
            min_confidence: Minimum confidence to accept a letter.
            stability_threshold: Number of consecutive detections needed to add a letter.
            space_delay: Seconds of 'space' detection to add a space.
        """
        self.min_confidence = min_confidence
        self.stability_threshold = stability_threshold
        self.space_delay = space_delay
        
        self.current_word: List[str] = []
        self.word_history: List[str] = []
        self.letter_buffer: Deque[tuple] = deque(maxlen=stability_threshold)
        self.last_space_time: float = 0.0
        self.last_letter: Optional[str] = None
        self.letter_count: int = 0
    
    def add_letter(self, letter: str, confidence: float) -> None:
        """Add a detected letter to the current word.
        
        Args:
            letter: Detected letter.
            confidence: Confidence of the detection.
        """
        current_time = time.time()
        
        # Handle 'nothing' class - ignore it
        if letter == 'nothing':
            return
        
        # Handle 'del' class - delete last letter
        if letter == 'del':
            if self.current_word:
                self.current_word.pop()
            self.letter_buffer.clear()
            self.last_letter = None
            self.letter_count = 0
            return
        
        # Handle 'space' class
        if letter == 'space':
            if current_time - self.last_space_time > self.space_delay:
                if self.current_word:
                    # Complete current word
                    word = ''.join(self.current_word)
                    if word:
                        self.word_history.append(word)
                    self.current_word.clear()
                self.last_space_time = current_time
            self.letter_buffer.clear()
            self.last_letter = None
            self.letter_count = 0
            return
        
        # Only process letters (A-Z)
        if letter not in ASL_CLASSES[:26]:  # First 26 are A-Z
            return
        
        # Check confidence threshold
        if confidence < self.min_confidence:
            return
        
        # Add to buffer
        self.letter_buffer.append((letter, confidence))
        
        # Check if we have enough stable detections
        if len(self.letter_buffer) >= self.stability_threshold:
            # Get most common letter in buffer
            letters = [item[0] for item in self.letter_buffer]
            most_common = max(set(letters), key=letters.count)
            
            # Only add if it's different from last letter
            if most_common != self.last_letter:
                self.current_word.append(most_common)
                self.last_letter = most_common
                self.letter_buffer.clear()
    
    def get_current_word(self) -> str:
        """Get the current word being formed.
        
        Returns:
            Current word as string.
        """
        return ''.join(self.current_word)
    
    def clear_word(self) -> None:
        """Clear the current word."""
        self.current_word.clear()
        self.letter_buffer.clear()
        self.last_letter = None
        self.letter_count = 0
    
    def get_word_history(self) -> List[str]:
        """Get history of formed words.
        
        Returns:
            List of completed words.
        """
        return self.word_history.copy()

