"""OCR Base Types and Abstractions.

Defines the common data structures and abstract base class that all OCR engines
must implement. This ensures the rest of the application is engine-agnostic.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class OCRLine:
    """A single line of OCR-detected text with its metadata."""
    text: str
    confidence: float  # 0.0 to 1.0
    bounding_box: List  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] quad points
    language: str = ""  # "ar", "en", or "" if unknown


@dataclass
class OCRResult:
    """Normalized OCR output from any engine.
    
    The `layout_data` field preserves backward compatibility with the existing
    app code that expects [{"bounding_box": ..., "text": ..., "confidence": ...}].
    """
    raw_text: str
    lines: List[OCRLine]
    confidence: float       # Average confidence across all lines [0..1]
    engine: str             # "paddle", "gemini_vision", etc.
    language: str           # "ar", "en", "mixed"
    metadata: Dict[str, Any] = field(default_factory=dict)
    layout_data: List[Dict] = field(default_factory=list)  # Backward-compatible format

    def to_legacy_dict(self) -> Dict[str, Any]:
        """Convert to the legacy dict format expected by app.py and ocr_engine.py.
        
        Returns {"raw_text": str, "layout_data": list}
        """
        return {
            "raw_text": self.raw_text,
            "layout_data": self.layout_data
        }


@dataclass
class QualityReport:
    """Result of OCR quality analysis."""
    score: float              # 0.0 = garbage, 1.0 = excellent
    needs_fallback: bool      # True if quality is too low
    reasons: List[str] = field(default_factory=list)  # Human-readable explanations
    numeric_tokens: List[str] = field(default_factory=list)  # Extracted numbers for validation


class OCREngine(ABC):
    """Abstract base class for all OCR engines.
    
    Each engine implementation must:
    1. Return normalized OCRResult objects
    2. Handle its own errors gracefully
    3. Report availability via is_available()
    4. Support the skip_tables flag
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this engine (e.g., 'paddle', 'gemini_vision')."""
        ...

    @property
    def display_name(self) -> str:
        """Human-readable name for UI display."""
        return self.name

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this engine can be used (dependencies installed, API keys present, etc.)."""
        ...

    @abstractmethod
    def extract(self, image_path: str, languages: Optional[List[str]] = None,
                skip_tables: bool = False) -> OCRResult:
        """Extract text from an image.
        
        Args:
            image_path: Absolute path to the image file.
            languages: List of language codes (e.g., ["ar", "en"]). None = auto-detect.
            skip_tables: If True, exclude table regions from OCR output.
            
        Returns:
            Normalized OCRResult.
            
        Raises:
            Exception: If extraction fails unrecoverably.
        """
        ...

    def _make_empty_result(self, engine_name: str = "") -> OCRResult:
        """Create an empty OCRResult for error cases."""
        return OCRResult(
            raw_text="",
            lines=[],
            confidence=0.0,
            engine=engine_name or self.name,
            language="",
            metadata={"error": True},
            layout_data=[]
        )
