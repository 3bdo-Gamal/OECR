"""OCR Engine Facade.

Backward-compatible wrapper that delegates to the new engine abstraction.
Existing code that imports extract_text_from_image from this module
continues to work without changes.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def extract_text_from_image(image_path: str) -> Dict[str, Any]:
    """Extract text from an image file.
    
    This is the backward-compatible entry point. Internally delegates
    to the OCR router which handles engine selection, quality analysis,
    and fallback logic.
    
    Args:
        image_path: Path to the image file.
        
    Returns:
        Dict with 'raw_text' (str) and 'layout_data' (list).
    """
    try:
        from core.ocr_router import extract_text
        result = extract_text(image_path)
        return result.to_legacy_dict()
    except Exception as e:
        # Fallback: try PaddleOCR directly
        try:
            from core.ocr_paddle import PaddleOCREngine
            engine = PaddleOCREngine()
            result = engine.extract(image_path)
            return result.to_legacy_dict()
        except Exception as fallback_err:
            logger.error(f"All OCR engines failed: {e}, fallback: {fallback_err}")
            raise Exception(f"OCR Engine Error: {str(e)}")