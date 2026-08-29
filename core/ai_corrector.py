"""AI-powered text correction module using Gemini API (Free Tier).

Sends raw OCR text to Google's Gemini model for intelligent correction
of character-level errors, especially for Arabic text where PaddleOCR
confuses similar-looking characters (ب/ت/ث/ن, ق/ف, etc.).

Free tier: 15 requests/minute, 1,500 requests/day — more than enough
for typical document processing.

Uses the new `google-genai` SDK (replaces deprecated `google-generativeai`).
"""

import os
import logging
from typing import Optional
try:
    from core.config import get_setting
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from core.config import get_setting

logger = logging.getLogger(__name__)

# Maximum characters per API call
MAX_CHUNK_SIZE = 8000


def _get_api_key() -> Optional[str]:
    """Get next available Gemini API key from the rotation pool."""
    try:
        from core.api_key_manager import APIKeyManager
        manager = APIKeyManager()
        key = manager.get_next_key()
        if key:
            return key
    except ImportError:
        pass
    
    # Fallback to single key
    key = get_setting("gemini_api_key", "")
    if key:
        return key
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    return os.environ.get("GEMINI_API_KEY", "")


def is_ai_correction_available() -> bool:
    """Check if AI correction is configured and available."""
    if not get_setting("ai_correction", False):
        return False
    key = _get_api_key()
    return bool(key and key.strip())


def _build_correction_prompt(raw_text: str, lang: str = "ar") -> str:
    """Build the correction prompt for the AI model."""
    if lang == "ar":
        return (
            "أنت مدقق نصوص متخصص في تصحيح أخطاء OCR للمستندات العربية المالية والقانونية.\n\n"
            "القواعد:\n"
            "1. صحّح أخطاء الحروف المتشابهة فقط (مثل: ب↔ت↔ث↔ن، ق↔ف، ح↔خ↔ج، د↔ذ، ر↔ز، س↔ش، ص↔ض، ط↔ظ، ع↔غ)\n"
            "2. صحّح الكلمات المكسورة أو المدمجة بالخطأ\n"
            "3. لا تُغيّر المعنى أو تُعيد صياغة الجمل\n"
            "4. لا تحذف أو تضيف محتوى جديد\n"
            "5. حافظ على نفس تنسيق الأسطر والفقرات\n"
            "6. الأرقام والتواريخ لا تتغير\n"
            "7. أعد النص المصحح فقط بدون أي تعليقات أو شرح\n\n"
            "النص المطلوب تصحيحه:\n\n"
            f"{raw_text}"
        )
    else:
        return (
            "You are an OCR text correction specialist.\n\n"
            "Rules:\n"
            "1. Fix only character-level OCR errors (misrecognized letters)\n"
            "2. Fix broken or merged words\n"
            "3. Do NOT change meaning or rephrase\n"
            "4. Do NOT add or remove content\n"
            "5. Keep the same line/paragraph formatting\n"
            "6. Numbers and dates stay unchanged\n"
            "7. Return ONLY the corrected text, no comments\n\n"
            "Text to correct:\n\n"
            f"{raw_text}"
        )


def _split_text_into_chunks(text: str, max_size: int = MAX_CHUNK_SIZE) -> list:
    """Split text into chunks at paragraph boundaries."""
    if len(text) <= max_size:
        return [text]
    
    chunks = []
    paragraphs = text.split('\n\n')
    current_chunk = ""
    
    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 > max_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = para
        else:
            current_chunk += ("\n\n" if current_chunk else "") + para
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks if chunks else [text]


def correct_text_with_ai(raw_text: str, progress_callback=None) -> str:
    """Correct OCR text using Gemini API.
    
    Args:
        raw_text: The raw OCR-extracted text to correct.
        progress_callback: Optional callback(message: str) for status updates.
        
    Returns:
        Corrected text, or original text if AI correction fails/unavailable.
    """
    if not raw_text or not raw_text.strip():
        return raw_text
    
    api_key = _get_api_key()
    if not api_key:
        logger.warning("AI correction skipped: no API key configured")
        return raw_text
    
    try:
        from google import genai
    except ImportError:
        logger.error("google-genai package not installed. Run: pip install google-genai")
        return raw_text
    
    lang = get_setting("ocr_lang", "ar")
    
    try:
        client = genai.Client(api_key=api_key)
        
        chunks = _split_text_into_chunks(raw_text)
        corrected_chunks = []
        
        for i, chunk in enumerate(chunks):
            if progress_callback:
                progress_callback(f"AI correcting chunk {i+1}/{len(chunks)}...")
            
            prompt = _build_correction_prompt(chunk, lang)
            
            response = client.models.generate_content(
                model=get_setting("gemini_model", "gemini-3.6-flash"),
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=0.1,  # Low = more faithful correction
                    max_output_tokens=8192,
                )
            )
            
            if response and response.text:
                corrected_chunks.append(response.text.strip())
            else:
                corrected_chunks.append(chunk)
        
        corrected_text = "\n\n".join(corrected_chunks)
        
        # Safety: if AI output length differs too much, keep original
        original_len = len(raw_text)
        corrected_len = len(corrected_text)
        if corrected_len < original_len * 0.5 or corrected_len > original_len * 1.5:
            logger.warning("AI correction output length differs too much, using original")
            return raw_text
        
        return corrected_text
        
    except Exception as e:
        logger.error(f"AI correction failed: {e}")
        if progress_callback:
            progress_callback(f"AI correction failed: {str(e)[:50]}... Using original text.")
        return raw_text
