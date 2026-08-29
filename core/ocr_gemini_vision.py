import os
import re
import time
import logging
import mimetypes
from typing import List, Optional

from core.ocr_base import OCREngine, OCRResult, OCRLine
from core.config import get_setting
from core.api_key_manager import APIKeyManager

logger = logging.getLogger(__name__)


class RateLimitError(Exception):
    """Raised when the Gemini API quota/rate limit is exhausted."""
    def __init__(self, message: str, reset_time: str = ""):
        self.reset_time = reset_time
        super().__init__(message)


class GeminiVisionOCREngine(OCREngine):
    """Gemini Vision OCR Engine implementation using the google-genai SDK."""

    @property
    def name(self) -> str:
        return 'gemini_vision'

    @property
    def display_name(self) -> str:
        return 'Gemini Vision'

    def is_available(self) -> bool:
        if not get_setting("allow_cloud_ocr", False):
            return False
            
        manager = APIKeyManager()
        if not manager.get_all_keys():
            return False
            
        try:
            import google.genai
            return True
        except ImportError:
            return False

    def _get_api_key(self) -> str:
        """Get the next available API key from the rotation pool."""
        manager = APIKeyManager()
        key = manager.get_next_key()
        if key:
            return key
        return os.environ.get("GEMINI_API_KEY", "")

    def extract(self, image_path: str, languages: Optional[List[str]] = None,
                skip_tables: bool = False) -> OCRResult:
        try:
            import google.genai
            from google.genai import types
        except ImportError:
            logger.error(f"[{self.name}] google-genai package is not installed.")
            raise RuntimeError("google-genai is required for GeminiVisionOCREngine.")

        api_key = self._get_api_key()
        if not api_key:
            raise ValueError("No Gemini API keys available (all exhausted or none configured).")

        # Using a sensible default if not set
        model_name = get_setting("gemini_model", "gemini-3.6-flash")
        client = google.genai.Client(api_key=api_key)
        current_key = api_key

        prompt_text = (
            "You are a HIGH-PRECISION OCR transcription engine.\n"
            "This is a financial document and accuracy is critical.\n"
            "DO NOT: summarize, translate, paraphrase, rewrite, correct spelling, infer missing chars, invent text, normalize financial values, change numbers/dates/currency amounts.\n"
            "PRESERVE: Arabic text, English text, Arabic and Western numerals, decimal/thousands separators, currency symbols, punctuation, dates, account/invoice numbers, company names, headings, labels, line structure, table rows/columns.\n"
            "Arabic must remain valid Unicode (NO manual reversal).\n"
            "If text is unclear, mark with [?] rather than guessing.\n"
            "For mixed Arabic/English, preserve both languages exactly.\n"
            "Numbers are especially sensitive — NEVER change a number.\n"
            "\n"
            "CRITICAL — ARABIC TEXT ORDER:\n"
            "Arabic is read RIGHT-TO-LEFT. You MUST output Arabic text in correct logical reading order.\n"
            "The first word in each output line must be the first word a native Arabic reader would read — which is the RIGHTMOST word on the page.\n"
            "Example: if the page visually shows (left to right): مصرية مساهمة شركة\n"
            "You must output: شركة مساهمة مصرية\n"
            "Because 'شركة' is on the right side and is read first in Arabic.\n"
            "Do NOT transcribe Arabic words in left-to-right visual order.\n"
        )

        if skip_tables:
            prompt_text += (
                "TABLES: When you encounter a table in the document, write '[--- Table skipped ---]' on its own line "
                "and DO NOT transcribe any text inside that table. However, you MUST continue transcribing ALL text "
                "that appears AFTER the table. Do NOT stop at the table. Process the ENTIRE page from top to bottom, "
                "only replacing table content with the marker.\n"
            )
            
        prompt_text += "\nOutput as plain text preserving the document's line structure."

        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
        except Exception as e:
            logger.error(f"[{self.name}] Failed to read image file {image_path}: {e}")
            raise

        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type:
            mime_type = "image/jpeg"

        manager = APIKeyManager()
        max_retries = 2
        retry_delay = 2.0
        network_retries = 1
        
        response_text = ""
        attempt = 0
        
        while attempt <= max_retries:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[
                        types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                        prompt_text
                    ],
                    config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=16384)
                )
                
                if response and response.text:
                    response_text = response.text
                    logger.info(f"[{self.name}] Successfully extracted text. Length: {len(response_text)}. Key: ...{current_key[-6:]}")
                    break
                else:
                    logger.warning(f"[{self.name}] Empty response returned.")
                    break
                    
            except Exception as e:
                status_code = getattr(e, 'code', None)
                if not status_code and hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                    status_code = e.response.status_code
                    
                error_msg = str(e).lower()
                
                # Check for rate limits (429) / quota exhausted
                if status_code == 429 or "429" in error_msg or "quota" in error_msg or "rate limit" in error_msg or "resource_exhausted" in error_msg:
                    # Parse reset time
                    reset_time = ""
                    reset_match = re.search(r'[Rr]esets?\s+in\s+([\dhmins\s]+)', str(e))
                    if reset_match:
                        reset_time = reset_match.group(1).strip()
                    
                    # Mark this key as exhausted
                    manager.mark_exhausted(current_key, reset_time)
                    
                    # Try next key
                    next_key = manager.get_next_key()
                    if next_key:
                        logger.info(f"[{self.name}] Rotating to next API key ...{next_key[-6:]}")
                        current_key = next_key
                        client = google.genai.Client(api_key=current_key)
                        retry_delay = 1.0  # Reset delay for fresh key
                        attempt = 0  # Reset attempts for new key
                        continue
                    else:
                        # All keys exhausted
                        msg = "All API keys have reached their limits."
                        if reset_time:
                            msg += f" Next reset in {reset_time}."
                        logger.error(f"[{self.name}] {msg}")
                        raise RateLimitError(msg, reset_time=reset_time)
                        
                # Check for server errors (5xx)
                elif (isinstance(status_code, int) and status_code >= 500) or any(x in error_msg for x in ["500", "502", "503", "504", "internal server error", "service unavailable"]):
                    if attempt < max_retries:
                        logger.warning(f"[{self.name}] Server error ({status_code}). Retrying in {retry_delay}s... (Attempt {attempt+1})")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        attempt += 1
                        continue
                    else:
                        logger.error(f"[{self.name}] Max retries reached for server errors.")
                        raise
                        
                # Check for Auth errors (401, 403)
                elif status_code in (401, 403) or "401" in error_msg or "403" in error_msg or "unauthorized" in error_msg or "forbidden" in error_msg or "api key not valid" in error_msg:
                    logger.error(f"[{self.name}] Authentication error ({status_code}). Not retrying.")
                    raise
                    
                # Other network/unknown errors
                else:
                    if network_retries > 0:
                        network_retries -= 1
                        logger.warning(f"[{self.name}] Network or unknown error: {e}. Retrying once...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        logger.error(f"[{self.name}] Unrecoverable error after retries: {e}")
                        raise
                        
        raw_text = response_text.strip()
        lines = []
        layout_data = []
        
        for line_text in raw_text.splitlines():
            line_text = line_text.strip()
            if not line_text:
                continue
                
            ocr_line = OCRLine(
                text=line_text,
                confidence=0.95,
                bounding_box=[]
            )
            lines.append(ocr_line)
            
            layout_data.append({
                "text": line_text,
                "confidence": 0.95,
                "bounding_box": []
            })
            
        return OCRResult(
            raw_text=raw_text,
            lines=lines,
            confidence=0.95 if lines else 0.0,
            engine=self.name,
            language="mixed",
            layout_data=layout_data
        )
