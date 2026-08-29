"""Gemini API Key Manager with automatic rotation.

Manages multiple API keys and rotates to the next available key when
one hits its rate limit. Thread-safe for concurrent page processing.
"""

import time
import logging
import threading
from typing import Optional, List, Tuple

from core.config import get_setting

logger = logging.getLogger(__name__)


class APIKeyManager:
    """Manages multiple Gemini API keys with automatic rotation.
    
    Keys are stored in settings as `gemini_api_keys` (list).
    Falls back to `gemini_api_key` (single string) for backward compatibility.
    
    When a key hits rate limit, it's marked exhausted with a reset timestamp.
    The manager rotates to the next available key automatically.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._exhausted_keys = {}  # key -> reset_timestamp (epoch)
        self._current_index = 0
        self._initialized = True

    def get_all_keys(self) -> List[str]:
        """Get all configured API keys (non-empty, stripped)."""
        # Try the new multi-key setting first
        keys = get_setting("gemini_api_keys", [])
        if isinstance(keys, list) and keys:
            return [k.strip() for k in keys if k and k.strip()]
        
        # Fallback to single key setting
        single_key = get_setting("gemini_api_key", "")
        if single_key and single_key.strip():
            return [single_key.strip()]
        
        return []

    def get_next_key(self) -> Optional[str]:
        """Get the next available (non-exhausted) API key.
        
        Returns None if all keys are exhausted.
        """
        with self._lock:
            keys = self.get_all_keys()
            if not keys:
                return None
            
            now = time.time()
            
            # Clean up expired exhaustions
            self._exhausted_keys = {
                k: t for k, t in self._exhausted_keys.items() 
                if t > now
            }
            
            # Try each key starting from current index
            for i in range(len(keys)):
                idx = (self._current_index + i) % len(keys)
                key = keys[idx]
                if key not in self._exhausted_keys:
                    self._current_index = (idx + 1) % len(keys)
                    return key
            
            return None

    def mark_exhausted(self, key: str, reset_time_str: str = ""):
        """Mark a key as rate-limited.
        
        Args:
            key: The API key that was rate-limited.
            reset_time_str: Reset time string like "4h30m12s" or "167h38m12s".
        """
        with self._lock:
            # Parse reset time string to seconds
            seconds = self._parse_reset_time(reset_time_str)
            if seconds <= 0:
                seconds = 3600  # Default 1 hour if we can't parse
            
            reset_at = time.time() + seconds
            self._exhausted_keys[key] = reset_at
            
            available = len(self.get_all_keys()) - len(self._exhausted_keys)
            logger.warning(
                f"API key ...{key[-6:]} exhausted. "
                f"Resets in {reset_time_str or f'{seconds}s'}. "
                f"{available} key(s) still available."
            )

    def all_exhausted(self) -> bool:
        """Check if all keys are exhausted."""
        with self._lock:
            keys = self.get_all_keys()
            if not keys:
                return True
            
            now = time.time()
            self._exhausted_keys = {
                k: t for k, t in self._exhausted_keys.items() 
                if t > now
            }
            
            return all(k in self._exhausted_keys for k in keys)

    def get_status(self) -> dict:
        """Get status of all keys for UI display."""
        with self._lock:
            keys = self.get_all_keys()
            now = time.time()
            
            # Clean expired
            self._exhausted_keys = {
                k: t for k, t in self._exhausted_keys.items() 
                if t > now
            }
            
            status = {
                "total": len(keys),
                "available": 0,
                "exhausted": 0,
                "keys": []
            }
            
            for i, key in enumerate(keys):
                masked = f"...{key[-6:]}" if len(key) > 6 else "***"
                if key in self._exhausted_keys:
                    remaining = int(self._exhausted_keys[key] - now)
                    hours, rem = divmod(remaining, 3600)
                    minutes = rem // 60
                    status["keys"].append({
                        "index": i + 1,
                        "masked": masked,
                        "status": "exhausted",
                        "resets_in": f"{hours}h{minutes}m"
                    })
                    status["exhausted"] += 1
                else:
                    status["keys"].append({
                        "index": i + 1,
                        "masked": masked,
                        "status": "available"
                    })
                    status["available"] += 1
            
            return status

    def reset_all(self):
        """Clear all exhaustion records (e.g., when keys are updated)."""
        with self._lock:
            self._exhausted_keys.clear()
            self._current_index = 0

    @staticmethod
    def _parse_reset_time(time_str: str) -> int:
        """Parse reset time string like '4h30m12s' or '167h38m12s' to seconds."""
        if not time_str:
            return 0
        
        import re
        total = 0
        hours = re.search(r'(\d+)\s*h', time_str)
        minutes = re.search(r'(\d+)\s*m', time_str)
        seconds = re.search(r'(\d+)\s*s', time_str)
        
        if hours:
            total += int(hours.group(1)) * 3600
        if minutes:
            total += int(minutes.group(1)) * 60
        if seconds:
            total += int(seconds.group(1))
        
        return total


# Module-level convenience function
def get_api_key() -> Optional[str]:
    """Get the next available API key. Returns None if all exhausted."""
    return APIKeyManager().get_next_key()
