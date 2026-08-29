import json
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional


def get_app_dir() -> Path:
    """Get the application base directory. Works in dev and PyInstaller frozen mode."""
    if getattr(sys, 'frozen', False):
        # Running as PyInstaller exe — settings go next to the exe
        return Path(sys.executable).parent
    return Path(__file__).parent.parent


def get_bundle_dir() -> Path:
    """Get the bundle data directory (where PyInstaller extracts data files)."""
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS)
    return Path(__file__).parent.parent


SETTINGS_FILE = get_app_dir() / "settings.json"

def load_settings() -> Dict[str, Any]:
    """
    Loads the settings dictionary from the local settings.json file.
    
    Returns:
        Dict[str, Any]: The loaded settings, or an empty dictionary if the file doesn't exist.
    """
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_settings(settings_dict: Dict[str, Any]) -> None:
    """
    Saves a dictionary of settings to the local settings.json file.

    Args:
        settings_dict (Dict[str, Any]): The dictionary of settings to save.
    """
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(settings_dict, f, indent=4)

def get_setting(key: str, default: Optional[Any] = None) -> Any:
    """
    Retrieves a specific setting by key.

    Args:
        key (str): The setting key to retrieve.
        default (Optional[Any]): The default value to return if the key is not found.

    Returns:
        Any: The value of the setting, or the default value.
    """
    settings = load_settings()
    return settings.get(key, default)

def set_setting(key: str, value: Any) -> None:
    """
    Updates or creates a specific setting and saves it to disk.

    Args:
        key (str): The setting key to set.
        value (Any): The value to store.
    """
    settings = load_settings()
    settings[key] = value
    save_settings(settings)
