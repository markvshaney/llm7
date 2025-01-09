# config/settings.py
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv


class Settings:
    """
    Settings management class that handles configuration from both config.yaml and environment variables.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self._load_config()
        self._load_environment()

    def _load_config(self):
        """Load configuration from config.yaml"""
        config_path = Path(__file__).parent / "config.yaml"

        try:
            with open(config_path, "r") as file:
                self._config = yaml.safe_load(file)
        except FileNotFoundError:
            self._config = {}
            print(f"Warning: Config file not found at {config_path}")
        except yaml.YAMLError as e:
            self._config = {}
            print(f"Error parsing config file: {e}")

    def _load_environment(self):
        """Load environment variables from .env file"""
        load_dotenv()

        # Override config values with environment variables if they exist
        for key in self._config.keys():
            env_value = os.getenv(key.upper())
            if env_value is not None:
                self._config[key] = env_value

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation (e.g., 'llm.defaults').
        """
        parts = key.split(".")
        current = self._config

        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default

        return current

    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value using dot notation (e.g., 'llm.defaults').
        """
        parts = key.split(".")
        current = self._config

        # Navigate to the correct nested level
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        # Set the value at the final level
        current[parts[-1]] = value

    @property
    def config(self) -> Dict[str, Any]:
        """Get the entire configuration dictionary"""
        return self._config.copy()

    def reload(self) -> None:
        """Reload configuration from files"""
        self._load_config()
        self._load_environment()


# Create a default instance
settings = Settings()
