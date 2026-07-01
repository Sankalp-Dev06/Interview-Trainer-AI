"""
Configuration Loader
Handles YAML config loading, environment variable injection, and validation.
"""

import os
import re
import yaml
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Loads and validates project configuration from YAML with env var substitution."""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self._config: Optional[Dict[str, Any]] = None

    def load(self) -> Dict[str, Any]:
        """Load configuration file with environment variable substitution."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            raw_content = f.read()

        # Substitute environment variables in the format ${VAR_NAME}
        resolved_content = self._substitute_env_vars(raw_content)

        self._config = yaml.safe_load(resolved_content)
        logger.info(f"Configuration loaded from {self.config_path}")
        return self._config

    def _substitute_env_vars(self, content: str) -> str:
        """Replace ${VAR_NAME} placeholders with environment variable values."""
        pattern = re.compile(r"\$\{([^}]+)\}")

        def replace_match(match):
            var_name = match.group(1)
            value = os.environ.get(var_name)
            if value is None:
                logger.warning(f"Environment variable '{var_name}' not set. Using placeholder.")
                return f"MISSING_{var_name}"
            return value

        return pattern.sub(replace_match, content)

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a nested config value using dot notation.
        Example: config.get('watsonx.llm.model_id')
        """
        if self._config is None:
            self.load()

        keys = key_path.split(".")
        value = self._config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    @property
    def config(self) -> Dict[str, Any]:
        if self._config is None:
            self.load()
        return self._config
