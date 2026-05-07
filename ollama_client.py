"""
Anthropic API client for LLM agent communication
"""
import os
import logging
from typing import List, Optional
import anthropic

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 2048


class OllamaClient:
    """Anthropic API client (interface-compatible with the original OllamaClient)"""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        base_url: str = "",       # kept for interface compatibility, unused
        **kwargs                  # absorbs repeat_penalty, repeat_last_n, min_p
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = "https://api.anthropic.com"
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    def generate(
        self,
        prompt: str,
        temperature: float = None,
        max_tokens: int = None
    ) -> str:
        """Generate text using Anthropic Messages API"""
        if temperature is None:
            temperature = self.temperature
        if max_tokens is None:
            max_tokens = self.max_tokens

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()
        except anthropic.AuthenticationError:
            logger.error("Invalid ANTHROPIC_API_KEY. Check your environment variable.")
            return ""
        except anthropic.RateLimitError as e:
            logger.error(f"Rate limit exceeded: {e}")
            return ""
        except anthropic.APIStatusError as e:
            logger.error(f"Anthropic API error ({e.status_code}): {e.message}")
            return ""
        except Exception as e:
            logger.error(f"Unexpected error calling Anthropic API: {e}")
            return ""

    def check_connection(self) -> bool:
        """Check Anthropic API connectivity and key validity"""
        if not os.environ.get("ANTHROPIC_API_KEY"):
            logger.error("ANTHROPIC_API_KEY environment variable is not set.")
            return False
        try:
            self.client.messages.create(
                model=self.model,
                max_tokens=1,
                messages=[{"role": "user", "content": "hi"}]
            )
            return True
        except anthropic.AuthenticationError:
            logger.error("Invalid ANTHROPIC_API_KEY.")
            return False
        except Exception as e:
            logger.error(f"Cannot connect to Anthropic API: {e}")
            return False

    def list_models(self) -> List[str]:
        """Return known Anthropic model IDs"""
        return [
            "claude-haiku-4-5-20251001",
            "claude-haiku-4-5",
            "claude-sonnet-4-6",
            "claude-opus-4-7",
        ]

    def check_model_exists(self) -> bool:
        """Check if the configured model is in the known list"""
        return self.model in self.list_models()
