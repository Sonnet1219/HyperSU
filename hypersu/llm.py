"""LLM client for HyperSU."""

import logging
import os

import httpx
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logger = logging.getLogger(__name__)


class LLMClient:
    """OpenAI-compatible LLM client with timeout handling."""

    def __init__(self, model_name: str):
        http_client = httpx.Client(timeout=60.0, trust_env=False)
        self.openai_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            http_client=http_client,
        )
        self.llm_config = {
            "model": model_name,
            "max_tokens": 2000,
            "temperature": 0,
        }

    def infer(self, messages):
        try:
            response = self.openai_client.chat.completions.create(
                **self.llm_config, messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.warning("LLM infer failed: %s", e)
            return ""
