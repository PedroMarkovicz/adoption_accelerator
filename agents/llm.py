"""
Generic LLM call wrapper for the agent system.

Provides a reusable function that handles client instantiation,
parameter configuration (reasoning, verbosity, max_tokens), error
handling, retry logic, and token usage logging.

All LLM nodes call this wrapper instead of the OpenAI client directly.
Uses the OpenAI Responses API (recommended for GPT-5 series).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from agents.config import get_openai_client, load_agent_config

logger = logging.getLogger(__name__)

# Maximum retry attempts for transient errors
_MAX_RETRIES = 3
_BASE_BACKOFF_SECONDS = 1.0


def call_llm(
    system_prompt: str,
    user_prompt: str,
    node_name: str,
    *,
    max_output_tokens: int | None = None,
    verbosity: str | None = None,
    json_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Call the OpenAI API using the Responses API.

    This is the single entry point for all LLM interactions in the
    agent system.  It reads configuration from the agent config,
    applies the appropriate reasoning effort, verbosity, and token
    limits, and handles retries for transient errors.

    Parameters
    ----------
    system_prompt : str
        System-level instructions for the model.
    user_prompt : str
        User-level prompt with the specific request data.
    node_name : str
        Name of the calling node (for logging and config lookup).
    max_output_tokens : int | None
        Override for max output tokens.  If ``None``, uses the
        per-node value from ``configs/agents/llm.yaml``.
    verbosity : str | None
        Override for verbosity (``"low"``, ``"medium"``).
        Defaults to the global setting from config.
    json_schema : dict | None
        If provided, enables structured output mode.  The dict
        should be a valid JSON Schema that the response must conform to.

    Returns
    -------
    dict
        Keys:
        - ``text``: The model's text output.
        - ``usage``: Token usage dict (``input_tokens``,
          ``output_tokens``, ``total_tokens``).
        - ``model``: Model ID used.
        - ``latency_ms``: Request latency in milliseconds.
    """
    cfg = load_agent_config()

    # Resolve per-node max tokens
    if max_output_tokens is None:
        token_map = {
            "explainer": cfg.llm.max_tokens.explainer,
            "recommender": cfg.llm.max_tokens.recommender,
            "description_writer": cfg.llm.max_tokens.description_writer,
        }
        max_output_tokens = token_map.get(node_name, 500)

    # Resolve verbosity
    if verbosity is None:
        verbosity = cfg.llm.verbosity

    # Build the input messages
    input_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Build API parameters for the Responses API.
    # Note: temperature is not supported on reasoning models (GPT-5 series).
    # When reasoning effort is set, the model manages its own sampling.
    api_params: dict[str, Any] = {
        "model": cfg.llm.model,
        "input": input_messages,
        "max_output_tokens": max_output_tokens,
        "reasoning": {"effort": cfg.llm.reasoning.effort},
        "text": {"format": {"type": "text"}},
    }

    # Apply structured output if requested
    if json_schema is not None:
        api_params["text"] = {
            "format": {
                "type": "json_schema",
                "name": f"{node_name}_output",
                "schema": json_schema,
                "strict": True,
            }
        }

    return _call_with_retry(api_params, node_name)


def _call_with_retry(
    api_params: dict[str, Any],
    node_name: str,
) -> dict[str, Any]:
    """Execute the API call with exponential backoff retry for transient errors."""
    import openai

    client = get_openai_client()
    last_error: Exception | None = None

    for attempt in range(_MAX_RETRIES):
        try:
            t0 = time.perf_counter()
            response = client.responses.create(**api_params)
            latency_ms = (time.perf_counter() - t0) * 1000

            # Extract text from the response
            text = _extract_response_text(response)

            # Extract usage info
            usage = {}
            if hasattr(response, "usage") and response.usage is not None:
                usage = {
                    "input_tokens": getattr(response.usage, "input_tokens", 0),
                    "output_tokens": getattr(response.usage, "output_tokens", 0),
                    "total_tokens": getattr(response.usage, "total_tokens", 0),
                }

            logger.info(
                "LLM call [%s]: model=%s, latency=%.0fms, "
                "tokens=%d/%d (in/out)",
                node_name,
                api_params["model"],
                latency_ms,
                usage.get("input_tokens", 0),
                usage.get("output_tokens", 0),
            )

            return {
                "text": text,
                "usage": usage,
                "model": api_params["model"],
                "latency_ms": round(latency_ms, 2),
            }

        except openai.RateLimitError as exc:
            last_error = exc
            backoff = _BASE_BACKOFF_SECONDS * (2 ** attempt)
            logger.warning(
                "LLM rate limit [%s] (attempt %d/%d), retrying in %.1fs: %s",
                node_name, attempt + 1, _MAX_RETRIES, backoff, exc,
            )
            time.sleep(backoff)

        except openai.APITimeoutError as exc:
            last_error = exc
            logger.warning(
                "LLM timeout [%s] (attempt %d/%d): %s",
                node_name, attempt + 1, _MAX_RETRIES, exc,
            )
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_BASE_BACKOFF_SECONDS)

        except openai.APIConnectionError as exc:
            last_error = exc
            logger.warning(
                "LLM connection error [%s] (attempt %d/%d): %s",
                node_name, attempt + 1, _MAX_RETRIES, exc,
            )
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_BASE_BACKOFF_SECONDS)

        except openai.AuthenticationError as exc:
            # Non-retryable
            logger.error("LLM authentication failed [%s]: %s", node_name, exc)
            raise

        except openai.BadRequestError as exc:
            # Non-retryable
            logger.error("LLM bad request [%s]: %s", node_name, exc)
            raise

    # All retries exhausted
    raise RuntimeError(
        f"LLM call failed for [{node_name}] after {_MAX_RETRIES} attempts: {last_error}"
    )


def _extract_response_text(response: Any) -> str:
    """Extract the text content from an OpenAI Responses API response."""
    # The Responses API returns output as a list of content blocks
    output = getattr(response, "output", None)
    if output is not None:
        for item in output:
            content = getattr(item, "content", None)
            if content is not None:
                for content_block in content:
                    if hasattr(content_block, "text"):
                        return content_block.text
            # Fallback: some response structures have text directly
            if hasattr(item, "text"):
                return item.text

    # Check for output_text convenience attribute
    output_text = getattr(response, "output_text", None)
    if output_text is not None:
        return output_text

    # Fallback for unexpected response structures
    logger.warning("Could not extract text from response, returning str()")
    return str(response)
