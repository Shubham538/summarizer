
"""
Functions:
- count_text_tokens(text, model, provider)
- count_list_tokens(texts, model, provider)
- estimate_crew_run_tokens(inputs_chunks, task_prompts, agent_system_prompts, model, provider, extra_messages)
"""

from __future__ import annotations
import argparse
import os
from typing import Iterable, List, Optional, Tuple

# --- OpenAI path (tiktoken) ---
import tiktoken

# --- Gemini path (lazy import) ---
def _lazy_genai_client():
    try:
        from google import genai  # pip install google-genai
    except Exception as e:
        raise RuntimeError(
            "Gemini counting requires the 'google-genai' SDK. Install with:\n"
            "  pip install google-genai\n"
            "and set GOOGLE_API_KEY (or GEMINI_API_KEY)."
        ) from e
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Gemini counting requires an API key. Set env var GOOGLE_API_KEY or GEMINI_API_KEY."
        )
    return genai.Client(api_key=api_key)


def _get_encoder(model: str = "gpt-4o-mini"):
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def _count_text_tokens_openai(text: Optional[str], model: str) -> int:
    if not text:
        return 0
    enc = _get_encoder(model)
    return len(enc.encode(text, disallowed_special=()))


def _count_text_tokens_gemini(text: Optional[str], model: str) -> int:
    if not text:
        return 0
    client = _lazy_genai_client()
    # The google-genai SDK returns an object with .total_tokens
    # We pass raw string or list of parts; string is accepted.
    resp = client.models.count_tokens(model=model, contents=text)
    # Try common shapes
    for key in ("total_tokens", "totalTokenCount"):
        if hasattr(resp, key):
            return int(getattr(resp, key))
        if isinstance(resp, dict) and key in resp:
            return int(resp[key])
    # Fallback: look for usage-like fields
    for key in ("total_tokens", "totalTokenCount"):
        if key in getattr(resp, "result", {}):
            return int(resp.result[key])
    # If structure unknown, best effort
    try:
        return int(resp.total_tokens)  # may raise
    except Exception:
        raise RuntimeError("Unexpected response from Gemini count_tokens; inspect object: %r" % (resp,))


def count_text_tokens(
    text: Optional[str],
    model: str = "gpt-4o-mini",
    provider: str = "openai",
) -> int:
    """Count tokens for a single string for the target provider/model."""
    provider = (provider or "openai").lower()
    if provider == "openai":
        return _count_text_tokens_openai(text, model)
    elif provider == "gemini":
        return _count_text_tokens_gemini(text, model)
    else:
        raise ValueError(f"Unknown provider '{provider}'. Use 'openai' or 'gemini'.")


def count_list_tokens(
    texts: Iterable[Optional[str]],
    model: str = "gpt-4o-mini",
    provider: str = "openai",
) -> Tuple[int, List[int]]:
    """Count total tokens for an iterable of strings; returns (total, per_item_counts)."""
    per_item = []
    total = 0
    for s in texts:
        n = count_text_tokens(s, model=model, provider=provider)
        per_item.append(n)
        total += n
    return total, per_item


def estimate_crew_run_tokens(
    inputs_chunks: Iterable[str] = (),
    task_prompts: Iterable[str] = (),
    agent_system_prompts: Iterable[str] = (),
    model: str = "gpt-4o-mini",
    provider: str = "openai",
    extra_messages: Iterable[str] = (),
) -> dict:
    """
    Provider-aware estimate for a Crew run.
    - OpenAI: static estimate using tiktoken (no API calls).
    - Gemini: uses google-genai count_tokens (API calls) for exact counts.
    """
    inputs_total, inputs_each = count_list_tokens(inputs_chunks, model=model, provider=provider)
    tasks_total, tasks_each = count_list_tokens(task_prompts, model=model, provider=provider)
    agents_total, agents_each = count_list_tokens(agent_system_prompts, model=model, provider=provider)
    extra_total, extra_each = count_list_tokens(extra_messages, model=model, provider=provider)

    breakdown = {
        "provider": provider,
        "model": model,
        "inputs_total": inputs_total,
        "inputs_each": inputs_each,
        "tasks_total": tasks_total,
        "tasks_each": tasks_each,
        "agents_total": agents_total,
        "agents_each": agents_each,
        "extra_total": extra_total,
        "extra_each": extra_each,
    }
    breakdown["grand_total"] = inputs_total + tasks_total + agents_total + extra_total
    return breakdown


def _main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Token counter for OpenAI & Gemini.")
    p.add_argument("--provider", choices=["openai", "gemini"], default="openai", help="LLM provider")
    p.add_argument("--model", default="gpt-4o-mini", help="Model name")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--text", help="Raw text to count")
    g.add_argument("--file", help="Path to a UTF-8 text file to count")
    args = p.parse_args(argv)

    if args.text:
        total = count_text_tokens(args.text, model=args.model, provider=args.provider)
        print(total)
        return 0

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            data = f.read()
        total = count_text_tokens(data, model=args.model, provider=args.provider)
        print(total)
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
