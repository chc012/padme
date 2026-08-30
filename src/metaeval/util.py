"""Small shared helpers."""

from __future__ import annotations

import json


def parse_json(content: str | None) -> dict:
    """Best-effort JSON out of an LLM response.

    Same escalation as `metaeval.judge` and `metaeval.steering.generate`, which each carry
    their own copy: raw parse, then a fenced block, then the outermost braces.
    Returns `{}` on failure -- callers must treat that as a failure signal rather
    than as an empty result.
    """
    text = (content or "").strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            inner = parts[1]
            newline = inner.find("\n")
            candidate = inner[newline:].strip() if newline != -1 else inner.strip()
            try:
                return json.loads(candidate)
            except (json.JSONDecodeError, ValueError):
                pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except (json.JSONDecodeError, ValueError):
            pass
    return {}
