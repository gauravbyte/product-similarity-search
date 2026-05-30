"""NL query parser: LLM first, local regex fallback (provider-agnostic).

Default request/response shaping targets a Gemini-compatible generateContent
endpoint; point LLM_API_URL at another provider and adjust the body + extraction
in _parse_llm to switch. With no key set, it falls back to the local regex parser.
"""
import json
import logging
import re

import requests

from .config import LLM_API_KEY, LLM_API_URL, LLM_SYSTEM_PROMPT, LLM_TIMEOUT, COLOURS

log = logging.getLogger(__name__)


def parse_query(query: str) -> tuple[dict, str]:
    try:
        return _parse_llm(query), "llm"
    except Exception as e:
        log.warning("LLM parse failed (%s), falling back to regex", e)
        return _parse_local(query), "local"


def _parse_llm(query: str) -> dict:
    if not LLM_API_KEY:
        raise RuntimeError("LLM_API_KEY not set")
    resp = requests.post(
        LLM_API_URL,
        headers={"Content-Type": "application/json", "X-goog-api-key": LLM_API_KEY},
        json={"contents": [{"parts": [{"text": f"{LLM_SYSTEM_PROMPT}\nInput: {query}"}]}]},
        timeout=LLM_TIMEOUT,
    )
    resp.raise_for_status()
    raw = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    m = re.search(r"\{[^}]+\}", raw)
    if not m:
        raise ValueError(f"No JSON in LLM response: {raw!r}")
    p = json.loads(m.group())
    return {
        "free_text": p.get("free_text") or query.strip(),
        "brand": p.get("brand"),
        "category": p.get("category"),
        "colour": p.get("colour"),
        "min_price": _float(p.get("min_price")),
        "max_price": _float(p.get("max_price")),
    }


_PRICE_PATS = [
    (r"\b(?:under|below|less than|up to)\s*(?:rs\.?|inr|₹)?\s*(\d+(?:\.\d+)?)", "max_price"),
    (r"\b(?:over|above|more than|at least)\s*(?:rs\.?|inr|₹)?\s*(\d+(?:\.\d+)?)", "min_price"),
]


def _parse_local(query: str) -> dict:
    text = query.strip()
    out = {"free_text": text, "brand": None, "category": None,
           "colour": None, "min_price": None, "max_price": None}

    for pat, key in _PRICE_PATS:
        m = re.search(pat, text, re.I)
        if m:
            out[key] = float(m.group(1))
            text = text.replace(m.group(0), " ")

    m = re.search(
        r"\bby\s+([a-z0-9][a-z0-9 &'.-]*?)(?=\s+\b(?:under|below|less|up|over|above|more|at|for|in)\b|$)",
        text, re.I,
    )
    if m:
        out["brand"] = m.group(1).strip()
        text = text.replace(m.group(0), " ")

    m = re.search(rf"\b({COLOURS})\b", text, re.I)
    if m:
        out["colour"] = m.group(1).lower()

    out["free_text"] = re.sub(r"\s+", " ", text).strip() or query.strip()
    return out


def _float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return None
