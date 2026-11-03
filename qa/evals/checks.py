"""Deterministic (no-LLM) checks used by the eval runner.

These are cheap, exact and repeatable, so we use them wherever we can and only ask an LLM
judge for things a rule cannot decide (is this answer faithful / correct?).
"""

from __future__ import annotations

import re

# Phrases that show the bot admitted it does not know (English + Hinglish).
ABSTAIN_MARKERS = (
    "don't know", "do not know", "dont know", "not sure", "don't have", "do not have",
    "not mentioned", "no information", "isn't in", "is not in", "not in the manual",
    "not covered in", "couldn't find", "could not find", "can't find", "cannot find",
    "not available in", "doesn't mention", "does not mention", "don't mention", "not listed",
    "pata nahi", "nahi pata", "jaankari nahi", "jankari nahi", "information nahi",
)

# Common Hindi words written in English letters. Two or more distinct ones => Hinglish reply.
HINGLISH_WORDS = {
    "hai", "hain", "karo", "kare", "karein", "kijiye", "ke", "ki", "ka", "ko", "mein", "nahi",
    "aap", "aapka", "aapke", "aapki", "hoga", "gaya", "gayi", "raha", "rahi", "bhi", "toh", "kya",
    "pe", "chahiye", "sakta", "sakte", "liye", "jaldi", "abhi", "ab", "lo", "dijiye", "hoon",
    "badlo", "mahine", "saal", "bacha", "bachi", "wala", "wali", "agar", "phir", "yeh", "woh",
}

SENSITIVE_ARG_KEYS = {"phone", "issue"}   # masked or free text: never exact-matched
_LIST_MARKER = re.compile(r"(?m)^\s*\d+[.)]\s")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def normalize(text: str) -> str:
    """Lowercase, unify quotes/dashes, drop thousands separators ("3,000" -> "3000")."""
    text = text.lower().replace("’", "'").replace("‘", "'")
    text = text.replace("–", "-").replace("—", "-").replace("‑", "-")
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)
    return re.sub(r"\s+", " ", text)


def _same_value(expected: object, actual: object) -> bool:
    """Compare tool argument values loosely: 3 == 3.0 == "3", case-insensitive strings."""
    try:
        return float(expected) == float(actual)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(expected).strip().lower() == str(actual).strip().lower()


def tool_call_matches(expected: dict, actual_name: str, actual_args: dict) -> bool:
    """True if the call has the expected name and every expected arg (except PII/free text)."""
    if expected["name"] != actual_name:
        return False
    for key, value in (expected.get("args") or {}).items():
        if key in SENSITIVE_ARG_KEYS:
            continue
        if key not in actual_args or not _same_value(value, actual_args[key]):
            return False
    return True


def check_tools(expected: list[dict], forbidden: list[str], calls: list[tuple[str, dict]]) -> tuple[bool, str]:
    """Every expected tool call must appear (any order); no forbidden tool may be called.

    `calls` is a list of (name, args). Returns (ok, reason-if-not-ok).
    """
    missing = [e for e in expected if not any(tool_call_matches(e, n, a) for n, a in calls)]
    used_forbidden = sorted({n for n, _ in calls if n in forbidden})
    problems = []
    if missing:
        problems.append("missing tool call(s): " + ", ".join(
            f"{m['name']}({m.get('args', {})})" for m in missing))
    if used_forbidden:
        problems.append("called forbidden tool(s): " + ", ".join(used_forbidden))
    return (not problems, "; ".join(problems))


def check_key_facts(answer: str, facts: list[str]) -> tuple[float, list[str]]:
    """Share of key facts found in the answer (case-insensitive) and the list of missing ones."""
    if not facts:
        return 1.0, []
    text = normalize(answer)
    missing = [f for f in facts if normalize(f) not in text]
    return (len(facts) - len(missing)) / len(facts), missing


def check_must_not_include(answer: str, phrases: list[str]) -> list[str]:
    """Phrases from `phrases` that wrongly appear in the answer."""
    text = normalize(answer)
    return [p for p in phrases if normalize(p) in text]


def abstained(answer: str) -> bool:
    """True if the answer says it does not know (instead of inventing something)."""
    text = normalize(answer)
    return any(marker in text for marker in ABSTAIN_MARKERS)


def unsupported_numbers(answer: str, sources: list[str]) -> list[str]:
    """Numbers in the answer that appear in none of the sources (manual chunks, tool results,
    the user's own messages). A cheap hallucination signal: invented specs are usually numbers."""
    answer = normalize(_LIST_MARKER.sub(" ", answer))   # drop "1." list numbering first
    source_numbers = set(_NUMBER.findall(normalize(" ".join(sources))))
    return sorted({n for n in _NUMBER.findall(answer) if n not in source_numbers})


def is_hinglish(text: str) -> bool:
    """Rough language check: does the reply use at least 2 different common Hindi words?"""
    words = set(re.findall(r"[a-z]+", text.lower()))
    return len(words & HINGLISH_WORDS) >= 2
