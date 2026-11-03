"""Schema check for the golden dataset (SPEC Contract 4).

Run:  python -m qa.golden.validate            (exit code 1 if anything is wrong)

Checks every line of ``qa/golden/golden.jsonl``:
* valid JSON, required fields present with the right types, known category/product
* ids are unique
* every ``expected_context_ids`` entry is a real manual chunk id
* tool names are real tools, fault names are real tools
* refusal cases have ``must_refuse: true``; other categories have it false
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

GOLDEN_PATH = Path(__file__).resolve().parent / "golden.jsonl"
ROOT = Path(__file__).resolve().parents[2]

CATEGORIES = {"manual_qa", "tool_use", "multi_turn", "hinglish", "refusal", "unanswerable", "recovery"}
PRODUCTS = {"aero", "lock", "pure", "general"}
TOOL_NAMES = {"get_device_status", "set_fan_speed", "check_warranty", "create_service_ticket"}
REQUIRED = {"id": str, "category": str, "product": str, "turns": list, "expected_answer": str,
            "expected_context_ids": list, "expected_tools": list, "must_refuse": bool}
OPTIONAL = {"fault": dict, "tags": list, "key_facts": list, "must_not_include": list,
            "forbidden_tools": list, "needs_review": bool, "notes": str, "source": dict}


def known_chunk_ids() -> set[str]:
    """All chunk ids of the manuals (from the retriever; falls back to docs/chunk_ids.txt)."""
    try:
        from assistant.retrieval import all_chunks
        return {c.id for c in all_chunks()}
    except Exception:  # noqa: BLE001 - assistant package missing or broken: use the text list
        path = ROOT / "docs" / "chunk_ids.txt"
        return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def load_golden(path: Path = GOLDEN_PATH) -> list[dict]:
    """Read the JSONL file into a list of case dicts (blank lines ignored)."""
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))
    return cases


def validate_case(case: dict, chunk_ids: set[str]) -> list[str]:
    """Return a list of problems with one case (empty list = valid)."""
    cid = case.get("id", "<no id>")
    errors = []
    for field, typ in REQUIRED.items():
        if field not in case:
            errors.append(f"{cid}: missing field '{field}'")
        elif not isinstance(case[field], typ):
            errors.append(f"{cid}: '{field}' must be {typ.__name__}")
    for field, typ in OPTIONAL.items():
        if field in case and not isinstance(case[field], typ):
            errors.append(f"{cid}: '{field}' must be {typ.__name__}")
    unknown = set(case) - set(REQUIRED) - set(OPTIONAL)
    if unknown:
        errors.append(f"{cid}: unknown fields {sorted(unknown)}")
    if errors:
        return errors

    if case["category"] not in CATEGORIES:
        errors.append(f"{cid}: unknown category '{case['category']}'")
    if case["product"] not in PRODUCTS:
        errors.append(f"{cid}: unknown product '{case['product']}'")
    if not case["turns"] or not all(isinstance(t, str) and t.strip() for t in case["turns"]):
        errors.append(f"{cid}: 'turns' must be a non-empty list of non-empty strings")
    if not case.get("needs_review") and not case["expected_answer"].strip():
        errors.append(f"{cid}: empty expected_answer")
    for ctx in case["expected_context_ids"]:
        if ctx not in chunk_ids:
            errors.append(f"{cid}: unknown context id '{ctx}'")
    for tool in case["expected_tools"]:
        if not isinstance(tool, dict) or tool.get("name") not in TOOL_NAMES:
            errors.append(f"{cid}: bad expected tool {tool!r}")
        elif not isinstance(tool.get("args", {}), dict):
            errors.append(f"{cid}: tool args must be an object")
    for name in case.get("forbidden_tools", []):
        if name not in TOOL_NAMES:
            errors.append(f"{cid}: unknown forbidden tool '{name}'")
    for name in case.get("fault", {}):
        if name not in TOOL_NAMES:
            errors.append(f"{cid}: fault for unknown tool '{name}'")
    if (case["category"] == "refusal") != case["must_refuse"]:
        errors.append(f"{cid}: must_refuse should be true only for category 'refusal'")
    if case["category"] == "multi_turn" and len(case["turns"]) < 2:
        errors.append(f"{cid}: multi_turn case needs at least 2 turns")
    if case["category"] == "recovery" and not case.get("fault"):
        errors.append(f"{cid}: recovery case needs a 'fault'")
    return errors


def validate_file(path: Path = GOLDEN_PATH) -> list[str]:
    """Validate the whole file; returns all problems found."""
    chunk_ids = known_chunk_ids()
    errors: list[str] = []
    try:
        cases = load_golden(path)
    except json.JSONDecodeError as exc:
        return [f"invalid JSON: {exc}"]
    seen: set[str] = set()
    for case in cases:
        errors += validate_case(case, chunk_ids)
        if case.get("id") in seen:
            errors.append(f"{case.get('id')}: duplicate id")
        seen.add(case.get("id"))
    return errors


def main() -> int:
    """CLI entry point."""
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else GOLDEN_PATH
    errors = validate_file(path)
    cases = load_golden(path) if not any(e.startswith("invalid JSON") for e in errors) else []
    counts: dict[str, int] = {}
    for c in cases:
        counts[c.get("category", "?")] = counts.get(c.get("category", "?"), 0) + 1
    print(f"{len(cases)} cases: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"smoke: {sum(1 for c in cases if 'smoke' in c.get('tags', []))}")
    for e in errors:
        print("ERROR", e)
    print("OK" if not errors else f"{len(errors)} problem(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
