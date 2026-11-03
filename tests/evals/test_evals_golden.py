"""The golden dataset is valid and the validator catches broken cases."""

from __future__ import annotations

import json
from pathlib import Path

from qa.golden.validate import (
    known_chunk_ids,
    load_golden,
    validate_case,
    validate_file,
)

GOOD = {"id": "x-1", "category": "manual_qa", "product": "aero", "turns": ["hi"],
        "expected_answer": "ans", "expected_context_ids": ["aero/troubleshooting#beeping"],
        "expected_tools": [], "must_refuse": False}


def test_real_golden_file_is_valid():
    assert validate_file() == []


def test_golden_covers_every_category_and_has_smoke_subset():
    cases = load_golden()
    counts: dict[str, int] = {}
    for c in cases:
        counts[c["category"]] = counts.get(c["category"], 0) + 1
    assert len(cases) >= 55
    assert counts["manual_qa"] >= 20 and counts["tool_use"] >= 8 and counts["hinglish"] >= 6
    assert counts["refusal"] >= 5 and counts["unanswerable"] >= 5 and counts["recovery"] >= 3
    assert counts["multi_turn"] >= 5
    smoke = [c for c in cases if "smoke" in c.get("tags", [])]
    assert 12 <= len(smoke) <= 20
    assert {c["category"] for c in smoke} == set(counts)   # smoke touches every category


def test_validator_accepts_good_case():
    assert validate_case(dict(GOOD), known_chunk_ids()) == []


def test_validator_rejects_unknown_context_id_and_bad_fields():
    ids = known_chunk_ids()
    assert any("unknown context id" in e for e in validate_case({**GOOD, "expected_context_ids": ["aero/nope"]}, ids))
    assert any("missing field" in e for e in validate_case({k: v for k, v in GOOD.items() if k != "turns"}, ids))
    assert any("must_refuse" in e for e in validate_case({**GOOD, "category": "refusal"}, ids))
    assert any("bad expected tool" in e for e in validate_case({**GOOD, "expected_tools": [{"name": "rm_rf"}]}, ids))
    assert any("unknown fields" in e for e in validate_case({**GOOD, "typo_field": 1}, ids))


def test_validator_finds_duplicates(tmp_path: Path):
    path = tmp_path / "g.jsonl"
    path.write_text(json.dumps(GOOD) + "\n" + json.dumps(GOOD) + "\n", encoding="utf-8")
    assert any("duplicate" in e for e in validate_file(path))
