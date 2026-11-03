"""Retrieval returns the right manual sections."""

import pytest

from assistant.retrieval import all_chunks, retrieve, slugify


def test_slugify_uses_hash_for_subsection() -> None:
    assert slugify("Troubleshooting: Beeping") == "troubleshooting#beeping"
    assert slugify("Speed Settings: Speeds 1 to 5") == "speed-settings#speeds-1-to-5"


def test_chunk_ids_are_unique_and_cover_all_products() -> None:
    chunks = all_chunks()
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))
    assert {c.product for c in chunks} == {"aero", "lock", "pure", "general"}
    assert 15 <= len(chunks) <= 30
    assert "aero/troubleshooting#beeping" in ids


@pytest.mark.parametrize("query, expected_id", [
    ("my fan is beeping 3 long beeps", "aero/troubleshooting#beeping"),
    ("what does the red blinking LED on the fan mean", "aero/troubleshooting#led-indicator-codes"),
    ("how do I enrol a new fingerprint on the lock", "lock/users#fingerprint-enrolment"),
    ("lock low battery warning", "lock/batteries#low-battery-warning"),
    ("when should I replace the RO membrane filter", "pure/filters#filter-life-and-replacement-schedule"),
    ("purifier error code E4", "pure/troubleshooting#error-codes"),
    ("what voids the warranty", "general/warranty#what-voids-the-warranty"),
    ("factory reset the fan", "aero/reset#factory-reset-and-wi-fi-reset"),
])
def test_top_results_contain_expected_chunk(query: str, expected_id: str) -> None:
    ids = [c.id for c in retrieve(query, k=3)]
    assert expected_id in ids, ids


def test_k_limits_results_and_empty_query() -> None:
    assert len(retrieve("fan speed", k=2)) == 2
    assert retrieve("", k=4) == []
