"""Retrieval quality metrics - no LLM needed, so they cost nothing.

We know which manual chunks SHOULD be retrieved for each golden case (`expected_context_ids`)
and which chunks the bot actually retrieved (`BotReply.context_ids`). From that:

* context_recall    = share of the expected chunks that were retrieved
                      ("did we find everything we needed?")
* context_precision = rank-aware precision (average precision): 1.0 when the needed chunks are
                      at the top of the list, lower when junk chunks are ranked above them
                      ("is the useful stuff ranked first?")
* context_precision_raw = plain share of retrieved chunks that were expected (with k=4 and one
                      expected chunk the best possible value is 0.25 - that is normal)

When Ragas is installed, recall and raw precision come from Ragas' ID-based metrics
(`IDBasedContextRecall`, `IDBasedContextPrecision`); otherwise from the same formulas below.
"""

from __future__ import annotations

import warnings


def recall(retrieved: list[str], expected: list[str]) -> float:
    """Share of expected ids that appear in retrieved."""
    if not expected:
        return 1.0
    return sum(1 for e in expected if e in retrieved) / len(expected)


def raw_precision(retrieved: list[str], expected: list[str]) -> float:
    """Share of retrieved ids that are expected."""
    if not retrieved:
        return 0.0
    return sum(1 for r in retrieved if r in expected) / len(retrieved)


def average_precision(retrieved: list[str], expected: list[str]) -> float:
    """Rank-aware precision: mean of precision@k at each rank k that holds a relevant chunk."""
    hits, total = 0, 0.0
    for k, cid in enumerate(retrieved, start=1):
        if cid in expected:
            hits += 1
            total += hits / k
    return total / hits if hits else 0.0


def _ragas_scores(retrieved: list[str], expected: list[str]) -> tuple[float, float] | None:
    """(recall, raw precision) from Ragas' ID-based metrics, or None if Ragas is unavailable."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from ragas.dataset_schema import SingleTurnSample
            from ragas.metrics import IDBasedContextPrecision, IDBasedContextRecall
    except Exception:  # noqa: BLE001 - ragas missing or API changed: fall back to our formulas
        return None
    sample = SingleTurnSample(retrieved_context_ids=retrieved, reference_context_ids=expected)
    try:
        rec = IDBasedContextRecall().single_turn_score(sample)
        prec = IDBasedContextPrecision().single_turn_score(sample)
    except Exception:  # noqa: BLE001
        return None
    return float(rec), float(prec)


def retrieval_scores(retrieved: list[str], expected: list[str], use_ragas: bool = True) -> dict:
    """All retrieval scores for one case. Empty `expected` => nothing to measure => {}."""
    if not expected:
        return {}
    from_ragas = _ragas_scores(retrieved, expected) if use_ragas else None
    rec, prec = from_ragas if from_ragas else (recall(retrieved, expected), raw_precision(retrieved, expected))
    return {"context_recall": round(rec, 4), "context_precision": round(average_precision(retrieved, expected), 4),
            "context_precision_raw": round(prec, 4)}


def ragas_available() -> bool:
    """True if Ragas' ID-based metrics can be used."""
    return _ragas_scores(["a"], ["a"]) is not None
