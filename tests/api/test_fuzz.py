"""Property-based fuzzing of the whole API with Schemathesis.

Schemathesis reads the app's OpenAPI document and uses Hypothesis to GENERATE hundreds of
requests: valid ones (random session ids, unicode messages, 1..2000 chars) and invalid
ones (wrong types, missing fields, too-long strings). For every response it checks
properties that must ALWAYS hold:

* not_a_server_error            - never a 5xx crash, whatever the input
* response_schema_conformance   - the body matches the documented response schema
* content_type_conformance      - the Content-Type is the documented one
* negative_data_rejection       - invalid input is rejected with a 4xx, not accepted
* positive_data_acceptance      - valid input is accepted

`status_code_conformance` is excluded: the spec does not document 404/503 yet - that gap is
tracked as a known bug in test_inprocess.py::test_openapi_documents_error_responses.

Runs against its own live server (fake LLM) so fuzzed fan speeds don't disturb other tests.
Set FUZZ_EXAMPLES to fuzz harder (default 40 per operation).
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
import schemathesis
from hypothesis import HealthCheck, settings
from schemathesis import checks

from tests.api.live_server import LiveServer

MAX_EXAMPLES = int(os.getenv("FUZZ_EXAMPLES", "40"))


@pytest.fixture(scope="module")
def fuzz_server() -> Iterator[LiveServer]:
    with LiveServer(llm_mode="fake") as server:
        yield server


@pytest.fixture(scope="module")
def fuzz_schema(fuzz_server: LiveServer):
    """The OpenAPI schema, loaded from the running server."""
    return schemathesis.openapi.from_url(fuzz_server.base_url + "/openapi.json")


schema = schemathesis.pytest.from_fixture("fuzz_schema")


@schema.parametrize()
@settings(max_examples=MAX_EXAMPLES, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much,
                                 HealthCheck.function_scoped_fixture])
def test_api_never_crashes_and_matches_schema(case, fuzz_server: LiveServer) -> None:
    case.call_and_validate(
        base_url=fuzz_server.base_url,
        checks=[checks.not_a_server_error, checks.response_schema_conformance,
                checks.content_type_conformance, checks.negative_data_rejection,
                checks.positive_data_acceptance],
    )
