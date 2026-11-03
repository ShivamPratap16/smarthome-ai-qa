# Golden dataset

`golden.jsonl` = one test case per line (SPEC Contract 4). 64 cases written by hand from
`manuals/*.md`: manual_qa 26, tool_use 10, multi_turn 6, hinglish 8, refusal 6,
unanswerable 5, recovery 3. 16 are tagged `smoke` (a fast subset that touches every category).

## Fields

| Field | Required | Meaning |
|---|---|---|
| `id` | yes | unique, e.g. `aero-beep-01` |
| `category` | yes | `manual_qa` `tool_use` `multi_turn` `hinglish` `refusal` `unanswerable` `recovery` |
| `product` | yes | `aero` `lock` `pure` `general` |
| `turns` | yes | user messages, in order (multi-turn = 2+) |
| `expected_answer` | yes | reference answer, copied from the manual (the judge compares against it) |
| `expected_context_ids` | yes | manual chunks that SHOULD be retrieved (ids in `docs/chunk_ids.txt`); `[]` for unanswerable |
| `expected_tools` | yes | tool calls that must happen (any order, across all turns). Args are a subset match; `phone` and `issue` are never compared (PII-masked / free text) |
| `must_refuse` | yes | true only for `refusal` cases |
| `fault` | no | injected device error, e.g. `{"set_fan_speed": [503, 503]}` (the bot retries once, so one 503 is recovered) |
| `key_facts` | no | short strings that must appear in the reply (deterministic check) |
| `must_not_include` | no | strings that must NOT appear (e.g. a raw phone number, a false "speed set") |
| `forbidden_tools` | no | tools that must not be called (e.g. ticket without a phone number) |
| `tags` | no | `smoke`, `regression` ... |
| `needs_review`, `notes`, `source` | no | used by regressions created from production failures |

## Adding a case

1. Find the fact in `manuals/*.md`; copy numbers exactly into `expected_answer` and `key_facts`.
2. Keep `key_facts` short and language-neutral (numbers, product words) so Hinglish replies pass too.
3. Run `python -m qa.golden.validate` - it checks the schema, unique ids and that every context id exists.
4. Run just your case: `python -m qa.evals.run --mode fake --ids <id>` (then `--mode record` once to save cassettes).
5. Regenerate promptfoo tests: `python -m qa.promptfoo.build_tests`.

## From a production failure

`python -m qa.regressions.from_failure --latest-failed` copies the user turns of the newest failed
chat in `data/chatlogs.db` into this file with `needs_review: true`. Write the correct
`expected_answer` (+ context ids / tools), delete `needs_review`, and the eval suite starts
checking it. Cases with `needs_review` are skipped by the runner until then.
