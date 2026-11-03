"""LLM-as-a-judge for the evals.

Two judges, both built on OUR `LLMClient` (so judge calls go through the same cassettes as the
bot: record once, replay for free in CI):

1. `RubricJudge` - ONE call per test case that returns several scores as JSON
   (faithfulness, answer_relevancy, correctness, hallucination). This is a "G-Eval style"
   combined rubric. One call instead of ~6 (what separate DeepEval metrics need) keeps the
   tiny free Gemini quota alive.
2. `GeminiDeepEvalModel` - an adapter so DeepEval's own metrics (FaithfulnessMetric, GEval)
   can use our Gemini client instead of OpenAI. We run those only on the `smoke` subset.

In `fake` mode `fake_judge_llm()` returns a rule-based judge (word-overlap heuristics), so the
whole pipeline runs offline in CI. Its numbers are NOT meaningful quality scores.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from assistant.llm import FakeLLM, LLMClient, LLMResponse

JUDGE_SYSTEM = """You are a strict quality evaluator for "Hawa Assist", a customer-support chatbot.
You get a JSON object with: question (the customer's conversation), answer (the bot's final reply),
context (manual sections and live tool results the bot was given) and reference (a correct answer
written by a human).

Score the answer and return ONLY a JSON object, no markdown:
{"faithfulness": <0..1>, "answer_relevancy": <0..1>, "correctness": <0..1>,
 "hallucination": <true|false>, "unsupported_claims": [<short strings>], "reason": "<one sentence>"}

Definitions:
- faithfulness: share of the factual claims in the answer that are supported by the context.
  Greetings, apologies, offers to raise a ticket and questions back to the user are not claims.
  If the answer makes no factual claims, faithfulness is 1.
- answer_relevancy: how directly and completely the answer addresses the customer's last question
  (1 = fully on topic, 0 = ignores the question).
- correctness: how well the answer agrees with the reference on the key facts (numbers, steps,
  yes/no). Extra correct details are fine. Reply language (English or Hinglish) does not matter.
- hallucination: true if the answer states ANY specific fact (number, step, feature, price,
  status) that is not in the context.
"""

JUDGE_KEYS = ("faithfulness", "answer_relevancy", "correctness")


@dataclass
class JudgeResult:
    """Scores for one case from the rubric judge."""

    faithfulness: float
    answer_relevancy: float
    correctness: float
    hallucination: bool
    unsupported_claims: list[str] = field(default_factory=list)
    reason: str = ""


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply (handles ```json fences and extra text)."""
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"no JSON object in judge reply: {text[:200]!r}")
    return json.loads(cleaned[start:end + 1])


def _clamp(value: object) -> float:
    try:
        return max(0.0, min(1.0, float(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def build_judge_input(question: str, answer: str, context: list[str], reference: str) -> str:
    """The user message sent to the judge: a JSON object (stable key order => stable cassettes)."""
    return json.dumps({"question": question, "answer": answer, "context": context,
                       "reference": reference}, ensure_ascii=False, indent=1)


class RubricJudge:
    """One-call-per-case judge on top of any LLMClient."""

    def __init__(self, llm: LLMClient):
        self.llm = llm
        self.calls = 0

    def score(self, question: str, answer: str, context: list[str], reference: str) -> JudgeResult:
        """Ask the judge LLM for all scores at once."""
        self.calls += 1
        reply = self.llm.generate(JUDGE_SYSTEM, [{"role": "user", "content": build_judge_input(
            question, answer, context, reference)}], None, 0.0)
        data = extract_json(reply.text)
        return JudgeResult(faithfulness=_clamp(data.get("faithfulness")),
                           answer_relevancy=_clamp(data.get("answer_relevancy")),
                           correctness=_clamp(data.get("correctness")),
                           hallucination=bool(data.get("hallucination", False)),
                           unsupported_claims=[str(c) for c in data.get("unsupported_claims") or []],
                           reason=str(data.get("reason", ""))[:300])


# --------------------------------------------------------------------------- fake judge (offline)
_WORD = re.compile(r"[a-z0-9][a-z0-9.\-]*")
_STOP = {"the", "a", "an", "is", "are", "to", "of", "and", "or", "in", "on", "for", "it", "your",
         "you", "i", "my", "with", "be", "can", "this", "that", "at", "by", "from", "if", "as",
         "according", "manual", "please", "will", "do", "not", "what", "how", "does"}


def _words(text: str) -> set[str]:
    return {w.strip(".-") for w in _WORD.findall(text.lower()) if w.strip(".-") not in _STOP}


def heuristic_scores(question: str, answer: str, context: list[str], reference: str) -> dict:
    """Word-overlap stand-in for a real judge. Only for offline plumbing tests."""
    a, q, r = _words(answer), _words(question), _words(reference)
    c = _words(" ".join(context)) | q
    faith = len(a & c) / len(a) if a else 1.0
    relevancy = min(1.0, 2 * len(a & q) / len(q)) if q else 0.0
    correctness = len(a & r) / len(r) if r else 0.0
    unsupported = sorted(a - c)[:5]
    return {"faithfulness": round(faith, 3), "answer_relevancy": round(relevancy, 3),
            "correctness": round(correctness, 3), "hallucination": faith < 0.5,
            "unsupported_claims": unsupported, "reason": "fake heuristic judge (word overlap)"}


def _fake_deepeval_reply(prompt: str) -> dict:
    """Canned JSON in the shape each DeepEval prompt asks for (truths / claims / verdicts / score)."""
    low = prompt.lower()
    if '"truths"' in low:
        return {"truths": ["(fake) the context states facts"]}
    if '"claims"' in low:
        return {"claims": ["(fake) the answer makes a claim"]}
    if '"verdicts"' in low:
        return {"verdicts": [{"verdict": "yes", "reason": "fake"}]}
    if '"score"' in low:
        return {"score": 7, "reason": "fake judge"}
    return {"reason": "fake judge"}


def _fake_judge_responder(system: str, messages: list[dict], tools: list[dict] | None) -> LLMResponse:
    if system != JUDGE_SYSTEM:                       # a DeepEval metric prompt
        return LLMResponse(text=json.dumps(_fake_deepeval_reply(messages[-1]["content"])))
    payload = json.loads(messages[-1]["content"])
    scores = heuristic_scores(payload["question"], payload["answer"], payload["context"],
                              payload["reference"])
    return LLMResponse(text=json.dumps(scores), tokens_in=0, tokens_out=0)


def fake_judge_llm() -> FakeLLM:
    """An LLMClient that answers judge prompts with heuristic JSON (no network)."""
    return FakeLLM(_fake_judge_responder)


# --------------------------------------------------------------------------- DeepEval adapter
def make_deepeval_model(llm: LLMClient, model_name: str = "gemini"):
    """Wrap our LLMClient as a DeepEval model (imported lazily: DeepEval is a heavy package)."""
    from deepeval.models import DeepEvalBaseLLM

    class GeminiDeepEvalModel(DeepEvalBaseLLM):
        """DeepEval calls generate(prompt, schema); we send it through our cached client."""

        def __init__(self) -> None:   # no super().__init__: it would try to load a real model
            self.llm = llm
            self.model = llm
            self.name = model_name
            self.calls = 0

        def load_model(self):
            return self.llm

        def generate(self, prompt: str, schema=None):
            self.calls += 1
            text = self.llm.generate("", [{"role": "user", "content": prompt}], None, 0.0).text
            if schema is None:
                return text
            try:
                return schema.model_validate(extract_json(text))
            except Exception:  # noqa: BLE001 - let DeepEval try its own JSON repair on the text
                return text

        async def a_generate(self, prompt: str, schema=None):
            return self.generate(prompt, schema)

        def get_model_name(self) -> str:
            return self.name

    return GeminiDeepEvalModel()
