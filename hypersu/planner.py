"""Standalone query planner for decomposing complex questions into sub-queries.

This module is intentionally not wired into HyperSU's main retrieval or QA
pipeline. It can be imported independently or used as a small CLI utility.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import re
from typing import Any

from hypersu.utils import LLM_Model


PLANNER_SYSTEM_PROMPT = """
You are a query planning agent that decomposes multi-hop questions into minimal, precise sub-queries for retrieval.

## Hard Rules
- Return **valid JSON only** (no markdown, no commentary).
- Every sub-query MUST revolve around entities or noun phrases **explicitly mentioned** in the original query. Do NOT invent, guess, or assume entities that are not present.
- You MAY introduce a bridging placeholder (e.g., "[birthplace]", "[that person]") when the query implicitly requires an intermediate answer, but the placeholder must be clearly derived from the query's own wording.
- Each sub-query must be a self-contained, independently searchable question.
- Prefer the **fewest** sub-queries that cover all reasoning hops (typically 2–4).
- Do NOT produce trivial paraphrases of the original query.
- Preserve the original query's intent exactly.

## Output Schema
{
  "is_complex": true,
  "reasoning": "short explanation of the decomposition logic",
  "sub_queries": [
    {
      "id": "sq1",
      "query": "sub-question text",
      "purpose": "what intermediate fact this retrieves",
      "answer_type": "entity|event|cause|comparison|attribute|timeline|other"
    }
  ],
  "synthesis_instruction": "how to chain sub-query answers into the final answer"
}

If the query is simple (single-hop), return:
{
  "is_complex": false,
  "reasoning": "short explanation",
  "sub_queries": [{"id": "sq1", "query": "<original query>", "purpose": "answer directly", "answer_type": "other"}],
  "synthesis_instruction": "answer directly from the best supporting evidence"
}

## Few-shot Examples

### 2-hop
Query: "What county is Erik Hort's birthplace a part of?"
```json
{
  "is_complex": true,
  "reasoning": "Need to first find Erik Hort's birthplace, then find what county that place belongs to.",
  "sub_queries": [
    {"id": "sq1", "query": "Where was Erik Hort born?", "purpose": "identify the birthplace entity", "answer_type": "entity"},
    {"id": "sq2", "query": "What county is [birthplace] a part of?", "purpose": "find the county of the birthplace", "answer_type": "entity"}
  ],
  "synthesis_instruction": "Use sq1's answer to resolve sq2, then return the county."
}
```

### 2-hop
Query: "What year did the publisher of Labyrinth end?"
```json
{
  "is_complex": true,
  "reasoning": "Need to identify the publisher of Labyrinth, then find when that publisher ended.",
  "sub_queries": [
    {"id": "sq1", "query": "Who published Labyrinth?", "purpose": "identify the publisher entity", "answer_type": "entity"},
    {"id": "sq2", "query": "What year did [publisher] end?", "purpose": "find the end year of the publisher", "answer_type": "attribute"}
  ],
  "synthesis_instruction": "Use sq1's answer to resolve sq2, return the year."
}
```

### 3-hop
Query: "How many times did plague occur in the place where Crucifixion's creator died?"
```json
{
  "is_complex": true,
  "reasoning": "Need the creator of Crucifixion, then where that person died, then plague occurrences there.",
  "sub_queries": [
    {"id": "sq1", "query": "Who created the Crucifixion?", "purpose": "identify the creator", "answer_type": "entity"},
    {"id": "sq2", "query": "Where did [creator] die?", "purpose": "identify the place of death", "answer_type": "entity"},
    {"id": "sq3", "query": "How many times did plague occur in [place]?", "purpose": "count plague occurrences", "answer_type": "attribute"}
  ],
  "synthesis_instruction": "Chain sq1 → sq2 → sq3 to get the final count."
}
```

### 4-hop
Query: "Where does the body of water by the city where the Southeast Library designer died empty into the Gulf of Mexico?"
```json
{
  "is_complex": true,
  "reasoning": "Need the designer of Southeast Library, where they died, the body of water by that city, and where it empties into the Gulf of Mexico.",
  "sub_queries": [
    {"id": "sq1", "query": "Who designed the Southeast Library?", "purpose": "identify the designer", "answer_type": "entity"},
    {"id": "sq2", "query": "In which city did [designer] die?", "purpose": "identify the city of death", "answer_type": "entity"},
    {"id": "sq3", "query": "What body of water is by [city]?", "purpose": "identify the body of water", "answer_type": "entity"},
    {"id": "sq4", "query": "Where does [body of water] empty into the Gulf of Mexico?", "purpose": "find the outlet into the Gulf", "answer_type": "entity"}
  ],
  "synthesis_instruction": "Chain sq1 → sq2 → sq3 → sq4 to get the final location."
}
```
""".strip()


@dataclass
class PlannedSubQuery:
    id: str
    query: str
    purpose: str
    answer_type: str = "other"


@dataclass
class QueryPlan:
    original_query: str
    is_complex: bool
    reasoning: str
    sub_queries: list[PlannedSubQuery] = field(default_factory=list)
    synthesis_instruction: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_query": self.original_query,
            "is_complex": self.is_complex,
            "reasoning": self.reasoning,
            "sub_queries": [asdict(item) for item in self.sub_queries],
            "synthesis_instruction": self.synthesis_instruction,
        }


def _strip_code_fences(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_json_object(text: str) -> dict[str, Any] | None:
    text = _strip_code_fences(text)
    if not text:
        return None

    try:
        loaded = json.loads(text)
        if isinstance(loaded, dict):
            return loaded
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        loaded = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return loaded if isinstance(loaded, dict) else None


def _fallback_plan(query: str, reason: str) -> QueryPlan:
    return QueryPlan(
        original_query=query,
        is_complex=False,
        reasoning=reason,
        sub_queries=[
            PlannedSubQuery(
                id="sq1",
                query=query,
                purpose="answer directly",
                answer_type="other",
            )
        ],
        synthesis_instruction="answer directly from the best supporting evidence",
    )


def _normalize_sub_queries(items: Any, original_query: str) -> list[PlannedSubQuery]:
    if not isinstance(items, list):
        return [
            PlannedSubQuery(
                id="sq1",
                query=original_query,
                purpose="answer directly",
                answer_type="other",
            )
        ]

    normalized: list[PlannedSubQuery] = []
    for idx, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        query = str(item.get("query", "")).strip()
        if not query:
            continue
        normalized.append(
            PlannedSubQuery(
                id=str(item.get("id") or f"sq{idx}"),
                query=query,
                purpose=str(item.get("purpose") or "gather supporting evidence").strip(),
                answer_type=str(item.get("answer_type") or "other").strip(),
            )
        )

    if normalized:
        return normalized

    return [
        PlannedSubQuery(
            id="sq1",
            query=original_query,
            purpose="answer directly",
            answer_type="other",
        )
    ]


class QueryPlanner:
    """Standalone planner agent for complex-query decomposition."""

    def __init__(self, llm_model_name: str = "gpt-4o-mini", max_subqueries: int = 5):
        self.llm_model = LLM_Model(llm_model_name)
        self.max_subqueries = max(1, max_subqueries)

    def _build_messages(self, query: str, extra_context: str | None = None) -> list[dict[str, str]]:
        user_payload = {
            "query": query,
            "max_subqueries": self.max_subqueries,
        }
        if extra_context:
            user_payload["extra_context"] = extra_context
        return [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, indent=2)},
        ]

    def plan(self, query: str, extra_context: str | None = None) -> QueryPlan:
        query = (query or "").strip()
        if not query:
            return _fallback_plan("", "empty query")

        raw_response = self.llm_model.infer(self._build_messages(query, extra_context))
        payload = _extract_json_object(raw_response)
        if payload is None:
            return _fallback_plan(query, "planner returned non-JSON output")

        sub_queries = _normalize_sub_queries(payload.get("sub_queries"), query)[: self.max_subqueries]
        return QueryPlan(
            original_query=query,
            is_complex=bool(payload.get("is_complex", len(sub_queries) > 1)),
            reasoning=str(payload.get("reasoning") or "").strip() or "planner generated a query plan",
            sub_queries=sub_queries,
            synthesis_instruction=(
                str(payload.get("synthesis_instruction") or "").strip()
                or "combine the sub-query evidence to answer the original question"
            ),
        )


def plan_query(query: str, llm_model_name: str = "gpt-4o-mini",
               max_subqueries: int = 5, extra_context: str | None = None) -> QueryPlan:
    """Convenience wrapper for one-off planning."""
    planner = QueryPlanner(llm_model_name=llm_model_name, max_subqueries=max_subqueries)
    return planner.plan(query=query, extra_context=extra_context)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone planner agent for decomposing complex queries."
    )
    parser.add_argument("--query", required=True, help="Query to decompose")
    parser.add_argument("--model", default="gpt-4o-mini", help="Planner LLM model name")
    parser.add_argument("--max-subqueries", type=int, default=5, help="Maximum sub-queries to keep")
    parser.add_argument(
        "--context",
        default=None,
        help="Optional extra planning context, such as corpus/domain hints",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    plan = plan_query(
        query=args.query,
        llm_model_name=args.model,
        max_subqueries=args.max_subqueries,
        extra_context=args.context,
    )
    print(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
