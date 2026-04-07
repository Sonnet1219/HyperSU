"""Direct-LLM entity extraction for HyperSU."""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

from hypersu.entity_normalization import (
    build_entity_embedding_text,
    is_low_value_mention,
    normalize_description,
    normalize_entity_name,
    normalize_entity_type,
)
from hypersu.utils import LLM_Model, compute_mdhash_id


logger = logging.getLogger(__name__)


NER_SYSTEM_PROMPT = """\
Extract only the key reusable entities from the text.

Rules:
- Use exact text spans from the source.
- Extract only entities that are specific enough to connect evidence across different semantic units.
- Favor people, groups, locations, events, artifacts, documents, organizations, medical entities, and other important named concepts.
- Provide a short grounded description for each extraction that explains who or what it is in this local context.
- Do not extract pronouns, generic references, vague abstractions, scene-setting objects, or long clause-like spans.
- Do not return overlapping duplicates.
- Use `extraction_class` as a coarse entity type.

Return a JSON array of objects, each with keys: "extraction_class", "extraction_text", "description".
Do NOT wrap the JSON in markdown fences or add any other text."""

FEW_SHOT_EXAMPLES = [
    {
        "text": (
            "The travellers reached Kynance Cove before visiting "
            "Landewednack, where the rector welcomed them."
        ),
        "extractions": [
            {
                "extraction_class": "location",
                "extraction_text": "Kynance Cove",
                "description": "coastal cove reached by the travellers",
            },
            {
                "extraction_class": "location",
                "extraction_text": "Landewednack",
                "description": "Cornish village visited after the cove",
            },
            {
                "extraction_class": "person",
                "extraction_text": "the rector",
                "description": "local rector who welcomed the visitors",
            },
        ],
    },
    {
        "text": (
            "The biopsy confirmed Hodgkin lymphoma, and the patient "
            "started ABVD chemotherapy."
        ),
        "extractions": [
            {
                "extraction_class": "test",
                "extraction_text": "biopsy",
                "description": "diagnostic procedure confirming the disease",
            },
            {
                "extraction_class": "medical_condition",
                "extraction_text": "Hodgkin lymphoma",
                "description": "disease diagnosis confirmed by biopsy",
            },
            {
                "extraction_class": "treatment",
                "extraction_text": "ABVD chemotherapy",
                "description": "chemotherapy regimen started for treatment",
            },
        ],
    },
]

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _build_messages(text: str) -> list[dict]:
    """Construct chat messages for entity extraction."""
    messages = [{"role": "system", "content": NER_SYSTEM_PROMPT}]
    for example in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": example["text"]})
        messages.append({
            "role": "assistant",
            "content": json.dumps(example["extractions"], ensure_ascii=False),
        })
    messages.append({"role": "user", "content": text})
    return messages


def _parse_extractions(response_text: str) -> list[dict]:
    """Parse LLM response into a list of extraction dicts."""
    if not response_text:
        return []
    text = response_text.strip()
    fence_match = _FENCE_RE.search(text)
    if fence_match:
        text = fence_match.group(1)
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Failed to parse NER response as JSON: %.200s", response_text)
        return []
    if not isinstance(result, list):
        logger.warning("NER response is not a JSON array: %.200s", response_text)
        return []
    valid = []
    for item in result:
        if isinstance(item, dict) and "extraction_text" in item:
            valid.append(item)
    return valid


class DirectLLMExtractor:
    """LLM-backed entity extraction via direct OpenAI API calls."""

    def __init__(self, model_id: str = "gpt-4o-mini", max_workers: int = 10):
        self.model_id = model_id
        self.max_workers = max(1, max_workers)
        self.llm = LLM_Model(model_id)
        logger.info(
            "DirectLLMExtractor loaded: model=%s, max_workers=%s",
            self.model_id,
            self.max_workers,
        )

    def _extract_single_su(self, su_text: str) -> list[dict]:
        """Extract entities from a single semantic unit text."""
        messages = _build_messages(su_text)
        response = self.llm.infer(messages)
        return _parse_extractions(response)

    def _build_mention_record(
        self, raw_extraction: dict, passage_hash_id: str, su_hash_id: str
    ) -> dict | None:
        surface_text = (raw_extraction.get("extraction_text") or "").strip()
        normalized_name = normalize_entity_name(surface_text)
        entity_type = normalize_entity_type(raw_extraction.get("extraction_class"))
        description = normalize_description(
            raw_extraction.get("description"),
            fallback_text=surface_text,
        )

        if is_low_value_mention(normalized_name, entity_type, description):
            return None

        mention_seed = f"{passage_hash_id}|{su_hash_id}|{normalized_name}|None|None"
        return {
            "mention_id": compute_mdhash_id(mention_seed, prefix="men-"),
            "passage_hash_id": passage_hash_id,
            "su_hash_id": su_hash_id,
            "surface_text": surface_text,
            "normalized_name": normalized_name,
            "entity_type": entity_type,
            "description": description,
            "char_start": None,
            "char_end": None,
            "grounded": False,
        }

    def extract_all_mentions(
        self, all_su_items: list[tuple[str, str, str]],
    ) -> dict[str, list[dict]]:
        """Extract mentions for all semantic units concurrently.

        Args:
            all_su_items: list of (su_hash_id, su_text, passage_hash_id) tuples
                across all passages.

        Returns:
            dict mapping su_hash_id -> list of mention dicts.
        """
        if not all_su_items:
            return {}

        mentions_by_su: dict[str, list[dict]] = {
            su_hash_id: [] for su_hash_id, _, _ in all_su_items
        }

        def _process_one(item):
            su_hash_id, su_text, passage_hash_id = item
            try:
                raw_extractions = self._extract_single_su(su_text)
            except Exception as exc:
                logger.warning(
                    "NER extraction failed for SU %s in passage %s: %s",
                    su_hash_id, passage_hash_id, exc,
                )
                return su_hash_id, []

            mentions = []
            seen = set()
            for raw in raw_extractions:
                mention = self._build_mention_record(raw, passage_hash_id, su_hash_id)
                if mention is None:
                    continue
                dedupe_key = (
                    mention["normalized_name"],
                    mention["entity_type"],
                    mention["description"],
                )
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                mentions.append(mention)
            return su_hash_id, mentions

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for su_hash_id, mentions in tqdm(
                executor.map(_process_one, all_su_items),
                total=len(all_su_items),
                desc="NER Extraction",
            ):
                mentions_by_su[su_hash_id] = mentions

        return mentions_by_su

    def extract_query_entities(self, query: str) -> list[dict]:
        """Extract query entities using the same schema as index-time mentions."""
        try:
            raw_extractions = self._extract_single_su(query)
        except Exception as exc:
            logger.warning(
                "NER query extraction failed; continuing without query entities. error=%s",
                exc,
            )
            return []

        query_mentions = []
        seen = set()
        for raw in raw_extractions:
            surface_text = (raw.get("extraction_text") or "").strip()
            normalized_name = normalize_entity_name(surface_text)
            entity_type = normalize_entity_type(raw.get("extraction_class"))
            description = normalize_description(
                raw.get("description"),
                fallback_text=surface_text,
            )
            if is_low_value_mention(normalized_name, entity_type, description):
                continue
            embedding_text = build_entity_embedding_text(normalized_name, description)
            dedupe_key = (normalized_name, entity_type, description)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            query_mentions.append({
                "surface_text": surface_text,
                "normalized_name": normalized_name,
                "entity_type": entity_type,
                "description": description,
                "embedding_text": embedding_text,
            })
        return query_mentions
