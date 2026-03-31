"""LangExtract-based entity extraction for HyperSU."""

from __future__ import annotations

import logging
import os

import langextract as lx

from hypersu.entity_normalization import (
    build_entity_embedding_text,
    is_low_value_mention,
    normalize_description,
    normalize_entity_name,
    normalize_entity_type,
)
from hypersu.utils import compute_mdhash_id


logger = logging.getLogger(__name__)


def _extract_description_value(extraction) -> str | None:
    """Read the best available local description from a LangExtract extraction."""
    if extraction.description:
        return extraction.description

    attributes = extraction.attributes or {}
    if not isinstance(attributes, dict):
        return None

    attr_description = attributes.get("description")
    if isinstance(attr_description, str):
        return attr_description
    if isinstance(attr_description, list):
        for value in attr_description:
            if isinstance(value, str) and value.strip():
                return value
    return None


LANGEXTRACT_PROMPT = """
Extract only the key reusable entities from the text.

Rules:
- Use exact text spans from the source.
- Extract only entities that are specific enough to connect evidence across different semantic units.
- Favor people, groups, locations, events, artifacts, documents, organizations, medical entities, and other important named concepts.
- Provide a short grounded description for each extraction that explains who or what it is in this local context.
- Do not extract pronouns, generic references, vague abstractions, scene-setting objects, or long clause-like spans.
- Do not return overlapping duplicates.
- Use `extraction_class` as a coarse entity type.
""".strip()


LANGEXTRACT_EXAMPLES = [
    lx.data.ExampleData(
        text=(
            "The travellers reached Kynance Cove before visiting "
            "Landewednack, where the rector welcomed them."
        ),
        extractions=[
            lx.data.Extraction(
                extraction_class="location",
                extraction_text="Kynance Cove",
                description="coastal cove reached by the travellers",
            ),
            lx.data.Extraction(
                extraction_class="location",
                extraction_text="Landewednack",
                description="Cornish village visited after the cove",
            ),
            lx.data.Extraction(
                extraction_class="person",
                extraction_text="the rector",
                description="local rector who welcomed the visitors",
            ),
        ],
    ),
    lx.data.ExampleData(
        text=(
            "The biopsy confirmed Hodgkin lymphoma, and the patient "
            "started ABVD chemotherapy."
        ),
        extractions=[
            lx.data.Extraction(
                extraction_class="test",
                extraction_text="biopsy",
                description="diagnostic procedure confirming the disease",
            ),
            lx.data.Extraction(
                extraction_class="medical_condition",
                extraction_text="Hodgkin lymphoma",
                description="disease diagnosis confirmed by biopsy",
            ),
            lx.data.Extraction(
                extraction_class="treatment",
                extraction_text="ABVD chemotherapy",
                description="chemotherapy regimen started for treatment",
            ),
        ],
    ),
]


class LangExtractExtractor:
    """LLM-backed entity extraction with grounded descriptions."""

    def __init__(
        self,
        model_id: str = "gpt-4o-mini",
        api_key: str | None = None,
        model_url: str | None = None,
        max_char_buffer: int = 1000,
        extraction_passes: int = 1,
        max_workers: int = 10,
        use_schema_constraints: bool = True,
    ):
        self.model_id = model_id
        self.api_key = api_key or os.getenv("LANGEXTRACT_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.model_url = model_url or os.getenv("OPENAI_BASE_URL")
        self.max_char_buffer = max_char_buffer
        self.extraction_passes = max(1, extraction_passes)
        self.max_workers = max(1, max_workers)
        self.use_schema_constraints = use_schema_constraints

        if self.model_id.startswith(("gpt-", "o1", "o3", "o4")) and not self.api_key:
            raise ValueError(
                "LangExtract extraction requires an API key. Set OPENAI_API_KEY or "
                "LANGEXTRACT_API_KEY before indexing."
            )

        logger.info(
            "LangExtract loaded: model=%s, max_char_buffer=%s, extraction_passes=%s, max_workers=%s",
            self.model_id,
            self.max_char_buffer,
            self.extraction_passes,
            self.max_workers,
        )

    def _extract_documents(self, documents):
        return lx.extract(
            text_or_documents=documents,
            prompt_description=LANGEXTRACT_PROMPT,
            examples=LANGEXTRACT_EXAMPLES,
            model_id=self.model_id,
            api_key=self.api_key,
            model_url=self.model_url,
            max_char_buffer=self.max_char_buffer,
            extraction_passes=self.extraction_passes,
            batch_length=max(self.max_workers, 10),
            max_workers=self.max_workers,
            use_schema_constraints=self.use_schema_constraints,
            fetch_urls=False,
            show_progress=False,
            temperature=0.0,
        )

    def _extract_mentions_from_su_batch_once(
        self, su_items, passage_hash_id: str
    ) -> dict[str, list[dict]]:
        """Run one batch extraction attempt for a set of semantic units."""
        documents = [
            lx.data.Document(text=su_text, document_id=su_hash_id)
            for su_hash_id, su_text in su_items
        ]
        document_results = self._extract_documents(documents)
        mentions_by_su = {su_hash_id: [] for su_hash_id, _ in su_items}

        for annotated_document in document_results:
            su_hash_id = annotated_document.document_id
            seen = set()
            for extraction in annotated_document.extractions or []:
                mention = self._build_mention_record(extraction, passage_hash_id, su_hash_id)
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
                mentions_by_su.setdefault(su_hash_id, []).append(mention)
        return mentions_by_su

    def _build_mention_record(self, extraction, passage_hash_id: str, su_hash_id: str):
        surface_text = (extraction.extraction_text or "").strip()
        normalized_name = normalize_entity_name(surface_text)
        entity_type = normalize_entity_type(extraction.extraction_class)
        description = normalize_description(
            _extract_description_value(extraction),
            fallback_text=surface_text,
        )

        if is_low_value_mention(normalized_name, entity_type, description):
            return None

        char_start = None
        char_end = None
        grounded = extraction.char_interval is not None
        if grounded:
            char_start = extraction.char_interval.start_pos
            char_end = extraction.char_interval.end_pos

        mention_seed = f"{passage_hash_id}|{su_hash_id}|{normalized_name}|{char_start}|{char_end}"
        return {
            "mention_id": compute_mdhash_id(mention_seed, prefix="men-"),
            "passage_hash_id": passage_hash_id,
            "su_hash_id": su_hash_id,
            "surface_text": surface_text,
            "normalized_name": normalized_name,
            "entity_type": entity_type,
            "description": description,
            "char_start": char_start,
            "char_end": char_end,
            "grounded": grounded,
        }

    def extract_mentions_from_su_batch(self, su_items, passage_hash_id: str) -> dict[str, list[dict]]:
        """Extract mentions for a batch of semantic units from one passage."""
        if not su_items:
            return {}
        try:
            return self._extract_mentions_from_su_batch_once(su_items, passage_hash_id)
        except ValueError as exc:
            if len(su_items) == 1:
                su_hash_id, _ = su_items[0]
                logger.warning(
                    "LangExtract failed for semantic unit %s in passage %s; "
                    "returning empty mentions. error=%s",
                    su_hash_id,
                    passage_hash_id,
                    exc,
                )
                return {su_hash_id: []}

            midpoint = max(1, len(su_items) // 2)
            logger.warning(
                "LangExtract batch extraction failed for %s semantic units in passage %s; "
                "retrying with smaller batches. error=%s",
                len(su_items),
                passage_hash_id,
                exc,
            )
            left_mentions = self.extract_mentions_from_su_batch(
                su_items[:midpoint], passage_hash_id
            )
            right_mentions = self.extract_mentions_from_su_batch(
                su_items[midpoint:], passage_hash_id
            )
            merged_mentions = {su_hash_id: [] for su_hash_id, _ in su_items}
            merged_mentions.update(left_mentions)
            merged_mentions.update(right_mentions)
            return merged_mentions

    def extract_query_entities(self, query: str) -> list[dict]:
        """Extract query entities using the same schema as index-time mentions."""
        try:
            result = self._extract_documents(query)
        except ValueError as exc:
            logger.warning(
                "LangExtract query extraction failed; continuing without query entities. "
                "error=%s",
                exc,
            )
            return []
        query_mentions = []
        seen = set()
        for extraction in result.extractions or []:
            surface_text = (extraction.extraction_text or "").strip()
            normalized_name = normalize_entity_name(surface_text)
            entity_type = normalize_entity_type(extraction.extraction_class)
            description = normalize_description(
                _extract_description_value(extraction),
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
