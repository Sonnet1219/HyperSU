"""Entity normalization and mention-to-node merging for HyperSU."""

from __future__ import annotations

from collections import Counter, defaultdict
from difflib import SequenceMatcher
import logging
import re

import numpy as np


logger = logging.getLogger(__name__)

_ARTICLE_RE = re.compile(r"^(?:the|an|a)\s+", re.IGNORECASE)
_SPACED_PAREN_RE = re.compile(r"\s+\([^)]*\)")
_MULTI_SPACE_RE = re.compile(r"\s{2,}")
_EDGE_PUNCT_RE = re.compile(r"^[^\w]+|[^\w]+$")
_POSSESSIVE_RE = re.compile(r"(?:'s|’s)$")

_LOW_SIGNAL_NAMES = {
    "he",
    "she",
    "they",
    "them",
    "him",
    "her",
    "his",
    "their",
    "hers",
    "theirs",
    "someone",
    "some one",
    "somebody",
    "something",
    "one of us",
    "one",
    "it",
    "its",
    "this",
    "that",
    "these",
    "those",
}

_LOW_SIGNAL_TYPES = {"pronoun", "reference", "generic", "other"}


def normalize_entity_name(text: str) -> str:
    """Normalize an entity surface form into a merge key."""
    text = (text or "").strip().lower()
    text = _ARTICLE_RE.sub("", text)
    text = text.replace("(s)", "")
    text = _SPACED_PAREN_RE.sub("", text)
    text = text.replace("-", " ").replace("/", " / ")
    text = _POSSESSIVE_RE.sub("", text)
    text = _EDGE_PUNCT_RE.sub("", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def normalize_entity_type(entity_type: str | None) -> str:
    """Normalize the coarse entity type for stable comparisons."""
    normalized = (entity_type or "entity").strip().lower()
    normalized = normalized.replace("-", "_").replace(" ", "_")
    normalized = _MULTI_SPACE_RE.sub("_", normalized)
    return normalized or "entity"


def normalize_description(description: str | None, fallback_text: str = "") -> str:
    """Normalize a local description while preserving simple readability."""
    text = (description or fallback_text or "").strip().lower()
    text = text.strip(" .;,:")
    text = _MULTI_SPACE_RE.sub(" ", text)
    return text


def build_entity_embedding_text(name: str, description: str | None = None) -> str:
    """Compose the text used for entity embeddings and retrieval matching."""
    name = normalize_entity_name(name)
    description = normalize_description(description)
    if description:
        return f"{name}: {description}"
    return name


def is_low_value_mention(name: str, entity_type: str, description: str) -> bool:
    """Filter mentions that are too generic to become useful graph nodes."""
    if not name:
        return True
    if entity_type in _LOW_SIGNAL_TYPES:
        return True
    if name in _LOW_SIGNAL_NAMES:
        return True
    if len(name) < 3:
        return True
    if len(name.split()) > 8:
        return True
    alnum_chars = sum(ch.isalnum() for ch in name)
    if alnum_chars < 3:
        return True
    if not description:
        return True
    return False


def _name_similarity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    tokens_a = set(a.split())
    tokens_b = set(b.split())
    jaccard = len(tokens_a & tokens_b) / max(len(tokens_a | tokens_b), 1)
    containment = 0.0
    if a in b or b in a:
        containment = min(len(a), len(b)) / max(len(a), len(b), 1)
    seq = SequenceMatcher(None, a, b).ratio()
    return max(jaccard, containment, seq)


def _choose_canonical_name(mentions: list[dict]) -> str:
    counts = Counter(m["normalized_name"] for m in mentions)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], len(item[0]), item[0]))
    return ranked[0][0]


def _choose_entity_type(mentions: list[dict]) -> str:
    counts = Counter(m["entity_type"] for m in mentions)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return ranked[0][0]


def _choose_canonical_description(
    descriptions: list[str], embedding_model, batch_size: int
) -> str:
    unique_descriptions = list(dict.fromkeys(d for d in descriptions if d))
    if not unique_descriptions:
        return ""
    if len(unique_descriptions) == 1:
        return unique_descriptions[0]

    embeddings = embedding_model.encode(
        unique_descriptions,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=batch_size,
    )
    embeddings = np.asarray(embeddings)
    similarity = embeddings @ embeddings.T
    mean_similarity = similarity.mean(axis=1)
    best_idx = int(np.argmax(mean_similarity))
    return unique_descriptions[best_idx]


def _make_cluster(mentions: list[dict], embedding_model, batch_size: int) -> dict:
    canonical_name = _choose_canonical_name(mentions)
    canonical_description = _choose_canonical_description(
        [m["description"] for m in mentions], embedding_model, batch_size
    )
    entity_type = _choose_entity_type(mentions)
    aliases = sorted({m["surface_text"].strip() for m in mentions if m["surface_text"].strip()})
    description_variants = list(
        dict.fromkeys(m["description"] for m in mentions if m["description"])
    )
    supporting_su_ids = sorted({m["su_hash_id"] for m in mentions})
    supporting_passage_ids = sorted({m["passage_hash_id"] for m in mentions})

    return {
        "canonical_name": canonical_name,
        "canonical_description": canonical_description,
        "entity_type": entity_type,
        "aliases": aliases,
        "description_variants": description_variants,
        "member_mentions": mentions,
        "supporting_su_ids": supporting_su_ids,
        "supporting_passage_ids": supporting_passage_ids,
        "embedding_text": build_entity_embedding_text(canonical_name, canonical_description),
    }


def _cluster_similarity(cluster_a: dict, cluster_b: dict) -> float:
    return _name_similarity(cluster_a["canonical_name"], cluster_b["canonical_name"])


def _can_merge(cluster_a: dict, cluster_b: dict, similarity: float) -> bool:
    name_score = _cluster_similarity(cluster_a, cluster_b)
    if name_score < 0.72:
        return False
    return similarity >= 0.90


def merge_entity_mentions(
    mentions: list[dict],
    su_text_by_hash: dict[str, str],
    embedding_model,
    similarity_threshold: float = 0.90,
    batch_size: int = 128,
):
    """Merge SU-level mentions into canonical entity nodes.

    Returns:
        entity_nodes: list[dict]
        passage_entities: {passage_hash_id: [entity_embedding_text, ...]}
        su_entities: {su_text: [entity_embedding_text, ...]}
        passage_entity_counts: {passage_hash_id: {entity_embedding_text: count}}
    """
    filtered_mentions = []
    for mention in mentions:
        mention = dict(mention)
        mention["normalized_name"] = normalize_entity_name(mention.get("normalized_name") or mention.get("surface_text", ""))
        mention["entity_type"] = normalize_entity_type(mention.get("entity_type"))
        mention["description"] = normalize_description(
            mention.get("description"),
            fallback_text=mention.get("surface_text", ""),
        )
        if is_low_value_mention(
            mention["normalized_name"], mention["entity_type"], mention["description"]
        ):
            continue
        filtered_mentions.append(mention)

    if not filtered_mentions:
        logger.info("Entity merge: no valid mentions after filtering")
        return [], {}, {}, {}

    exact_groups = defaultdict(list)
    for mention in filtered_mentions:
        # Hard-cluster by normalized name only so same-name mentions can vote on type.
        exact_groups[mention["normalized_name"]].append(mention)

    initial_clusters = [
        _make_cluster(group_mentions, embedding_model, batch_size)
        for group_mentions in exact_groups.values()
    ]
    logger.info(
        "Entity merge: %d mentions -> %d exact clusters",
        len(filtered_mentions),
        len(initial_clusters),
    )

    if len(initial_clusters) == 1:
        merged_clusters = initial_clusters
    else:
        cluster_embeddings = embedding_model.encode(
            [cluster["embedding_text"] for cluster in initial_clusters],
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=batch_size,
        )
        cluster_embeddings = np.asarray(cluster_embeddings)
        similarity_matrix = cluster_embeddings @ cluster_embeddings.T

        parent = list(range(len(initial_clusters)))

        def find(idx: int) -> int:
            while parent[idx] != idx:
                parent[idx] = parent[parent[idx]]
                idx = parent[idx]
            return idx

        def union(a: int, b: int) -> None:
            root_a = find(a)
            root_b = find(b)
            if root_a != root_b:
                parent[root_a] = root_b

        rows, cols = np.where(similarity_matrix >= similarity_threshold)
        for i, j in zip(rows.tolist(), cols.tolist()):
            if i >= j:
                continue
            similarity = float(similarity_matrix[i, j])
            if _can_merge(initial_clusters[i], initial_clusters[j], similarity):
                union(i, j)

        merged_members = defaultdict(list)
        for idx, cluster in enumerate(initial_clusters):
            merged_members[find(idx)].extend(cluster["member_mentions"])

        merged_clusters = [
            _make_cluster(group_mentions, embedding_model, batch_size)
            for group_mentions in merged_members.values()
        ]

    entity_nodes = []
    passage_entities = defaultdict(list)
    su_entities = defaultdict(list)
    passage_entity_counts = defaultdict(lambda: defaultdict(float))

    for cluster in merged_clusters:
        mentions_in_cluster = cluster["member_mentions"]
        embedding_text = cluster["embedding_text"]
        canonical_name = cluster["canonical_name"]
        canonical_description = cluster["canonical_description"]

        entity_node = {
            "canonical_name": canonical_name,
            "canonical_description": canonical_description,
            "entity_type": cluster["entity_type"],
            "aliases": cluster["aliases"],
            "description_variants": cluster["description_variants"],
            "supporting_su_ids": cluster["supporting_su_ids"],
            "supporting_passage_ids": cluster["supporting_passage_ids"],
            "mention_ids": [m["mention_id"] for m in mentions_in_cluster],
            "mention_count": len(mentions_in_cluster),
            "su_count": len(cluster["supporting_su_ids"]),
            "passage_count": len(cluster["supporting_passage_ids"]),
            "embedding_text": embedding_text,
        }
        entity_nodes.append(entity_node)

        seen_sus = set()
        seen_passages = set()
        for mention in mentions_in_cluster:
            su_hash_id = mention["su_hash_id"]
            passage_hash_id = mention["passage_hash_id"]
            su_text = su_text_by_hash.get(su_hash_id)
            if su_text is None:
                continue
            if su_hash_id not in seen_sus:
                su_entities[su_text].append(embedding_text)
                seen_sus.add(su_hash_id)
            if passage_hash_id not in seen_passages:
                passage_entities[passage_hash_id].append(embedding_text)
                seen_passages.add(passage_hash_id)
            passage_entity_counts[passage_hash_id][embedding_text] += 1.0

    entity_nodes.sort(key=lambda node: node["embedding_text"])
    for passage_hash_id in passage_entities:
        passage_entities[passage_hash_id] = list(dict.fromkeys(passage_entities[passage_hash_id]))
    for su_text in su_entities:
        su_entities[su_text] = list(dict.fromkeys(su_entities[su_text]))

    logger.info(
        "Entity merge: %d mentions -> %d canonical entities",
        len(filtered_mentions),
        len(entity_nodes),
    )
    return entity_nodes, dict(passage_entities), dict(su_entities), {
        passage_hash_id: dict(weights)
        for passage_hash_id, weights in passage_entity_counts.items()
    }
