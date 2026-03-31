"""GraphRAG-Bench benchmark data loading and result formatting."""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# HuggingFace cached dataset paths (downloaded via `datasets` or `hf_hub_download`)
_HF_CACHE_BASE = Path.home() / ".cache/huggingface/hub/datasets--GraphRAG-Bench--GraphRAG-Bench"
_SNAPSHOT_DIR = None  # resolved lazily


def _get_snapshot_dir() -> Path:
    """Resolve the latest snapshot directory from HuggingFace cache."""
    global _SNAPSHOT_DIR
    if _SNAPSHOT_DIR is not None:
        return _SNAPSHOT_DIR
    snapshots = _HF_CACHE_BASE / "snapshots"
    if not snapshots.exists():
        raise FileNotFoundError(
            f"GraphRAG-Bench dataset not found at {snapshots}. "
            "Please download it first: "
            "huggingface-cli download GraphRAG-Bench/GraphRAG-Bench --repo-type dataset"
        )
    # Use the latest snapshot (there's usually just one)
    dirs = sorted(snapshots.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    if not dirs:
        raise FileNotFoundError(f"No snapshots found in {snapshots}")
    _SNAPSHOT_DIR = dirs[0]
    return _SNAPSHOT_DIR


CORPUS_FILES = {
    "medical": "Datasets/Corpus/medical.json",
    "novel": "Datasets/Corpus/novel.json",
}
QUESTION_FILES = {
    "medical": "Datasets/Questions/medical_questions.json",
    "novel": "Datasets/Questions/novel_questions.json",
}


def load_corpus(corpus_name: str) -> list[dict]:
    """Load corpus entries from local GraphRAG-Bench cache.

    Each entry has keys: corpus_name, context.
    - medical.json: single dict -> wrapped as one-element list
    - novel.json: list of 20 dicts, one per novel

    Args:
        corpus_name: "medical" or "novel"

    Returns:
        List of dicts with "corpus_name" and "context" keys
    """
    snapshot = _get_snapshot_dir()
    path = snapshot / CORPUS_FILES[corpus_name]
    with open(path, "r") as f:
        data = json.load(f)
    # medical.json is a single dict, novel.json is a list
    if isinstance(data, dict):
        data = [data]
    logger.info("Loaded %s corpus: %d entries", corpus_name, len(data))
    for entry in data:
        logger.info("  %s: %d chars", entry["corpus_name"], len(entry["context"]))
    return data


def load_questions(corpus_name: str) -> list[dict]:
    """Load questions from local GraphRAG-Bench cache.

    Each question dict has: id, source, question, answer, question_type,
    evidence, evidence_relations

    Args:
        corpus_name: "medical" or "novel"

    Returns:
        List of question dicts
    """
    snapshot = _get_snapshot_dir()
    path = snapshot / QUESTION_FILES[corpus_name]
    with open(path, "r") as f:
        questions = json.load(f)
    logger.info("Loaded %d %s questions", len(questions), corpus_name)

    type_counts = {}
    for q in questions:
        qt = q.get("question_type", "unknown")
        type_counts[qt] = type_counts.get(qt, 0) + 1
    for qt, count in sorted(type_counts.items()):
        logger.info("  %s: %d questions", qt, count)

    return questions


def format_results(retrieval_results: list[dict], questions: list[dict]) -> list[dict]:
    """Convert HyperSU output to GraphRAG-Bench unified JSON format.

    Output matches the official format expected by:
        python -m Evaluation.generation_eval --data_file <output>
        python -m Evaluation.retrieval_eval --data_file <output>

    Required fields: id, question, source, context (List[str]),
    evidence (str), question_type, generated_answer, ground_truth

    Args:
        retrieval_results: Output from HyperSU qa() -- list of dicts with
            question, sorted_passage, pred_answer, gold_answer
        questions: Original question dicts with id, source, question_type, evidence

    Returns:
        List of dicts in GraphRAG-Bench unified format
    """
    formatted = []
    for result, question in zip(retrieval_results, questions):
        formatted.append({
            "id": question["id"],
            "question": result["question"],
            "source": question["source"],
            "context": result.get("sorted_passage", []),  # List[str]
            "evidence": question.get("evidence", ""),
            "question_type": question.get("question_type", ""),
            "generated_answer": result.get("pred_answer", ""),
            "ground_truth": question["answer"],
        })
    return formatted
