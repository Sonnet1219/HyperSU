"""Run HyperSU on multi-hop QA datasets (HotpotQA, 2WikiMultiHop, MuSiQue)."""

import argparse
import json
import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

# Ensure project root is on sys.path so `hypersu` is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from hypersu.engine import HyperSU
from hypersu.llm import LLMClient
from hypersu.utils import setup_logging
from evaluate import Evaluator

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")

# HuggingFace cached dataset path
_HF_CACHE_BASE = Path.home() / ".cache/huggingface/hub/datasets--Zly0523--linear-rag"
_SNAPSHOT_DIR = None

AVAILABLE_DATASETS = ("hotpotqa", "2wikimultihop", "musique")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run HyperSU on multi-hop QA datasets")
    parser.add_argument("--spacy_model", type=str, default="en_core_web_trf", help="The spacy model to use")
    parser.add_argument("--embedding_model", type=str, default="BAAI/bge-large-en-v1.5", help="The path of embedding model to use")
    parser.add_argument("--dataset_name", type=str, default="novel", help="The dataset to use")
    parser.add_argument("--llm_model", type=str, default="gpt-4o-mini", help="The LLM model to use")
    parser.add_argument("--max_workers", type=int, default=16, help="The max number of workers to use")
    parser.add_argument("--ner_model", type=str, default="gpt-4o-mini", help="The NER extraction model to use")
    parser.add_argument("--expansion_max_hops", type=int, default=3, help="Max BFS hops from seed entities")
    parser.add_argument("--expansion_top_k", type=int, default=15, help="New entities discovered per hop")
    parser.add_argument("--hop_decay", type=float, default=0.5, help="Score decay per hop")
    parser.add_argument("--scoring_lambda", type=float, default=0.7, help="Dual-channel fusion weight")
    parser.add_argument("--reranker_model", type=str, default="Qwen/Qwen3-Reranker-4B", help="Local or Hub path for the reranker model")
    parser.add_argument("--reranker_candidate_top_k", type=int, default=30, help="How many retrieval candidates to pass into the reranker")
    parser.add_argument("--reranker_batch_size", type=int, default=16, help="Batch size for reranker scoring")
    parser.add_argument("--reranker_max_length", type=int, default=4096, help="Max token length for reranker inputs")
    parser.add_argument("--question_offset", type=int, default=0, help="Start index of questions (0-based)")
    parser.add_argument("--question_limit", type=int, default=None, help="Optional limit for the number of questions to run")
    parser.add_argument("--skip_evaluation", action="store_true", help="Skip the answer evaluation stage")
    parser.add_argument("--use_reranker", action="store_true", help="Enable reranker (default: off)")
    parser.add_argument("--use_planner", action="store_true", help="Enable sub-query decomposition via planner")
    parser.add_argument("--planner_model", type=str, default="gpt-4o-mini", help="LLM model for the query planner")
    parser.add_argument("--planner_max_subqueries", type=int, default=5, help="Maximum sub-queries per question")
    return parser.parse_args()


def _get_snapshot_dir() -> Path:
    """Resolve the latest snapshot directory from HuggingFace cache."""
    global _SNAPSHOT_DIR
    if _SNAPSHOT_DIR is not None:
        return _SNAPSHOT_DIR
    snapshots = _HF_CACHE_BASE / "snapshots"
    if not snapshots.exists():
        raise FileNotFoundError(
            f"Multi-hop QA dataset not found at {snapshots}. "
            "Please download it first: "
            "huggingface-cli download Zly0523/linear-rag --repo-type dataset"
        )
    dirs = sorted(snapshots.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    if not dirs:
        raise FileNotFoundError(f"No snapshots found in {snapshots}")
    _SNAPSHOT_DIR = dirs[0]
    return _SNAPSHOT_DIR


def load_dataset(dataset_name):
    """Load dataset from HuggingFace cache (Zly0523/linear-rag)."""
    snapshot = _get_snapshot_dir()
    dataset_dir = snapshot / dataset_name
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset '{dataset_name}' not found in {snapshot}. "
            f"Available: {AVAILABLE_DATASETS}"
        )
    with open(dataset_dir / "questions.json", "r", encoding="utf-8") as f:
        questions = json.load(f)
    with open(dataset_dir / "chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    passages = [f'{idx}:{chunk}' for idx, chunk in enumerate(chunks)]
    logger.info("Loaded %s: %d questions, %d passages", dataset_name, len(questions), len(passages))
    return questions, passages


def main():
    time = datetime.now()
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
    args = parse_arguments()
    output_dir = os.path.join(PROJECT_ROOT, f"results/{args.dataset_name}/{time_str}")
    questions, passages = load_dataset(args.dataset_name)
    if args.question_offset > 0:
        questions = questions[args.question_offset:]
    if args.question_limit is not None:
        questions = questions[:args.question_limit]
    setup_logging(f"{output_dir}/log.txt")

    model = HyperSU(
        save_dir=os.path.join(PROJECT_ROOT, f"index_store/{args.dataset_name}"),
        llm_model_name=args.llm_model,
        embedding_model_name=args.embedding_model,
        spacy_model=args.spacy_model,
        max_workers=args.max_workers,
        ner_model_id=args.ner_model,
        expansion_max_hops=args.expansion_max_hops,
        expansion_top_k=args.expansion_top_k,
        hop_decay=args.hop_decay,
        scoring_lambda=args.scoring_lambda,
        reranker_model_name=args.reranker_model,
        reranker_candidate_top_k=args.reranker_candidate_top_k,
        reranker_batch_size=args.reranker_batch_size,
        reranker_max_length=args.reranker_max_length,
        use_reranker=args.use_reranker,
        use_planner=args.use_planner,
        planner_model_name=args.planner_model,
        planner_max_subqueries=args.planner_max_subqueries,
    )
    model.index(passages)
    queries = [q["question"] for q in questions]
    results = model.rag_qa(queries)
    # Attach gold answers for evaluation
    predictions = []
    for result, q in zip(results, questions):
        predictions.append({
            "question": result["query"],
            "pred_answer": result["answer"],
            "gold_answer": q["answer"],
            "sorted_passage": result["passages"],
            "sorted_passage_scores": result["scores"],
        })
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/predictions.json", "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=4)
    if not args.skip_evaluation:
        llm_model = LLMClient(args.llm_model)
        evaluator = Evaluator(llm_model=llm_model, predictions_path=f"{output_dir}/predictions.json")
        evaluator.evaluate(max_workers=args.max_workers)


if __name__ == "__main__":
    main()
