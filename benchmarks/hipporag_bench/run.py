"""Run HyperSU on HippoRAG datasets (HotpotQA, 2WikiMultiHopQA, MuSiQue).

Datasets: osunlp/HippoRAG_v2 (HuggingFace)
Evaluation: SQuAD-style EM / F1 aligned with HippoRAG
"""

import argparse
import json
import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from hypersu.engine import HyperSU
from hypersu.utils import setup_logging
from evaluate import evaluate

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")

AVAILABLE_DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")

# ---------------------------------------------------------------------------
# HippoRAG dataset path resolution
# ---------------------------------------------------------------------------

_HF_CACHE_BASE = Path.home() / ".cache/huggingface/hub/datasets--osunlp--HippoRAG_v2"


def _get_snapshot_dir() -> Path:
    snapshots = _HF_CACHE_BASE / "snapshots"
    if not snapshots.exists():
        raise FileNotFoundError(
            f"HippoRAG dataset not found at {snapshots}. "
            "Download it first:\n"
            "  python -c \"from huggingface_hub import snapshot_download; "
            "snapshot_download('osunlp/HippoRAG_v2', repo_type='dataset')\""
        )
    dirs = sorted(snapshots.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    if not dirs:
        raise FileNotFoundError(f"No snapshots found in {snapshots}")
    return dirs[0]


# ---------------------------------------------------------------------------
# Gold answer extraction — aligned with HippoRAG main.py get_gold_answers()
# ---------------------------------------------------------------------------

def get_gold_answers(samples):
    """Extract gold answers per question, matching HippoRAG's logic exactly."""
    gold_answers = []
    for sample in samples:
        gold_ans = None
        if "answer" in sample or "gold_ans" in sample:
            gold_ans = sample["answer"] if "answer" in sample else sample["gold_ans"]
        elif "reference" in sample:
            gold_ans = sample["reference"]
        assert gold_ans is not None, f"No gold answer found in sample: {list(sample.keys())}"

        if isinstance(gold_ans, str):
            gold_ans = [gold_ans]
        assert isinstance(gold_ans, list)

        gold_ans = set(gold_ans)
        if "answer_aliases" in sample:
            gold_ans.update(sample["answer_aliases"])

        gold_answers.append(gold_ans)
    return gold_answers


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(dataset_name):
    snapshot = _get_snapshot_dir()
    questions_path = snapshot / f"{dataset_name}.json"
    corpus_path = snapshot / f"{dataset_name}_corpus.json"

    if not questions_path.exists():
        raise FileNotFoundError(
            f"Dataset '{dataset_name}' not found at {questions_path}. "
            f"Available: {AVAILABLE_DATASETS}"
        )

    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    # Build passages: "idx:title\ntext" for adjacency linking
    passages = []
    for i, item in enumerate(corpus):
        title = item.get("title", "")
        text = item.get("text", "")
        passages.append(f"{i}:{title}\n{text}")

    logger.info(
        "Loaded %s: %d questions, %d passages",
        dataset_name, len(questions), len(passages),
    )
    return questions, passages


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run HyperSU on HippoRAG datasets")
    parser.add_argument("--dataset_name", type=str, default="hotpotqa",
                        choices=AVAILABLE_DATASETS,
                        help="Dataset to evaluate on")
    parser.add_argument("--spacy_model", type=str, default="en_core_web_trf")
    parser.add_argument("--embedding_model", type=str, default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--llm_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--max_workers", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Embedding encode batch size (reduce if OOM)")
    parser.add_argument("--ner_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--expansion_max_hops", type=int, default=3)
    parser.add_argument("--expansion_top_k", type=int, default=15)
    parser.add_argument("--hop_decay", type=float, default=0.5)
    parser.add_argument("--scoring_lambda", type=float, default=0.7)
    parser.add_argument("--use_reranker", action="store_true")
    parser.add_argument("--reranker_model", type=str, default="Qwen/Qwen3-Reranker-4B")
    parser.add_argument("--reranker_candidate_top_k", type=int, default=30)
    parser.add_argument("--reranker_batch_size", type=int, default=16)
    parser.add_argument("--reranker_max_length", type=int, default=4096)
    parser.add_argument("--use_planner", action="store_true")
    parser.add_argument("--planner_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--planner_max_subqueries", type=int, default=5)
    parser.add_argument("--disable_backward", action="store_true",
                        help="Ablation: disable backward expansion (forward only)")
    parser.add_argument("--question_offset", type=int, default=0,
                        help="Start index of questions (0-based)")
    parser.add_argument("--question_limit", type=int, default=None,
                        help="Max number of questions to run QA on")
    parser.add_argument("--skip_index", action="store_true",
                        help="Skip indexing (reuse existing index_store)")
    parser.add_argument("--index_only", action="store_true",
                        help="Only build the index, skip QA and evaluation")
    parser.add_argument("--skip_evaluation", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args = parse_arguments()

    output_dir = os.path.join(
        PROJECT_ROOT, f"results/hipporag_bench/{args.dataset_name}/{time_str}"
    )
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(f"{output_dir}/log.txt")

    questions, passages = load_dataset(args.dataset_name)

    model = HyperSU(
        save_dir=os.path.join(PROJECT_ROOT, f"index_store/hipporag_{args.dataset_name}"),
        llm_model_name=args.llm_model,
        embedding_model_name=args.embedding_model,
        spacy_model=args.spacy_model,
        max_workers=args.max_workers,
        batch_size=args.batch_size,
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
        disable_backward=args.disable_backward,
    )

    # ── Index (full corpus, skip if already built) ──
    if args.skip_index:
        logger.info("Skipping index (--skip_index). Loading existing index from %s",
                     model.config.save_dir)
        model.index(passages)  # still needed to load embeddings & build hypergraph
        # TODO: add a lightweight load-only path if index is expensive to rebuild
    else:
        model.index(passages)

    if args.index_only:
        logger.info("Index complete. Exiting (--index_only).")
        return

    # ── Slice questions for QA ──
    if args.question_offset > 0:
        questions = questions[args.question_offset:]
    if args.question_limit is not None:
        questions = questions[:args.question_limit]

    logger.info(
        "Running QA on %d questions (offset=%d, limit=%s)",
        len(questions), args.question_offset, args.question_limit,
    )

    queries = [q["question"] for q in questions]
    results = model.rag_qa(queries)

    # ── Gold answers (HippoRAG-aligned) ──
    gold_answers = get_gold_answers(questions)

    # ── Save predictions ──
    predictions = []
    for result, q, golds in zip(results, questions, gold_answers):
        predictions.append({
            "question": result["query"],
            "pred_answer": result["answer"],
            "gold_answers": list(golds),
            "passages": result["passages"],
            "scores": result["scores"],
        })

    with open(f"{output_dir}/predictions.json", "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=4)

    # ── Evaluate ──
    if not args.skip_evaluation:
        predicted_answers = [p["pred_answer"] for p in predictions]
        avg_em, avg_f1, per_example = evaluate(gold_answers, predicted_answers)

        logger.info("Evaluation Results:")
        logger.info("  EM:  %.4f", avg_em)
        logger.info("  F1:  %.4f", avg_f1)

        # Attach per-example scores
        for pred, scores in zip(predictions, per_example):
            pred["EM"] = scores["EM"]
            pred["F1"] = scores["F1"]

        with open(f"{output_dir}/predictions.json", "w", encoding="utf-8") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=4)

        eval_results = {
            "dataset": args.dataset_name,
            "num_questions": len(predictions),
            "question_offset": args.question_offset,
            "question_limit": args.question_limit,
            "EM": avg_em,
            "F1": avg_f1,
            "config": {
                "llm_model": args.llm_model,
                "embedding_model": args.embedding_model,
                "expansion_max_hops": args.expansion_max_hops,
                "expansion_top_k": args.expansion_top_k,
                "scoring_lambda": args.scoring_lambda,
                "use_reranker": args.use_reranker,
                "use_planner": args.use_planner,
                "disable_backward": args.disable_backward,
            },
        }
        with open(f"{output_dir}/evaluation_results.json", "w", encoding="utf-8") as f:
            json.dump(eval_results, f, ensure_ascii=False, indent=4)

        print(f"\n{'='*50}")
        print(f"Dataset: {args.dataset_name}")
        print(f"Questions: {len(predictions)} (offset={args.question_offset})")
        print(f"EM:  {avg_em:.4f} ({avg_em*100:.2f}%)")
        print(f"F1:  {avg_f1:.4f} ({avg_f1*100:.2f}%)")
        print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
