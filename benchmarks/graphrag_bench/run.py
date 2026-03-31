"""Run HyperSU on GraphRAG-Bench benchmark.

Pipeline:
  1. Load corpus entries & questions from local HuggingFace cache
  2. For each corpus entry: chunk -> index -> retrieve + QA for matching questions
  3. Save predictions in official unified JSON format
  4. Run official Evaluation scripts (generation_eval + retrieval_eval)
"""

import argparse
import json
import os
import subprocess
import sys
import warnings
from collections import defaultdict
from datetime import datetime
from dotenv import load_dotenv

# Ensure project root is on sys.path so `hypersu` is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

from hypersu.engine import HyperSU
from hypersu.chunker import chunk_corpus_by_tokens
from hypersu.utils import setup_logging
from bench import load_corpus, load_questions, format_results

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")

# Path to the cloned official GraphRAG-Benchmark repo
EVAL_REPO_DIR = os.path.expanduser("~/GraphRAG-Benchmark")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run HyperSU on GraphRAG-Bench",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Corpus ---
    parser.add_argument("--corpus_name", type=str, default="medical",
                        choices=["medical", "novel"], help="Which corpus to use")
    parser.add_argument("--corpus_entry", type=str, default=None,
                        help="Optional single corpus entry to process, e.g. Novel-30752")

    # --- Models ---
    parser.add_argument("--embedding_model", type=str, default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--llm_model", type=str, default=os.getenv("LLM_MODEL_NAME", "gpt-4o-mini"))
    parser.add_argument("--spacy_model", type=str, default="en_core_web_trf")

    # --- Entity extraction ---
    parser.add_argument("--langextract_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--langextract_max_char_buffer", type=int, default=1000)
    parser.add_argument("--langextract_extraction_passes", type=int, default=1)

    # --- Chunking ---
    parser.add_argument("--chunk_size", type=int, default=1200, help="Token size for corpus chunking")
    parser.add_argument("--chunk_overlap", type=int, default=100, help="Token overlap between chunks")
    parser.add_argument("--semantic_unit_percentile", type=int, default=60,
                        help="Kamradt percentile for semantic unit boundary detection")

    # --- Retrieval ---
    parser.add_argument("--max_workers", type=int, default=int(os.getenv("MAX_WORKERS", "16")))
    parser.add_argument("--expansion_max_hops", type=int, default=3, help="Max BFS hops from seed entities")
    parser.add_argument("--expansion_top_k", type=int, default=15, help="New entities discovered per hop")
    parser.add_argument("--hop_decay", type=float, default=0.5, help="Score decay per hop")
    parser.add_argument("--conductance_floor", type=float, default=0.5, help="SU conductance floor")
    parser.add_argument("--conductance_gamma", type=float, default=1.0, help="SU conductance power exponent")
    parser.add_argument("--scoring_lambda", type=float, default=0.7, help="Dual-channel fusion weight (1.0=dense, 0.0=entity)")

    # --- Reranker ---
    parser.add_argument("--rerank", action="store_true",
                        help="Enable reranking")
    parser.add_argument("--reranker_model", type=str, default="Qwen/Qwen3-Reranker-4B")
    parser.add_argument("--reranker_candidate_top_k", type=int, default=30)
    parser.add_argument("--reranker_batch_size", type=int, default=16)
    parser.add_argument("--reranker_max_length", type=int, default=4096)

    # --- Question filter ---
    parser.add_argument("--question_types", type=str, nargs="+", default=None,
                        help="Only run these question types (e.g. 'Complex Reasoning')")
    parser.add_argument("--question_limit", type=int, default=None,
                        help="Limit number of questions (for testing)")

    # --- Pipeline control ---
    parser.add_argument("--skip_qa", action="store_true",
                        help="Skip QA generation (only do indexing)")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Skip official evaluation (only generate predictions)")

    # --- Evaluation settings ---
    parser.add_argument("--eval_mode", type=str, default="API",
                        choices=["API", "ollama"], help="Evaluation LLM mode")
    parser.add_argument("--eval_model", type=str, default=None,
                        help="LLM for evaluation (defaults to --llm_model)")
    parser.add_argument("--eval_base_url", type=str, default=None,
                        help="Base URL for evaluation LLM (defaults to OPENAI_BASE_URL)")
    parser.add_argument("--eval_embedding_model", type=str, default="BAAI/bge-large-en-v1.5",
                        help="Embedding model for evaluation")
    parser.add_argument("--eval_num_samples", type=int, default=None,
                        help="Limit samples per question type for evaluation")

    return parser.parse_args()


def run_official_eval(predictions_path: str, output_dir: str, args):
    """Run official GraphRAG-Benchmark evaluation scripts."""
    eval_model = args.eval_model or args.llm_model
    eval_base_url = args.eval_base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    # Set LLM_API_KEY for official eval (it reads this env var)
    eval_env = os.environ.copy()
    eval_env["LLM_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
    # Ensure the Evaluation package is importable
    eval_env["PYTHONPATH"] = EVAL_REPO_DIR + os.pathsep + eval_env.get("PYTHONPATH", "")

    gen_output = os.path.abspath(os.path.join(output_dir, "generation_eval.json"))
    ret_output = os.path.abspath(os.path.join(output_dir, "retrieval_eval.json"))

    base_cmd = [
        sys.executable, "-m",
    ]

    gen_cmd = base_cmd + [
        "Evaluation.generation_eval",
        "--mode", args.eval_mode,
        "--model", eval_model,
        "--base_url", eval_base_url,
        "--embedding_model", args.eval_embedding_model,
        "--data_file", predictions_path,
        "--output_file", gen_output,
        "--detailed_output",
    ]

    ret_cmd = base_cmd + [
        "Evaluation.retrieval_eval",
        "--mode", args.eval_mode,
        "--model", eval_model,
        "--base_url", eval_base_url,
        "--embedding_model", args.eval_embedding_model,
        "--data_file", predictions_path,
        "--output_file", ret_output,
        "--detailed_output",
    ]

    if args.eval_num_samples:
        gen_cmd += ["--num_samples", str(args.eval_num_samples)]
        ret_cmd += ["--num_samples", str(args.eval_num_samples)]

    # Run from the official repo directory so imports resolve
    print("\n" + "=" * 60)
    print("Running official generation evaluation...")
    print("=" * 60)
    result = subprocess.run(gen_cmd, cwd=EVAL_REPO_DIR, env=eval_env,
                            capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"WARNING: generation_eval exited with code {result.returncode}")
        print(result.stderr[-1000:] if result.stderr else "")

    print("\n" + "=" * 60)
    print("Running official retrieval evaluation...")
    print("=" * 60)
    result = subprocess.run(ret_cmd, cwd=EVAL_REPO_DIR, env=eval_env,
                            capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"WARNING: retrieval_eval exited with code {result.returncode}")
        print(result.stderr[-1000:] if result.stderr else "")

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    for label, path in [("Generation", gen_output), ("Retrieval", ret_output)]:
        if os.path.exists(path):
            with open(path) as f:
                results = json.load(f)
            print(f"\n--- {label} Metrics ---")
            for qtype, metrics in results.items():
                scores = metrics.get("average_scores", metrics)
                print(f"  {qtype}:")
                for metric, score in scores.items():
                    if isinstance(score, float):
                        print(f"    {metric}: {score:.4f}")
                    else:
                        print(f"    {metric}: {score}")
        else:
            print(f"\n--- {label}: results file not found ---")


def main():
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args = parse_arguments()
    output_dir = os.path.join(PROJECT_ROOT, f"results/graphrag-bench/{args.corpus_name}/{time_str}")
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(f"{output_dir}/log.txt")

    # Load corpus entries and questions
    corpus_entries = load_corpus(args.corpus_name)
    all_questions = load_questions(args.corpus_name)

    if args.corpus_entry is not None:
        corpus_entries = [entry for entry in corpus_entries if entry["corpus_name"] == args.corpus_entry]
        all_questions = [q for q in all_questions if q.get("source") == args.corpus_entry]
        if not corpus_entries:
            raise ValueError(
                f"Corpus entry '{args.corpus_entry}' not found in {args.corpus_name} corpus."
            )
        print(f"Restricted to corpus entry: {args.corpus_entry}")

    # Filter question types
    if args.question_types is not None:
        allowed = set(args.question_types)
        all_questions = [q for q in all_questions if q.get("question_type") in allowed]
        print(f"Filtered to question types: {args.question_types}")

    # Apply question limit
    if args.question_limit is not None:
        all_questions = all_questions[:args.question_limit]
    print(f"Total questions to process: {len(all_questions)}")

    # Group questions by source -> corpus_name
    questions_by_source = defaultdict(list)
    for q in all_questions:
        questions_by_source[q["source"]].append(q)

    # Map corpus_name -> corpus entry
    corpus_by_name = {entry["corpus_name"]: entry["context"] for entry in corpus_entries}

    # Find which corpus entries have matching questions
    if args.skip_qa and args.corpus_entry is not None:
        sources_needed = {args.corpus_entry}
    else:
        sources_needed = set(questions_by_source.keys())
    sources_available = set(corpus_by_name.keys())
    missing = sources_needed - sources_available
    if missing:
        print(f"WARNING: Questions reference unknown sources: {missing}")

    all_formatted = []

    for corpus_name in sorted(sources_needed & sources_available):
        questions = questions_by_source[corpus_name]
        corpus_text = corpus_by_name[corpus_name]
        print(f"\n{'='*60}")
        print(f"Processing {corpus_name}: {len(questions)} questions, {len(corpus_text)} chars")
        print(f"{'='*60}")

        # Create HyperSU instance with per-corpus index
        model = HyperSU(
            save_dir=os.path.join(PROJECT_ROOT, f"index_store/graphrag-bench-{args.corpus_name}/{corpus_name}"),
            llm_model_name=args.llm_model,
            embedding_model_name=args.embedding_model,
            spacy_model=args.spacy_model,
            max_workers=args.max_workers,
            chunk_token_size=args.chunk_size,
            chunk_overlap_token_size=args.chunk_overlap,
            semantic_unit_percentile=args.semantic_unit_percentile,
            expansion_max_hops=args.expansion_max_hops,
            expansion_top_k=args.expansion_top_k,
            hop_decay=args.hop_decay,
            conductance_floor=args.conductance_floor,
            conductance_gamma=args.conductance_gamma,
            scoring_lambda=args.scoring_lambda,
            use_reranker=args.rerank,
            reranker_model_name=args.reranker_model,
            reranker_candidate_top_k=args.reranker_candidate_top_k,
            reranker_batch_size=args.reranker_batch_size,
            reranker_max_length=args.reranker_max_length,
            langextract_model_id=args.langextract_model,
            langextract_max_char_buffer=args.langextract_max_char_buffer,
            langextract_extraction_passes=args.langextract_extraction_passes,
        )

        # Chunk corpus with semantic boundary awareness
        chunks = chunk_corpus_by_tokens(
            corpus_text, args.chunk_size, args.chunk_overlap,
            nlp_model=model.spacy_model,
            embedding_model=model.embedding_model,
        )
        passages = list(chunks)
        print(f"  Chunked into {len(passages)} passages")

        # Index
        model.index(passages)
        print(f"  Indexing complete for {corpus_name}.")

        if args.skip_qa:
            continue

        # QA
        queries = [q["question"] for q in questions]
        results = model.rag_qa(queries, qa_mode="graphrag_bench")

        # Format results
        legacy_results = []
        for result, q in zip(results, questions):
            legacy_results.append({
                "question": result["query"],
                "sorted_passage": result["passages"],
                "pred_answer": result["answer"],
                "gold_answer": q["answer"],
            })
        formatted = format_results(legacy_results, questions)
        all_formatted.extend(formatted)
        print(f"  QA complete for {corpus_name}: {len(formatted)} results")

        # Free model to reclaim GPU memory before next corpus
        import gc
        del model
        gc.collect()
        import torch
        torch.cuda.empty_cache()

    if args.skip_qa:
        print("Skipping QA generation (--skip_qa). Done.")
        return

    # Save all predictions
    predictions_path = os.path.join(output_dir, "predictions.json")
    with open(predictions_path, "w", encoding="utf-8") as f:
        json.dump(all_formatted, f, ensure_ascii=False, indent=2)
    print(f"\nPredictions saved to {predictions_path} ({len(all_formatted)} total)")

    # Run official evaluation
    if not args.skip_eval:
        if not os.path.isdir(EVAL_REPO_DIR):
            print(f"WARNING: Official eval repo not found at {EVAL_REPO_DIR}")
            print("Clone it: git clone https://github.com/GraphRAG-Bench/GraphRAG-Benchmark.git ~/GraphRAG-Benchmark")
            print("Skipping evaluation.")
        else:
            run_official_eval(os.path.abspath(predictions_path), output_dir, args)
    else:
        print("Skipping evaluation (--skip_eval).")


if __name__ == "__main__":
    main()
