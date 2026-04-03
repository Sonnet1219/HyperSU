"""SQuAD-style EM / F1 evaluation aligned with HippoRAG.

Reference: MRQA official eval & HippoRAG src/hipporag/evaluation/qa_eval.py
"""

import re
import string
from collections import Counter
from typing import List, Set, Union

import numpy as np


# ---------------------------------------------------------------------------
# Text normalisation — identical to HippoRAG's eval_utils.normalize_answer
# ---------------------------------------------------------------------------

def normalize_answer(answer: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(answer))))


# ---------------------------------------------------------------------------
# Exact Match — identical to HippoRAG QAExactMatch
# ---------------------------------------------------------------------------

def compute_exact_match(gold: str, predicted: str) -> float:
    return 1.0 if normalize_answer(gold) == normalize_answer(predicted) else 0.0


# ---------------------------------------------------------------------------
# Token-level F1 — identical to HippoRAG QAF1Score.compute_f1
# ---------------------------------------------------------------------------

def compute_f1(gold: str, predicted: str) -> float:
    gold_tokens = normalize_answer(gold).split()
    predicted_tokens = normalize_answer(predicted).split()
    common = Counter(predicted_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(predicted_tokens)
    recall = 1.0 * num_same / len(gold_tokens)
    return 2 * (precision * recall) / (precision + recall)


# ---------------------------------------------------------------------------
# Multi-answer aggregation — np.max across all gold answers per question,
# exactly as HippoRAG does.
# ---------------------------------------------------------------------------

def evaluate(
    gold_answers: List[Union[List[str], Set[str]]],
    predicted_answers: List[str],
):
    """Compute corpus-level EM and F1.

    Args:
        gold_answers: Per-question list/set of acceptable gold answers.
        predicted_answers: Per-question predicted answer string.

    Returns:
        (avg_em, avg_f1, per_example): corpus-level averages and per-example
        dicts with keys ``EM`` and ``F1``.
    """
    assert len(gold_answers) == len(predicted_answers), (
        f"Length mismatch: {len(gold_answers)} gold vs {len(predicted_answers)} predicted"
    )

    per_example = []
    total_em = 0.0
    total_f1 = 0.0

    for gold_list, predicted in zip(gold_answers, predicted_answers):
        gold_list = list(gold_list)
        em_scores = [compute_exact_match(g, predicted) for g in gold_list]
        f1_scores = [compute_f1(g, predicted) for g in gold_list]

        best_em = float(np.max(em_scores))
        best_f1 = float(np.max(f1_scores))

        per_example.append({"EM": best_em, "F1": best_f1})
        total_em += best_em
        total_f1 += best_f1

    n = len(gold_answers) if gold_answers else 1
    avg_em = total_em / n
    avg_f1 = total_f1 / n

    return avg_em, avg_f1, per_example
