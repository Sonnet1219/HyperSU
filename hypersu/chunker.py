"""Semantic-aware chunking: fixed token target + semantic boundary splitting."""

import numpy as np
import tiktoken
import logging

logger = logging.getLogger(__name__)

_enc = tiktoken.encoding_for_model("gpt-4")


def _token_count(text):
    """Return token count for a string."""
    return len(_enc.encode(text))


def chunk_corpus_by_tokens(text, chunk_size=1200, overlap=100,
                           nlp_model=None, embedding_model=None,
                           search_window_ratio=0.2):
    """Split text into chunks of ~chunk_size tokens, cutting at semantic boundaries.

    When nlp_model and embedding_model are provided, the function:
      1. Splits the entire text into sentences via spaCy
      2. Accumulates sentences until approaching chunk_size tokens
      3. Within a search window (chunk_size ± search_window_ratio), finds the
         sentence boundary with the largest semantic gap (cosine distance)
      4. Cuts there, producing chunks that respect both token budget and semantics

    Falls back to pure fixed-window splitting if models are not provided.

    Args:
        text: Full corpus text
        chunk_size: Target token count per chunk (default 1200)
        overlap: Number of overlap sentences carried into the next chunk
        nlp_model: Loaded spaCy model for sentence segmentation (optional)
        embedding_model: SentenceTransformer model for embeddings (optional)
        search_window_ratio: Fraction of chunk_size to search around target (default 0.2)
    """
    if nlp_model is None or embedding_model is None:
        return _chunk_fixed(text, chunk_size, overlap)

    return _chunk_semantic(text, chunk_size, overlap, nlp_model, embedding_model,
                           search_window_ratio)


def _chunk_fixed(text, chunk_size, overlap):
    """Pure fixed-window token chunking (legacy fallback)."""
    tokens = _enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = _enc.decode(chunk_tokens)
        chunks.append(chunk_text)
        if end >= len(tokens):
            break
        start += chunk_size - overlap
    return chunks


def _chunk_semantic(text, chunk_size, overlap_sents, nlp_model, embedding_model,
                    search_window_ratio):
    """Fixed token target + semantic boundary chunking."""
    # 1. Sentence segmentation
    nlp_model.max_length = max(nlp_model.max_length, len(text) + 100)
    doc = nlp_model(text)
    sentences = [s.text.strip() for s in doc.sents if len(s.text.strip()) > 5]

    if len(sentences) == 0:
        return [text.strip()] if text.strip() else []

    # 2. Compute token count for each sentence
    sent_tokens = [_token_count(s) for s in sentences]

    # 3. Embed all sentences (for cosine distance computation)
    embeddings = embedding_model.encode(
        sentences, normalize_embeddings=True, show_progress_bar=False
    )

    # 4. Compute adjacent cosine distances
    distances = []
    for i in range(len(sentences) - 1):
        sim = float(np.dot(embeddings[i], embeddings[i + 1]))
        distances.append(1.0 - sim)
    # Sentinel for last sentence (no split after last)
    distances.append(0.0)

    # 5. Greedily build chunks, cutting at best semantic boundary near chunk_size
    lo = int(chunk_size * (1 - search_window_ratio))  # e.g. 960
    hi = int(chunk_size * (1 + search_window_ratio))  # e.g. 1440

    chunks = []
    start_idx = 0

    while start_idx < len(sentences):
        # Accumulate sentences until we exceed hi or run out
        cumulative = 0
        end_idx = start_idx
        while end_idx < len(sentences) and cumulative + sent_tokens[end_idx] <= hi:
            cumulative += sent_tokens[end_idx]
            end_idx += 1

        # If we've consumed all remaining sentences, emit final chunk
        if end_idx >= len(sentences):
            chunk_text = " ".join(sentences[start_idx:end_idx])
            chunks.append(chunk_text)
            break

        # Find candidate split points: sentence boundaries where cumulative tokens ∈ [lo, hi]
        # A split "after sentence i" means the chunk is sentences[start_idx:i+1]
        cumsum = 0
        candidates = []  # (sentence_index, cumulative_tokens, semantic_distance)
        for i in range(start_idx, end_idx):
            cumsum += sent_tokens[i]
            if lo <= cumsum <= hi and i + 1 < len(sentences):
                candidates.append((i, cumsum, distances[i]))

        if candidates:
            # Pick the boundary with the largest semantic gap
            best = max(candidates, key=lambda c: c[2])
            split_after = best[0]
        else:
            # No candidate in window — split at the last sentence that fits under chunk_size
            cumsum = 0
            split_after = start_idx
            for i in range(start_idx, end_idx):
                cumsum += sent_tokens[i]
                if cumsum <= chunk_size:
                    split_after = i

        chunk_text = " ".join(sentences[start_idx:split_after + 1])
        chunks.append(chunk_text)

        # Overlap: carry last `overlap_sents` sentences into next chunk
        start_idx = max(split_after + 1 - overlap_sents, split_after + 1)
        # Ensure forward progress
        if start_idx <= split_after:
            start_idx = split_after + 1

    logger.info("Semantic chunking: %d sentences -> %d chunks (target %d tokens)",
                len(sentences), len(chunks), chunk_size)
    return chunks


def _extract_sentences(text, nlp_model):
    """Extract sentences from text using spaCy sentence segmentation."""
    doc = nlp_model(text)
    return [s.text.strip() for s in doc.sents if len(s.text.strip()) > 10]


def _kamradt_semantic_units(sentences, embeddings, percentile=60):
    """
    Kamradt Percentile semantic chunking.

    Compute cosine distance between adjacent sentence embeddings.
    Split at distances above the given percentile threshold.
    Returns list of lists of sentence indices (groups).
    """
    if len(sentences) <= 1:
        return [list(range(len(sentences)))]

    # Compute adjacent cosine distances
    distances = []
    for i in range(len(sentences) - 1):
        sim = float(np.dot(embeddings[i], embeddings[i + 1]))
        distances.append(1.0 - sim)

    threshold = np.percentile(distances, percentile)

    units = []
    current = [0]
    for i in range(len(distances)):
        if distances[i] > threshold:
            units.append(current)
            current = [i + 1]
        else:
            current.append(i + 1)
    units.append(current)
    return units


def _balance_semantic_units(groups, sentences, embeddings, min_words, max_words):
    """Balance SU lengths by merging short units and splitting long ones.

    Pass 1 – Merge: repeatedly find the shortest SU (< min_words) and merge it
             with its more semantically similar neighbor.
    Pass 2 – Split: for any SU exceeding max_words, cut at the internal sentence
             boundary with the largest semantic gap; recurse if needed.

    Args:
        groups: list of lists of sentence indices (from Kamradt)
        sentences: list of sentence strings
        embeddings: normalized sentence embeddings (numpy array)
        min_words: minimum word count per SU
        max_words: maximum word count per SU

    Returns:
        list of lists of sentence indices (balanced groups)
    """

    def _word_count(group):
        return sum(len(sentences[i].split()) for i in group)

    def _group_embedding(group):
        emb = np.mean(embeddings[group], axis=0)
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 1e-8 else emb

    def _similarity(group_a, group_b):
        return float(np.dot(_group_embedding(group_a), _group_embedding(group_b)))

    # ── Pass 1: Merge short SUs ──
    while True:
        # Find the shortest SU below min_words
        shortest_idx = -1
        shortest_wc = min_words  # only merge if strictly below
        for i, g in enumerate(groups):
            wc = _word_count(g)
            if wc < shortest_wc:
                shortest_wc = wc
                shortest_idx = i
        if shortest_idx == -1:
            break

        # Decide merge direction by semantic similarity
        left_sim = (_similarity(groups[shortest_idx - 1], groups[shortest_idx])
                    if shortest_idx > 0 else -1.0)
        right_sim = (_similarity(groups[shortest_idx], groups[shortest_idx + 1])
                     if shortest_idx < len(groups) - 1 else -1.0)

        if left_sim >= right_sim and left_sim > -1.0:
            # Merge into left neighbor
            groups[shortest_idx - 1] = groups[shortest_idx - 1] + groups[shortest_idx]
            groups.pop(shortest_idx)
        elif right_sim > -1.0:
            # Merge into right neighbor
            groups[shortest_idx] = groups[shortest_idx] + groups[shortest_idx + 1]
            groups.pop(shortest_idx + 1)
        else:
            break  # single SU left, nothing to merge

    # ── Pass 2: Split long SUs ──
    def _split_group(group):
        if _word_count(group) <= max_words or len(group) <= 1:
            return [group]

        # Find internal sentence boundary with largest semantic gap
        best_dist = -1.0
        best_pos = -1
        for i in range(len(group) - 1):
            # Check that neither half would be below min_words
            left_half = group[:i + 1]
            right_half = group[i + 1:]
            if _word_count(left_half) < min_words or _word_count(right_half) < min_words:
                continue
            dist = 1.0 - float(np.dot(embeddings[group[i]], embeddings[group[i + 1]]))
            if dist > best_dist:
                best_dist = dist
                best_pos = i

        if best_pos == -1:
            # Cannot split without violating min_words, accept as-is
            return [group]

        left = group[:best_pos + 1]
        right = group[best_pos + 1:]
        return _split_group(left) + _split_group(right)

    balanced = []
    for g in groups:
        balanced.extend(_split_group(g))

    return balanced


def create_semantic_units(chunk_text, nlp_model, embedding_model, percentile=60,
                          min_words=20, max_words=200):
    """
    Split a chunk into semantic units using Kamradt Percentile method,
    then balance SU lengths to stay within [min_words, max_words].

    1. Extract sentences via spaCy
    2. Embed sentences with BGE
    3. Run Kamradt to group adjacent sentences
    4. Balance: merge short SUs into neighbors, split long SUs at semantic gaps
    5. Return list of semantic unit texts

    Args:
        chunk_text: Text of a single chunk (~1200 tokens)
        nlp_model: Loaded spaCy model for sentence segmentation
        embedding_model: SentenceTransformer model for embeddings
        percentile: Kamradt breakpoint percentile (default 60)
        min_words: Minimum word count per SU (default 20)
        max_words: Maximum word count per SU (default 200)

    Returns:
        list[str]: List of semantic unit texts (each is 1+ joined sentences)
    """
    sentences = _extract_sentences(chunk_text, nlp_model)

    if len(sentences) == 0:
        # Fallback: treat the whole chunk as one semantic unit
        return [chunk_text.strip()] if chunk_text.strip() else []

    if len(sentences) == 1:
        return sentences

    # Embed all sentences
    embeddings = embedding_model.encode(
        sentences, normalize_embeddings=True, show_progress_bar=False
    )

    # Run Kamradt grouping
    groups = _kamradt_semantic_units(sentences, embeddings, percentile)

    # Balance SU lengths
    groups = _balance_semantic_units(groups, sentences, embeddings, min_words, max_words)

    # Join sentences in each group into a semantic unit text
    su_texts = []
    for group in groups:
        su_text = " ".join(sentences[i] for i in group)
        su_texts.append(su_text)

    return su_texts
