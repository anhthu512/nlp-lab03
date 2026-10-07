"""Core word-context representation for LAB 03.

This module intentionally implements the required co-occurrence pipeline without
using Word2Vec.  Word2Vec is used only in the experiment notebook.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence

import numpy as np
from scipy.sparse import csr_matrix


def build_vocabulary(
    corpus: Iterable[Sequence[str]],
    min_count: int = 1,
    max_size: int | None = None,
) -> tuple[dict[str, int], list[str]]:
    """Build a deterministic token-to-index mapping from a tokenized corpus."""
    if min_count < 1:
        raise ValueError("min_count must be at least 1")
    counts = Counter(token for sentence in corpus for token in sentence)
    items = [(token, count) for token, count in counts.items() if count >= min_count]
    items.sort(key=lambda item: (-item[1], item[0]))
    if max_size is not None:
        if max_size < 1:
            raise ValueError("max_size must be positive")
        items = items[:max_size]
    index_to_word = [token for token, _ in items]
    vocabulary = {token: index for index, token in enumerate(index_to_word)}
    return vocabulary, index_to_word


def build_cooccurrence_matrix(
    corpus: Iterable[Sequence[str]],
    vocabulary: dict[str, int],
    window_size: int = 1,
) -> csr_matrix:
    """Count symmetric target-context pairs inside a fixed context window."""
    if window_size < 1:
        raise ValueError("window_size must be at least 1")

    pair_counts: Counter[tuple[int, int]] = Counter()
    for sentence in corpus:
        ids = [vocabulary.get(token) for token in sentence]
        for target_pos, target_id in enumerate(ids):
            if target_id is None:
                continue
            left = max(0, target_pos - window_size)
            right = min(len(ids), target_pos + window_size + 1)
            for context_pos in range(left, right):
                if context_pos == target_pos:
                    continue
                context_id = ids[context_pos]
                if context_id is not None:
                    pair_counts[(target_id, context_id)] += 1

    size = len(vocabulary)
    if not pair_counts:
        return csr_matrix((size, size), dtype=np.float64)
    rows, columns, values = zip(
        *((row, column, float(value)) for (row, column), value in pair_counts.items())
    )
    return csr_matrix((values, (rows, columns)), shape=(size, size), dtype=np.float64)


def cosine_similarity(vector_x, vector_y) -> float:
    """Return cosine similarity for dense arrays or scipy sparse row vectors."""
    if hasattr(vector_x, "multiply"):
        dot = float(vector_x.multiply(vector_y).sum())
        norm_x = float(np.sqrt(vector_x.multiply(vector_x).sum()))
        norm_y = float(np.sqrt(vector_y.multiply(vector_y).sum()))
    else:
        x = np.asarray(vector_x, dtype=np.float64).ravel()
        y = np.asarray(vector_y, dtype=np.float64).ravel()
        if x.shape != y.shape:
            raise ValueError("vectors must have the same shape")
        dot = float(np.dot(x, y))
        norm_x = float(np.linalg.norm(x))
        norm_y = float(np.linalg.norm(y))
    denominator = norm_x * norm_y
    return dot / denominator if denominator else 0.0


def most_similar(
    word: str,
    matrix: csr_matrix,
    vocabulary: dict[str, int],
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Return the top-k cosine neighbors of a word in a row-vector matrix."""
    if word not in vocabulary:
        raise KeyError(f"Out-of-vocabulary word: {word}")
    if top_k < 1:
        raise ValueError("top_k must be positive")

    index_to_word = [""] * len(vocabulary)
    for token, index in vocabulary.items():
        index_to_word[index] = token

    target_index = vocabulary[word]
    target = matrix.getrow(target_index)
    row_norms = np.sqrt(matrix.multiply(matrix).sum(axis=1)).A1
    target_norm = row_norms[target_index]
    if target_norm == 0:
        return []

    dots = (matrix @ target.T).toarray().ravel()
    denominators = row_norms * target_norm
    scores = np.divide(dots, denominators, out=np.zeros_like(dots), where=denominators != 0)
    scores[target_index] = -np.inf
    ranked = np.argsort(-scores, kind="stable")
    return [
        (index_to_word[index], float(scores[index]))
        for index in ranked[:top_k]
        if np.isfinite(scores[index])
    ]
