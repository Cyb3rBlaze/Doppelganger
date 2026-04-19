"""One-shot dream-mode pass over the unified memory graph."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from app.services import unified_memory

DEFAULT_DREAM_SEMANTIC_THRESHOLD = 0.74
DEFAULT_DREAM_COMBINED_THRESHOLD = 0.63
DEFAULT_DREAM_MAX_NEW_EDGES_PER_NODE = 8
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "she",
    "that",
    "the",
    "their",
    "them",
    "there",
    "they",
    "this",
    "to",
    "was",
    "we",
    "were",
    "with",
    "you",
    "your",
}

SELECT_EMBEDDED_MEMORY_NODES_SQL = """
SELECT
    node_id,
    node_type,
    title,
    content,
    metadata,
    embedding
FROM memory_nodes
WHERE embedding IS NOT NULL
ORDER BY node_id ASC
"""

SELECT_EXISTING_MEMORY_EDGES_SQL = """
SELECT
    source_node_id,
    target_node_id
FROM memory_edges
"""

DELETE_DREAM_EDGES_SQL = """
DELETE FROM memory_edges
WHERE edge_types @> '["dream"]'::jsonb
"""


@dataclass(frozen=True)
class DreamNode:
    """One embedded memory node available for dream-mode linking."""

    node_id: str
    node_type: str
    title: str
    content: str
    metadata: dict[str, Any]
    embedding: list[float]
    keywords: set[str]


def _normalize_keywords(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9_'-]{2,}", text.lower())
        if token not in STOPWORDS
    }


def build_dream_keywords(
    *,
    node_type: str,
    title: str | None,
    content: str,
    metadata: dict[str, Any],
) -> set[str]:
    """Build coarse relevance keywords for a memory node."""
    text_parts = [node_type, title or "", content]
    for key in ("channel", "direction", "source_kind", "source_path", "document_id"):
        value = metadata.get(key)
        if isinstance(value, str):
            text_parts.append(value)
    return _normalize_keywords("\n".join(text_parts))


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Compute cosine similarity between two embeddings."""
    if not left or not right or len(left) != len(right):
        return 0.0

    dot_product = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for left_value, right_value in zip(left, right, strict=True):
        dot_product += left_value * right_value
        left_norm += left_value * left_value
        right_norm += right_value * right_value

    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    return dot_product / math.sqrt(left_norm * right_norm)


def relevance_similarity(left: DreamNode, right: DreamNode) -> float:
    """Compute a lightweight relevance-overlap score between two nodes."""
    if not left.keywords or not right.keywords:
        return 0.0

    shared_keywords = left.keywords & right.keywords
    if not shared_keywords:
        return 0.0

    overlap_score = len(shared_keywords) / min(len(left.keywords), len(right.keywords))
    metadata_bonus = 0.0

    left_channel = left.metadata.get("channel")
    right_channel = right.metadata.get("channel")
    if isinstance(left_channel, str) and left_channel == right_channel:
        metadata_bonus += 0.05

    left_document_id = left.metadata.get("document_id")
    right_document_id = right.metadata.get("document_id")
    if isinstance(left_document_id, str) and left_document_id == right_document_id:
        metadata_bonus += 0.08

    return min(1.0, overlap_score + metadata_bonus)


def build_dream_edge_types(left: DreamNode, right: DreamNode, relevance_score: float) -> list[str]:
    """Build edge-type labels for a dream-generated relation."""
    edge_types = ["dream", "dream_semantic"]
    if relevance_score > 0:
        edge_types.append("dream_relevance")
    if left.node_type != right.node_type:
        edge_types.append("dream_cross_type")

    node_type_pair = {left.node_type, right.node_type}
    if node_type_pair == {"message", "document_chunk"}:
        edge_types.append("message_document")
    elif node_type_pair == {"session_summary", "document_chunk"}:
        edge_types.append("summary_document")

    return edge_types


def load_embedded_memory_nodes(*, target_dsn: str) -> list[DreamNode]:
    """Load embedded unified-memory nodes for dream-mode comparison."""
    psycopg = unified_memory._load_psycopg()
    with psycopg.connect(target_dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute(SELECT_EMBEDDED_MEMORY_NODES_SQL)
            rows = cursor.fetchall()

    nodes: list[DreamNode] = []
    for row in rows:
        node_id, node_type, title, content, metadata, embedding = row
        normalized_metadata = unified_memory._normalize_json_dict(metadata)
        normalized_embedding = unified_memory._normalize_embedding(embedding)
        if not normalized_embedding:
            continue
        nodes.append(
            DreamNode(
                node_id=node_id,
                node_type=node_type,
                title=str(title or ""),
                content=str(content or ""),
                metadata=normalized_metadata,
                embedding=normalized_embedding,
                keywords=build_dream_keywords(
                    node_type=node_type,
                    title=title,
                    content=str(content or ""),
                    metadata=normalized_metadata,
                ),
            )
        )
    return nodes


def load_existing_edge_pairs(*, target_dsn: str) -> set[tuple[str, str]]:
    """Load existing undirected edge pairs to avoid duplicate dream edges."""
    psycopg = unified_memory._load_psycopg()
    with psycopg.connect(target_dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute(SELECT_EXISTING_MEMORY_EDGES_SQL)
            rows = cursor.fetchall()

    return {
        tuple(sorted((str(source_node_id), str(target_node_id))))
        for source_node_id, target_node_id in rows
    }


def run_dream_iteration(
    *,
    target_dsn: str | None = None,
    semantic_threshold: float = DEFAULT_DREAM_SEMANTIC_THRESHOLD,
    combined_threshold: float = DEFAULT_DREAM_COMBINED_THRESHOLD,
    max_new_edges_per_node: int = DEFAULT_DREAM_MAX_NEW_EDGES_PER_NODE,
) -> dict[str, int | float]:
    """Run one dream-mode edge-discovery pass over the unified memory graph."""
    resolved_dsn = target_dsn or unified_memory.get_unified_memory_dsn()
    if not resolved_dsn:
        raise RuntimeError(f"{unified_memory.UNIFIED_MEMORY_DSN_ENV} is not set.")
    if max_new_edges_per_node <= 0:
        raise RuntimeError("max_new_edges_per_node must be greater than zero.")

    unified_memory.ensure_unified_memory_schema(
        resolved_dsn,
        unified_memory.get_embedding_dimension(),
    )
    nodes = load_embedded_memory_nodes(target_dsn=resolved_dsn)
    existing_pairs = load_existing_edge_pairs(target_dsn=resolved_dsn)

    candidate_pairs: list[tuple[float, float, float, DreamNode, DreamNode]] = []
    compared_pair_count = 0
    for left_index in range(len(nodes)):
        left_node = nodes[left_index]
        for right_index in range(left_index + 1, len(nodes)):
            right_node = nodes[right_index]
            pair_key = tuple(sorted((left_node.node_id, right_node.node_id)))
            if pair_key in existing_pairs:
                continue

            compared_pair_count += 1
            semantic_score = cosine_similarity(left_node.embedding, right_node.embedding)
            relevance_score = relevance_similarity(left_node, right_node)
            combined_score = semantic_score * 0.75 + relevance_score * 0.25
            if (
                semantic_score < semantic_threshold
                and combined_score < combined_threshold
            ):
                continue
            candidate_pairs.append(
                (combined_score, semantic_score, relevance_score, left_node, right_node)
            )

    candidate_pairs.sort(
        key=lambda item: (item[0], item[1], item[2], item[3].node_id, item[4].node_id),
        reverse=True,
    )

    kept_pairs: list[tuple[float, float, float, DreamNode, DreamNode]] = []
    per_node_edge_counts: dict[str, int] = {}
    for candidate in candidate_pairs:
        _, _, _, left_node, right_node = candidate
        if per_node_edge_counts.get(left_node.node_id, 0) >= max_new_edges_per_node:
            continue
        if per_node_edge_counts.get(right_node.node_id, 0) >= max_new_edges_per_node:
            continue
        kept_pairs.append(candidate)
        per_node_edge_counts[left_node.node_id] = per_node_edge_counts.get(left_node.node_id, 0) + 1
        per_node_edge_counts[right_node.node_id] = per_node_edge_counts.get(right_node.node_id, 0) + 1

    psycopg = unified_memory._load_psycopg()
    with psycopg.connect(resolved_dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute(DELETE_DREAM_EDGES_SQL)
            created_edge_count = 0
            for combined_score, semantic_score, relevance_score, left_node, right_node in kept_pairs:
                edge_types = build_dream_edge_types(left_node, right_node, relevance_score)
                signals = {
                    "dream": combined_score,
                    "semantic": semantic_score,
                    "relevance": relevance_score,
                }
                for source_node_id, target_node_id in (
                    (left_node.node_id, right_node.node_id),
                    (right_node.node_id, left_node.node_id),
                ):
                    unified_memory._upsert_memory_edge(
                        cursor,
                        unified_memory.MemoryEdgeRecord(
                            source_node_id=source_node_id,
                            target_node_id=target_node_id,
                            score=combined_score,
                            edge_types=edge_types,
                            signals=signals,
                        ),
                    )
                    created_edge_count += 1
        connection.commit()

    return {
        "embedded_node_count": len(nodes),
        "compared_pair_count": compared_pair_count,
        "candidate_pair_count": len(candidate_pairs),
        "created_pair_count": len(kept_pairs),
        "created_edge_count": created_edge_count,
        "semantic_threshold": semantic_threshold,
        "combined_threshold": combined_threshold,
        "max_new_edges_per_node": max_new_edges_per_node,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the dream-mode CLI parser."""
    parser = argparse.ArgumentParser(
        description="Run one dream-mode pass over unified memory and add dream edges."
    )
    parser.add_argument(
        "--semantic-threshold",
        type=float,
        default=DEFAULT_DREAM_SEMANTIC_THRESHOLD,
        help="Minimum semantic similarity that can qualify a pair on its own.",
    )
    parser.add_argument(
        "--combined-threshold",
        type=float,
        default=DEFAULT_DREAM_COMBINED_THRESHOLD,
        help="Minimum blended semantic/relevance score required for a new edge.",
    )
    parser.add_argument(
        "--max-new-edges-per-node",
        type=int,
        default=DEFAULT_DREAM_MAX_NEW_EDGES_PER_NODE,
        help="Cap the number of new dream edges each node can gain per pass.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run one CLI dream-mode iteration."""
    parser = build_argument_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = run_dream_iteration(
        semantic_threshold=args.semantic_threshold,
        combined_threshold=args.combined_threshold,
        max_new_edges_per_node=args.max_new_edges_per_node,
    )
    print(json.dumps({"status": "ok", "dream_mode": result}, indent=2))


if __name__ == "__main__":
    main()
