"""Personalized PageRank for graph-based memory retrieval (Upgrade 3)."""

import sqlite3
from typing import List, Tuple, Dict, Any
from .config import get_logger
from .database import get_db

logger = get_logger(__name__)


def personalized_pagerank(
    db_path: str,
    seed_memory_ids: List[int],
    damping: float = 0.85,
    iterations: int = 20,
    top_k: int = 10
) -> List[Tuple[int, float]]:
    """
    Run Personalized PageRank starting from seed_memory_ids.

    Uses the existing graph topology (memory_relations table) to find
    memories 2-3 hops away that are contextually relevant.

    Args:
        db_path: Path to database (unused, kept for API compatibility)
        seed_memory_ids: Starting memory IDs to run PPR from
        damping: Damping factor (default 0.85, standard for PageRank)
        iterations: Number of iterations (default 20)
        top_k: Return top K memories by score

    Returns:
        List of (memory_id, score) tuples sorted by score descending
    """
    if not seed_memory_ids:
        return []

    conn = get_db()

    # Get all memory IDs that are active
    all_memory_ids = [row[0] for row in conn.execute(
        "SELECT id FROM memories WHERE active = 1"
    ).fetchall()]

    if not all_memory_ids:
        conn.close()
        return []

    # Build adjacency list from memory_relations
    # Note: memory_relations has from_memory_id -> to_memory_id edges
    adjacency = {}
    out_degree = {}

    for mem_id in all_memory_ids:
        adjacency[mem_id] = []
        out_degree[mem_id] = 0

    # Get all edges
    edges = conn.execute("""
        SELECT source_id, target_id
        FROM memory_relations
        WHERE source_id IN (SELECT id FROM memories WHERE active = 1)
          AND target_id IN (SELECT id FROM memories WHERE active = 1)
    """).fetchall()

    conn.close()

    for from_id, to_id in edges:
        if from_id in adjacency and to_id in adjacency:
            adjacency[from_id].append(to_id)
            out_degree[from_id] += 1

    # Initialize scores
    scores = {mem_id: 0.0 for mem_id in all_memory_ids}

    # Set seed scores (uniform distribution)
    seed_score = 1.0 / len(seed_memory_ids)
    for seed_id in seed_memory_ids:
        if seed_id in scores:
            scores[seed_id] = seed_score

    # Build reverse adjacency once (O(N+E)) instead of checking all pairs (O(N²))
    reverse_adjacency = {mid: [] for mid in all_memory_ids}
    for from_id, neighbors in adjacency.items():
        for to_id in neighbors:
            if to_id in reverse_adjacency:
                reverse_adjacency[to_id].append(from_id)

    # Power iteration
    for _ in range(iterations):
        new_scores = {mem_id: 0.0 for mem_id in all_memory_ids}

        for mem_id in all_memory_ids:
            # Teleport back to seed nodes
            teleport_prob = (1 - damping) * seed_score if mem_id in seed_memory_ids else 0.0
            new_scores[mem_id] = teleport_prob

            # Random walk from incoming edges
            for incoming_id in reverse_adjacency.get(mem_id, []):
                degree = out_degree.get(incoming_id, 0)
                if degree > 0:
                    new_scores[mem_id] += damping * scores[incoming_id] / degree

        scores = new_scores

    # Remove seed nodes from results (we already know about them)
    for seed_id in seed_memory_ids:
        scores.pop(seed_id, None)

    # Sort by score and return top K
    sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
    return sorted_scores[:top_k]


def ppr_boost_search_results(
    search_results: List[Dict[str, Any]],
    ppr_weight: float = 0.3
) -> List[Dict[str, Any]]:
    """
    Boost search results using PPR scores.

    Takes search results, runs PPR seeded from top results, and merges scores:
    final_score = (1 - ppr_weight) * original_score + ppr_weight * ppr_score

    Args:
        search_results: List of memory dicts with 'id' and 'score' keys
        ppr_weight: Weight for PPR boost (default 0.3 = 30%)

    Returns:
        Re-ranked search results with updated scores
    """
    if not search_results:
        return []

    # Use top 5 results as seeds (or fewer if less than 5 results)
    num_seeds = min(5, len(search_results))
    seed_ids = [r['id'] for r in search_results[:num_seeds]]

    # Run PPR
    ppr_scores = personalized_pagerank(
        db_path="",  # Will use get_db() internally
        seed_memory_ids=seed_ids,
        top_k=50  # Get more candidates for merging
    )

    # Build PPR score dict
    ppr_dict = {mem_id: score for mem_id, score in ppr_scores}

    # Normalize PPR scores to 0-1 range
    if ppr_scores:
        max_ppr = max(score for _, score in ppr_scores)
        if max_ppr > 0:
            ppr_dict = {mem_id: score / max_ppr for mem_id, score in ppr_dict.items()}

    # Merge scores
    boosted_results = []
    for result in search_results:
        mem_id = result['id']
        original_score = result.get('score', 0.0)
        ppr_score = ppr_dict.get(mem_id, 0.0)

        # Normalize original score if needed (assume it's already 0-1 range)
        # Merge: 70% original + 30% PPR
        final_score = (1 - ppr_weight) * original_score + ppr_weight * ppr_score

        result_copy = result.copy()
        result_copy['score'] = final_score
        result_copy['ppr_boost'] = ppr_score
        boosted_results.append(result_copy)

    # Re-sort by new score
    boosted_results.sort(key=lambda x: -x['score'])

    return boosted_results
