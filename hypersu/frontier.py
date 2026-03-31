"""Hop-wise frontier expansion on a hypergraph for multi-hop entity discovery.

Vertices  = entities
Hyperedges = semantic units (each hyperedge connects all entities that co-occur in one SU)

Algorithm: explicit BFS-like hop-by-hop exploration.  Each hop discovers
NEW entities through query-relevant SUs, then those new entities become the
frontier for the next hop.
"""

import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


def _compute_conductance(su_embeddings, query_embs, floor, gamma, device):
    """Compute SU conductance weights from one or more query embeddings.

    Args:
        su_embeddings: (n_su, dim) array of semantic-unit embeddings.
        query_embs: (m, dim) array where m >= 1.  When a planner is used each
            row is a sub-query embedding; otherwise m == 1 (original query).
        floor: conductance lower-bound threshold (delta).
        gamma: sharpening exponent.

    For each SU e the gating weight is:
        w(e) = clamp((max_i sim(sq_i, e) - floor) / (1 - floor), 0, 1) ^ gamma
    The *max* ensures an SU is activated whenever it is highly relevant to
    **any** sub-query.
    """
    query_embs = np.asarray(query_embs)
    if query_embs.ndim == 1:
        query_embs = query_embs.reshape(1, -1)
    # (n_su, m) similarities, then take max across sub-queries → (n_su,)
    all_sims = np.dot(su_embeddings, query_embs.T)
    max_sims = np.max(all_sims, axis=1)
    w_q = torch.from_numpy(max_sims).float().to(device).clamp(min=0)
    w_base = torch.clamp(w_q - floor, min=0) / max(1.0 - floor, 1e-8)
    return w_base.pow(gamma)


def frontier_expansion(config, hypergraph, entity_hash_ids,
                       su_embeddings, question_embedding,
                       seed_entity_indices, seed_entity_hash_ids,
                       seed_entity_scores,
                       su_hash_ids=None,
                       sub_query_embeddings=None):
    """Hop-wise frontier expansion on a hypergraph.

    Each hop:
      1. Find SUs reachable from the current frontier, gated by query relevance.
      2. Through those SUs, discover new (not yet activated) entities.
      3. Score each candidate via max-aggregation:
            score(v) = max over reachable SUs e containing v:
                       w[e] × max(frontier_score of frontier entities in e)
      4. Keep top-K new entities; they become the next frontier.

    Args:
        su_hash_ids: list of SU hash IDs matching the SU embedding index order.
            When provided, the function tracks which SUs were traversed during
            expansion and returns their bridge scores.
        sub_query_embeddings: numpy array (m, dim) of sub-query embeddings
            from the planner. When provided, conductance uses
            max_i sim(sq_i, e) instead of sim(q, e).

    Returns:
        activated_entities: dict of entity_hash_id → (entity_idx, score)
        activated_su_scores: dict of su_hash_id → best bridge_score across hops.
            Empty dict if su_hash_ids is None.
    """
    device = hypergraph.device
    num_vertices = hypergraph.num_vertices
    num_hyperedges = hypergraph.num_hyperedges
    top_k = config.expansion_top_k
    max_hops = config.expansion_max_hops
    hop_decay = config.hop_decay
    floor = config.conductance_floor
    gamma = config.conductance_gamma

    track_sus = su_hash_ids is not None

    H = hypergraph.H

    # ── Phase A: Query-conditioned SU conductance ──
    if sub_query_embeddings is not None:
        conductance_embs = np.asarray(sub_query_embeddings)
        if conductance_embs.ndim == 1:
            conductance_embs = conductance_embs.reshape(1, -1)
    else:
        conductance_embs = question_embedding.reshape(1, -1)

    w_conductance = _compute_conductance(su_embeddings, conductance_embs, floor, gamma, device)

    num_active = int((w_conductance > 0).sum().item())
    logger.info(
        "Conductance (floor=%.2f, gamma=%.2f): %d/%d active SUs, %d suppressed",
        floor, gamma, num_active, num_hyperedges, num_hyperedges - num_active,
    )

    # ── Phase B: Initialise seeds ──
    activated = {}
    for idx, score in zip(seed_entity_indices, seed_entity_scores):
        activated[idx] = float(score)

    frontier = set(activated.keys())

    # Precompute COO structure once
    H_coo = H.coalesce()
    h_indices = H_coo.indices()
    h_v_idx = h_indices[0]
    h_e_idx = h_indices[1]

    # Track best bridge score per SU across all hops
    su_best_bridge = torch.zeros(num_hyperedges, device=device) if track_sus else None

    # ── Phase C: Hop-by-hop frontier expansion ──
    for hop in range(1, max_hops + 1):
        if not frontier:
            break

        # 1) Build frontier score vector
        frontier_scores = torch.zeros(num_vertices, device=device)
        for idx in frontier:
            frontier_scores[idx] = activated[idx]

        # 2) Scatter max frontier scores to SUs via COO
        is_frontier = frontier_scores[h_v_idx] > 0
        frontier_signal = torch.where(is_frontier, frontier_scores[h_v_idx],
                                       torch.tensor(0.0, device=device))
        max_frontier_per_su = torch.zeros(num_hyperedges, device=device)
        max_frontier_per_su.scatter_reduce_(0, h_e_idx, frontier_signal, reduce="amax")

        # 3) Bridge score = conductance × max frontier score
        bridge_scores = w_conductance * max_frontier_per_su

        # Track activated SUs: keep max bridge_score across hops
        if track_sus:
            su_best_bridge = torch.max(su_best_bridge, bridge_scores)

        # 4) Scatter max bridge scores to candidate entities via COO
        candidate_scores = torch.full((num_vertices,), -1.0, device=device)
        edge_bridge = bridge_scores[h_e_idx]
        candidate_scores.scatter_reduce_(0, h_v_idx, edge_bridge, reduce="amax")

        # Zero out already-activated entities
        activated_tensor = torch.zeros(num_vertices, dtype=torch.bool, device=device)
        for idx in activated:
            activated_tensor[idx] = True
        candidate_scores[activated_tensor] = -1.0

        # 5) Select top-K new entities with positive scores
        num_positive = int((candidate_scores > 0).sum().item())
        if num_positive == 0:
            logger.debug("Hop %d: no new candidates found, stopping.", hop)
            break

        k = min(top_k, num_positive)
        top_vals, top_idx = torch.topk(candidate_scores, k)

        decay = hop_decay ** hop
        new_frontier = set()
        for idx_t, val_t in zip(top_idx.cpu().tolist(), top_vals.cpu().tolist()):
            if val_t <= 0:
                break
            activated[idx_t] = val_t * decay
            new_frontier.add(idx_t)

        logger.debug(
            "Hop %d: %d new entities (top score=%.4f, decay=%.3f), "
            "%d total activated",
            hop, len(new_frontier),
            top_vals[0].item() if len(top_vals) > 0 else 0, decay,
            len(activated),
        )

        frontier = new_frontier

    logger.info(
        "Frontier expansion finished: %d hops, %d activated entities",
        min(hop, max_hops) if frontier else hop - 1, len(activated),
    )

    # ── Phase D: Collect results ──
    activated_entities = {}
    for idx, hash_id, score in zip(seed_entity_indices, seed_entity_hash_ids, seed_entity_scores):
        activated_entities[hash_id] = (idx, score)

    for entity_idx, score in activated.items():
        hash_id = entity_hash_ids[entity_idx]
        if hash_id not in activated_entities:
            activated_entities[hash_id] = (entity_idx, score)

    # Collect activated SU scores
    activated_su_scores = {}
    if track_sus:
        active_mask = su_best_bridge > 0
        active_indices = active_mask.nonzero(as_tuple=True)[0].cpu().tolist()
        bridge_vals = su_best_bridge.cpu()
        for su_idx in active_indices:
            activated_su_scores[su_hash_ids[su_idx]] = float(bridge_vals[su_idx])
        logger.info("Activated SUs: %d / %d", len(activated_su_scores), num_hyperedges)

    return activated_entities, activated_su_scores
