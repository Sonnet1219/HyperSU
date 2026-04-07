"""Knowledge graph: entity-passage edge weights and entity-SU hyperedge mapping."""

import re
import numpy as np
import torch
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class Hypergraph:
    """Sparse hypergraph backed by an incidence matrix H.

    H[v, e] = 1  iff vertex v belongs to hyperedge e.
    """

    def __init__(self, entity_hash_id_to_su_hash_ids,
                 entity_embedding_store, su_embedding_store, device):
        self.device = device
        self.num_vertices = len(entity_embedding_store.hash_id_to_text)
        self.num_hyperedges = len(su_embedding_store.hash_id_to_text)

        # Build incidence matrix H  (|V| x |E|)
        indices = []
        for entity_hash_id, su_hash_ids in entity_hash_id_to_su_hash_ids.items():
            v_idx = entity_embedding_store.hash_id_to_idx[entity_hash_id]
            for su_hash_id in su_hash_ids:
                e_idx = su_embedding_store.hash_id_to_idx[su_hash_id]
                indices.append([v_idx, e_idx])

        if indices:
            idx_tensor = torch.tensor(indices, dtype=torch.long).t()
            val_tensor = torch.ones(idx_tensor.shape[1], dtype=torch.float32)
            self.H = torch.sparse_coo_tensor(
                idx_tensor, val_tensor,
                (self.num_vertices, self.num_hyperedges), device=device,
            ).coalesce()
        else:
            self.H = torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long),
                torch.zeros(0, dtype=torch.float32),
                (self.num_vertices, self.num_hyperedges), device=device,
            )

        nnz = self.H._nnz()
        ones_e = torch.ones(self.num_hyperedges, 1, device=device)
        d_v = torch.sparse.mm(self.H, ones_e).squeeze()
        ones_v = torch.ones(self.num_vertices, 1, device=device)
        d_e = torch.sparse.mm(self.H.t(), ones_v).squeeze()

        logger.info(
            "Hypergraph built: %d vertices, %d hyperedges, nnz=%d, "
            "sparsity=%.2f%%, D_v range [%.0f, %.0f], D_e range [%.0f, %.0f]",
            self.num_vertices, self.num_hyperedges, nnz,
            (1 - nnz / max(self.num_vertices * self.num_hyperedges, 1)) * 100,
            d_v.min(), d_v.max(), d_e.min(), d_e.max(),
        )


class KnowledgeGraph:
    """Entity-passage edge weights and entity-SU hyperedge mapping
    for frontier expansion and dual-channel passage scoring."""

    def __init__(self):
        self.edge_weights = defaultdict(dict)
        self.entity_to_su_ids = {}
        self.passage_to_su_ids = {}
        self._hypergraph = None

    def get_state(self) -> dict:
        """Serialize graph state (excluding _hypergraph which is rebuilt at retrieval time)."""
        return {
            "edge_weights": dict(self.edge_weights),
            "entity_to_su_ids": dict(self.entity_to_su_ids),
            "passage_to_su_ids": dict(self.passage_to_su_ids),
        }

    def load_state(self, state: dict) -> None:
        """Restore graph state from a previously saved dict."""
        self.edge_weights = defaultdict(dict, state["edge_weights"])
        self.entity_to_su_ids = dict(state["entity_to_su_ids"])
        self.passage_to_su_ids = dict(state["passage_to_su_ids"])
        self._hypergraph = None

    # ====== Indexing ======

    def build_node_edge_maps(self, passage_entities, su_entities):
        """Extract entity/SU node sets and build bidirectional edge maps.

        Args:
            passage_entities: dict mapping passage_hash_id -> list of entity texts
            su_entities: dict mapping su_text -> list of entity texts

        Returns:
            entity_nodes, su_nodes, passage_to_entities, entity_to_su, su_to_entity
        """
        entity_nodes = set()
        su_nodes = set()
        passage_to_entities = defaultdict(set)
        entity_to_su = defaultdict(set)
        su_to_entity = defaultdict(set)
        for passage_hash_id, entities in passage_entities.items():
            for entity in entities:
                entity_nodes.add(entity)
                passage_to_entities[passage_hash_id].add(entity)
        for su, entities in su_entities.items():
            su_nodes.add(su)
            for entity in entities:
                entity_to_su[entity].add(su)
                su_to_entity[su].add(entity)
        return entity_nodes, su_nodes, passage_to_entities, entity_to_su, su_to_entity

    def build_entity_su_mapping(self, entity_to_su, entity_store, su_store):
        """Convert text-level entity->SU mapping to hash_id-level and store.

        Args:
            entity_to_su: dict mapping entity_text -> set of su_texts
            entity_store: EmbeddingStore for entities
            su_store: EmbeddingStore for semantic units
        """
        self.entity_to_su_ids = {}
        for entity, sus in entity_to_su.items():
            entity_hash_id = entity_store.text_to_hash_id[entity]
            self.entity_to_su_ids[entity_hash_id] = [
                su_store.text_to_hash_id[s] for s in sus
            ]

    def link_entities_to_passages(
        self,
        passage_to_entities,
        passage_store,
        entity_store,
        passage_entity_counts=None,
    ):
        """Create weighted edges between passages and their contained entities."""
        passage_entity_count = {}
        passage_total_count = defaultdict(float)
        for passage_hash_id, entities in passage_to_entities.items():
            passage_text = passage_store.hash_id_to_text[passage_hash_id]
            weights = (passage_entity_counts or {}).get(passage_hash_id, {})
            for entity in entities:
                entity_hash_id = entity_store.text_to_hash_id[entity]
                if entity in weights:
                    count = float(weights[entity])
                else:
                    count = float(passage_text.count(entity))
                passage_entity_count[(passage_hash_id, entity_hash_id)] = count
                passage_total_count[passage_hash_id] += count
        for (passage_hash_id, entity_hash_id), count in passage_entity_count.items():
            total = passage_total_count[passage_hash_id]
            weight = count / total if total > 0 else 0.0
            self.edge_weights[passage_hash_id][entity_hash_id] = weight

    def link_adjacent_passages(self, passage_store):
        """Connect sequentially adjacent passages based on their index prefix."""
        passage_id_to_text = passage_store.get_hash_id_to_text()
        index_pattern = re.compile(r'^(\d+):')
        indexed_items = [
            (int(match.group(1)), node_key)
            for node_key, text in passage_id_to_text.items()
            if (match := index_pattern.match(text.strip()))
        ]
        indexed_items.sort(key=lambda x: x[0])
        for i in range(len(indexed_items) - 1):
            current_node = indexed_items[i][1]
            next_node = indexed_items[i + 1][1]
            self.edge_weights[current_node][next_node] = 1.0

    # ====== Retrieval ======

    def build_hypergraph(self, entity_store, su_store, device):
        """Build Hypergraph incidence matrix from entity-SU mapping."""
        self._hypergraph = Hypergraph(self.entity_to_su_ids, entity_store, su_store, device)
        return self._hypergraph


def dense_retrieval(passage_embeddings, query_embedding):
    """Rank all passages by cosine similarity to the query."""
    query_emb = query_embedding.reshape(1, -1)
    similarities = np.dot(passage_embeddings, query_emb.T).flatten()
    sorted_indices = np.argsort(similarities)[::-1]
    sorted_scores = similarities[sorted_indices].tolist()
    return sorted_indices, sorted_scores
