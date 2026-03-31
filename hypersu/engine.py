"""HyperSU: Hypergraph-based retrieval-augmented generation engine."""

from __future__ import annotations

import copy
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import spacy
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from hypersu.chunker import create_semantic_units
from hypersu.config import HyperSUConfig
from hypersu.embedding_store import EmbeddingStore
from hypersu.entity_normalization import merge_entity_mentions
from hypersu.frontier import frontier_expansion
from hypersu.knowledge_graph import KnowledgeGraph, dense_retrieval
from hypersu.ner import LangExtractExtractor
from hypersu.planner import QueryPlanner
from hypersu.reranker import QwenReranker
from hypersu.utils import LLM_Model, compute_mdhash_id


logger = logging.getLogger(__name__)


class HyperSU:
    def __init__(self, save_dir="./output", llm_model_name="gpt-4o-mini",
                 embedding_model_name="BAAI/bge-large-en-v1.5", **kwargs):
        self.config = HyperSUConfig(
            save_dir=save_dir,
            embedding_model_name=embedding_model_name,
            llm_model_name=llm_model_name,
            **kwargs,
        )
        logger.info("Initializing HyperSU with config: %s", self.config)
        logger.info("Retrieval method: Frontier Expansion")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", self.device)

        self.embedding_model = SentenceTransformer(
            embedding_model_name, device=str(self.device)
        )
        self._llm_model = None

        self._init_embedding_stores()

        spacy.prefer_gpu()
        self.spacy_model = spacy.load(self.config.spacy_model)
        logger.info("spaCy sentence splitter loaded with model: %s", self.config.spacy_model)

        self.ner_extractor = LangExtractExtractor(
            model_id=self.config.langextract_model_id,
            api_key=self.config.langextract_api_key,
            model_url=self.config.langextract_model_url,
            max_char_buffer=self.config.langextract_max_char_buffer,
            extraction_passes=self.config.langextract_extraction_passes,
            max_workers=self.config.max_workers,
            use_schema_constraints=self.config.langextract_use_schema_constraints,
        )

        self.knowledge_graph = KnowledgeGraph()
        self.reranker = None
        self.planner = None
        if self.config.use_planner:
            self.planner = QueryPlanner(
                llm_model_name=self.config.planner_model_name,
                max_subqueries=self.config.planner_max_subqueries,
            )
            logger.info("Query planner enabled (model=%s, max_sq=%d)",
                        self.config.planner_model_name, self.config.planner_max_subqueries)
        self.entity_nodes_by_hash = {}

    @property
    def llm_model(self):
        if self._llm_model is None:
            self._llm_model = LLM_Model(self.config.llm_model_name)
        return self._llm_model

    def _init_embedding_stores(self):
        save_dir = self.config.save_dir
        self.passage_embedding_store = EmbeddingStore(
            self.embedding_model,
            db_filename=os.path.join(save_dir, "passage_embedding.parquet"),
            batch_size=self.config.batch_size,
            namespace="passage",
        )
        self.entity_embedding_store = EmbeddingStore(
            self.embedding_model,
            db_filename=os.path.join(save_dir, "entity_embedding.parquet"),
            batch_size=self.config.batch_size,
            namespace="entity",
        )
        self.su_embedding_store = EmbeddingStore(
            self.embedding_model,
            db_filename=os.path.join(save_dir, "su_embedding.parquet"),
            batch_size=self.config.batch_size,
            namespace="su",
        )

    # ====== Indexing ======

    def index(self, docs):
        """Build the hypergraph index from documents."""
        self.passage_embedding_store.insert_text(docs)
        hash_id_to_passage = self.passage_embedding_store.get_hash_id_to_text()

        cached_passage_to_su_ids, cached_su_data, uncached_passage_ids = \
            self._load_cached_extractions(hash_id_to_passage.keys())
        logger.info(
            "Extraction cache status: cached_passages=%s, uncached_passages=%s",
            len(cached_passage_to_su_ids),
            len(uncached_passage_ids),
        )

        if uncached_passage_ids:
            new_hash_id_to_passage = {
                hash_id: hash_id_to_passage[hash_id]
                for hash_id in uncached_passage_ids
            }
            logger.info(
                "Creating semantic units via Kamradt Percentile (p=%s)...",
                self.config.semantic_unit_percentile,
            )

            total_su_count = 0
            for passage_hash_id, passage_text in tqdm(
                new_hash_id_to_passage.items(),
                desc="Semantic Unit Chunking",
            ):
                su_texts = create_semantic_units(
                    passage_text,
                    self.spacy_model,
                    self.embedding_model,
                    self.config.semantic_unit_percentile,
                )
                su_items = []
                su_hash_ids = []
                seen_su_ids = set()
                for su_text in su_texts:
                    su_hash_id = compute_mdhash_id(su_text, prefix="su-")
                    if su_hash_id in seen_su_ids:
                        continue
                    seen_su_ids.add(su_hash_id)
                    su_items.append((su_hash_id, su_text))
                    su_hash_ids.append(su_hash_id)
                cached_passage_to_su_ids[passage_hash_id] = su_hash_ids
                total_su_count += len(su_items)

                su_mentions = self.ner_extractor.extract_mentions_from_su_batch(
                    su_items, passage_hash_id=passage_hash_id
                )
                for su_hash_id, su_text in su_items:
                    cached_su_data[su_hash_id] = {
                        "text": su_text,
                        "mentions": su_mentions.get(su_hash_id, []),
                    }

            logger.info(
                "Created %s semantic units from %s passages (avg %.1f SU/passage)",
                total_su_count,
                len(new_hash_id_to_passage),
                total_su_count / max(len(new_hash_id_to_passage), 1),
            )
        else:
            logger.info("All passages already have cached extraction results; skipping extraction.")

        self._persist_extraction_data(cached_passage_to_su_ids, cached_su_data)

        # Rebuild SU/entity stores from the extraction cache so the graph stays consistent.
        self.su_embedding_store.clear()
        self.entity_embedding_store.clear()
        self.knowledge_graph = KnowledgeGraph()

        su_text_by_hash = {
            su_hash_id: payload["text"]
            for su_hash_id, payload in cached_su_data.items()
            if payload.get("text")
        }
        self.su_embedding_store.insert_text(list(su_text_by_hash.values()))

        all_mentions = []
        for payload in cached_su_data.values():
            all_mentions.extend(payload.get("mentions", []))

        entity_nodes, passage_entities, su_entities, passage_entity_counts = \
            merge_entity_mentions(
                all_mentions,
                su_text_by_hash,
                self.embedding_model,
                similarity_threshold=self.config.entity_merge_threshold,
                batch_size=self.config.batch_size,
            )

        entity_texts = [node["embedding_text"] for node in entity_nodes]
        self.entity_embedding_store.insert_text(entity_texts)
        self._persist_entity_nodes(entity_nodes)

        _, _, passage_to_entities, entity_to_su, _ = \
            self.knowledge_graph.build_node_edge_maps(passage_entities, su_entities)

        self.knowledge_graph.build_entity_su_mapping(
            entity_to_su,
            self.entity_embedding_store,
            self.su_embedding_store,
        )
        self.knowledge_graph.link_entities_to_passages(
            passage_to_entities,
            self.passage_embedding_store,
            self.entity_embedding_store,
            passage_entity_counts=passage_entity_counts,
        )
        self.knowledge_graph.link_adjacent_passages(self.passage_embedding_store)
        self.knowledge_graph.passage_to_su_ids = dict(cached_passage_to_su_ids)

    def _load_cached_extractions(self, passage_hash_ids):
        self._ner_cache_path = os.path.join(self.config.save_dir, "ner_results.json")
        if not os.path.exists(self._ner_cache_path):
            return {}, {}, set(passage_hash_ids)

        with open(self._ner_cache_path, encoding="utf-8") as handle:
            cached = json.load(handle)

        if cached.get("schema_version") != 2:
            logger.warning(
                "Ignoring legacy extraction cache at %s because schema_version != 2",
                self._ner_cache_path,
            )
            return {}, {}, set(passage_hash_ids)

        passage_to_su_ids = cached.get("passage_hash_id_to_su_hash_ids", {})
        su_to_data = cached.get("su_to_data", {})
        uncached_ids = set(passage_hash_ids) - set(passage_to_su_ids.keys())
        return passage_to_su_ids, su_to_data, uncached_ids

    def _persist_extraction_data(self, passage_to_su_ids, su_to_data):
        os.makedirs(os.path.dirname(self._ner_cache_path), exist_ok=True)
        with open(self._ner_cache_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "schema_version": 2,
                    "passage_hash_id_to_su_hash_ids": passage_to_su_ids,
                    "su_to_data": su_to_data,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )

    def _persist_entity_nodes(self, entity_nodes):
        entity_nodes_path = os.path.join(self.config.save_dir, "entity_nodes.json")
        self.entity_nodes_by_hash = {}
        serializable_nodes = {}
        for entity_node in entity_nodes:
            entity_text = entity_node["embedding_text"]
            entity_hash_id = self.entity_embedding_store.text_to_hash_id.get(entity_text)
            if entity_hash_id is None:
                continue
            stored_node = dict(entity_node)
            stored_node["hash_id"] = entity_hash_id
            self.entity_nodes_by_hash[entity_hash_id] = stored_node
            serializable_nodes[entity_hash_id] = stored_node

        with open(entity_nodes_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "schema_version": 1,
                    "entities": serializable_nodes,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )

    # ====== Retrieval ======

    def retrieve(self, queries, num_to_retrieve=None):
        """Retrieve relevant passages for a list of queries."""
        top_k = num_to_retrieve or self.config.retrieval_top_k
        use_rerank = self.config.use_reranker

        if use_rerank:
            self._lazy_load_reranker()
            logger.info(
                "Reranking enabled: top %s candidates -> top %s final passages",
                self.config.reranker_candidate_top_k,
                top_k,
            )
        else:
            logger.info("Reranking disabled. Using dual-channel scoring (top %s).", top_k)

        self._prepare_retrieval_cache()

        retrieval_results = []
        for query in tqdm(queries, desc="Retrieving"):
            query_embedding = self.embedding_model.encode(
                query,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=self.config.batch_size,
                prompt=self.config.query_instruction_prefix,
            )

            # ── Sub-query decomposition (planner) ──
            sub_query_embeddings = None
            sub_query_texts = None
            if self.planner is not None:
                plan = self.planner.plan(query)
                if plan.is_complex and len(plan.sub_queries) > 1:
                    sub_query_texts = [sq.query for sq in plan.sub_queries]
                    sub_query_embeddings = self.embedding_model.encode(
                        sub_query_texts,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                        batch_size=self.config.batch_size,
                        prompt=self.config.query_instruction_prefix,
                    )
                    logger.debug("Planner decomposed into %d sub-queries: %s",
                                 len(sub_query_texts), sub_query_texts)

            _, seed_indices, seed_texts, seed_hash_ids, seed_scores = \
                self.extract_seed_entities(query)

            if seed_texts:
                sorted_hash_ids, sorted_scores = self._diffuse_from_seeds(
                    query,
                    query_embedding,
                    seed_indices,
                    seed_hash_ids,
                    seed_scores,
                    sub_query_embeddings=sub_query_embeddings,
                )
                if use_rerank:
                    candidate_hash_ids = sorted_hash_ids[:self.config.reranker_candidate_top_k]
                    candidate_passages = [
                        self.passage_embedding_store.hash_id_to_text[hash_id]
                        for hash_id in candidate_hash_ids
                    ]
                    _, final_scores, final_passages = self._rerank_passages(
                        query,
                        candidate_hash_ids,
                        candidate_passages,
                        top_k,
                    )
                else:
                    top_hash_ids = sorted_hash_ids[:top_k]
                    final_scores = [float(score) for score in sorted_scores[:top_k]]
                    final_passages = [
                        self.passage_embedding_store.hash_id_to_text[hash_id]
                        for hash_id in top_hash_ids
                    ]
            else:
                sorted_indices, sorted_scores = dense_retrieval(
                    self._passage_embeddings,
                    query_embedding,
                )
                if use_rerank:
                    candidate_indices = sorted_indices[:self.config.reranker_candidate_top_k]
                    candidate_hash_ids = [
                        self.passage_embedding_store.hash_ids[idx]
                        for idx in candidate_indices
                    ]
                    candidate_passages = [
                        self.passage_embedding_store.texts[idx]
                        for idx in candidate_indices
                    ]
                    _, final_scores, final_passages = self._rerank_passages(
                        query,
                        candidate_hash_ids,
                        candidate_passages,
                        top_k,
                    )
                else:
                    top_indices = sorted_indices[:top_k]
                    final_scores = [
                        float(sorted_scores[idx]) for idx in range(len(top_indices))
                    ]
                    final_passages = [
                        self.passage_embedding_store.texts[idx]
                        for idx in top_indices
                    ]

            result_entry = {
                "query": query,
                "passages": final_passages,
                "scores": final_scores,
            }
            if sub_query_texts is not None:
                result_entry["sub_queries"] = sub_query_texts
            retrieval_results.append(result_entry)
        return retrieval_results

    def _prepare_retrieval_cache(self):
        """Cache embeddings and graph lookups. Called once before retrieval loop."""
        self._entity_hash_ids = list(self.entity_embedding_store.hash_id_to_text.keys())
        self._entity_embeddings = np.asarray(self.entity_embedding_store.embeddings)
        self._passage_hash_ids = list(self.passage_embedding_store.hash_id_to_text.keys())
        self._passage_embeddings = np.asarray(self.passage_embedding_store.embeddings)
        self._su_embeddings = np.asarray(self.su_embedding_store.embeddings)
        self._su_hash_ids = list(self.su_embedding_store.hash_id_to_text.keys())

        logger.info("Building hypergraph...")
        self.knowledge_graph.build_hypergraph(
            self.entity_embedding_store,
            self.su_embedding_store,
            self.device,
        )

    def extract_seed_entities(self, query):
        """Extract query entities and map them onto indexed canonical entities."""
        if len(self._entity_hash_ids) == 0 or self._entity_embeddings.size == 0:
            return [], [], [], [], []

        query_mentions = self.ner_extractor.extract_query_entities(query)
        if not query_mentions:
            return [], [], [], [], []

        query_entity_embeddings = self.embedding_model.encode(
            [mention["embedding_text"] for mention in query_mentions],
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=self.config.batch_size,
            prompt=self.config.query_instruction_prefix,
        )
        similarities = np.dot(self._entity_embeddings, query_entity_embeddings.T)
        seed_indices = []
        seed_texts = []
        seed_hash_ids = []
        seed_scores = []
        for query_idx in range(len(query_mentions)):
            scores = similarities[:, query_idx]
            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])
            best_hash_id = self._entity_hash_ids[best_idx]
            best_metadata = self.entity_nodes_by_hash.get(best_hash_id, {})
            best_text = best_metadata.get(
                "canonical_name",
                self.entity_embedding_store.hash_id_to_text[best_hash_id],
            )
            seed_indices.append(best_idx)
            seed_texts.append(best_text)
            seed_hash_ids.append(best_hash_id)
            seed_scores.append(best_score)
        return query_mentions, seed_indices, seed_texts, seed_hash_ids, seed_scores

    def _diffuse_from_seeds(self, query, query_embedding, seed_indices,
                            seed_hash_ids, seed_scores,
                            sub_query_embeddings=None):
        """Bidirectional frontier expansion + SU-based passage scoring.

        Forward pass:  expand from query seed entities → activated SUs with scores.
        Backward pass: extract candidate entities from dense-retrieval top passages,
                       expand backwards → activated SUs with scores.
        Meet:          SUs traversed by BOTH forward and backward get a bonus.
        Scoring:       each passage is scored by its top-m SU scores (capped mean),
                       fused with dense retrieval similarity.
        """
        # ── Forward expansion ──
        forward_activated, forward_su_scores = frontier_expansion(
            config=self.config,
            hypergraph=self.knowledge_graph._hypergraph,
            entity_hash_ids=self._entity_hash_ids,
            su_embeddings=self._su_embeddings,
            question_embedding=query_embedding,
            seed_entity_indices=seed_indices,
            seed_entity_hash_ids=seed_hash_ids,
            seed_entity_scores=seed_scores,
            su_hash_ids=self._su_hash_ids,
            sub_query_embeddings=sub_query_embeddings,
        )

        # ── Backward seed extraction from dense-retrieval top passages ──
        dense_sims = np.dot(self._passage_embeddings, query_embedding)
        top_passage_indices = np.argsort(dense_sims)[::-1][:self.config.backward_seed_top_k]

        bw_seed_map = {}  # entity_hash_id -> (entity_idx, score)
        for p_idx in top_passage_indices:
            p_hash_id = self._passage_hash_ids[int(p_idx)]
            p_score = float(dense_sims[p_idx])
            if p_hash_id not in self.knowledge_graph.edge_weights:
                continue
            for entity_hash_id in self.knowledge_graph.edge_weights[p_hash_id]:
                if entity_hash_id in forward_activated:
                    continue
                if entity_hash_id not in self.entity_embedding_store.hash_id_to_idx:
                    continue
                entity_idx = self.entity_embedding_store.hash_id_to_idx[entity_hash_id]
                if entity_hash_id not in bw_seed_map or p_score > bw_seed_map[entity_hash_id][1]:
                    bw_seed_map[entity_hash_id] = (entity_idx, p_score)

        # ── Backward expansion ──
        backward_su_scores = {}
        if bw_seed_map:
            bw_hash_ids = list(bw_seed_map.keys())
            bw_indices = [bw_seed_map[h][0] for h in bw_hash_ids]
            bw_scores = [bw_seed_map[h][1] for h in bw_hash_ids]

            bw_config = copy.copy(self.config)
            bw_config.expansion_max_hops = self.config.backward_max_hops

            _, backward_su_scores = frontier_expansion(
                config=bw_config,
                hypergraph=self.knowledge_graph._hypergraph,
                entity_hash_ids=self._entity_hash_ids,
                su_embeddings=self._su_embeddings,
                question_embedding=query_embedding,
                seed_entity_indices=bw_indices,
                seed_entity_hash_ids=bw_hash_ids,
                seed_entity_scores=bw_scores,
                su_hash_ids=self._su_hash_ids,
                sub_query_embeddings=sub_query_embeddings,
            )

        # ── SU-level meet: SUs traversed by both forward and backward ──
        meeting_sus = set(forward_su_scores.keys()) & set(backward_su_scores.keys())
        bonus = self.config.meeting_su_bonus

        # Build final SU scores: forward scores + bonus for meeting SUs
        su_scores = {}
        for su_hash_id, score in forward_su_scores.items():
            if su_hash_id in meeting_sus:
                su_scores[su_hash_id] = score * bonus
            else:
                su_scores[su_hash_id] = score

        logger.info(
            "Bidirectional: forward_sus=%d, backward_sus=%d, meeting_sus=%d",
            len(forward_su_scores), len(backward_su_scores), len(meeting_sus),
        )

        # ── Passage scoring: top-m capped mean of SU scores + dense ──
        top_m = self.config.su_score_top_m
        passage_to_su = self.knowledge_graph.passage_to_su_ids

        su_coverage_scores = np.zeros(len(self._passage_hash_ids))
        for passage_idx, passage_hash_id in enumerate(self._passage_hash_ids):
            passage_sus = passage_to_su.get(passage_hash_id, [])
            if not passage_sus:
                continue
            # Collect scores for SUs in this passage that were activated
            scores = []
            for su_hash_id in passage_sus:
                if su_hash_id in su_scores:
                    scores.append(su_scores[su_hash_id])
            if not scores:
                continue
            # Top-m capped mean
            scores.sort(reverse=True)
            top_scores = scores[:top_m]
            su_coverage_scores[passage_idx] = sum(top_scores) / top_m

        dense_max = dense_sims.max()
        dense_norm = dense_sims / dense_max if dense_max > 0 else dense_sims
        coverage_max = su_coverage_scores.max()
        coverage_norm = su_coverage_scores / coverage_max if coverage_max > 0 else su_coverage_scores

        lam = self.config.scoring_lambda
        final_scores = lam * dense_norm + (1 - lam) * coverage_norm

        sorted_indices = np.argsort(final_scores)[::-1]
        sorted_hash_ids = [self._passage_hash_ids[idx] for idx in sorted_indices]
        sorted_scores = final_scores[sorted_indices].tolist()
        return sorted_hash_ids, sorted_scores

    def _lazy_load_reranker(self):
        if self.reranker is not None:
            return
        self.reranker = QwenReranker(
            model_name=self.config.reranker_model_name,
            batch_size=self.config.reranker_batch_size,
            max_length=self.config.reranker_max_length,
            instruction=self.config.reranker_instruction,
        )

    def _rerank_passages(self, query, candidate_hash_ids, candidate_passages, top_k):
        """Rerank candidate passages using the reranker model."""
        if not candidate_passages:
            return [], [], []
        rerank_scores = self.reranker.score(query, candidate_passages)
        reranked_indices = np.argsort(np.asarray(rerank_scores))[::-1]
        selected_indices = reranked_indices[:top_k]

        final_hash_ids = [candidate_hash_ids[idx] for idx in selected_indices]
        final_scores = [float(rerank_scores[idx]) for idx in selected_indices]
        final_passages = [candidate_passages[idx] for idx in selected_indices]
        return final_hash_ids, final_scores, final_passages

    # ====== QA ======

    MULTIHOP_SYSTEM_PROMPT = (
        "As an advanced reading comprehension assistant, your task is to analyze text passages "
        "and corresponding questions meticulously. Your response start after \"Thought: \", "
        "where you will methodically break down the reasoning process, illustrating how you "
        "arrive at conclusions. Conclude with \"Answer: \" to present a concise, definitive "
        "response, devoid of additional elaborations."
    )

    MULTIHOP_ONESHOT_USER = (
        "[1] The Last Horse (Spanish:El último caballo) is a 1950 Spanish comedy film "
        "directed by Edgar Neville starring Fernando Fernán Gómez.\n"
        "[2] The University of Southampton, which was founded in 1862 and received its "
        "Royal Charter as a university in 1952, has over 22,000 students. The university "
        "is ranked in the top 100 research universities in the world in the Academic "
        "Ranking of World Universities 2010. In 2010, the THES - QS World University "
        "Rankings positioned the University of Southampton in the top 80 universities in "
        "the world.\n"
        "[3] Stanton Township is a township in Champaign County, Illinois, USA. As of "
        "the 2010 census, its population was 505 and it contained 202 housing units.\n"
        "[4] Neville A. Stanton is a British Professor of Human Factors and Ergonomics "
        "at the University of Southampton. Prof Stanton is a Chartered Engineer (C.Eng), "
        "Chartered Psychologist (C.Psychol) and Chartered Ergonomist (C.ErgHF). He has "
        "written and edited over a forty books and over three hundered peer-reviewed "
        "journal papers on applications of the subject.\n"
        "[5] Finding Nemo Theatrical release poster Directed by Andrew Stanton Produced "
        "by Graham Walters Screenplay by Andrew Stanton Bob Peterson David Reynolds.\n"
        "\nQuestion: When was Neville A. Stanton's employer founded?\n"
    )

    MULTIHOP_ONESHOT_ASSISTANT = (
        "Thought: The employer of Neville A. Stanton is University of Southampton. "
        "The University of Southampton was founded in 1862.\n"
        "Answer: 1862."
    )

    GRAPHRAG_BENCH_SYSTEM_PROMPT = (
        "You are a careful reading comprehension assistant. You will receive several "
        "passages from a specialized document and a factual question.\n\n"
        "Rules:\n"
        "1. Some passages may be irrelevant. Identify and ignore them.\n"
        "2. Answer based ONLY on the passages. Do not add outside information.\n"
        "3. Use the exact words and phrases from the passages — do not paraphrase.\n"
        "4. Reuse key phrases from the question to frame your answer.\n"
        "5. Be concise:\n"
        "   - If the question asks for a specific entity (who, where, what name), "
        "answer with just that entity or a short noun phrase.\n"
        "   - Otherwise, answer in one sentence at most.\n"
        "6. Do not explain, justify, or add any extra commentary.\n\n"
        "Format:\n"
        "Answer: <your answer>"
    )

    def rag_qa(self, queries, num_to_retrieve=None, qa_mode="multihop"):
        """Retrieve passages and generate answers for a list of queries.

        Args:
            qa_mode: "multihop" for HotpotQA/MuSiQue/2WikiMultiHop,
                     "graphrag_bench" for GraphRAG-Bench (Medical/Novel).
        """
        retrieval_results = self.retrieve(queries, num_to_retrieve)

        if qa_mode == "graphrag_bench":
            system_prompt = self.GRAPHRAG_BENCH_SYSTEM_PROMPT
        else:
            system_prompt = self.MULTIHOP_SYSTEM_PROMPT

        all_messages = []
        for result in retrieval_results:
            prompt_user = ""
            for i, passage in enumerate(result["passages"], 1):
                prompt_user += f"[{i}] {passage}\n"
            prompt_user += f"\nQuestion: {result['query']}\n"
            sub_queries = result.get("sub_queries")
            if sub_queries:
                prompt_user += "\nReasoning clues (answer these in order to reach the final answer):\n"
                for i, sq in enumerate(sub_queries, 1):
                    prompt_user += f"  Step {i}: {sq}\n"
            prompt_user += "\n"
            if qa_mode == "multihop":
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": self.MULTIHOP_ONESHOT_USER},
                    {"role": "assistant", "content": self.MULTIHOP_ONESHOT_ASSISTANT},
                    {"role": "user", "content": prompt_user},
                ]
            else:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_user},
                ]
            all_messages.append(messages)

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            all_qa_results = list(tqdm(
                executor.map(self.llm_model.infer, all_messages),
                total=len(all_messages),
                desc="QA Reading (Parallel)",
            ))

        for qa_result, result in zip(all_qa_results, retrieval_results):
            if "Answer:" in qa_result:
                answer = qa_result.split("Answer:")[-1].strip()
            else:
                # No Answer: line — take the last non-empty line as the answer
                lines = [l.strip() for l in qa_result.strip().splitlines() if l.strip()]
                answer = lines[-1] if lines else qa_result.strip()
            result["answer"] = answer
        return retrieval_results
