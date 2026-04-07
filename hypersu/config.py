from dataclasses import dataclass


@dataclass
class HyperSUConfig:
    save_dir: str = "./index_store"
    embedding_model_name: str = "BAAI/bge-large-en-v1.5"
    llm_model_name: str = "gpt-4o-mini"
    query_instruction_prefix: str = "Represent this sentence for searching relevant passages: "
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    spacy_model: str = "en_core_web_trf"
    batch_size: int = 128
    max_workers: int = 16
    retrieval_top_k: int = 5
    scoring_lambda: float = 0.5
    expansion_max_hops: int = 4
    expansion_top_k: int = 30
    hop_decay: float = 0.5
    conductance_floor: float = 0.5
    conductance_gamma: float = 1.0
    semantic_unit_percentile: int = 60
    use_reranker: bool = False
    reranker_model_name: str = "Qwen/Qwen3-Reranker-4B"
    reranker_candidate_top_k: int = 30
    reranker_batch_size: int = 8
    reranker_max_length: int = 4096
    reranker_instruction: str = (
        "Given a multi-hop question, judge whether the document contains evidence "
        "that helps answer the question, either directly or as an intermediate bridge."
    )
    ner_model_id: str = "gpt-4o-mini"
    entity_merge_threshold: float = 0.90
    # Bidirectional expansion
    backward_seed_top_k: int = 10
    backward_max_hops: int = 3
    meeting_su_bonus: float = 2.0
    su_score_top_m: int = 3
    # Ablation
    disable_backward: bool = False
    # Planner
    use_planner: bool = False
    planner_model_name: str = "gpt-4o-mini"
    planner_max_subqueries: int = 3
