# HyperSU

A hypergraph-based retrieval framework for multi-hop question answering.

## What's different

Most graph-based RAG systems build pairwise entity-relation graphs via LLM extraction. HyperSU takes a different route:

**Semantic units as hyperedges.** Instead of asking an LLM to rewrite the corpus into relation triples or summaries, we chunk it into semantic units (coherent sentence groups within a passage) and treat each unit directly as a hyperedge connecting all entities that co-occur in it. The original text is preserved as-is, and we get a sparse entity–SU hypergraph where one edge can link multiple entities at once — a natural fit for multi-hop reasoning.

**Bidirectional frontier expansion.** Most graph-based RAG methods underutilize graph structure — they typically collect evidence within 1-hop of query entities and stop there. We push further with a bidirectional strategy: a forward pass expands from query seed entities through the hypergraph, while a backward pass seeds from dense-retrieval top passages and expands in reverse. SUs visited by *both* directions get a score bonus. This meet-in-the-middle design helps bridge long reasoning chains that single-direction expansion tends to miss.

## How it works

### Indexing

1. Chunk corpus into passages, split passages into semantic units
2. Extract entity mentions from each SU, normalize into canonical entities
3. Build the hypergraph: entities are vertices, SUs are hyperedges
4. Precompute embeddings for passages, entities, and SUs

### Retrieval

1. Decompose the question into sub-queries (planner)
2. Extract seed entities from the query, map to hypergraph vertices
3. Forward expansion: propagate from seeds through gated hyperedges
4. Backward expansion: seed from dense-retrieval top passages, expand in reverse
5. Merge: SUs hit by both passes get a bonus
6. Project activated SUs back to passages, fuse with dense similarity scores
7. Optionally rerank, then send top passages to a reader LLM

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_trf

export OPENAI_API_KEY="your-key"
export OPENAI_BASE_URL="your-base-url"
```

## Usage

```bash
# Multi-hop QA (HotpotQA / MuSiQue / 2WikiMultihopQA)
python benchmarks/multihop/run.py \
  --dataset_name hotpotqa \
  --llm_model gpt-4o-mini \
  --langextract_model gpt-4o-mini \
  --expansion_max_hops 3 \
  --expansion_top_k 15 \
  --scoring_lambda 0.7

# GraphRAG-Bench
python benchmarks/graphrag_bench/run.py \
  --corpus_name medical \
  --llm_model gpt-4o-mini \
  --langextract_model gpt-4o-mini \
  --expansion_max_hops 3 \
  --expansion_top_k 15
```

## Key parameters

| Parameter | What it controls |
|---|---|
| `--expansion_max_hops` | Max hops in hypergraph expansion |
| `--expansion_top_k` | Vertices kept per hop |
| `--conductance_floor` | Gate threshold for hyperedge activation |
| `--conductance_gamma` | Sharpening exponent for gate scores |
| `--scoring_lambda` | Fusion weight: dense similarity vs. hypergraph coverage |
| `--backward_seed_top_k` | Passages used to seed backward expansion |
| `--backward_max_hops` | Max hops for backward pass |

## Project structure

```
hypersu/
  engine.py          # main pipeline (indexing + retrieval + QA)
  frontier.py        # hypergraph frontier expansion
  knowledge_graph.py # hypergraph construction and storage
  chunker.py         # corpus chunking and SU splitting
  ner.py             # entity extraction
  planner.py         # sub-query decomposition
  reranker.py        # passage reranking
  config.py          # all configuration
```
