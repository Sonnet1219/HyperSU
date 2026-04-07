# HyperSU

A hypergraph-based retrieval framework for multi-hop question answering.

## What's different

Most graph-based RAG systems build pairwise entity-relation graphs via LLM extraction. HyperSU takes a different route:

**Semantic units as hyperedges.** We chunk the corpus into semantic units (coherent sentence groups) and treat each SU as a hyperedge connecting all co-occurring entities. Unlike pairwise knowledge graphs that flatten n-ary relations into multiple triples and lose joint context, a single hyperedge naturally preserves the complete relational structure — three or more entities participating in one event, one causal chain, or one comparison stay linked in a single edge. Original text is kept as-is; no LLM rewriting needed.

**Planner-guided conductance gating.** A planner agent decomposes the input question into minimal sub-queries, each targeting one reasoning hop. These sub-query embeddings jointly condition the hyperedge conductance gate: an SU is activated when it is semantically relevant to *any* sub-query. This lets a single retrieval pass cover the full multi-hop reasoning chain without iterative LLM calls per hop.

**Bidirectional frontier expansion.** We run two concurrent propagation processes on the hypergraph. The *forward pass* seeds from query-linked entities and expands outward hop-by-hop: at each hop, activated entities scatter scores through conductance-gated hyperedges to discover new entities, retaining the top-K per round. The *backward pass* seeds from an answer candidate pool (entities anchored to high-confidence passages) and propagates inward along the same gating mechanism. SUs traversed by both directions receive a convergence bonus. This meet-in-the-middle design recovers reasoning paths that single-direction expansion cannot reach within bounded hops.

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
  --ner_model gpt-4o-mini \
  --expansion_max_hops 3 \
  --expansion_top_k 15 \
  --scoring_lambda 0.7

# GraphRAG-Bench
python benchmarks/graphrag_bench/run.py \
  --corpus_name medical \
  --llm_model gpt-4o-mini \
  --ner_model gpt-4o-mini \
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
