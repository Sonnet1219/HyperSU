"""All prompt templates for HyperSU: extraction, QA, and clue agent."""

from __future__ import annotations

import json


# ============================================================
# Entity Extraction Prompts
# ============================================================

EXTRACTION_SYSTEM_PROMPT = """\
Extract only the key reusable entities from the text.

Rules:
- Use exact text spans from the source.
- Extract only entities that are specific enough to connect evidence across different semantic units.
- Favor people, groups, locations, events, artifacts, documents, organizations, medical entities, and other important named concepts.
- Provide a short grounded description for each extraction that explains who or what it is in this local context.
- Do not extract pronouns, generic references, vague abstractions, scene-setting objects, or long clause-like spans.
- Do not return overlapping duplicates.
- Use `extraction_class` as a coarse entity type.

Return a JSON array of objects, each with keys: "extraction_class", "extraction_text", "description".
Do NOT wrap the JSON in markdown fences or add any other text."""

EXTRACTION_FEW_SHOT_EXAMPLES = [
    {
        "text": (
            "The travellers reached Kynance Cove before visiting "
            "Landewednack, where the rector welcomed them."
        ),
        "extractions": [
            {
                "extraction_class": "location",
                "extraction_text": "Kynance Cove",
                "description": "coastal cove reached by the travellers",
            },
            {
                "extraction_class": "location",
                "extraction_text": "Landewednack",
                "description": "Cornish village visited after the cove",
            },
            {
                "extraction_class": "person",
                "extraction_text": "the rector",
                "description": "local rector who welcomed the visitors",
            },
        ],
    },
    {
        "text": (
            "The biopsy confirmed Hodgkin lymphoma, and the patient "
            "started ABVD chemotherapy."
        ),
        "extractions": [
            {
                "extraction_class": "test",
                "extraction_text": "biopsy",
                "description": "diagnostic procedure confirming the disease",
            },
            {
                "extraction_class": "medical_condition",
                "extraction_text": "Hodgkin lymphoma",
                "description": "disease diagnosis confirmed by biopsy",
            },
            {
                "extraction_class": "treatment",
                "extraction_text": "ABVD chemotherapy",
                "description": "chemotherapy regimen started for treatment",
            },
        ],
    },
]


def build_extraction_messages(text: str) -> list[dict]:
    """Construct chat messages for entity extraction."""
    messages = [{"role": "system", "content": EXTRACTION_SYSTEM_PROMPT}]
    for example in EXTRACTION_FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": example["text"]})
        messages.append({
            "role": "assistant",
            "content": json.dumps(example["extractions"], ensure_ascii=False),
        })
    messages.append({"role": "user", "content": text})
    return messages


# ============================================================
# QA Prompts — Multihop (HotpotQA / 2WikiMultiHop / MuSiQue)
# ============================================================

MULTIHOP_SYSTEM_PROMPT = (
    "As an advanced reading comprehension assistant, your task is to analyze text passages "
    "and corresponding questions meticulously. Your response start after \"Thought: \", "
    "where you will methodically break down the reasoning process, illustrating how you "
    "arrive at conclusions. Conclude with \"Answer: \" to present a concise, definitive "
    "response, devoid of additional elaborations."
)

MULTIHOP_ONESHOT_USER = (
    "[1] The Last Horse (Spanish:El \u00faltimo caballo) is a 1950 Spanish comedy film "
    "directed by Edgar Neville starring Fernando Fern\u00e1n G\u00f3mez.\n"
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


# ============================================================
# QA Prompts — GraphRAG-Bench (Medical / Novel)
# ============================================================

GRAPHRAG_BENCH_SYSTEM_PROMPT = (
    "You are a careful reading comprehension assistant. You will receive several "
    "passages from a document and a question.\n\n"
    "Instructions:\n"
    "1. Some passages may be irrelevant. Identify and use only the relevant ones.\n"
    "2. Synthesize information across passages when the question requires it.\n"
    "3. Preserve key terms and proper nouns from the passages in your answer.\n"
    "4. Answer in 1\u20132 concise sentences. Be direct \u2014 no hedging or filler.\n"
    "5. If the question asks for a specific entity (who, where, what name), "
    "answer with just that entity or a short noun phrase.\n"
    "6. Do not add information beyond what the passages support.\n"
    "7. If the passages genuinely do not contain the answer, say so briefly.\n\n"
    "Format:\n"
    "Reasoning: <which passages are relevant and how they connect>\n"
    "Answer: <your answer>"
)
