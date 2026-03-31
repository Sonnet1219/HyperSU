import logging
from typing import Sequence
import importlib.util

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


logger = logging.getLogger(__name__)


class QwenReranker:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Reranker-4B",
        batch_size: int = 2,
        max_length: int = 4096,
        instruction: str = (
            "Given a multi-hop question, judge whether the document contains "
            "evidence that helps answer the question, either directly or as an intermediate bridge."
        ),
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.instruction = instruction
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Loading reranker model: %s", self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {
            "dtype": torch.bfloat16 if self.device.type == "cuda" else torch.float32,
            "attn_implementation": "sdpa",
        }

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)

        self.model.to(self.device)
        self.model.eval()
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.prefix = (
            "<|im_start|>system\n"
            "Judge whether the Document meets the requirements based on the Query and the "
            "Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
            "<|im_end|>\n<|im_start|>user\n"
        )
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        logger.info(
            "Reranker ready on %s with batch_size=%s, max_length=%s",
            self.device,
            self.batch_size,
            self.max_length,
        )

    def _format_pair(self, query: str, document: str) -> str:
        return (
            f"<Instruct>: {self.instruction}\n"
            f"<Query>: {query}\n"
            f"<Document>: {document}"
        )

    def _prepare_inputs(self, pairs: Sequence[tuple[str, str]]):
        usable_length = self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        serialized_pairs = [self._format_pair(query, document) for query, document in pairs]
        inputs = self.tokenizer(
            serialized_pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=usable_length,
        )
        for idx, token_ids in enumerate(inputs["input_ids"]):
            inputs["input_ids"][idx] = self.prefix_tokens + token_ids + self.suffix_tokens
        padded_inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt")
        return {key: value.to(self.device) for key, value in padded_inputs.items()}

    @torch.inference_mode()
    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []

        all_scores: list[float] = []
        for start in range(0, len(documents), self.batch_size):
            batch_documents = documents[start:start + self.batch_size]
            batch_pairs = [(query, document) for document in batch_documents]
            inputs = self._prepare_inputs(batch_pairs)
            # Only compute logits for the last token to save memory
            outputs = self.model(**inputs)
            last_logits = outputs.logits[:, -1, :]
            true_logits = last_logits[:, self.token_true_id]
            false_logits = last_logits[:, self.token_false_id]
            pair_logits = torch.stack([false_logits, true_logits], dim=1)
            batch_scores = torch.softmax(pair_logits, dim=1)[:, 1]
            all_scores.extend(batch_scores.float().cpu().tolist())
            del outputs, last_logits, inputs
            torch.cuda.empty_cache()
        return all_scores
