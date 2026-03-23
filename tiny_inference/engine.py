from __future__ import annotations

import json
import time
from typing import Any, Iterable, Optional

import torch
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

from .config import GenerationConfig
from .decoding import decode_stream, decode_tokens
from .tools import parse_tool_calls


def parse_messages(messages_json: str) -> list[dict[str, Any]]:
    if not messages_json:
        return []
    data = json.loads(messages_json)
    if isinstance(data, list):
        return data
    raise ValueError("messages must be a JSON list.")


class TinyQwenEngine:
    def __init__(self, model_name_or_path: str):
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        self.model = Qwen3_5ForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float32,
        ).to(self.device)
        self.model.config._attn_implementation = "eager"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def _prepare_inputs(
        self,
        prompt: Optional[str],
        messages: Optional[list[dict[str, Any]]],
        enable_thinking: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if messages:
            encoded = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=enable_thinking,
            )
        else:
            encoded = self.tokenizer(
                prompt or "",
                return_tensors="pt",
            )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(self.device)
        return input_ids, attention_mask

    def generate(
        self,
        prompt: Optional[str] = None,
        messages: Optional[list[dict[str, Any]]] = None,
        gen_config: Optional[GenerationConfig] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        benchmark: bool = False,
    ) -> dict[str, Any]:
        gen_config = gen_config or GenerationConfig()
        input_ids, attention_mask = self._prepare_inputs(prompt, messages, gen_config.enable_thinking)

        start = time.time()
        generated_ids, full_ids = decode_tokens(
            model=self.model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            gen_config=gen_config,
        )
        end = time.time()

        text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        tool_calls = parse_tool_calls(text)

        usage = {
            "prompt_tokens": int(input_ids.shape[1]),
            "completion_tokens": int(len(generated_ids)),
            "total_tokens": int(full_ids.shape[1]),
        }

        result: dict[str, Any] = {
            "text": text,
            "tool_calls": tool_calls,
            "usage": usage,
        }

        if benchmark:
            elapsed = max(end - start, 1e-6)
            result["metrics"] = {
                "elapsed_s": elapsed,
                "tokens_per_s": len(generated_ids) / elapsed,
            }
        return result

    def generate_stream(
        self,
        prompt: Optional[str] = None,
        messages: Optional[list[dict[str, Any]]] = None,
        gen_config: Optional[GenerationConfig] = None,
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> Iterable[str]:
        gen_config = gen_config or GenerationConfig()
        input_ids, attention_mask = self._prepare_inputs(prompt, messages, gen_config.enable_thinking)        
        yield from decode_stream(
            model=self.model,
            tokenizer=self.tokenizer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            gen_config=gen_config,
        )
