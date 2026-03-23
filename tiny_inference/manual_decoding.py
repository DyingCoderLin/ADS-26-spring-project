from __future__ import annotations

import torch

from .config import GenerationConfig
from .manual_qwen3_5 import qwen3_5_text_forward
from .sampling import sample_next_token


def _eos_token_ids(model) -> set[int]:
    """Match HF GenerationMixin: stop if *any* configured EOS id is sampled."""
    raw = getattr(model.config, "eos_token_id", None)
    if raw is None:
        return set()
    if isinstance(raw, (list, tuple)):
        return {int(x) for x in raw if x is not None}
    return {int(raw)}


@torch.no_grad()
def decode_tokens_manual(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    gen_config: GenerationConfig,
) -> tuple[list[int], torch.Tensor]:
    model.eval()
    generated: list[int] = []
    eos_ids = _eos_token_ids(model)

    for step in range(gen_config.max_new_tokens):
        logits, _ = qwen3_5_text_forward(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=False,
        )

        logits = logits[:, -1, :]
        next_token = sample_next_token(logits, gen_config)
        input_ids = torch.cat([input_ids, next_token], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)
        token_id = int(next_token.item())
        generated.append(token_id)
        if token_id in eos_ids:
            break

    return generated, input_ids


@torch.no_grad()
def decode_stream_manual(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    gen_config: GenerationConfig,
):
    model.eval()
    eos_ids = _eos_token_ids(model)

    for step in range(gen_config.max_new_tokens):
        logits, _ = qwen3_5_text_forward(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=False,
        )

        logits = logits[:, -1, :]
        next_token = sample_next_token(logits, gen_config)
        input_ids = torch.cat([input_ids, next_token], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)
        token_id = int(next_token.item())
        if token_id in eos_ids:
            break
        piece = tokenizer.decode([token_id], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if piece:
            yield piece
