from __future__ import annotations

import torch

from .config import GenerationConfig
from .manual_decoding import decode_stream_manual, decode_tokens_manual


def decode_tokens(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    gen_config: GenerationConfig,
) -> tuple[list[int], torch.Tensor]:
    return decode_tokens_manual(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        gen_config=gen_config,
    )


def decode_stream(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    gen_config: GenerationConfig,
):
    yield from decode_stream_manual(
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        attention_mask=attention_mask,
        gen_config=gen_config,
    )
