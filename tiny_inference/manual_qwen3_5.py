from __future__ import annotations

import torch
from transformers.masking_utils import create_causal_mask
from .manual_attention import qwen3_5_attention_forward
from .manual_linear import qwen3_5_linear_attn_forward


@torch.no_grad()
def qwen3_5_text_forward(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    past_key_values: None,
    use_cache: bool,
) -> tuple[torch.Tensor, None]:
    text_model = model.model
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    inputs_embeds = text_model.embed_tokens(input_ids)

    past_seen_tokens = 0
    cache_position = torch.arange(
        past_seen_tokens,
        past_seen_tokens + inputs_embeds.shape[1],
        device=inputs_embeds.device,
    )

    position_ids = cache_position.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
    text_position_ids = position_ids[0]
    position_ids = position_ids[1:]

    causal_mask = create_causal_mask(
        config=text_model.config,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=None,
        position_ids=text_position_ids,
    )
    linear_attn_mask = text_model._update_linear_attn_mask(attention_mask, cache_position)

    hidden_states = inputs_embeds
    position_embeddings = text_model.rotary_emb(hidden_states, position_ids)

    for decoder_layer in text_model.layers[: text_model.config.num_hidden_layers]:
        layer_mask = linear_attn_mask if decoder_layer.layer_type == "linear_attention" else causal_mask
        residual = hidden_states
        hidden_states = decoder_layer.input_layernorm(hidden_states)

        if decoder_layer.layer_type == "linear_attention":
            hidden_states = qwen3_5_linear_attn_forward(
                decoder_layer.linear_attn,
                hidden_states=hidden_states,
                attention_mask=layer_mask,
            )
        else:
            hidden_states, _ = qwen3_5_attention_forward(
                decoder_layer.self_attn,
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=layer_mask,
            )

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
        hidden_states = decoder_layer.mlp(hidden_states)
        hidden_states = residual + hidden_states

    hidden_states = text_model.norm(hidden_states)
    logits = model.lm_head(hidden_states)
    return logits, None
