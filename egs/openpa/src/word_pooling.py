"""Word representation pooling utilities."""

import torch


def mean_pool_word_bpe_states(hidden_states, word_starts, word_ends):
    """Mean-pool half-open decoder spans for every word."""
    if word_starts.shape != word_ends.shape:
        raise ValueError("word_starts and word_ends must have identical shapes")
    if torch.any(word_starts > word_ends):
        raise ValueError("BPE span start must not exceed its end")

    valid_spans = word_ends > word_starts
    outside_sequence = valid_spans & (
        (word_starts < 0) | (word_ends > hidden_states.size(1))
    )
    if torch.any(outside_sequence):
        raise ValueError("BPE span is outside the decoder sequence")
    token_positions = torch.arange(
        hidden_states.size(1), device=hidden_states.device
    ).view(1, 1, -1)
    span_mask = (token_positions >= word_starts.unsqueeze(-1)) & (
        token_positions < word_ends.unsqueeze(-1)
    )
    span_mask = span_mask & valid_spans.unsqueeze(-1)
    span_count = span_mask.sum(dim=-1, keepdim=True).clamp_min(1)
    return torch.einsum(
        "bwt,btd->bwd", span_mask.to(hidden_states.dtype), hidden_states
    ) / span_count
