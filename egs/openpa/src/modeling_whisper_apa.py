"""Whisper with decoder-probe heads for pronunciation assessment."""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from transformers import WhisperForConditionalGeneration
from transformers.modeling_outputs import ModelOutput


class IndependentScoreHeads(nn.Module):
    """Predict each APA dimension with its own normalized regression head."""

    def __init__(self, hidden_size, num_scores, dropout=0.0):
        super().__init__()
        self.heads = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 1),
            )
            for _ in range(num_scores)
        )

    def forward(self, hidden_states):
        return torch.cat([head(hidden_states) for head in self.heads], dim=-1)

    def forward_aspects(self, hidden_states):
        """Apply head i only to the representation pooled for aspect i."""
        if hidden_states.ndim != 3 or hidden_states.size(1) != len(self.heads):
            raise ValueError(
                "Aspect-specific states must have shape "
                f"[batch, {len(self.heads)}, hidden]"
            )
        return torch.cat(
            [head(hidden_states[:, index]) for index, head in enumerate(self.heads)],
            dim=-1,
        )


class MaskedAttentionPooling(nn.Module):
    """Learn one normalized weight per valid decoder word representation."""

    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, hidden_states, mask):
        scores = self.attention(hidden_states).squeeze(-1)
        scores = scores.masked_fill(~mask.bool(), -torch.inf)
        weights = torch.softmax(scores.float(), dim=-1).to(hidden_states.dtype)
        return torch.sum(hidden_states * weights.unsqueeze(-1), dim=1)


class AspectSpecificAttentionPooling(nn.Module):
    """Learn an independent masked word-attention distribution per score."""

    def __init__(self, hidden_size, num_scores):
        super().__init__()
        self.pools = nn.ModuleList(
            MaskedAttentionPooling(hidden_size) for _ in range(num_scores)
        )

    def forward(self, hidden_states, mask):
        return torch.stack(
            [pool(hidden_states, mask) for pool in self.pools],
            dim=1,
        )


class BidirectionalUtteranceContext(nn.Module):
    """Add lightweight full-context refinement to causal word states."""

    def __init__(self, hidden_size, context_size=128, num_heads=4, dropout=0.):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, context_size),
        )
        self.context_layer = nn.TransformerEncoderLayer(
            d_model=context_size,
            nhead=num_heads,
            dim_feedforward=2 * context_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.output_projection = nn.Linear(context_size, hidden_size)

    def forward(self, hidden_states, mask):
        context = self.input_projection(hidden_states)
        context = self.context_layer(
            context,
            src_key_padding_mask=~mask.bool(),
        )
        return hidden_states + self.output_projection(context)


class SharedHandcraftedFusion(nn.Module):
    """Group-normalize and append the fixed handcrafted feature vector."""

    def __init__(
        self,
        num_fluency_features,
        num_prosodic_features,
    ):
        super().__init__()
        self.num_fluency_features = num_fluency_features
        self.fluency_normalization = nn.LayerNorm(num_fluency_features)
        self.prosodic_normalization = nn.LayerNorm(num_prosodic_features)

    def forward(self, hidden_states, features):
        features = features.float()
        fluency_features = features[:, : self.num_fluency_features]
        prosodic_features = features[:, self.num_fluency_features :]
        normalized_fluency = self.fluency_normalization(fluency_features)
        normalized_prosodic = self.prosodic_normalization(prosodic_features)
        if hidden_states.ndim == 3:
            normalized_fluency = normalized_fluency.unsqueeze(1).expand(
                -1, hidden_states.size(1), -1
            )
            normalized_prosodic = normalized_prosodic.unsqueeze(1).expand(
                -1, hidden_states.size(1), -1
            )
        return torch.cat(
            (
                hidden_states.float(),
                normalized_fluency,
                normalized_prosodic,
            ),
            dim=-1,
        )


@dataclass
class WhisperAPAOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    word_logits: Optional[torch.FloatTensor] = None
    utterance_logits: Optional[torch.FloatTensor] = None
    asr_loss: Optional[torch.FloatTensor] = None
    word_loss: Optional[torch.FloatTensor] = None
    utterance_loss: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None


class WhisperForPronunciationAssessment(WhisperForConditionalGeneration):
    """Multi-task Whisper whose APA features come from <score> positions."""

    def __init__(self, config):
        super().__init__(config)
        hidden_size = config.d_model
        num_word_scores = getattr(config, "num_word_scores", 1)
        num_utterance_scores = getattr(config, "num_utterance_scores", 4)
        head_dropout = getattr(config, "apa_head_dropout", 0.0)
        self.handcrafted_features = getattr(
            config,
            "handcrafted_features",
            getattr(config, "shared_handcrafted_fusion", False),
        )
        self.word_score_head = IndependentScoreHeads(
            hidden_size, num_word_scores, head_dropout
        )
        self.word_representation = getattr(config, "word_representation", "score-token")
        if self.word_representation not in {"score-token", "bpe-mean"}:
            raise ValueError(
                "word_representation must be 'score-token' or 'bpe-mean', "
                f"got {self.word_representation!r}"
            )
        num_fluency_features = int(getattr(config, "num_fluency_features", 8))
        num_prosodic_features = int(getattr(config, "num_prosodic_features", 3))
        shared_feature_size = num_fluency_features + num_prosodic_features
        utterance_hidden_size = (
            hidden_size + shared_feature_size
            if self.handcrafted_features
            else hidden_size
        )
        self.utterance_score_head = IndependentScoreHeads(
            utterance_hidden_size,
            num_utterance_scores,
            head_dropout,
        )
        representation_dropout = getattr(
            config,
            "utterance_representation_dropout",
            getattr(config, "handcrafted_high_dim_dropout", 0.0),
        )
        self.utterance_representation_dropout = nn.Dropout(
            representation_dropout
        )
        self.shared_handcrafted_fusion = (
            SharedHandcraftedFusion(
                num_fluency_features,
                num_prosodic_features,
            )
            if self.handcrafted_features
            else nn.Identity()
        )
        utterance_pooling = getattr(config, "utterance_pooling", "mean")
        if utterance_pooling not in {"mean", "attention", "aspect-attention"}:
            raise ValueError(
                "Utterance pooling must be 'mean', 'attention', or "
                "'aspect-attention', "
                f"got {utterance_pooling!r}"
            )
        self.utterance_pooling = utterance_pooling
        if utterance_pooling == "attention":
            self.utterance_pool = MaskedAttentionPooling(hidden_size)
        elif utterance_pooling == "aspect-attention":
            self.utterance_pool = AspectSpecificAttentionPooling(
                hidden_size, num_utterance_scores
            )
        else:
            self.utterance_pool = nn.Identity()
        self.asr_loss_weight = getattr(config, "asr_loss_weight", 1.0)
        self.use_utterance_context_layer = getattr(
            config, "utterance_context_layer", False
        )
        self.utterance_context = (
            BidirectionalUtteranceContext(hidden_size)
            if self.use_utterance_context_layer
            else nn.Identity()
        )
        self.mask_score_token_asr_loss = getattr(
            config, "mask_score_token_asr_loss", False
        )
        self.word_loss_weight = getattr(config, "word_loss_weight", 1.0)
        self.binary_word_label_indices = set(
            getattr(config, "binary_word_label_indices", [])
        )
        self.stress_loss_weight = getattr(config, "stress_loss_weight", 1.0)
        self.stress_incorrect_weight = getattr(
            config, "stress_incorrect_weight", 1.0
        )
        self.utterance_loss_weight = getattr(config, "utterance_loss_weight", 1.0)
        self.post_init()

    def forward(
        self,
        input_features=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        decoder_inputs_embeds=None,
        decoder_position_ids=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
        score_positions=None,
        word_start_positions=None,
        word_end_positions=None,
        word_labels=None,
        word_label_mask=None,
        word_loss_weights=None,
        word_syllable_counts=None,
        word_stress_all_labels=None,
        utterance_labels=None,
        utterance_loss_weights=None,
        fluency_features=None,
        prosodic_features=None,
        **kwargs,
    ):
        del word_syllable_counts, word_stress_all_labels, kwargs
        return_dict = True if return_dict is None else return_dict
        if not return_dict:
            raise ValueError("WhisperForPronunciationAssessment requires return_dict=True")

        outputs = self.model(
            input_features=input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            decoder_position_ids=decoder_position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=bool(output_hidden_states),
            return_dict=True,
            cache_position=cache_position,
        )
        causal_decoder_hidden = outputs.last_hidden_state
        lm_logits = self.proj_out(causal_decoder_hidden)

        asr_loss = None
        if labels is not None and self.asr_loss_weight != 0:
            asr_labels = labels
            if self.mask_score_token_asr_loss:
                score_token_id = getattr(self.config, "score_token_id", None)
                if score_token_id is None:
                    raise ValueError(
                        "score_token_id is required when mask_score_token_asr_loss=True"
                    )
                asr_labels = labels.masked_fill(labels.eq(score_token_id), -100)
            if asr_labels.ne(-100).any():
                asr_loss = F.cross_entropy(
                    lm_logits.reshape(-1, self.config.vocab_size), asr_labels.reshape(-1),
                    ignore_index=-100,
                )

        word_logits = None
        utterance_logits = None
        word_loss = None
        utterance_loss = None
        word_ends = word_end_positions if word_end_positions is not None else score_positions
        if word_ends is not None:
            if word_ends.dtype != torch.long:
                word_ends = word_ends.long()
            batch_size, max_words = word_ends.shape
            if self.word_representation == "bpe-mean":
                from word_pooling import mean_pool_word_bpe_states

                if word_start_positions is None:
                    raise ValueError(
                        "word_start_positions is required for BPE word representation"
                    )
                word_hidden = mean_pool_word_bpe_states(
                    causal_decoder_hidden,
                    word_start_positions,
                    word_ends,
                )
            else:
                gather_index = word_ends.unsqueeze(-1).expand(
                    -1, -1, causal_decoder_hidden.size(-1)
                )
                word_hidden = causal_decoder_hidden.gather(1, gather_index)
            word_logits = self.word_score_head(word_hidden)

            if word_label_mask is None:
                word_label_mask = torch.ones_like(word_ends, dtype=torch.bool)
            word_label_mask = word_label_mask.bool()
            utterance_word_hidden = (
                self.utterance_context(word_hidden, word_label_mask)
                if self.use_utterance_context_layer
                else word_hidden
            )
            valid_count = word_label_mask.sum(dim=1, keepdim=True).clamp_min(1)
            decoder_mean_repr = (
                utterance_word_hidden * word_label_mask.unsqueeze(-1)
            ).sum(dim=1) / valid_count
            if self.utterance_pooling in {"attention", "aspect-attention"}:
                utterance_repr = self.utterance_pool(
                    utterance_word_hidden, word_label_mask
                )
            else:
                utterance_repr = decoder_mean_repr
            utterance_repr = self.utterance_representation_dropout(
                utterance_repr
            )
            if self.handcrafted_features:
                if fluency_features is None or prosodic_features is None:
                    raise ValueError(
                        "fluency_features and prosodic_features are required "
                        "when handcrafted_features=True"
                    )
                shared_features = torch.cat(
                    (fluency_features.float(), prosodic_features.float()), dim=-1
                )
                utterance_repr = self.shared_handcrafted_fusion(
                    utterance_repr, shared_features
                )
            if self.utterance_pooling == "aspect-attention":
                utterance_logits = self.utterance_score_head.forward_aspects(
                    utterance_repr
                )
            else:
                utterance_logits = self.utterance_score_head(utterance_repr)

            if word_labels is not None and word_label_mask.any():
                if word_loss_weights is None:
                    word_loss_weights = torch.ones_like(
                        word_label_mask, dtype=word_logits.dtype
                    )
                else:
                    word_loss_weights = word_loss_weights.to(word_logits.dtype)
                per_target_losses = []
                for index in range(word_logits.size(-1)):
                    valid = (
                        word_label_mask
                        & word_labels[:, :, index].ne(-100.0)
                        & word_loss_weights.gt(0)
                    )
                    if not valid.any():
                        continue
                    if index in self.binary_word_label_indices:
                        binary_target = word_labels[:, :, index].float()[valid]
                        binary_loss = F.binary_cross_entropy_with_logits(
                            word_logits[:, :, index][valid],
                            binary_target,
                            reduction="none",
                        )
                        class_weight = torch.where(
                            binary_target.eq(0),
                            binary_loss.new_tensor(self.stress_incorrect_weight),
                            binary_loss.new_tensor(1.0),
                        )
                        class_weight = class_weight * word_loss_weights[valid]
                        target_loss = (
                            (binary_loss * class_weight).sum() / class_weight.sum()
                        ) * self.stress_loss_weight
                    else:
                        target_loss = F.mse_loss(
                            word_logits[:, :, index][valid],
                            word_labels[:, :, index].float()[valid],
                            reduction="none",
                        )
                        weights = word_loss_weights[valid]
                        target_loss = (target_loss * weights).sum() / weights.sum()
                    per_target_losses.append(target_loss)
                if per_target_losses:
                    word_loss = sum(per_target_losses) / len(per_target_losses)
            if utterance_labels is not None:
                valid_utterance = utterance_labels.ne(-100.0)
                if valid_utterance.any():
                    element_loss = F.mse_loss(
                        utterance_logits,
                        utterance_labels.float(),
                        reduction="none",
                    )
                    if utterance_loss_weights is None:
                        utterance_loss_weights = torch.ones(
                            utterance_logits.size(0),
                            device=utterance_logits.device,
                            dtype=utterance_logits.dtype,
                        )
                    weights = utterance_loss_weights.to(
                        device=utterance_logits.device,
                        dtype=utterance_logits.dtype,
                    ).unsqueeze(-1).expand_as(element_loss)
                    weights = weights * valid_utterance.to(weights.dtype)
                    utterance_loss = (element_loss * weights).sum() / valid_utterance.sum()

        losses = []
        if asr_loss is not None:
            losses.append(self.asr_loss_weight * asr_loss)
        if word_loss is not None:
            losses.append(self.word_loss_weight * word_loss)
        if utterance_loss is not None:
            losses.append(self.utterance_loss_weight * utterance_loss)
        loss = sum(losses) if losses else None

        return WhisperAPAOutput(
            loss=loss,
            logits=lm_logits,
            word_logits=word_logits,
            utterance_logits=utterance_logits,
            asr_loss=asr_loss,
            word_loss=word_loss,
            utterance_loss=utterance_loss,
            past_key_values=outputs.past_key_values,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_attentions=outputs.encoder_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
        )


def initialize_score_token_weights(model, score_token_id):
    """Give the newly appended token a deterministic, reloadable initialization."""
    with torch.no_grad():
        input_weight = model.get_input_embeddings().weight
        source_input = torch.cat((input_weight[:score_token_id], input_weight[score_token_id + 1 :]))
        input_weight[score_token_id].copy_(source_input.mean(dim=0))
        output_layer = model.get_output_embeddings()
        if output_layer is not None and output_layer.weight.data_ptr() != input_weight.data_ptr():
            output_weight = output_layer.weight
            source_output = torch.cat((output_weight[:score_token_id], output_weight[score_token_id + 1 :]))
            output_weight[score_token_id].copy_(source_output.mean(dim=0))
