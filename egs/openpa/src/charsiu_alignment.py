"""Minimal English Charsiu forced alignment used by open-response evaluation."""

import re
import unicodedata
from itertools import chain, groupby
from pathlib import Path

import numpy as np
import nltk
import torch
from nltk.tokenize import TweetTokenizer
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)


def _monotonic_forced_align(frame_scores, phone_ids):
    """Assign every frame to a phone with Charsiu's stay/advance DTW steps."""
    frames, phones = frame_scores.shape[0], len(phone_ids)
    if not phones or frames < phones:
        raise ValueError(f"Cannot align {phones} phones to {frames} frames")
    cost = np.full((frames, phones), np.inf, dtype=np.float64)
    backtrack = np.zeros((frames, phones), dtype=np.int8)
    cost[0, 0] = -frame_scores[0, phone_ids[0]]
    for frame in range(1, frames):
        first = max(0, phones - (frames - frame))
        last = min(phones - 1, frame)
        for phone in range(first, last + 1):
            stay = cost[frame - 1, phone]
            advance = cost[frame - 1, phone - 1] if phone else np.inf
            if advance < stay:
                previous, backtrack[frame, phone] = advance, 1
            else:
                previous = stay
            cost[frame, phone] = previous - frame_scores[frame, phone_ids[phone]]
    if not np.isfinite(cost[-1, -1]):
        raise ValueError("Charsiu monotonic alignment has no valid path")
    alignment = [0] * frames
    phone = phones - 1
    for frame in range(frames - 1, -1, -1):
        alignment[frame] = phone
        if frame and backtrack[frame, phone]:
            phone -= 1
    return alignment


class CharsiuForcedAligner:
    """Small inference-only equivalent of Charsiu's English forced aligner."""

    def __init__(
        self,
        model_name="charsiu/en_w2v2_fc_10ms",
        tokenizer_name="charsiu/tokenizer_en_cmu",
        device="cuda:0",
        local_files_only=False,
        cache_dir=None,
        silence_threshold=4,
        resolution=0.01,
    ):
        project_nltk_data = Path(__file__).resolve().parent / ".cache" / "nltk_data"
        if project_nltk_data.is_dir():
            nltk.data.path.insert(0, str(project_nltk_data))
        try:
            from g2p_en import G2p
            from g2p_en.expand import normalize_numbers
        except ImportError as error:
            raise RuntimeError(
                "Charsiu alignment requires g2p-en; install project requirements"
            ) from error

        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
            tokenizer_name,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
        )
        extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=16000,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=False,
        )
        self.processor = Wav2Vec2Processor(
            feature_extractor=extractor, tokenizer=tokenizer
        )
        self.model = Wav2Vec2ForCTC.from_pretrained(
            model_name,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
        ).eval().to(device)
        self.device = torch.device(device)
        self.g2p = G2p()
        self.normalize_numbers = normalize_numbers
        self.word_tokenize = TweetTokenizer().tokenize
        self.silence = "[SIL]"
        self.silence_id = tokenizer.convert_tokens_to_ids(self.silence)
        self.silence_threshold = silence_threshold
        self.resolution = resolution

    def _words_and_phones(self, text):
        normalized = self.normalize_numbers(str(text))
        normalized = "".join(
            character
            for character in unicodedata.normalize("NFD", normalized)
            if unicodedata.category(character) != "Mn"
        ).lower()
        normalized = re.sub(r"[^ a-z'.,?!\-]", "", normalized)
        normalized = normalized.replace("i.e.", "that is").replace(
            "e.g.", "for example"
        )
        words = self.word_tokenize(normalized)
        grouped_phones = list(
            tuple(group)
            for is_phone, group in groupby(self.g2p(normalized), key=lambda item: item != " ")
            if is_phone
        )
        pairs = [
            (word, phones)
            for word, phones in zip(words, grouped_phones)
            if phones and re.search(r"\w+\d?", phones[0])
        ]
        words = [word for word, _ in pairs]
        phones = [phones for _, phones in pairs]
        if not words:
            raise ValueError(f"Charsiu text produced no words: {text!r}")
        return words, phones

    def _silence_mask(self, scores):
        prediction = np.argmax(scores, axis=-1)
        mask = []
        for phone_id, group in groupby(prediction):
            group = list(group)
            if phone_id == self.silence_id and len(group) < self.silence_threshold:
                mask.extend([-1] * len(group))
            else:
                mask.extend(group)
        return np.asarray(mask)

    def align(self, waveform, text):
        words, phones_by_word = self._words_and_phones(text)
        phones = list(chain.from_iterable(phones_by_word))
        phone_ids = [
            self.processor.tokenizer.convert_tokens_to_ids(re.sub(r"\d", "", phone))
            for phone in phones
        ]
        inputs = self.processor(
            np.asarray(waveform, dtype=np.float32),
            sampling_rate=16000,
            return_tensors="pt",
        ).input_values.to(self.device)
        with torch.no_grad():
            scores = torch.softmax(self.model(inputs).logits, dim=-1)[0].cpu().numpy()
        silence_mask = self._silence_mask(scores)
        nonsilence = np.flatnonzero(silence_mask != self.silence_id)
        if not len(nonsilence):
            raise ValueError("Charsiu found no active speech")
        aligned_indices = _monotonic_forced_align(scores[nonsilence], phone_ids)
        frame_alignment = []
        cursor = 0
        for value in silence_mask:
            if value == self.silence_id:
                frame_alignment.append(None)
            else:
                frame_alignment.append(aligned_indices[cursor])
                cursor += 1

        phone_spans = []
        frame = 0
        for phone_index, group in groupby(frame_alignment):
            length = len(list(group))
            phone_spans.append(
                (
                    frame * self.resolution,
                    (frame + length) * self.resolution,
                    phone_index,
                )
            )
            frame += length

        phone_words = [
            (word_index, word)
            for word_index, (word, word_phones) in enumerate(zip(words, phones_by_word))
            for _ in word_phones
        ]
        assigned_spans = []
        for start, end, phone_index in phone_spans:
            if phone_index is None:
                continue
            word_index, word = phone_words[phone_index]
            assigned_spans.append((start, end, word_index, word))
        word_spans = []
        for (_, word), group in groupby(
            assigned_spans, key=lambda item: (item[2], item[3])
        ):
            group = list(group)
            word_spans.append({
                "word": word,
                "start": round(float(group[0][0]), 3),
                "end": round(float(group[-1][1]), 3),
            })
        return word_spans
