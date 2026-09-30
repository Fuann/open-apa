"""SpeechOcean762 preprocessing and variable-word-count batching."""

from collections import Counter
from dataclasses import dataclass
import io
from pathlib import Path
from typing import Dict, List

import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F

from multipa_normalization import normalize_multipa_word_units


SCORE_TOKEN = "<score>"
UTTERANCE_LABELS = ("accuracy", "fluency", "prosodic", "total")
DEFAULT_WORD_LABELS = ("accuracy", "total")
_DURATION_STATISTIC_NAMES = (
    "sum",
    "mean",
    "std",
    "median",
    "mean_absolute_deviation",
)
_SEGMENT_STATISTICS_FEATURE_NAMES = tuple(
    f"log_{segment}_duration_{statistic}"
    for segment in ("speech_segment", "pause_segment")
    for statistic in _DURATION_STATISTIC_NAMES
)
FLUENCY_FEATURE_NAMES = (
    "log_utterance_duration",
    "speech_ratio",
    "internal_pause_ratio",
    *_SEGMENT_STATISTICS_FEATURE_NAMES,
)
PROSODIC_FEATURE_NAMES = (
    "log_energy_std",
    "log_energy_mean_absolute_deviation",
    "energy_delta_absolute_mean",
    "energy_delta_std",
)
PREPROCESSING_SCHEMA_VERSION = 17
_SILERO_VAD_MODEL = None


def decode_audio(audio: Dict, target_sampling_rate: int = 16000):
    """Return mono audio without depending on datasets/torchcodec decoding."""
    if "array" in audio:
        waveform = np.asarray(audio["array"], dtype=np.float32)
        sampling_rate = int(audio["sampling_rate"])
    else:
        source = io.BytesIO(audio["bytes"]) if audio.get("bytes") else audio["path"]
        waveform, sampling_rate = sf.read(source, dtype="float32", always_2d=True)
        waveform = waveform.mean(axis=1)

    if waveform.ndim > 1:
        waveform = waveform.mean(axis=0)
    if sampling_rate != target_sampling_rate:
        target_length = round(len(waveform) * target_sampling_rate / sampling_rate)
        waveform = F.interpolate(
            torch.from_numpy(waveform)[None, None],
            size=target_length,
            mode="linear",
            align_corners=False,
        )[0, 0].numpy()
        sampling_rate = target_sampling_rate
    return waveform, sampling_rate


def _boolean_runs(mask):
    """Return (value, start, end) runs with end exclusive."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    boundaries = np.flatnonzero(mask[1:] != mask[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [mask.size]))
    return [(bool(mask[start]), int(start), int(end)) for start, end in zip(starts, ends)]


def _silero_speech_mask(
    waveform,
    sampling_rate,
    frame_count,
    hop_length,
    threshold,
):
    """Return a frame mask from lazily loaded Silero VAD timestamps."""
    global _SILERO_VAD_MODEL
    try:
        from silero_vad import get_speech_timestamps, load_silero_vad
    except ImportError as exc:
        raise RuntimeError(
            "Silero fluency VAD requires silero-vad. Install requirements.txt "
            "or run `python -m pip install silero-vad`."
        ) from exc
    if sampling_rate not in {8000, 16000}:
        raise ValueError(
            f"Silero VAD supports 8 kHz or 16 kHz audio, got {sampling_rate} Hz"
        )
    if _SILERO_VAD_MODEL is None:
        _SILERO_VAD_MODEL = load_silero_vad()
    timestamps = get_speech_timestamps(
        torch.from_numpy(waveform),
        _SILERO_VAD_MODEL,
        threshold=threshold,
        sampling_rate=sampling_rate,
        min_speech_duration_ms=100,
        min_silence_duration_ms=150,
        speech_pad_ms=0,
        return_seconds=False,
    )
    speech = np.zeros(frame_count, dtype=bool)
    for segment in timestamps:
        start = max(0, int(segment["start"]) // hop_length)
        end = min(
            frame_count,
            int(np.ceil(int(segment["end"]) / float(hop_length))),
        )
        speech[start:end] = True
    return speech


def _duration_statistics(durations):
    """Return log1p(sum/mean/std/median/mean absolute deviation)."""
    durations = np.asarray(durations, dtype=np.float64)
    if durations.size == 0:
        return np.zeros(len(_DURATION_STATISTIC_NAMES), dtype=np.float32)
    mean = durations.mean()
    statistics = np.asarray(
        [
            durations.sum(),
            mean,
            durations.std(),
            np.median(durations),
            np.mean(np.abs(durations - mean)),
        ],
        dtype=np.float64,
    )
    return np.log1p(statistics).astype(np.float32)


def extract_prosodic_features(
    waveform,
    sampling_rate,
    vad_method="energy",
    vad_threshold=0.5,
    analysis=None,
):
    """Extract the fixed 4-D speech-frame log-energy feature set."""
    if analysis is None:
        _, analysis = extract_fluency_features(
            waveform,
            sampling_rate,
            vad_method=vad_method,
            vad_threshold=vad_threshold,
            return_analysis=True,
        )
    speech = analysis["speech"]
    rms = analysis["rms"]

    log_energy = np.log(rms[speech] + 1e-7)
    if log_energy.size:
        energy_mean = log_energy.mean()
        energy_std = log_energy.std()
        energy_mad = np.mean(np.abs(log_energy - energy_mean))
    else:
        energy_std = energy_mad = 0.0
    energy_delta = np.diff(log_energy)
    if energy_delta.size:
        energy_delta_absolute_mean = np.mean(np.abs(energy_delta))
        energy_delta_std = energy_delta.std()
    else:
        energy_delta_absolute_mean = energy_delta_std = 0.0
    features = np.asarray(
        [
            energy_std,
            energy_mad,
            energy_delta_absolute_mean,
            energy_delta_std,
        ],
        dtype=np.float32,
    )
    if not np.isfinite(features).all():
        raise ValueError("Prosodic feature extraction produced a non-finite value")
    return features


def extract_fluency_features(
    waveform,
    sampling_rate,
    vad_method="energy",
    vad_threshold=0.5,
    return_analysis=False,
):
    """Extract the fixed 13-D alignment-free fluency feature set."""
    if vad_method not in {"energy", "silero"}:
        raise ValueError(f"fluency VAD must be 'energy' or 'silero', got {vad_method!r}")
    if not 0.0 <= vad_threshold <= 1.0:
        raise ValueError("fluency VAD threshold must be between 0 and 1")
    waveform = np.ascontiguousarray(
        np.asarray(waveform, dtype=np.float32).reshape(-1)
    )
    duration = waveform.size / float(sampling_rate)

    # Frame grid shared by energy and Silero speech/pause features.
    frame_length = max(1, round(0.025 * sampling_rate))
    hop_length = max(1, round(0.010 * sampling_rate))
    if waveform.size < frame_length:
        waveform = np.pad(waveform, (0, frame_length - waveform.size))
    frame_count = 1 + (waveform.size - frame_length) // hop_length
    frames = np.lib.stride_tricks.as_strided(
        waveform,
        shape=(frame_count, frame_length),
        strides=(waveform.strides[0] * hop_length, waveform.strides[0]),
        writeable=False,
    )
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1) + 1e-12)
    if vad_method == "silero":
        speech = _silero_speech_mask(
            waveform,
            sampling_rate,
            frame_count,
            hop_length,
            vad_threshold,
        )
    else:
        log_energy = np.log(rms + 1e-7)
        low, high = np.percentile(log_energy, [10.0, 90.0])
        if high - low < 0.5:
            speech = np.full(frame_count, rms.mean() >= 1e-4, dtype=bool)
        else:
            threshold = low + 0.35 * (high - low)
            speech = (log_energy >= threshold) & (rms >= 1e-4)

    # Smooth VAD: bridge <=100 ms gaps and remove <100 ms speech bursts.
    merge_gap_frames = max(1, round(0.100 * sampling_rate / hop_length))
    minimum_speech_frames = max(1, round(0.100 * sampling_rate / hop_length))
    minimum_pause_frames = max(1, round(0.150 * sampling_rate / hop_length))
    for value, start, end in _boolean_runs(speech):
        if not value and start > 0 and end < frame_count and end - start <= merge_gap_frames:
            speech[start:end] = True
    for value, start, end in _boolean_runs(speech):
        if value and end - start < minimum_speech_frames:
            speech[start:end] = False

    # Keep internal pauses >=150 ms; exclude leading/trailing silence.
    speech_runs = [(start, end) for value, start, end in _boolean_runs(speech) if value]
    pause_runs = []
    if speech_runs:
        first_speech, last_speech = speech_runs[0][0], speech_runs[-1][1]
        pause_runs = [
            (start, end)
            for value, start, end in _boolean_runs(speech)
            if not value
            and start >= first_speech
            and end <= last_speech
            and end - start >= minimum_pause_frames
        ]

    seconds_per_frame = hop_length / float(sampling_rate)
    speech_durations = np.asarray(
        [(end - start) * seconds_per_frame for start, end in speech_runs],
        dtype=np.float64,
    )
    pause_durations = np.asarray(
        [(end - start) * seconds_per_frame for start, end in pause_runs],
        dtype=np.float64,
    )
    # NOTE: 5 speech segment duration statistics (without min/max)
    speech_statistics = _duration_statistics(speech_durations)
    # NOTE: 5 internal pause duration statistics (without min/max)
    pause_statistics = _duration_statistics(pause_durations)
    # NOTE: log_utterance_duration
    log_utterance_duration = np.log1p(duration)
    # NOTE: speech_ratio
    speech_ratio = float(speech.mean()) if speech.size else 0.0
    # NOTE: internal_pause_ratio
    pause_total = float(pause_durations.sum()) if pause_durations.size else 0.0
    internal_pause_ratio = pause_total / max(duration, 1e-6)
    features = np.concatenate(
        (
            np.asarray(
                [log_utterance_duration, speech_ratio, internal_pause_ratio],
                dtype=np.float32,
            ),
            speech_statistics,
            pause_statistics,
        )
    )
    if not np.isfinite(features).all():
        raise ValueError("Fluency feature extraction produced a non-finite value")
    if return_analysis:
        return features, {
            "waveform": waveform,
            "rms": rms,
            "speech": speech,
            "hop_length": hop_length,
        }
    return features


def build_apa_token_ids(tokenizer, words: List[str], score_token_id: int) -> List[int]:
    """Build Whisper labels, inserting exactly one score token per dataset word."""
    token_ids = list(tokenizer.prefix_tokens)
    # The model supplies decoder_start_token_id itself. Keeping it in labels would
    # create two consecutive <|startoftranscript|> tokens after shifting.
    if token_ids and token_ids[0] == tokenizer.convert_tokens_to_ids("<|startoftranscript|>"):
        token_ids = token_ids[1:]
    for index, word in enumerate(words):
        # Whisper's byte-level BPE encodes the leading blank as part of non-first words.
        piece = word if index == 0 else " " + word
        piece_ids = tokenizer(piece, add_special_tokens=False).input_ids
        if not piece_ids:
            raise ValueError(f"Word produced no tokens: {word!r}")
        token_ids.extend(piece_ids)
        token_ids.append(score_token_id)
    token_ids.append(tokenizer.eos_token_id)
    return token_ids


def truncate_apa_words(tokenizer, words, score_token_id, max_target_positions):
    """Keep the longest whole-word prefix that fits Whisper's decoder."""
    if max_target_positions is None:
        return words
    prefix_ids = list(tokenizer.prefix_tokens)
    start_token_id = tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    if prefix_ids and prefix_ids[0] == start_token_id:
        prefix_ids = prefix_ids[1:]
    # Reserve one position for EOS. Each retained word also needs one <score>.
    length = len(prefix_ids) + 1
    retained = []
    for index, word in enumerate(words):
        piece = word if index == 0 else " " + word
        piece_ids = tokenizer(piece, add_special_tokens=False).input_ids
        if length + len(piece_ids) + 1 > max_target_positions:
            break
        retained.append(word)
        length += len(piece_ids) + 1
    if not retained:
        raise ValueError(
            f"No APA word fits within max_target_positions={max_target_positions}"
        )
    return retained


def align_hypothesis_words(reference, hypothesis):
    """Return a reference index for each hypothesis word via Levenshtein alignment."""
    rows, columns = len(reference), len(hypothesis)
    cost = [[0] * (columns + 1) for _ in range(rows + 1)]
    operation = [[None] * (columns + 1) for _ in range(rows + 1)]
    for row in range(1, rows + 1):
        cost[row][0], operation[row][0] = row, "delete"
    for column in range(1, columns + 1):
        cost[0][column], operation[0][column] = column, "insert"
    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            diagonal = "match" if reference[row - 1] == hypothesis[column - 1] else "substitute"
            candidates = [
                (cost[row - 1][column - 1] + (diagonal != "match"), diagonal),
                (cost[row - 1][column] + 1, "delete"),
                (cost[row][column - 1] + 1, "insert"),
            ]
            cost[row][column], operation[row][column] = min(
                candidates, key=lambda item: item[0]
            )
    mapping = [None] * columns
    row, column = rows, columns
    while row or column:
        current = operation[row][column]
        if current in {"match", "substitute"}:
            mapping[column - 1] = row - 1
            row -= 1
            column -= 1
        elif current == "delete":
            row -= 1
        else:
            column -= 1
    return mapping


def _word_targets(item, word_label_names, normalize_word_labels, binary_stress,
                  stress_multisyllabic_only):
    syllable_count = sum(
        str(phone)[-1:] in {"0", "1", "2"} for phone in item.get("phones", [])
    )
    stress_value = float(item.get("stress", -100.0))
    stress_all = (
        1.0 if binary_stress and stress_value == 10.0
        else 0.0 if binary_stress and stress_value == 5.0
        else -100.0
    )
    labels = []
    for name in word_label_names:
        value = float(item[name])
        if name == "stress" and binary_stress:
            if stress_multisyllabic_only and syllable_count < 2:
                value = -100.0
            elif value == 10:
                value = 1.0
            elif value == 5:
                value = 0.0
            else:
                raise ValueError(f"Unexpected stress label {value}; expected 5 or 10")
        elif normalize_word_labels:
            value /= 10.0
        labels.append(value)
    return labels, syllable_count, stress_all


def substitution_loss_weight(reference_accuracy, minimum_weight):
    """Trust ASR substitutions more when the human word accuracy is low."""
    accuracy = min(max(float(reference_accuracy), 0.0), 10.0)
    return minimum_weight + (1.0 - minimum_weight) * (1.0 - accuracy / 10.0)


class MixedTranscriptDataset(torch.utils.data.Dataset):
    """Sample GT/L2/general decoder transcript views without duplicating audio."""

    def __init__(
        self,
        clean_dataset,
        raw_dataset,
        processor,
        score_token_id,
        word_label_names,
        l2_records,
        general_records,
        probabilities=(0.4, 0.4, 0.2),
        asr_view_utterance_loss_weight=0.3,
        substitution_weight=0.5,
        adaptive_substitution_loss=True,
        normalize_word_labels=False,
        binary_stress=False,
        stress_multisyllabic_only=False,
        text_normalization="multipa",
        max_target_positions=None,
    ):
        if len(clean_dataset) != len(raw_dataset):
            raise ValueError("Clean and raw training datasets must have equal length")
        probabilities = torch.tensor(probabilities, dtype=torch.float)
        if probabilities.numel() != 3 or (probabilities < 0).any() or probabilities.sum() <= 0:
            raise ValueError("Transcript probabilities must be three non-negative values")
        if asr_view_utterance_loss_weight < 0:
            raise ValueError("ASR-view utterance loss weight must be non-negative")
        self.clean_dataset = clean_dataset
        self.probabilities = probabilities / probabilities.sum()
        self.asr_view_utterance_loss_weight = float(
            asr_view_utterance_loss_weight
        )
        self.views = []
        sources = (("l2", l2_records), ("general", general_records))
        alignment_counts = {name: Counter() for name, _ in sources}
        truncation_counts = Counter()
        for index in range(len(raw_dataset)):
            example = raw_dataset[index]
            audio_path = example["audio"].get("path")
            audio_id = Path(audio_path).name if audio_path else str(index)
            reference_items = example["words"]
            reference_words = [str(item["text"]).strip() for item in reference_items]
            normalized_reference = normalize_multipa_word_units(reference_words)
            example_views = {}
            for view_name, records in sources:
                if audio_id not in records:
                    raise ValueError(f"{view_name} transcript is missing {audio_id}")
                hypothesis_words = [word["word"] for word in records[audio_id]["words"]]
                if text_normalization == "multipa":
                    hypothesis_words = normalize_multipa_word_units(hypothesis_words)
                elif text_normalization != "none":
                    raise ValueError(f"Unknown text normalization: {text_normalization!r}")
                hypothesis_words = [word for word in hypothesis_words if word]
                if not hypothesis_words:
                    hypothesis_words = ["nospeech"]
                original_word_count = len(hypothesis_words)
                hypothesis_words = truncate_apa_words(
                    processor.tokenizer,
                    hypothesis_words,
                    score_token_id,
                    max_target_positions,
                )
                if len(hypothesis_words) != original_word_count:
                    truncation_counts[view_name] += 1
                alignment_reference = (
                    normalized_reference if text_normalization == "multipa" else reference_words
                )
                mapping = align_hypothesis_words(alignment_reference, hypothesis_words)
                aligned_reference = {index for index in mapping if index is not None}
                alignment_counts[view_name]["deletion"] += (
                    len(alignment_reference) - len(aligned_reference)
                )
                labels, syllables, stress_all, weights = [], [], [], []
                for word, reference_index in zip(hypothesis_words, mapping):
                    if reference_index is None:
                        alignment_counts[view_name]["insertion"] += 1
                        labels.append([-100.0] * len(word_label_names))
                        syllables.append(-100)
                        stress_all.append(-100.0)
                        weights.append(0.0)
                        continue
                    target, count, stress = _word_targets(
                        reference_items[reference_index],
                        word_label_names,
                        normalize_word_labels,
                        binary_stress,
                        stress_multisyllabic_only,
                    )
                    labels.append(target)
                    syllables.append(count)
                    stress_all.append(stress)
                    if word == alignment_reference[reference_index]:
                        weights.append(1.0)
                        operation = "match"
                    else:
                        weights.append(
                            substitution_loss_weight(
                                reference_items[reference_index]["accuracy"],
                                substitution_weight,
                            )
                            if adaptive_substitution_loss
                            else substitution_weight
                        )
                        operation = "substitution"
                    alignment_counts[view_name][operation] += 1
                example_views[view_name] = {
                    "labels": build_apa_token_ids(
                        processor.tokenizer, hypothesis_words, score_token_id
                    ),
                    "word_labels": labels,
                    "word_syllable_counts": syllables,
                    "word_stress_all_labels": stress_all,
                    "word_loss_weights": weights,
                    "utterance_loss_weight": self.asr_view_utterance_loss_weight,
                    "disable_asr_loss": True,
                }
            self.views.append(example_views)
        for view_name, counts in alignment_counts.items():
            print(f"Mixed transcript {view_name} alignment: {dict(counts)}", flush=True)
        if truncation_counts:
            print(
                "Mixed transcript sequences truncated to decoder limit: "
                f"{dict(truncation_counts)}",
                flush=True,
            )

    def __len__(self):
        return len(self.clean_dataset)

    def __getitem__(self, index):
        feature = dict(self.clean_dataset[index])
        view_index = int(torch.multinomial(self.probabilities, 1).item())
        if view_index == 0:
            feature["word_loss_weights"] = [1.0] * len(feature["word_labels"])
            feature["utterance_loss_weight"] = 1.0
            feature["disable_asr_loss"] = False
        else:
            feature.update(self.views[index]["l2" if view_index == 1 else "general"])
        return feature


def preprocess_example(
    example: Dict,
    processor,
    score_token_id: int,
    word_label_names=DEFAULT_WORD_LABELS,
    normalize_word_labels=False,
    normalize_utterance_labels=False,
    binary_stress=False,
    stress_multisyllabic_only=False,
    text_normalization="none",
    fluency_vad="energy",
    fluency_vad_threshold=0.5,
    handcrafted_features=False,
    preprocessing_schema_version=PREPROCESSING_SCHEMA_VERSION,
) -> Dict:
    del preprocessing_schema_version
    audio = example["audio"]
    waveform, sampling_rate = decode_audio(audio)
    words = [item["text"].strip() for item in example["words"]]
    if text_normalization == "multipa":
        words = normalize_multipa_word_units(words)
    elif text_normalization != "none":
        raise ValueError(f"Unknown text normalization: {text_normalization!r}")
    if any(not word for word in words):
        raise ValueError("Text normalization produced an empty word")
    word_labels = []
    word_syllable_counts = []
    word_stress_all_labels = []
    for item in example["words"]:
        syllable_count = sum(
            str(phone)[-1:] in {"0", "1", "2"}
            for phone in item.get("phones", [])
        )
        word_syllable_counts.append(syllable_count)
        stress_value = float(item.get("stress", -100.0))
        word_stress_all_labels.append(
            1.0 if binary_stress and stress_value == 10.0
            else 0.0 if binary_stress and stress_value == 5.0
            else -100.0
        )
        labels = []
        for name in word_label_names:
            value = float(item[name])
            if name == "stress" and binary_stress:
                if stress_multisyllabic_only and syllable_count < 2:
                    value = -100.0
                elif value == 10:
                    value = 1.0
                elif value == 5:
                    value = 0.0
                else:
                    raise ValueError(f"Unexpected stress label {value}; expected 5 or 10")
            elif normalize_word_labels:
                value /= 10.0
            labels.append(value)
        word_labels.append(labels)
    if len(words) != len(word_labels) or not words:
        raise ValueError("Each non-empty dataset word must have pronunciation labels")
    acoustic_inputs = processor.feature_extractor(
        waveform,
        sampling_rate=sampling_rate,
        return_attention_mask=True,
    )
    input_features = acoustic_inputs.input_features[0]
    utterance_labels = [float(example[name]) for name in UTTERANCE_LABELS]
    if normalize_utterance_labels:
        utterance_labels = [value / 10.0 for value in utterance_labels]
    if handcrafted_features:
        fluency_features, acoustic_analysis = extract_fluency_features(
            waveform,
            sampling_rate,
            vad_method=fluency_vad,
            vad_threshold=fluency_vad_threshold,
            return_analysis=True,
        )
        prosodic_features = extract_prosodic_features(
            waveform,
            sampling_rate,
            vad_method=fluency_vad,
            vad_threshold=fluency_vad_threshold,
            analysis=acoustic_analysis,
        )
    else:
        fluency_features = np.zeros(len(FLUENCY_FEATURE_NAMES), dtype=np.float32)
        prosodic_features = np.zeros(len(PROSODIC_FEATURE_NAMES), dtype=np.float32)
    return {
        "input_features": input_features,
        "attention_mask": acoustic_inputs.attention_mask[0],
        "fluency_features": fluency_features,
        "prosodic_features": prosodic_features,
        "labels": build_apa_token_ids(processor.tokenizer, words, score_token_id),
        "word_labels": word_labels,
        "word_syllable_counts": word_syllable_counts,
        "word_stress_all_labels": word_stress_all_labels,
        "utterance_labels": utterance_labels,
    }


@dataclass
class DataCollatorWhisperAPA:
    processor: object
    score_token_id: int
    max_target_positions: int = None
    word_representation: str = "score-token"

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        if self.word_representation not in {"score-token", "bpe-mean"}:
            raise ValueError(f"Unknown word representation: {self.word_representation}")
        label_token_ids = []
        bpe_boundaries = []
        for index, feature in enumerate(features):
            original_ids = feature["labels"]
            score_indices = [
                position for position, token_id in enumerate(original_ids)
                if token_id == self.score_token_id
            ]
            if len(score_indices) != len(feature["word_labels"]):
                raise ValueError(
                    f"Found {len(score_indices)} score delimiters but "
                    f"{len(feature['word_labels'])} word labels"
                )
            if self.word_representation == "bpe-mean":
                token_ids = [
                    token_id for token_id in original_ids
                    if token_id != self.score_token_id
                ]
                first_start = len(self.processor.tokenizer.prefix_tokens)
                lengths = []
                # labels omit <|startoftranscript|>, so their first word starts
                # one position earlier than it does in decoder_input_ids.
                previous_score = len(self.processor.tokenizer.prefix_tokens) - 2
                for score_index in score_indices:
                    length = score_index - previous_score - 1
                    if length <= 0:
                        raise ValueError("Found an empty word BPE span")
                    lengths.append(length)
                    previous_score = score_index
                starts, ends = [], []
                cursor = first_start
                for length in lengths:
                    starts.append(cursor)
                    cursor += length
                    ends.append(cursor)
                bpe_boundaries.append((starts, ends))
            else:
                token_ids = original_ids
                bpe_boundaries.append(None)
            label_token_ids.append(token_ids)
            if self.max_target_positions is not None and len(token_ids) > self.max_target_positions:
                raise ValueError(
                    f"Batch item {index} has {len(token_ids)} decoder tokens; "
                    f"Whisper supports at most {self.max_target_positions}"
                )
            invalid_ids = [
                token_id for token_id in token_ids
                if token_id < 0 or token_id >= len(self.processor.tokenizer)
            ]
            if invalid_ids:
                raise ValueError(
                    f"Batch item {index} contains token IDs outside tokenizer vocab "
                    f"size {len(self.processor.tokenizer)}: {invalid_ids[:5]}"
                )
        audio = [
            {
                "input_features": x["input_features"],
                "attention_mask": x["attention_mask"],
            }
            for x in features
        ]
        batch = self.processor.feature_extractor.pad(audio, return_tensors="pt")
        label_features = [{"input_ids": token_ids} for token_ids in label_token_ids]
        label_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = label_batch["input_ids"].masked_fill(label_batch.attention_mask.ne(1), -100)

        # Whisper shifts labels internally. Construct it here so score positions refer
        # to actual decoder inputs (and therefore actual <score> hidden states).
        decoder_input_ids = torch.full_like(labels, self.processor.tokenizer.pad_token_id)
        decoder_input_ids[:, 0] = self.processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        decoder_input_ids[:, 1:] = labels[:, :-1].masked_fill(labels[:, :-1].eq(-100), self.processor.tokenizer.pad_token_id)
        decoder_attention_mask = decoder_input_ids.ne(self.processor.tokenizer.pad_token_id).long()

        if self.word_representation == "bpe-mean":
            positions = [
                torch.tensor(ends, dtype=torch.long)
                for _, ends in bpe_boundaries
            ]
        else:
            positions = [
                row.eq(self.score_token_id).nonzero(as_tuple=False).flatten()
                for row in decoder_input_ids
            ]
        max_words = max(len(x) for x in positions)
        score_positions = torch.zeros(len(features), max_words, dtype=torch.long)
        word_start_positions = torch.zeros(len(features), max_words, dtype=torch.long)
        num_word_scores = len(features[0]["word_labels"][0])
        word_labels = torch.full((len(features), max_words, num_word_scores), -100.0)
        word_label_mask = torch.zeros(len(features), max_words, dtype=torch.bool)
        word_syllable_counts = torch.full((len(features), max_words), -100, dtype=torch.long)
        word_stress_all_labels = torch.full((len(features), max_words), -100.0)
        word_loss_weights = torch.zeros(len(features), max_words, dtype=torch.float)
        for i, (pos, feature) in enumerate(zip(positions, features)):
            if len(pos) != len(feature["word_labels"]):
                raise ValueError(f"Found {len(pos)} score tokens but {len(feature['word_labels'])} word labels")
            score_positions[i, : len(pos)] = pos
            # In decoder_input_ids, the first word starts immediately after the
            # Whisper prefix. Later words start immediately after the preceding
            # <score> delimiter. The corresponding score position is the
            # exclusive end of each word's BPE span.
            if self.word_representation == "bpe-mean":
                starts = pos.new_tensor(bpe_boundaries[i][0])
            else:
                starts = torch.cat(
                    (
                        pos.new_tensor([len(self.processor.tokenizer.prefix_tokens)]),
                        pos[:-1] + 1,
                    )
                )
            if torch.any(starts >= pos):
                raise ValueError("Found an empty word BPE span")
            word_start_positions[i, : len(pos)] = starts
            word_labels[i, : len(pos)] = torch.tensor(feature["word_labels"], dtype=torch.float)
            word_label_mask[i, : len(pos)] = True
            word_syllable_counts[i, : len(pos)] = torch.tensor(
                feature["word_syllable_counts"], dtype=torch.long
            )
            word_stress_all_labels[i, : len(pos)] = torch.tensor(
                feature["word_stress_all_labels"], dtype=torch.float
            )
            word_loss_weights[i, : len(pos)] = torch.tensor(
                feature.get("word_loss_weights", [1.0] * len(pos)), dtype=torch.float
            )

        disable_asr_loss = torch.tensor([
            feature.get("disable_asr_loss", False) for feature in features
        ])
        labels[disable_asr_loss] = -100
        utterance_loss_weights = torch.tensor(
            [feature.get("utterance_loss_weight", 1.0) for feature in features],
            dtype=torch.float,
        )

        batch.update({
            "labels": labels,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "score_positions": score_positions,
            "word_end_positions": score_positions,
            "word_start_positions": word_start_positions,
            "word_labels": word_labels,
            "word_label_mask": word_label_mask,
            "word_loss_weights": word_loss_weights,
            "word_syllable_counts": word_syllable_counts,
            "word_stress_all_labels": word_stress_all_labels,
            "utterance_labels": torch.tensor([x["utterance_labels"] for x in features], dtype=torch.float),
            "utterance_loss_weights": utterance_loss_weights,
            "fluency_features": torch.tensor(
                np.stack([x["fluency_features"] for x in features]),
                dtype=torch.float,
            ),
            "prosodic_features": torch.tensor(
                np.stack([x["prosodic_features"] for x in features]),
                dtype=torch.float,
            ),
            "multipa_fallback": torch.tensor([
                feature.get("multipa_fallback", False) for feature in features
            ], dtype=torch.bool),
        })
        return batch
