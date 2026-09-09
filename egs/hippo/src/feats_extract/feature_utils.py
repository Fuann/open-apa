"""Pure helpers for the checkpoint-compatible Hippo feature layout."""
import re
import string
import numpy as np

UTT = ('accuracy', 'completeness', 'fluency', 'prosodic', 'total')
WORD = ('accuracy', 'stress', 'total')
# Same 0–10 defaults as ../multipa/src/evaluate_speechocean762.py.
UTT_FALLBACK = dict(accuracy=1., fluency=0., prosodic=0., total=0.)
WORD_FALLBACK = dict(accuracy=0., stress=5., total=1.)
SSL_MODELS = {'hu': 'facebook/hubert-large-ll60k',
              'w2v': 'facebook/wav2vec2-xls-r-300m',
              'wlm': 'microsoft/wavlm-large'}
BERT_MODEL = 'answerdotai/ModernBERT-base'


def normalize_transcript(text):
    # Match MultiPA: remove punctuation, lowercase, then convert numeric tokens.
    from num2words import num2words
    text = text.translate(str.maketrans('', '', string.punctuation.replace("'", ''))).replace('  ', ' ').lower()
    try:
        int(text.replace(' ', ''))
        text = ' '.join(text)
    except ValueError:
        pass
    return ' '.join(num2words(token) if token.isdigit() else token for token in text.split())


def aligned_labels(reference, predicted, labels, zero_substitutions=True):
    """Needleman-Wunsch (+1/-1/-1); ties: diagonal, left, up (string2string)."""
    # https://string2string.readthedocs.io/en/stable/_modules/string2string/alignment/classical.html
    n, m = len(reference), len(predicted)
    if len(labels) != n:
        raise ValueError('Reference label count mismatch')
    score = np.zeros((n + 1, m + 1), dtype=int)
    score[:, 0] = -np.arange(n + 1)
    score[0, :] = -np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            score[i, j] = max(score[i-1, j-1] + (1 if reference[i-1] == predicted[j-1] else -1),
                              score[i-1, j] - 1, score[i, j-1] - 1)
    result = np.zeros((m,) + np.asarray(labels).shape[1:], dtype=np.float32)
    i, j = n, m
    while i or j:
        if i and j and score[i, j] == score[i-1, j-1] + (1 if reference[i-1] == predicted[j-1] else -1):
            if not zero_substitutions or reference[i-1] == predicted[j-1]:
                result[j-1] = labels[i-1]
            i, j = i-1, j-1
        elif j and score[i, j] == score[i, j-1] - 1:
            j -= 1
        else:
            i -= 1
    return result


def pad_rows(rows, length, value=0):
    rows = np.asarray(rows, dtype=np.float32)
    if len(rows) > length:
        raise ValueError(f'{len(rows)} phones exceeds --max-phones {length}; increase it, never truncate')
    out = np.full((length,) + rows.shape[1:], value, dtype=np.float32)
    out[:len(rows)] = rows
    return out


def word_mask(word_ids):
    word_ids = np.asarray(word_ids)
    return ((word_ids[:, None] == word_ids[None, :]) & (word_ids[:, None] >= 0)).astype(np.float32)


def phone_targets(ref, words, phones_by_word, vocab, closed=False):
    phones = [re.sub(r'\d', '', p) for group in phones_by_word for p in group]
    if not phones:
        raise ValueError('No phones in transcript')
    if any(p not in vocab or vocab[p] == 0 for p in phones):
        raise ValueError(f'Unknown phone(s): {set(phones) - set(vocab)}')
    ids = np.asarray([vocab[p] for p in phones], dtype=np.int64)
    word_ids = np.repeat(np.arange(len(words)), [len(group) for group in phones_by_word])
    reference_words = [w['text'].lower() for w in ref['words']]
    wl = np.array([[w[k] for k in WORD] for w in ref['words']])
    pl = [s for w in ref['words'] for s in w['phones-accuracy']]
    reference_phones = [re.sub(r'\d', '', p) for w in ref['words'] for p in w['phones']]
    if closed:
        if words != reference_words or phones != reference_phones:
            raise ValueError('Closed-response reference mismatch')
        wp, pp = wl, np.asarray(pl)
    else:
        wp = aligned_labels(reference_words, words, wl)
        # Historical Hippo retains substituted reference phone scores, but zeros insertions.
        pp = aligned_labels(reference_phones, phones, pl, zero_substitutions=False)
    return ids, np.column_stack((pp, ids)), np.column_stack((wp[word_ids], word_ids))
