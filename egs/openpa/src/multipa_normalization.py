"""Text normalization used by the MultiPA open-response evaluation."""

import re
import string

import num2words


def remove_punctuation_except_apostrophe(text):
    """Remove ASCII punctuation while retaining apostrophes."""
    punctuation = string.punctuation.replace("'", "")
    return str(text).translate(str.maketrans("", "", punctuation))


def convert_numbers_to_words(sentence):
    """Match MultiPA's digit conversion, including digit-wise numeric prompts."""
    sentence = str(sentence)
    compact = sentence.replace(" ", "")
    if compact.isdigit():
        # MultiPA contains four-digit prompts that are read one digit at a time.
        sentence = " ".join(
            num2words.num2words(int(character)) for character in compact
        )
    else:
        sentence = " ".join(
            num2words.num2words(int(token)) if token.isdigit() else token
            for token in sentence.split()
        )
    return re.sub(r"\s+", " ", sentence).strip()


def normalize_multipa_text(text):
    """Apply the paper's punctuation, lowercase, and number normalization."""
    text = remove_punctuation_except_apostrophe(text).lower()
    return convert_numbers_to_words(text)


def normalize_multipa_word_units(words):
    """Normalize transcript words without changing their label/timestamp count."""
    cleaned = [remove_punctuation_except_apostrophe(word).lower() for word in words]
    compact = "".join(cleaned)
    if compact.isdigit():
        return [
            " ".join(num2words.num2words(int(character)) for character in word)
            for word in cleaned
        ]
    return [
        num2words.num2words(int(word)) if word.isdigit() else word
        for word in cleaned
    ]
