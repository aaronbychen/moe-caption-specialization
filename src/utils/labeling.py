def map_pos_to_category(token):
    """
    Map spaCy POS tags to coarse semantic categories (5 classes).
    """
    if token.pos_ in {"NOUN", "PROPN", "PRON"}:
        return "object"
    elif token.pos_ == "ADJ":
        return "attribute"
    elif token.pos_ in {"ADP", "SCONJ"}:
        return "relation"
    elif token.pos_ in {"VERB", "AUX"}:
        return "action"
    else:
        return "functional"


def map_pos_to_fine_category(token):
    """
    Map spaCy POS tags to fine-grained semantic categories (9 classes).
    Splits coarse groups into more specific linguistic roles.
    """
    if token.pos_ == "NOUN":
        return "noun"
    elif token.pos_ == "PROPN":
        return "proper_noun"
    elif token.pos_ == "PRON":
        return "pronoun"
    elif token.pos_ == "ADJ":
        return "adjective"
    elif token.pos_ == "VERB":
        return "verb"
    elif token.pos_ == "AUX":
        return "auxiliary"
    elif token.pos_ in {"ADP", "SCONJ"}:
        return "relation"
    elif token.pos_ == "ADV":
        return "adverb"
    else:
        return "functional"


def normalize_t5_piece(token_str):
    """
    Convert a T5 subword token into a comparable text fragment.
    Example:
      '▁dog' -> 'dog'
      's'    -> 's'
      '.'    -> '.'
      '</s>' -> ''
    """
    if token_str == "</s>":
        return ""
    return token_str.replace("\u2581", "")
