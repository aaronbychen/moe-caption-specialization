def map_pos_to_category(token):
    """
    Map spaCy POS tags to coarse semantic categories.
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

def normalize_t5_piece(token_str):
    """
    Convert a T5 subword token into a comparable text fragment.
    """
    if token_str == "</s>":
        return ""
    return token_str.replace("▁", "")

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
    return token_str.replace("▁", "")