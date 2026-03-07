import os
import torch
import spacy
from transformers import T5Tokenizer, T5EncoderModel
from src.utils.labeling import map_pos_to_category, normalize_t5_piece


def main():
    captions = [
        "A dog jumps over a log.",
        "Two children sit at a table.",
        "A red bus is parked on the street."
    ]

    # load models
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5EncoderModel.from_pretrained("t5-small")
    model.eval()

    nlp = spacy.load("en_core_web_sm")

    # tokenize captions with T5
    inputs = tokenizer(
        captions,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    hidden_states = outputs.last_hidden_state
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    aligned_rows = []

    for caption_id, caption in enumerate(captions):
        doc = nlp(caption)
        spacy_words = [token.text for token in doc]
        spacy_categories = [map_pos_to_category(token) for token in doc]

        valid_len = attention_mask[caption_id].sum().item()
        ids = input_ids[caption_id][:valid_len].tolist()
        t5_tokens = tokenizer.convert_ids_to_tokens(ids)
        t5_vectors = hidden_states[caption_id][:valid_len]

        print("\n" + "=" * 80)
        print(f"Caption {caption_id}: {caption}")
        print("spaCy words:", spacy_words)
        print("T5 tokens:  ", t5_tokens)

        word_idx = 0
        piece_buffer = ""

        for sub_idx, sub_token in enumerate(t5_tokens):
            piece = normalize_t5_piece(sub_token)

            if piece == "":
                continue

            if word_idx >= len(spacy_words):
                break

            piece_buffer += piece
            target_word = spacy_words[word_idx]

            # compare in lowercase for robustness
            if piece_buffer.lower() == target_word.lower():
                row = {
                    "caption_id": caption_id,
                    "caption": caption,
                    "word": target_word,
                    "category": spacy_categories[word_idx],
                    "subword_token": sub_token,
                    "vector": t5_vectors[sub_idx].cpu()
                }
                aligned_rows.append(row)

                print(
                    f"Aligned: word='{target_word}', "
                    f"category='{spacy_categories[word_idx]}', "
                    f"matched_subword='{sub_token}'"
                )

                word_idx += 1
                piece_buffer = ""

        print(f"Aligned {word_idx} / {len(spacy_words)} words.")
    
    print("\n" + "=" * 80)
    print(f"Total aligned rows: {len(aligned_rows)}")
    
    os.makedirs("artifacts", exist_ok=True)
    save_path = "artifacts/aligned_token_table.pt"
    torch.save(aligned_rows, save_path)

    print(f"Saved aligned token table to {save_path}")


if __name__ == "__main__":
    main()