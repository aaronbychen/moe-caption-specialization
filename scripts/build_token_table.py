import os
import torch
import spacy
from datasets import load_dataset
from transformers import T5Tokenizer, T5EncoderModel
from src.utils.labeling import map_pos_to_category, map_pos_to_fine_category, normalize_t5_piece


def process_split(captions, split_name, tokenizer, model, nlp, caption_id_offset=0):
    aligned_rows = []
    batch_size = 16

    for batch_start in range(0, len(captions), batch_size):
        batch_captions = captions[batch_start:batch_start + batch_size]

        inputs = tokenizer(batch_captions, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)

        hidden_states = outputs.last_hidden_state
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        for _id, caption in enumerate(batch_captions):
            caption_id = caption_id_offset + batch_start + _id
            doc = nlp(caption)
            spacy_words = [token.text for token in doc]
            spacy_categories = [map_pos_to_category(token) for token in doc]
            spacy_fine_categories = [map_pos_to_fine_category(token) for token in doc]

            valid_len = attention_mask[_id].sum().item()
            ids = input_ids[_id][:valid_len].tolist()
            t5_tokens = tokenizer.convert_ids_to_tokens(ids)
            t5_vectors = hidden_states[_id][:valid_len]

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

                if piece_buffer.lower() == target_word.lower():
                    aligned_rows.append({
                        "caption_id": caption_id,
                        "split": split_name,
                        "word": target_word,
                        "category": spacy_categories[word_idx],
                        "fine_category": spacy_fine_categories[word_idx],
                        "vector": t5_vectors[sub_idx].detach().cpu().clone()
                    })
                    word_idx += 1
                    piece_buffer = ""

        batch_num = batch_start // batch_size + 1
        total_batches = (len(captions) + batch_size - 1) // batch_size
        if batch_num % 200 == 0 or batch_num == total_batches:
            print(f"  [{split_name}] batch {batch_num}/{total_batches}")

    return aligned_rows


def main():
    # Load COCO train (50k) and validation (5k)
    splits = [
        ("train[:50000]", "train"),
        ("validation", "val"),
    ]

    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = T5EncoderModel.from_pretrained("t5-base")
    model.eval()
    nlp = spacy.load("en_core_web_sm")

    all_rows = []
    offset = 0

    for split_spec, split_name in splits:
        dataset = load_dataset("phiyodr/coco2017", split=split_spec)
        captions = [ex["captions"][0].strip() for ex in dataset if ex.get("captions")]
        print(f"Loaded {len(captions)} captions for {split_name}")

        rows = process_split(captions, split_name, tokenizer, model, nlp, caption_id_offset=offset)
        all_rows.extend(rows)
        offset += len(captions)
        print(f"  {split_name}: {len(rows)} aligned tokens")

    print(f"\nTotal: {len(all_rows)} aligned tokens")

    os.makedirs("artifacts", exist_ok=True)
    chunk_size = 250000
    total_chunks = (len(all_rows) + chunk_size - 1) // chunk_size
    for i in range(total_chunks):
        chunk = all_rows[i * chunk_size:(i + 1) * chunk_size]
        path = f"artifacts/aligned_token_table_part{i+1}.pt"
        torch.save(chunk, path)
        print(f"Saved {path} ({len(chunk)} rows)")


if __name__ == "__main__":
    main()
