import os
import torch
import spacy
from datasets import load_dataset
from transformers import T5Tokenizer, T5EncoderModel
from src.utils.labeling import map_pos_to_category, normalize_t5_piece


def main():
    # load COCO training split and use the first caption from each example
    dataset = load_dataset("phiyodr/coco2017", split="train[:10000]")
    captions = []
    for example in dataset:
        if "captions" in example and len(example["captions"]) > 0:
            captions.append(example["captions"][0].strip())

    print(f"Loaded {len(captions)} captions from phiyodr/coco2017.")

    # load models
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = T5EncoderModel.from_pretrained("t5-base")
    model.eval()

    nlp = spacy.load("en_core_web_sm")

    aligned_rows = []

    batch_size = 16

    for batch_start in range(0, len(captions), batch_size):
        batch_captions = captions[batch_start:batch_start + batch_size]

        # tokenize captions with T5
        inputs = tokenizer(
            batch_captions,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        with torch.no_grad():
            outputs = model(**inputs)

        hidden_states = outputs.last_hidden_state  # (B, seq_len, 512)
        input_ids = inputs["input_ids"]  # (B, seq_len), 1 id for each token
        attention_mask = inputs["attention_mask"]  # (B, seq_len), seq_len = length of longest tokenized caption in the specific batch

        for _id, caption in enumerate(batch_captions):
            caption_id = batch_start + _id
            doc = nlp(caption)
            spacy_words = [token.text for token in doc]
            spacy_categories = [map_pos_to_category(token) for token in doc]

            valid_len = attention_mask[_id].sum().item()  # <= seq_len
            ids = input_ids[_id][:valid_len].tolist()  # (valid_len)
            t5_tokens = tokenizer.convert_ids_to_tokens(ids)  # (valid_len)
            t5_vectors = hidden_states[_id][:valid_len]  # (valid_len, 768)

            # print("\n" + "=" * 80)
            # print(f"Caption {caption_id}: {caption}")
            # print("spaCy words:", spacy_words)
            # print("T5 tokens:  ", t5_tokens)

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
                        "vector": t5_vectors[sub_idx].detach().cpu().clone()
                    }
                    aligned_rows.append(row)

                    # print(
                    #     f"Aligned: word='{target_word}', "
                    #     f"category='{spacy_categories[word_idx]}', "
                    #     f"matched_subword='{sub_token}'"
                    # )

                    word_idx += 1
                    piece_buffer = ""

            # print(f"Aligned {word_idx} / {len(spacy_words)} words.")
        current_batch = batch_start // batch_size + 1
        total_batches = (len(captions) + batch_size - 1) // batch_size
        if current_batch % 100 == 0 or current_batch == total_batches:
            print(f"Processed batch {current_batch} / {total_batches}")
    
    print("\n" + "=" * 80)
    print(f"Total aligned rows: {len(aligned_rows)}")
    
    chunk_size = 250000
    total_chunks = (len(aligned_rows) + chunk_size - 1) // chunk_size

    os.makedirs("artifacts", exist_ok=True)

    for i in range(total_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(aligned_rows))
        chunk = aligned_rows[start_idx:end_idx]

        save_path = f"artifacts/aligned_token_table_part{i+1}.pt"
        torch.save(chunk, save_path)
        print(f"Saved chunk {i+1}/{total_chunks} to {save_path}")


if __name__ == "__main__":
    main()