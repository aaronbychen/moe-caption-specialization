import os
import torch
import spacy
from datasets import load_dataset
from transformers import AutoTokenizer, SwitchTransformersForConditionalGeneration
from src.utils.labeling import map_pos_to_category, map_pos_to_fine_category, normalize_t5_piece


def process_split(captions, split_name, tokenizer, model, moe_routers, moe_block_indices, nlp, caption_id_offset=0):
    aligned_rows = []
    batch_size = 8

    for batch_start in range(0, len(captions), batch_size):
        batch_captions = captions[batch_start:batch_start + batch_size]

        inputs = tokenizer(batch_captions, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model.encoder(**inputs, output_hidden_states=True)

        hidden_states_all = outputs.hidden_states
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        batch_len, seq_len = input_ids.shape

        # Router logits per MoE layer
        all_layer_logits = []
        all_layer_experts = []
        for router, block_idx in zip(moe_routers, moe_block_indices):
            h = hidden_states_all[block_idx]
            h_flat = h.reshape(-1, h.shape[-1])
            with torch.no_grad():
                logits = router.classifier(h_flat)
            logits = logits.view(batch_len, seq_len, -1)
            all_layer_logits.append(logits)
            all_layer_experts.append(logits.argmax(dim=-1))

        expert_ids_first = all_layer_experts[0]
        all_layer_probs = torch.cat([torch.softmax(lg, dim=-1) for lg in all_layer_logits], dim=-1)

        for i, caption in enumerate(batch_captions):
            caption_id = caption_id_offset + batch_start + i
            doc = nlp(caption)
            spacy_words = [token.text for token in doc]
            spacy_categories = [map_pos_to_category(token) for token in doc]
            spacy_fine_categories = [map_pos_to_fine_category(token) for token in doc]

            valid_len = attention_mask[i].sum().item()
            ids = input_ids[i][:valid_len].tolist()
            t5_tokens = tokenizer.convert_ids_to_tokens(ids)
            experts_first = expert_ids_first[i][:valid_len]
            probs_all = all_layer_probs[i][:valid_len]

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
                        "expert_id": experts_first[sub_idx].item(),
                        "all_layer_probs": probs_all[sub_idx].cpu(),
                    })
                    word_idx += 1
                    piece_buffer = ""

        batch_num = batch_start // batch_size + 1
        total_batches = (len(captions) + batch_size - 1) // batch_size
        if batch_num % 200 == 0 or batch_num == total_batches:
            print(f"  [{split_name}] batch {batch_num}/{total_batches}")

    return aligned_rows


def main():
    tokenizer = AutoTokenizer.from_pretrained("google/switch-base-8")
    model = SwitchTransformersForConditionalGeneration.from_pretrained("google/switch-base-8", torch_dtype=torch.float32)
    model.eval()

    moe_routers = []
    moe_block_indices = []
    for i, block in enumerate(model.encoder.block):
        if hasattr(block.layer[-1], 'mlp') and hasattr(block.layer[-1].mlp, 'router'):
            moe_routers.append(block.layer[-1].mlp.router)
            moe_block_indices.append(i)
    print(f"Found {len(moe_routers)} MoE layers at blocks {moe_block_indices}")

    nlp = spacy.load("en_core_web_sm")

    splits = [
        ("train[:50000]", "train"),
        ("validation", "val"),
    ]

    all_rows = []
    offset = 0

    for split_spec, split_name in splits:
        dataset = load_dataset("phiyodr/coco2017", split=split_spec)
        captions = [ex["captions"][0].strip() for ex in dataset if ex.get("captions")]
        print(f"Loaded {len(captions)} captions for {split_name}")

        rows = process_split(captions, split_name, tokenizer, model, moe_routers, moe_block_indices, nlp, caption_id_offset=offset)
        all_rows.extend(rows)
        offset += len(captions)
        print(f"  {split_name}: {len(rows)} aligned tokens")

    print(f"\nTotal: {len(all_rows)} aligned tokens")

    os.makedirs("artifacts", exist_ok=True)
    torch.save(all_rows, "artifacts/switch_token_table_8.pt")
    print("Saved artifacts/switch_token_table_8.pt")


if __name__ == "__main__":
    main()
