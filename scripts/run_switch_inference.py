import os
import torch
import spacy
from datasets import load_dataset
from transformers import AutoTokenizer, SwitchTransformersForConditionalGeneration
from src.utils.labeling import map_pos_to_category, map_pos_to_fine_category, normalize_t5_piece


def main():
    dataset = load_dataset("phiyodr/coco2017", split="train[:10000]")
    captions = []
    for example in dataset:
        if "captions" in example and len(example["captions"]) > 0:
            captions.append(example["captions"][0].strip())

    print(f"Loaded {len(captions)} captions from phiyodr/coco2017.")

    tokenizer = AutoTokenizer.from_pretrained("google/switch-base-8")
    model = SwitchTransformersForConditionalGeneration.from_pretrained("google/switch-base-8", torch_dtype=torch.float32)
    model.eval()

    # collect all MoE layer routers
    moe_routers = []
    for block in model.encoder.block:
        if hasattr(block.layer[-1], 'mlp') and hasattr(block.layer[-1].mlp, 'router'):
            moe_routers.append(block.layer[-1].mlp.router)
    print(f"Found {len(moe_routers)} MoE layers in encoder.")

    nlp = spacy.load("en_core_web_sm")
    aligned_rows = []

    batch_size = 8
    for batch_start in range(0, len(captions), batch_size):
        batch_captions = captions[batch_start:batch_start + batch_size]

        inputs = tokenizer(batch_captions, return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            # get hidden states from all layers
            outputs = model.encoder(**inputs, output_hidden_states=True)

        hidden_states_all = outputs.hidden_states  # tuple of (batch, seq, 768) per layer
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        batch_len, seq_len = input_ids.shape

        # compute router logits for each MoE layer
        # MoE layers are at blocks 1,3,5,7,9,11 -> hidden_states indices 2,4,6,8,10,12
        # (hidden_states[0] = embedding, hidden_states[i] = output of block i-1)
        moe_block_indices = [1, 3, 5, 7, 9, 11]
        all_layer_logits = []  # list of (batch, seq, 8)
        all_layer_experts = []  # list of (batch, seq)

        for router, block_idx in zip(moe_routers, moe_block_indices):
            # input to block i is hidden_states[i] (output of previous block)
            h = hidden_states_all[block_idx]  # (batch, seq, 768)
            h_flat = h.reshape(-1, h.shape[-1])
            with torch.no_grad():
                logits = router.classifier(h_flat)  # (batch*seq, 8)
            logits = logits.view(batch_len, seq_len, -1)
            all_layer_logits.append(logits)
            all_layer_experts.append(logits.argmax(dim=-1))

        # first MoE layer expert (backward compat)
        expert_ids_first = all_layer_experts[0]
        # all-layer soft probs concatenated: (batch, seq, 48)
        all_layer_probs = torch.cat([torch.softmax(lg, dim=-1) for lg in all_layer_logits], dim=-1)

        if batch_start == 0:
            print(f"Per-layer logits shape: {all_layer_logits[0].shape}")
            print(f"All-layer probs shape: {all_layer_probs.shape}")
            print(f"First layer expert assignments: {expert_ids_first[0][:10]}")

        for i, caption in enumerate(batch_captions):
            caption_id = batch_start + i
            doc = nlp(caption)
            spacy_words = [token.text for token in doc]
            spacy_categories = [map_pos_to_category(token) for token in doc]
            spacy_fine_categories = [map_pos_to_fine_category(token) for token in doc]

            valid_len = attention_mask[i].sum().item()
            ids = input_ids[i][:valid_len].tolist()
            t5_tokens = tokenizer.convert_ids_to_tokens(ids)
            experts_first = expert_ids_first[i][:valid_len]
            probs_all = all_layer_probs[i][:valid_len]  # (valid_len, 48)

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
                    row = {
                        "caption_id": caption_id,
                        "word": target_word,
                        "category": spacy_categories[word_idx],
                        "fine_category": spacy_fine_categories[word_idx],
                        "expert_id": experts_first[sub_idx].item(),
                        "all_layer_probs": probs_all[sub_idx].cpu(),
                    }
                    aligned_rows.append(row)
                    word_idx += 1
                    piece_buffer = ""

        batch_num = batch_start // batch_size + 1
        total_batches = (len(captions) + batch_size - 1) // batch_size
        if batch_num % 50 == 0 or batch_num == total_batches:
            print(f"Processed batch {batch_num} / {total_batches}")

    print(f"\nTotal aligned rows: {len(aligned_rows)}")

    os.makedirs("artifacts", exist_ok=True)
    save_path = "artifacts/switch_token_table_8.pt"
    torch.save(aligned_rows, save_path)
    print(f"Saved Switch token table to {save_path}")


if __name__ == "__main__":
    main()
