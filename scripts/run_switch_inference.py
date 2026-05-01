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

    # load Switch model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/switch-base-8")
    model = SwitchTransformersForConditionalGeneration.from_pretrained("google/switch-base-8", torch_dtype=torch.float32)
    model.eval()

    nlp = spacy.load("en_core_web_sm")

    aligned_rows = []

    # process in small batches to avoid OOM
    batch_size = 8
    for batch_start in range(0, len(captions), batch_size):
        batch_captions = captions[batch_start:batch_start + batch_size]
        
        inputs = tokenizer(
            batch_captions,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        with torch.no_grad():
            outputs = model.encoder(**inputs)

        # manually compute router logits from hidden states
        hidden_states = outputs.last_hidden_state
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # get the first MoE layer's router
        first_moe_layer = None
        for layer in model.encoder.block:
            if hasattr(layer.layer[-1], 'mlp') and hasattr(layer.layer[-1].mlp, 'router'):
                first_moe_layer = layer.layer[-1]
                break
        
        # pass through layer norm and get router logits
        batch_len = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        
        # we need to recompute through the first MoE layer to get routing
        # simpler approach: just use the router classifier on the hidden states
        router = first_moe_layer.mlp.router
        
        # flatten hidden states
        hidden_flat = hidden_states.view(-1, hidden_states.shape[-1])
        
        # get full router logits before top-k
        with torch.no_grad():
            router_logits = router.classifier(hidden_flat)  # (batch*seq, num_experts)
        
        if batch_start == 0:
            print(f"Router logits shape: {router_logits.shape}")
            print(f"Sample logits:\n{router_logits[:5]}")
            print(f"Expert assignments: {router_logits[:5].argmax(dim=-1)}")
        
        expert_ids = router_logits.argmax(dim=-1).view(batch_len, seq_len)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        for i, caption in enumerate(batch_captions):
            caption_id = batch_start + i
            doc = nlp(caption)
            spacy_words = [token.text for token in doc]
            spacy_categories = [map_pos_to_category(token) for token in doc]
            spacy_fine_categories = [map_pos_to_fine_category(token) for token in doc]

            valid_len = attention_mask[i].sum().item()
            ids = input_ids[i][:valid_len].tolist()
            t5_tokens = tokenizer.convert_ids_to_tokens(ids)
            experts = expert_ids[i][:valid_len]

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
                        "expert_id": experts[sub_idx].item()
                    }
                    aligned_rows.append(row)

                    word_idx += 1
                    piece_buffer = ""

        print(f"Processed batch {batch_start // batch_size + 1} / {(len(captions) + batch_size - 1) // batch_size}")

    print(f"\nTotal aligned rows: {len(aligned_rows)}")
    
    os.makedirs("artifacts", exist_ok=True)
    save_path = "artifacts/switch_token_table_10000.pt"
    torch.save(aligned_rows, save_path)

    print(f"Saved Switch token table to {save_path}")


if __name__ == "__main__":
    main()
