import torch
from transformers import T5Tokenizer, T5EncoderModel

def main():
    captions = [
        "A dog jumps over a log.",
        "Two children sit at a table.",
        "A red bus is parked on the street."
    ]

    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5EncoderModel.from_pretrained(model_name)
    model.eval()

    inputs = tokenizer(
        captions,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    # hidden states: [batch_size, seq_len, hidden_dim]
    hidden_states = outputs.last_hidden_state
    print("hidden_states shape:", hidden_states.shape)

    # print each caption's token
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    for i, caption in enumerate(captions):
        print("\n" + "=" * 60)
        print(f"Caption {i+1}: {caption}")

        valid_len = attention_mask[i].sum().item()
        ids = input_ids[i][:valid_len].tolist()
        tokens = tokenizer.convert_ids_to_tokens(ids)

        print("Tokens:")
        print(tokens)

        print("Per-caption hidden state shape:")
        print(hidden_states[i][:valid_len].shape)

if __name__ == "__main__":
    main()