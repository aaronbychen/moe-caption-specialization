import spacy
from src.utils.labeling import map_pos_to_category

def main():
    captions = [
        "A dog jumps over a log.",
        "Two children sit at a table.",
        "A red bus is parked on the street."
    ]

    # load the English spaCy pipeline
    nlp = spacy.load("en_core_web_sm")

    for i, caption in enumerate(captions):
        # run POS tagging and dependency parsing
        doc = nlp(caption)

        print("\n" + "=" * 70)
        print(f"Caption {i+1}: {caption}")
        print(f"{'TEXT':<15}{'POS':<12}{'DEP':<12}{'CATEGORY':<12}")
        print("-" * 55)

        for token in doc:
            category = map_pos_to_category(token)
            print(f"{token.text:<15}{token.pos_:<12}{token.dep_:<12}{category:<12}")

if __name__ == "__main__":
    main()