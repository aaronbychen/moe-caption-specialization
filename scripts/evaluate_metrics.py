import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import glob
import torch
import numpy as np
from collections import Counter, defaultdict
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, v_measure_score, f1_score


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def majority_mapping(categories, assignments, n_experts):
    """Learn expert/cluster -> dominant category mapping."""
    expert_cats = defaultdict(list)
    for cat, assign in zip(categories, assignments):
        expert_cats[assign].append(cat)
    mapping = {}
    for eid in range(n_experts):
        tokens = expert_cats.get(eid, [])
        mapping[eid] = Counter(tokens).most_common(1)[0][0] if tokens else None
    return mapping

def apply_mapping(assignments, mapping):
    return [mapping.get(a) for a in assignments]

def score(true, pred):
    acc = sum(1 for t, p in zip(true, pred) if t == p) / len(true)
    mf1 = f1_score(true, pred, average="macro", zero_division=0)
    ari = adjusted_rand_score(true, pred)
    return {"accuracy": acc, "macro_f1": mf1, "ari": ari}

def word_identity_baseline(train_words, train_cats, test_words, test_cats):
    """Majority word -> category mapping learned on train, evaluated on test."""
    word_cats = defaultdict(list)
    for w, c in zip(train_words, train_cats):
        word_cats[w.lower()].append(c)
    word_map = {w: Counter(cats).most_common(1)[0][0] for w, cats in word_cats.items()}
    # fallback = overall majority class
    fallback = Counter(train_cats).most_common(1)[0][0]
    preds = [word_map.get(w.lower(), fallback) for w in test_words]
    return score(test_cats, preds)


# ---------------------------------------------------------------------------
# Caption-level train/test split
# ---------------------------------------------------------------------------

def split_by_caption(data, test_ratio=0.2, seed=42):
    """Split data by caption_id so no caption appears in both train and test."""
    caption_ids = sorted(set(row["caption_id"] for row in data))
    rng = np.random.RandomState(seed)
    rng.shuffle(caption_ids)
    split_idx = int(len(caption_ids) * (1 - test_ratio))
    train_ids = set(caption_ids[:split_idx])
    train = [r for r in data if r["caption_id"] in train_ids]
    test = [r for r in data if r["caption_id"] not in train_ids]
    return train, test


# ---------------------------------------------------------------------------
# Evaluation with proper train/test protocol
# ---------------------------------------------------------------------------

def eval_clustering(label, train_cats, train_features, test_cats, test_features, n):
    """Fit KMeans on train, predict on test, learn mapping on train, score on test."""
    km = KMeans(n_clusters=n, random_state=42, n_init="auto")
    km.fit(train_features)
    train_clusters = km.predict(train_features).tolist()
    test_clusters = km.predict(test_features).tolist()
    mapping = majority_mapping(train_cats, train_clusters, n)
    test_preds = apply_mapping(test_clusters, mapping)
    result = score(test_cats, test_preds)
    # ARI on test (assignment vs true label)
    result["ari"] = adjusted_rand_score(test_cats, test_clusters)
    print(f"  [{label}] Acc={result['accuracy']:.4f} F1={result['macro_f1']:.4f} ARI={result['ari']:.4f}")
    return result

def eval_expert_ids(label, train_cats, train_ids, test_cats, test_ids, n):
    """Learn mapping on train expert IDs, score on test."""
    mapping = majority_mapping(train_cats, train_ids, n)
    test_preds = apply_mapping(test_ids, mapping)
    result = score(test_cats, test_preds)
    result["ari"] = adjusted_rand_score(test_cats, test_ids)
    print(f"  [{label}] Acc={result['accuracy']:.4f} F1={result['macro_f1']:.4f} ARI={result['ari']:.4f}")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Load T5 data
    all_files = glob.glob("artifacts/**/aligned_token_table_part*.pt", recursive=True)
    t5_data = []
    for file in all_files:
        t5_data.extend(torch.load(file))
    print(f"Loaded {len(t5_data)} T5 aligned tokens.")

    # Load Switch data
    sw_data = None
    for sw_path in glob.glob("artifacts/**/switch_token_table_8.pt", recursive=True):
        sw_data = torch.load(sw_path)
        break
    has_switch = sw_data is not None
    has_all_layer = has_switch and "all_layer_probs" in sw_data[0]
    has_fine = "fine_category" in t5_data[0] if t5_data else False

    print(f"Switch data: {len(sw_data) if sw_data else 0} tokens, all_layer={has_all_layer}, fine={has_fine}")

    # Caption-level train/test split
    t5_train, t5_test = split_by_caption(t5_data)
    print(f"T5 split: {len(t5_train)} train, {len(t5_test)} test")

    if has_switch:
        sw_train, sw_test = split_by_caption(sw_data)
        print(f"Switch split: {len(sw_train)} train, {len(sw_test)} test")

    n = 8
    output = {}

    for cat_key, label in [("category", "coarse")] + ([("fine_category", "fine")] if has_fine else []):
        print(f"\n{'#'*60}")
        print(f"FAIRNESS BENCHMARK | {label} | N={n}")
        print(f"{'#'*60}")

        # Extract categories and features
        train_cats = [r[cat_key] for r in t5_train]
        test_cats = [r[cat_key] for r in t5_test]
        train_vecs = torch.stack([r["vector"] for r in t5_train]).numpy().astype("float32")
        test_vecs = torch.stack([r["vector"] for r in t5_test]).numpy().astype("float32")

        results = {}

        # --- Baselines ---
        majority_class = Counter(train_cats).most_common(1)[0][0]
        majority_preds = [majority_class] * len(test_cats)
        results["majority"] = score(test_cats, majority_preds)
        n_classes = len(set(train_cats))
        results["majority"]["note"] = f"always predict '{majority_class}'"
        results["random"] = {"accuracy": 1.0 / n_classes, "macro_f1": 0.0, "ari": 0.0}
        print(f"  [Majority baseline] Acc={results['majority']['accuracy']:.4f}")
        print(f"  [Random baseline] Acc={results['random']['accuracy']:.4f}")

        # --- Word identity baseline ---
        train_words = [r["word"] for r in t5_train]
        test_words = [r["word"] for r in t5_test]
        results["word_identity"] = word_identity_baseline(train_words, train_cats, test_words, test_cats)
        print(f"  [Word identity] Acc={results['word_identity']['accuracy']:.4f} F1={results['word_identity']['macro_f1']:.4f}")

        # --- T5 768D KMeans baseline ---
        results["t5_768d"] = eval_clustering("T5 768D KMeans", train_cats, train_vecs, test_cats, test_vecs, n)

        # --- T5 PCA-8D ---
        pca8 = PCA(n_components=8, random_state=42).fit(train_vecs)
        results["t5_pca_8d"] = eval_clustering("T5 PCA-8D", train_cats, pca8.transform(train_vecs), test_cats, pca8.transform(test_vecs), n)

        # --- T5 PCA-32D ---
        pca32 = PCA(n_components=32, random_state=42).fit(train_vecs)
        results["t5_pca_32d"] = eval_clustering("T5 PCA-32D", train_cats, pca32.transform(train_vecs), test_cats, pca32.transform(test_vecs), n)

        # --- Switch ---
        if has_switch:
            sw_train_cats = [r.get(cat_key, r["category"]) for r in sw_train]
            sw_test_cats = [r.get(cat_key, r["category"]) for r in sw_test]
            sw_train_ids = [r["expert_id"] for r in sw_train]
            sw_test_ids = [r["expert_id"] for r in sw_test]

            # Switch hard expert ID
            results["switch_hard"] = eval_expert_ids("Switch hard expert", sw_train_cats, sw_train_ids, sw_test_cats, sw_test_ids, n)

            if has_all_layer:
                sw_train_probs = torch.stack([r["all_layer_probs"] for r in sw_train]).numpy().astype("float32")
                sw_test_probs = torch.stack([r["all_layer_probs"] for r in sw_test]).numpy().astype("float32")

                # Switch all-layer 48D
                results["switch_all_layer_48d"] = eval_clustering("Switch all-layer 48D", sw_train_cats, sw_train_probs, sw_test_cats, sw_test_probs, n)

                # Switch all-layer PCA-8D
                sw_pca8 = PCA(n_components=8, random_state=42).fit(sw_train_probs)
                results["switch_all_layer_pca_8d"] = eval_clustering("Switch all-layer PCA-8D", sw_train_cats, sw_pca8.transform(sw_train_probs), sw_test_cats, sw_pca8.transform(sw_test_probs), n)

        output[label] = results

    # --- Print summary table ---
    for label, results in output.items():
        print(f"\n{'='*70}")
        print(f"SUMMARY TABLE | {label}")
        print(f"{'='*70}")
        print(f"{'Feature':<30} {'Dim':>5} {'Accuracy':>10} {'Macro-F1':>10} {'ARI':>8}")
        print("-" * 70)
        dim_map = {
            "random": "-", "majority": "-", "word_identity": "-",
            "switch_hard": "8", "t5_pca_8d": "8", "switch_all_layer_pca_8d": "8",
            "t5_pca_32d": "32", "switch_all_layer_48d": "48", "t5_768d": "768",
        }
        for key in ["random", "majority", "word_identity", "switch_hard",
                     "t5_pca_8d", "switch_all_layer_pca_8d", "t5_pca_32d",
                     "t5_768d", "switch_all_layer_48d"]:
            if key not in results:
                continue
            r = results[key]
            dim = dim_map.get(key, "?")
            print(f"{key:<30} {dim:>5} {r['accuracy']:>10.4f} {r['macro_f1']:>10.4f} {r['ari']:>8.4f}")

    # --- Metadata ---
    output["metadata"] = {
        "protocol": "caption-level 80/20 train/test split",
        "pca_fit": "train only",
        "kmeans_fit": "train only",
        "mapping_fit": "train only",
        "evaluation": "test only",
        "seed": 42,
        "n_clusters": n,
        "t5_train_tokens": len(t5_train),
        "t5_test_tokens": len(t5_test),
    }

    with open("artifacts/eval_metrics.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nSaved metrics to artifacts/eval_metrics.json")

if __name__ == "__main__":
    main()
