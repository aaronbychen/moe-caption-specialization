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
from sklearn.metrics import (adjusted_rand_score, f1_score,
                             classification_report)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def majority_mapping(categories, assignments, n_experts):
    expert_cats = defaultdict(list)
    for cat, assign in zip(categories, assignments):
        expert_cats[assign].append(cat)
    return {eid: Counter(expert_cats.get(eid, [])).most_common(1)[0][0]
            if expert_cats.get(eid) else None for eid in range(n_experts)}

def apply_mapping(assignments, mapping):
    return [mapping.get(a) for a in assignments]

def score(true, pred):
    acc = sum(1 for t, p in zip(true, pred) if t == p) / len(true)
    mf1 = f1_score(true, pred, average="macro", zero_division=0)
    ari = adjusted_rand_score(true, pred)
    return {"accuracy": acc, "macro_f1": mf1, "ari": ari}

def word_identity_baseline(train_words, train_cats, test_words, test_cats):
    word_cats = defaultdict(list)
    for w, c in zip(train_words, train_cats):
        word_cats[w.lower()].append(c)
    word_map = {w: Counter(cats).most_common(1)[0][0] for w, cats in word_cats.items()}
    fallback = Counter(train_cats).most_common(1)[0][0]
    preds = [word_map.get(w.lower(), fallback) for w in test_words]
    return score(test_cats, preds)

def split_data(data):
    """Split by 'split' field. Falls back to caption-level 80/20 if no split field."""
    if data and "split" in data[0]:
        train = [r for r in data if r["split"] == "train"]
        test = [r for r in data if r["split"] == "val"]
        return train, test
    # fallback
    caption_ids = sorted(set(r["caption_id"] for r in data))
    rng = np.random.RandomState(42)
    rng.shuffle(caption_ids)
    split_idx = int(len(caption_ids) * 0.8)
    train_ids = set(caption_ids[:split_idx])
    return ([r for r in data if r["caption_id"] in train_ids],
            [r for r in data if r["caption_id"] not in train_ids])


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def eval_clustering(train_cats, train_features, test_cats, test_features, n, km_seed=42):
    km = KMeans(n_clusters=n, random_state=km_seed, n_init="auto")
    km.fit(train_features)
    train_clusters = km.predict(train_features).tolist()
    test_clusters = km.predict(test_features).tolist()
    mapping = majority_mapping(train_cats, train_clusters, n)
    test_preds = apply_mapping(test_clusters, mapping)
    result = score(test_cats, test_preds)
    result["ari"] = adjusted_rand_score(test_cats, test_clusters)
    result["preds"] = test_preds
    return result

def eval_expert_ids(train_cats, train_ids, test_cats, test_ids, n):
    mapping = majority_mapping(train_cats, train_ids, n)
    test_preds = apply_mapping(test_ids, mapping)
    result = score(test_cats, test_preds)
    result["ari"] = adjusted_rand_score(test_cats, test_ids)
    result["preds"] = test_preds
    return result


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

FEATURE_ORDER = ["random", "majority", "word_identity", "switch_hard",
                 "t5_pca_8d", "switch_all_layer_pca_8d", "t5_pca_32d",
                 "t5_pca_48d", "t5_768d", "switch_all_layer_48d"]
DIM_MAP = {"random": "-", "majority": "-", "word_identity": "-",
           "switch_hard": "8", "t5_pca_8d": "8", "switch_all_layer_pca_8d": "8",
           "t5_pca_32d": "32", "t5_pca_48d": "48",
           "switch_all_layer_48d": "48", "t5_768d": "768"}

def run_benchmark(t5_train, t5_test, sw_train, sw_test, cat_key, n=8):
    train_cats = [r[cat_key] for r in t5_train]
    test_cats = [r[cat_key] for r in t5_test]
    train_vecs = torch.stack([r["vector"] for r in t5_train]).numpy().astype("float32")
    test_vecs = torch.stack([r["vector"] for r in t5_test]).numpy().astype("float32")

    results = {}

    # Baselines
    majority_class = Counter(train_cats).most_common(1)[0][0]
    results["majority"] = score(test_cats, [majority_class] * len(test_cats))
    results["random"] = {"accuracy": 1.0 / len(set(train_cats)), "macro_f1": 0.0, "ari": 0.0}

    # Word identity
    results["word_identity"] = word_identity_baseline(
        [r["word"] for r in t5_train], train_cats,
        [r["word"] for r in t5_test], test_cats)

    # T5 768D
    results["t5_768d"] = eval_clustering(train_cats, train_vecs, test_cats, test_vecs, n)

    # T5 PCA-8D
    pca8 = PCA(n_components=8, random_state=42).fit(train_vecs)
    results["t5_pca_8d"] = eval_clustering(train_cats, pca8.transform(train_vecs), test_cats, pca8.transform(test_vecs), n)

    # T5 PCA-32D
    pca32 = PCA(n_components=32, random_state=42).fit(train_vecs)
    results["t5_pca_32d"] = eval_clustering(train_cats, pca32.transform(train_vecs), test_cats, pca32.transform(test_vecs), n)

    # T5 PCA-48D (exact dimension match with Switch all-layer)
    pca48 = PCA(n_components=48, random_state=42).fit(train_vecs)
    results["t5_pca_48d"] = eval_clustering(train_cats, pca48.transform(train_vecs), test_cats, pca48.transform(test_vecs), n)

    # Switch
    if sw_train:
        sw_train_cats = [r.get(cat_key, r["category"]) for r in sw_train]
        sw_test_cats = [r.get(cat_key, r["category"]) for r in sw_test]

        results["switch_hard"] = eval_expert_ids(
            sw_train_cats, [r["expert_id"] for r in sw_train],
            sw_test_cats, [r["expert_id"] for r in sw_test], n)

        has_all_layer = "all_layer_probs" in sw_train[0]
        if has_all_layer:
            sw_train_p = torch.stack([r["all_layer_probs"] for r in sw_train]).numpy().astype("float32")
            sw_test_p = torch.stack([r["all_layer_probs"] for r in sw_test]).numpy().astype("float32")

            results["switch_all_layer_48d"] = eval_clustering(sw_train_cats, sw_train_p, sw_test_cats, sw_test_p, n)

            sw_pca8 = PCA(n_components=8, random_state=42).fit(sw_train_p)
            results["switch_all_layer_pca_8d"] = eval_clustering(
                sw_train_cats, sw_pca8.transform(sw_train_p), sw_test_cats, sw_pca8.transform(sw_test_p), n)

    return results, test_cats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Load data
    all_files = glob.glob("artifacts/aligned_token_table_part*.pt")
    t5_data = []
    for f in all_files:
        t5_data.extend(torch.load(f))
    print(f"Loaded {len(t5_data)} T5 tokens")

    sw_data = None
    for p in glob.glob("artifacts/switch_token_table_8.pt"):
        sw_data = torch.load(p)
        break

    has_fine = "fine_category" in t5_data[0] if t5_data else False

    # Split
    t5_train, t5_test = split_data(t5_data)
    sw_train, sw_test = split_data(sw_data) if sw_data else (None, None)
    print(f"T5: {len(t5_train)} train, {len(t5_test)} test")
    if sw_train:
        print(f"Switch: {len(sw_train)} train, {len(sw_test)} test")

    output = {}

    for cat_key, label in [("category", "coarse")] + ([("fine_category", "fine")] if has_fine else []):
        print(f"\n{'#'*60}")
        print(f"BENCHMARK | {label}")
        print(f"{'#'*60}")

        results, test_cats = run_benchmark(t5_train, t5_test, sw_train, sw_test, cat_key)

        # Print summary
        print(f"\n{'Feature':<30} {'Dim':>4} {'Accuracy':>10} {'Macro-F1':>10} {'ARI':>8}")
        print("-" * 70)
        for key in FEATURE_ORDER:
            if key not in results:
                continue
            r = results[key]
            print(f"{key:<30} {DIM_MAP.get(key,'-'):>4} {r['accuracy']:>10.4f} {r['macro_f1']:>10.4f} {r['ari']:>8.4f}")

        # Per-class F1
        print(f"\n--- Per-class F1 | {label} ---")
        for key in FEATURE_ORDER:
            if key not in results or "preds" not in results[key]:
                continue
            print(f"\n  [{key}]")
            for line in classification_report(test_cats, results[key]["preds"], zero_division=0).strip().split("\n"):
                print(f"    {line}")

        # Clean preds before saving
        for key in results:
            results[key].pop("preds", None)
        output[label] = results

    # --- KMeans seed robustness (coarse only) ---
    print(f"\n{'#'*60}")
    print("KMEANS SEED ROBUSTNESS | coarse")
    print(f"{'#'*60}")

    cat_key = "category"
    train_cats = [r[cat_key] for r in t5_train]
    test_cats = [r[cat_key] for r in t5_test]
    train_vecs = torch.stack([r["vector"] for r in t5_train]).numpy().astype("float32")
    test_vecs = torch.stack([r["vector"] for r in t5_test]).numpy().astype("float32")

    km_seeds = [42, 123, 7]
    seed_features = {"t5_768d": (train_vecs, test_vecs, train_cats, test_cats)}

    # T5 PCA-48D
    pca48 = PCA(n_components=48, random_state=42).fit(train_vecs)
    seed_features["t5_pca_48d"] = (pca48.transform(train_vecs), pca48.transform(test_vecs), train_cats, test_cats)

    if sw_train:
        sw_train_cats = [r.get(cat_key, r["category"]) for r in sw_train]
        sw_test_cats = [r.get(cat_key, r["category"]) for r in sw_test]
        sw_train_p = torch.stack([r["all_layer_probs"] for r in sw_train]).numpy().astype("float32")
        sw_test_p = torch.stack([r["all_layer_probs"] for r in sw_test]).numpy().astype("float32")
        seed_features["switch_all_layer_48d"] = (sw_train_p, sw_test_p, sw_train_cats, sw_test_cats)

    seed_results = {}
    for fname, (tr_f, te_f, tr_c, te_c) in seed_features.items():
        runs = [eval_clustering(tr_c, tr_f, te_c, te_f, 8, km_seed=s) for s in km_seeds]
        accs = [r["accuracy"] for r in runs]
        f1s = [r["macro_f1"] for r in runs]
        aris = [r["ari"] for r in runs]
        print(f"  {fname:<30} Acc={np.mean(accs):.4f}+/-{np.std(accs):.4f} "
              f"F1={np.mean(f1s):.4f}+/-{np.std(f1s):.4f} ARI={np.mean(aris):.4f}+/-{np.std(aris):.4f}")
        seed_results[fname] = {
            "accuracy_mean": np.mean(accs), "accuracy_std": np.std(accs),
            "macro_f1_mean": np.mean(f1s), "macro_f1_std": np.std(f1s),
            "ari_mean": np.mean(aris), "ari_std": np.std(aris),
        }
    output["kmeans_seed_robustness"] = seed_results

    output["metadata"] = {
        "protocol": "COCO train 50k (fit) + validation 5k (eval)",
        "pca_fit": "train only", "kmeans_fit": "train only",
        "mapping_fit": "train only", "evaluation": "val only",
        "n_clusters": 8,
        "t5_train": len(t5_train), "t5_test": len(t5_test),
    }

    with open("artifacts/eval_metrics.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nSaved artifacts/eval_metrics.json")

if __name__ == "__main__":
    main()
