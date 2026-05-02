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

def split_by_caption(data, test_ratio=0.2, seed=42):
    caption_ids = sorted(set(row["caption_id"] for row in data))
    rng = np.random.RandomState(seed)
    rng.shuffle(caption_ids)
    split_idx = int(len(caption_ids) * (1 - test_ratio))
    train_ids = set(caption_ids[:split_idx])
    train = [r for r in data if r["caption_id"] in train_ids]
    test = [r for r in data if r["caption_id"] not in train_ids]
    return train, test


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def eval_clustering(train_cats, train_features, test_cats, test_features, n):
    km = KMeans(n_clusters=n, random_state=42, n_init="auto")
    km.fit(train_features)
    train_clusters = km.predict(train_features).tolist()
    test_clusters = km.predict(test_features).tolist()
    mapping = majority_mapping(train_cats, train_clusters, n)
    test_preds = apply_mapping(test_clusters, mapping)
    result = score(test_cats, test_preds)
    result["ari"] = adjusted_rand_score(test_cats, test_clusters)
    result["preds"] = test_preds  # for per-class analysis
    return result

def eval_expert_ids(train_cats, train_ids, test_cats, test_ids, n):
    mapping = majority_mapping(train_cats, train_ids, n)
    test_preds = apply_mapping(test_ids, mapping)
    result = score(test_cats, test_preds)
    result["ari"] = adjusted_rand_score(test_cats, test_ids)
    result["preds"] = test_preds
    return result


# ---------------------------------------------------------------------------
# Single-seed benchmark run
# ---------------------------------------------------------------------------

def run_benchmark(t5_data, sw_data, cat_key, seed, n=8):
    """Run full benchmark for one seed. Returns dict of results."""
    t5_train, t5_test = split_by_caption(t5_data, seed=seed)
    has_switch = sw_data is not None
    has_all_layer = has_switch and "all_layer_probs" in sw_data[0]

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

    # Switch
    if has_switch:
        sw_train, sw_test = split_by_caption(sw_data, seed=seed)
        sw_train_cats = [r.get(cat_key, r["category"]) for r in sw_train]
        sw_test_cats = [r.get(cat_key, r["category"]) for r in sw_test]

        results["switch_hard"] = eval_expert_ids(
            sw_train_cats, [r["expert_id"] for r in sw_train],
            sw_test_cats, [r["expert_id"] for r in sw_test], n)

        if has_all_layer:
            sw_train_p = torch.stack([r["all_layer_probs"] for r in sw_train]).numpy().astype("float32")
            sw_test_p = torch.stack([r["all_layer_probs"] for r in sw_test]).numpy().astype("float32")

            results["switch_all_layer_48d"] = eval_clustering(sw_train_cats, sw_train_p, sw_test_cats, sw_test_p, n)

            sw_pca8 = PCA(n_components=8, random_state=42).fit(sw_train_p)
            results["switch_all_layer_pca_8d"] = eval_clustering(
                sw_train_cats, sw_pca8.transform(sw_train_p), sw_test_cats, sw_pca8.transform(sw_test_p), n)

    results["_test_cats"] = test_cats
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

FEATURE_ORDER = ["random", "majority", "word_identity", "switch_hard",
                 "t5_pca_8d", "switch_all_layer_pca_8d", "t5_pca_32d",
                 "t5_768d", "switch_all_layer_48d"]
DIM_MAP = {"random": "-", "majority": "-", "word_identity": "-",
           "switch_hard": "8", "t5_pca_8d": "8", "switch_all_layer_pca_8d": "8",
           "t5_pca_32d": "32", "switch_all_layer_48d": "48", "t5_768d": "768"}

def main():
    all_files = glob.glob("artifacts/**/aligned_token_table_part*.pt", recursive=True)
    t5_data = []
    for file in all_files:
        t5_data.extend(torch.load(file))
    print(f"Loaded {len(t5_data)} T5 aligned tokens.")

    sw_data = None
    for sw_path in glob.glob("artifacts/**/switch_token_table_8.pt", recursive=True):
        sw_data = torch.load(sw_path)
        break
    has_fine = "fine_category" in t5_data[0] if t5_data else False
    print(f"Switch: {len(sw_data) if sw_data else 0} tokens, fine={has_fine}")

    seeds = [42, 123, 7]
    output = {}

    for cat_key, label in [("category", "coarse")] + ([("fine_category", "fine")] if has_fine else []):
        print(f"\n{'#'*60}")
        print(f"{label.upper()} | 3-seed robustness")
        print(f"{'#'*60}")

        # Collect results across seeds
        all_runs = {}
        for seed in seeds:
            print(f"\n--- seed={seed} ---")
            run = run_benchmark(t5_data, sw_data, cat_key, seed)
            for key in FEATURE_ORDER:
                if key not in run:
                    continue
                r = run[key]
                print(f"  [{key}] Acc={r['accuracy']:.4f} F1={r['macro_f1']:.4f} ARI={r['ari']:.4f}")
                all_runs.setdefault(key, []).append(r)

        # Aggregate mean ± std
        print(f"\n{'='*80}")
        print(f"SUMMARY | {label} | mean +/- std over {len(seeds)} seeds")
        print(f"{'='*80}")
        print(f"{'Feature':<30} {'Dim':>4} {'Accuracy':>16} {'Macro-F1':>16} {'ARI':>16}")
        print("-" * 90)

        agg = {}
        for key in FEATURE_ORDER:
            if key not in all_runs:
                continue
            runs = all_runs[key]
            accs = [r["accuracy"] for r in runs]
            f1s = [r["macro_f1"] for r in runs]
            aris = [r["ari"] for r in runs]
            dim = DIM_MAP.get(key, "?")
            print(f"{key:<30} {dim:>4} "
                  f"{np.mean(accs):.4f}+/-{np.std(accs):.4f} "
                  f"{np.mean(f1s):.4f}+/-{np.std(f1s):.4f} "
                  f"{np.mean(aris):.4f}+/-{np.std(aris):.4f}")
            agg[key] = {
                "dim": dim,
                "accuracy_mean": np.mean(accs), "accuracy_std": np.std(accs),
                "macro_f1_mean": np.mean(f1s), "macro_f1_std": np.std(f1s),
                "ari_mean": np.mean(aris), "ari_std": np.std(aris),
            }
        output[label] = agg

        # Per-class F1 for seed=42 (primary seed)
        print(f"\n--- Per-class F1 (seed=42) | {label} ---")
        primary = run_benchmark(t5_data, sw_data, cat_key, seed=42)
        test_cats = primary["_test_cats"]
        for key in FEATURE_ORDER:
            if key not in primary or "preds" not in primary[key]:
                continue
            preds = primary[key]["preds"]
            print(f"\n  [{key}]")
            report = classification_report(test_cats, preds, zero_division=0)
            for line in report.strip().split("\n"):
                print(f"    {line}")

    output["metadata"] = {
        "protocol": "caption-level 80/20 split, 3 seeds",
        "seeds": seeds,
        "pca_fit": "train only", "kmeans_fit": "train only",
        "mapping_fit": "train only", "evaluation": "test only",
        "n_clusters": 8,
    }

    with open("artifacts/eval_metrics.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nSaved metrics to artifacts/eval_metrics.json")

if __name__ == "__main__":
    main()
