import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import glob
import math
import torch
import numpy as np
from collections import Counter, defaultdict
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, v_measure_score, f1_score

def compute_load_metrics(assignments, n_experts):
    counts = Counter(assignments)
    expert_counts = [counts.get(i, 0) for i in range(n_experts)]
    mean_load = np.mean(expert_counts)
    std_load = np.std(expert_counts)
    load_cv = std_load / mean_load if mean_load > 0 else 0.0
    return expert_counts, load_cv

def compute_purity(categories, assignments, n_experts):
    expert_cats = defaultdict(list)
    for cat, assign in zip(categories, assignments):
        expert_cats[assign].append(cat)
    purities = {}
    weighted_sum = 0.0
    total = len(categories)
    for eid in range(n_experts):
        tokens = expert_cats.get(eid, [])
        if not tokens:
            purities[eid] = 0.0
            continue
        dominant_count = Counter(tokens).most_common(1)[0][1]
        purities[eid] = dominant_count / len(tokens)
        weighted_sum += dominant_count
    avg_purity = weighted_sum / total if total > 0 else 0.0
    return purities, avg_purity

def compute_entropy_per_category(categories, assignments):
    cat_to_assign = defaultdict(list)
    for cat, assign in zip(categories, assignments):
        cat_to_assign[cat].append(assign)
    entropies = {}
    for cat, assigns in cat_to_assign.items():
        counts = Counter(assigns)
        total = len(assigns)
        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0: entropy -= p * math.log2(p)
        entropies[cat] = entropy
    avg_entropy = sum(entropies.values()) / len(entropies) if entropies else 0.0
    return entropies, avg_entropy

def majority_mapping_accuracy(categories, assignments, n_experts):
    expert_cats = defaultdict(list)
    for cat, assign in zip(categories, assignments):
        expert_cats[assign].append(cat)
    mapping = {}
    for eid in range(n_experts):
        tokens = expert_cats.get(eid, [])
        mapping[eid] = Counter(tokens).most_common(1)[0][0] if tokens else None
    preds = [mapping.get(a) for a in assignments]
    correct = sum(1 for p, c in zip(preds, categories) if p == c)
    accuracy = correct / len(categories) if categories else 0.0
    macro_f1 = f1_score(categories, preds, average="macro", zero_division=0)
    return accuracy, macro_f1, mapping

def compute_baselines(categories):
    counts = Counter(categories)
    total = len(categories)
    majority_class = counts.most_common(1)[0][0]
    majority_acc = counts[majority_class] / total
    majority_preds = [majority_class] * total
    majority_f1 = f1_score(categories, majority_preds, average="macro", zero_division=0)
    n_classes = len(counts)
    random_acc = 1.0 / n_classes
    return {
        "majority_accuracy": majority_acc, "majority_f1": majority_f1,
        "random_accuracy": random_acc, "majority_class": majority_class,
    }

def eval_model(label, categories, assignments, n):
    ari = adjusted_rand_score(categories, assignments)
    vmeasure = v_measure_score(categories, assignments)
    cat_entropy, avg_entropy = compute_entropy_per_category(categories, assignments)
    counts, load_cv = compute_load_metrics(assignments, n)
    purities, avg_purity = compute_purity(categories, assignments, n)
    accuracy, macro_f1, mapping = majority_mapping_accuracy(categories, assignments, n)

    print(f"\n[{label}]")
    print(f"  Accuracy: {accuracy:.4f} | Macro-F1: {macro_f1:.4f}")
    print(f"  ARI: {ari:.4f} | V-Meas: {vmeasure:.4f} | Entropy: {avg_entropy:.4f} | Purity: {avg_purity:.4f}")

    return {
        "accuracy": accuracy, "macro_f1": macro_f1,
        "ari": ari, "v_measure": vmeasure,
        "avg_entropy": avg_entropy, "load_cv": load_cv,
        "avg_purity": avg_purity, "expert_counts": counts,
    }

def run_eval(t5_data, t5_vectors, switch_data_map, cat_key, label_suffix):
    t5_categories = [row[cat_key] for row in t5_data]
    results = {}

    baselines = compute_baselines(t5_categories)
    print(f"\n  Baselines: majority_acc={baselines['majority_accuracy']:.4f} "
          f"majority_f1={baselines['majority_f1']:.4f} random_acc={baselines['random_accuracy']:.4f}")
    results["baselines"] = baselines

    for n in [4, 8, 16]:
        print(f"\n{'='*60}")
        print(f"N={n} | {label_suffix}")
        print(f"{'='*60}")

        kmeans = KMeans(n_clusters=n, random_state=42, n_init="auto")
        t5_clusters = kmeans.fit_predict(t5_vectors)
        entry = {"t5_768d": eval_model(f"T5 K-Means 768D (k={n})", t5_categories, t5_clusters.tolist(), n)}

        sw_data = switch_data_map.get(n)
        if sw_data:
            sw_ids = [row["expert_id"] for row in sw_data]
            sw_cats = [row.get(cat_key, row["category"]) for row in sw_data]
            entry["switch_hard"] = eval_model(f"Switch hard expert (N={n})", sw_cats, sw_ids, n)

        results[f"n_{n}"] = entry

    return results

def run_fairness_benchmark(t5_data, t5_vectors, switch_data, cat_key, label_suffix):
    """Compare features at equal dimensionality."""
    categories = [row[cat_key] for row in t5_data]
    n = 8  # fixed for fairness comparison
    results = {}

    print(f"\n{'#'*60}")
    print(f"FEATURE-BUDGET FAIRNESS BENCHMARK | {label_suffix}")
    print(f"{'#'*60}")

    baselines = compute_baselines(categories)
    print(f"  Baselines: majority={baselines['majority_accuracy']:.4f} random={baselines['random_accuracy']:.4f}")
    results["baselines"] = baselines

    # --- T5 768D (upper bound) ---
    km_768 = KMeans(n_clusters=n, random_state=42, n_init="auto")
    t5_768_clusters = km_768.fit_predict(t5_vectors)
    results["t5_768d"] = eval_model("T5 768D (upper bound)", categories, t5_768_clusters.tolist(), n)

    # --- T5 PCA-8D ---
    t5_pca8 = PCA(n_components=8, random_state=42).fit_transform(t5_vectors)
    km_pca8 = KMeans(n_clusters=n, random_state=42, n_init="auto")
    t5_pca8_clusters = km_pca8.fit_predict(t5_pca8)
    results["t5_pca_8d"] = eval_model("T5 PCA-8D", categories, t5_pca8_clusters.tolist(), n)

    # --- T5 PCA-32D ---
    t5_pca32 = PCA(n_components=32, random_state=42).fit_transform(t5_vectors)
    km_pca32 = KMeans(n_clusters=n, random_state=42, n_init="auto")
    t5_pca32_clusters = km_pca32.fit_predict(t5_pca32)
    results["t5_pca_32d"] = eval_model("T5 PCA-32D", categories, t5_pca32_clusters.tolist(), n)

    if switch_data:
        sw_cats = [row.get(cat_key, row["category"]) for row in switch_data]

        # --- Switch hard expert ID (8-way) ---
        sw_ids = [row["expert_id"] for row in switch_data]
        results["switch_hard_8d"] = eval_model("Switch hard expert ID", sw_cats, sw_ids, n)

        # --- Switch all-layer routing (48D -> K-Means 8) ---
        has_all_layer = "all_layer_probs" in switch_data[0]
        if has_all_layer:
            sw_all_probs = torch.stack([row["all_layer_probs"] for row in switch_data]).numpy().astype("float32")

            km_sw48 = KMeans(n_clusters=n, random_state=42, n_init="auto")
            sw_48_clusters = km_sw48.fit_predict(sw_all_probs)
            results["switch_all_layer_48d"] = eval_model("Switch all-layer 48D", sw_cats, sw_48_clusters.tolist(), n)

            # --- Switch all-layer PCA-8D (for fair 8D comparison) ---
            sw_pca8 = PCA(n_components=8, random_state=42).fit_transform(sw_all_probs)
            km_sw_pca8 = KMeans(n_clusters=n, random_state=42, n_init="auto")
            sw_pca8_clusters = km_sw_pca8.fit_predict(sw_pca8)
            results["switch_all_layer_pca_8d"] = eval_model("Switch all-layer PCA-8D", sw_cats, sw_pca8_clusters.tolist(), n)

    return results

def main():
    all_files = glob.glob("artifacts/**/aligned_token_table_part*.pt", recursive=True)
    t5_data = []
    for file in all_files:
        t5_data.extend(torch.load(file))
    t5_vectors = torch.stack([row["vector"] for row in t5_data]).numpy().astype("float32")

    print(f"Loaded {len(t5_data)} T5 aligned tokens.")

    has_fine = "fine_category" in t5_data[0] if t5_data else False

    switch_data_map = {}
    for n in [4, 8, 16]:
        for sw_path in glob.glob(f"artifacts/**/switch_token_table_{n}.pt", recursive=True):
            switch_data_map[n] = torch.load(sw_path)
            break

    output = {}

    # standard evaluation
    for cat_key, label, n_classes in [("category", "coarse", 5)] + ([("fine_category", "fine", 9)] if has_fine else []):
        print(f"\n{'#'*60}")
        print(f"{label.upper()} CATEGORIES ({n_classes} classes)")
        print(f"{'#'*60}")
        output[label] = run_eval(t5_data, t5_vectors, switch_data_map, cat_key, label)

    # fairness benchmark (N=8 only, coarse)
    sw_data_8 = switch_data_map.get(8)
    output["fairness_coarse"] = run_fairness_benchmark(t5_data, t5_vectors, sw_data_8, "category", "coarse")
    if has_fine:
        output["fairness_fine"] = run_fairness_benchmark(t5_data, t5_vectors, sw_data_8, "fine_category", "fine")

    with open("artifacts/eval_metrics.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nSaved metrics to artifacts/eval_metrics.json")

if __name__ == "__main__":
    main()
