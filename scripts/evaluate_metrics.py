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
    """Map each expert/cluster to its dominant category, then compute accuracy + macro-F1."""
    expert_cats = defaultdict(list)
    for cat, assign in zip(categories, assignments):
        expert_cats[assign].append(cat)

    # build majority mapping
    mapping = {}
    for eid in range(n_experts):
        tokens = expert_cats.get(eid, [])
        if tokens:
            mapping[eid] = Counter(tokens).most_common(1)[0][0]
        else:
            mapping[eid] = None

    # predict using mapping
    preds = [mapping.get(a) for a in assignments]
    correct = sum(1 for p, c in zip(preds, categories) if p == c)
    accuracy = correct / len(categories) if categories else 0.0
    macro_f1 = f1_score(categories, preds, average="macro", zero_division=0)

    return accuracy, macro_f1, mapping

def compute_baselines(categories):
    """Random and majority baselines."""
    counts = Counter(categories)
    total = len(categories)
    # majority baseline: always predict the most common class
    majority_class = counts.most_common(1)[0][0]
    majority_acc = counts[majority_class] / total
    # majority F1
    majority_preds = [majority_class] * total
    majority_f1 = f1_score(categories, majority_preds, average="macro", zero_division=0)
    # random baseline: expected accuracy = sum(p_i^2) for uniform expert assignment
    # with majority mapping, random assignment gives ~largest class fraction
    n_classes = len(counts)
    random_acc = 1.0 / n_classes  # expected with uniform random
    return {
        "majority_accuracy": majority_acc,
        "majority_f1": majority_f1,
        "random_accuracy": random_acc,
        "majority_class": majority_class,
        "class_distribution": {k: v / total for k, v in counts.items()}
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
    print(f"  ARI: {ari:.4f} | V-Meas: {vmeasure:.4f} | Entropy: {avg_entropy:.4f} | Purity: {avg_purity:.4f} | Load CV: {load_cv:.4f}")
    print(f"  Mapping: {mapping}")

    return {
        "accuracy": accuracy, "macro_f1": macro_f1,
        "ari": ari, "v_measure": vmeasure,
        "avg_entropy": avg_entropy, "load_cv": load_cv,
        "avg_purity": avg_purity,
        "expert_counts": counts,
        "majority_mapping": {str(k): v for k, v in mapping.items()}
    }

def run_eval(t5_data, t5_vectors, switch_data_map, cat_key, label_suffix):
    t5_categories = [row[cat_key] for row in t5_data]
    results = {}

    # baselines
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
        entry = {"t5_baseline": eval_model(f"T5 K-Means (k={n})", t5_categories, t5_clusters.tolist(), n)}

        sw_data = switch_data_map.get(n)
        if sw_data:
            sw_ids = [row["expert_id"] for row in sw_data]
            sw_cats = [row.get(cat_key, row["category"]) for row in sw_data]
            entry["switch_model"] = eval_model(f"Switch MoE (N={n})", sw_cats, sw_ids, n)

        results[f"n_{n}"] = entry

    return results

def main():
    all_files = glob.glob("artifacts/**/aligned_token_table_part*.pt", recursive=True)
    t5_data = []
    for file in all_files:
        t5_data.extend(torch.load(file))
    t5_vectors = torch.stack([row["vector"] for row in t5_data]).numpy().astype("float32")

    print(f"Loaded {len(t5_data)} T5 aligned tokens.")

    has_fine = "fine_category" in t5_data[0] if t5_data else False
    print(f"Fine-grained categories available: {has_fine}")

    switch_data_map = {}
    for n in [4, 8, 16]:
        for sw_path in glob.glob(f"artifacts/**/switch_token_table_{n}.pt", recursive=True):
            switch_data_map[n] = torch.load(sw_path)
            break

    print("\n" + "#" * 60)
    print("COARSE CATEGORIES (5 classes)")
    print("#" * 60)
    coarse_results = run_eval(t5_data, t5_vectors, switch_data_map, "category", "coarse")
    output = {"coarse": coarse_results}

    if has_fine:
        print("\n" + "#" * 60)
        print("FINE-GRAINED CATEGORIES (9 classes)")
        print("#" * 60)
        fine_results = run_eval(t5_data, t5_vectors, switch_data_map, "fine_category", "fine")
        output["fine"] = fine_results

    with open("artifacts/eval_metrics.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nSaved metrics to artifacts/eval_metrics.json")

if __name__ == "__main__":
    main()
