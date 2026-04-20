import json
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, v_measure_score


def main():
    # load T5 aligned token table
    t5_data = torch.load("artifacts/aligned_token_table.pt")
    t5_vectors = torch.stack([row["vector"] for row in t5_data])
    t5_categories = [row["category"] for row in t5_data]
    
    # load Switch token table
    switch_data = torch.load("artifacts/switch_token_table.pt")
    switch_expert_ids = [row["expert_id"] for row in switch_data]
    switch_categories = [row["category"] for row in switch_data]
    
    # run KMeans on T5 vectors
    kmeans = KMeans(n_clusters=8, random_state=42)
    t5_clusters = kmeans.fit_predict(t5_vectors.numpy())
    
    # compute metrics for T5 baseline
    t5_ari = adjusted_rand_score(t5_categories, t5_clusters)
    t5_vmeasure = v_measure_score(t5_categories, t5_clusters)
    
    # compute metrics for Switch model
    switch_ari = adjusted_rand_score(switch_categories, switch_expert_ids)
    switch_vmeasure = v_measure_score(switch_categories, switch_expert_ids)
    
    # print results in formatted table
    print("\n" + "=" * 60)
    print("EVALUATION METRICS: Semantic Category Alignment")
    print("=" * 60)
    print(f"{'Model':<20} {'ARI':<15} {'V-Measure':<15}")
    print("-" * 60)
    print(f"{'T5 Baseline':<20} {t5_ari:<15.4f} {t5_vmeasure:<15.4f}")
    print(f"{'Switch-base-8':<20} {switch_ari:<15.4f} {switch_vmeasure:<15.4f}")
    print("=" * 60)
    print(f"\nT5 samples: {len(t5_data)}")
    print(f"Switch samples: {len(switch_data)}")
    print()

    results = {
        "t5_baseline": {"ari": t5_ari, "v_measure": t5_vmeasure, "n_samples": len(t5_data)},
        "switch_base_8": {"ari": switch_ari, "v_measure": switch_vmeasure, "n_samples": len(switch_data)},
    }
    with open("artifacts/eval_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved metrics to artifacts/eval_metrics.json")


if __name__ == "__main__":
    main()
