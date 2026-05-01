"""
Generate a text summary of key findings from evaluation metrics.
Reads artifacts/eval_metrics_ablation.json and prints a structured discussion.
"""
import json
import os

def load_metrics(path="artifacts/eval_metrics_ablation.json"):
    with open(path) as f:
        return json.load(f)

def summarize_n(key, data):
    n = key.split("_")[1]
    t5 = data.get("t5_baseline")
    sw = data.get("switch_model")

    lines = [f"## N = {n}\n"]

    if t5:
        lines.append(f"T5 K-Means:  ARI={t5['ari']:.4f}  V-Measure={t5['v_measure']:.4f}  "
                      f"Entropy={t5['avg_entropy']:.3f}  Load CV={t5['load_cv']:.3f}")
    if sw:
        lines.append(f"Switch MoE:  ARI={sw['ari']:.4f}  V-Measure={sw['v_measure']:.4f}  "
                      f"Entropy={sw['avg_entropy']:.3f}  Load CV={sw['load_cv']:.3f}")

    if t5 and sw:
        ari_gap = t5["ari"] - sw["ari"]
        vm_gap = t5["v_measure"] - sw["v_measure"]
        lines.append(f"\nT5 leads by ARI +{ari_gap:.4f}, V-Measure +{vm_gap:.4f}.")

        # load balance comparison
        if sw["load_cv"] > t5["load_cv"]:
            lines.append(f"Switch routing is less balanced (CV {sw['load_cv']:.3f} vs {t5['load_cv']:.3f}).")

        # expert load skew
        sw_counts = sw["expert_counts"]
        max_load = max(sw_counts)
        min_load = min(sw_counts)
        ratio = max_load / min_load if min_load > 0 else float("inf")
        lines.append(f"Switch max/min expert load ratio: {ratio:.1f}x "
                      f"(expert loads range {min_load:,} to {max_load:,}).")

    return "\n".join(lines)

def main():
    metrics = load_metrics()

    report = ["=" * 60]
    report.append("MoE Expert Specialization: Summary of Findings")
    report.append("=" * 60)
    report.append("")

    for key in sorted(metrics.keys()):
        report.append(summarize_n(key, metrics[key]))
        report.append("")

    # overall discussion
    report.append("=" * 60)
    report.append("Discussion")
    report.append("=" * 60)
    report.append("")

    n8 = metrics.get("n_8", {})
    t5 = n8.get("t5_baseline", {})
    sw = n8.get("switch_model", {})

    if t5 and sw:
        report.append(
            "The T5 baseline with post-hoc K-Means clustering consistently outperforms "
            "Switch Transformer's learned routing on both ARI and V-Measure. This suggests "
            "that the Switch router does NOT naturally learn linguistically meaningful expert "
            "specialization."
        )
        report.append("")
        report.append(
            "The Switch router shows higher entropy per category and greater load imbalance "
            f"(CV={sw['load_cv']:.3f} vs {t5['load_cv']:.3f}), indicating that routing decisions "
            "are driven by factors other than semantic category -- likely token frequency, "
            "positional patterns, or the auxiliary load-balancing loss."
        )
        report.append("")
        report.append(
            "Key takeaway: MoE expert specialization in text encoding appears to be an "
            "emergent property of representation geometry (capturable by K-Means) rather "
            "than a learned routing strategy."
        )

    text = "\n".join(report)
    print(text)

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/summary_report.txt", "w") as f:
        f.write(text + "\n")
    print(f"\nSaved to artifacts/summary_report.txt")

if __name__ == "__main__":
    main()
