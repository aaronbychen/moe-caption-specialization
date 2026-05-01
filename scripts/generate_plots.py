import os
import torch
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.cluster import KMeans

FONT = {"title": 18, "label": 14, "tick": 12}


def make_heatmap(ct_norm, cmap, title, save_path, n_experts):
    width = max(12, n_experts * 0.4)
    fig, ax = plt.subplots(figsize=(width, 6))
    sns.heatmap(ct_norm, annot=(n_experts <= 16), fmt=".2f", cmap=cmap, ax=ax,
                annot_kws={"size": FONT["tick"]})
    ax.set_title(title, fontsize=FONT["title"], pad=14)
    ax.set_xlabel("Semantic Category", fontsize=FONT["label"])
    ax.set_ylabel(ax.get_ylabel(), fontsize=FONT["label"])
    ax.tick_params(axis="both", labelsize=FONT["tick"])
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def print_top_words(df, id_col, label, max_ids=8):
    print(f"\n{'='*60}")
    print(f"Top 10 words per {label}")
    print("=" * 60)
    for gid in sorted(df[id_col].unique())[:max_ids]:
        words = df.loc[df[id_col] == gid, "word"].str.lower()
        top = [w for w, _ in Counter(words).most_common(10)]
        print(f"  {label} {gid}: {', '.join(top)}")


def main():
    os.makedirs("artifacts", exist_ok=True)

    # --- T5 ---
    all_files = glob.glob("artifacts/aligned_token_table_part*.pt")
    t5_data = []
    for file in all_files:
        t5_data.extend(torch.load(file))
    vectors = torch.stack([r["vector"] for r in t5_data]).numpy()

    for n in [8]:
        print(f"\n----Generating Plots for N={n}----")

        t5_df = pd.DataFrame([{"word": r["word"], "category": r["category"],
                               "vector": r["vector"].numpy()} for r in t5_data])
        t5_df["cluster"] = KMeans(n_clusters=n, random_state=42).fit_predict(vectors)
            
        ct_t5 = pd.crosstab(t5_df["cluster"], t5_df["category"])
        ct_t5_norm = ct_t5.div(ct_t5.sum(axis=0), axis=1)  # column-normalize
        ct_t5_norm.index.name = "Cluster ID"

        make_heatmap(ct_t5_norm, "Blues",
                    f"T5 Baseline (k={n}): Latent Clusters vs. Semantic Categories",
                    f"artifacts/t5_heatmap_{n}.png", n)
        print_top_words(t5_df, "cluster", "T5 Cluster")

        # --- Switch ---
        try:
            sw_data = torch.load(f"artifacts/switch_token_table_{n}.pt")
        except FileNotFoundError:
            print(f"Switch data for n={n} not found. Skipping plot.")
            continue
        sw_df = pd.DataFrame([{"word": r["word"], "category": r["category"],
                                "expert_id": r["expert_id"]} for r in sw_data])

        ct_sw = pd.crosstab(sw_df["expert_id"], sw_df["category"])
        ct_sw_norm = ct_sw.div(ct_sw.sum(axis=0), axis=1)
        ct_sw_norm.index.name = "Expert ID"

        make_heatmap(ct_sw_norm, "Oranges",
                    f"Switch MoE (N={n}): Routing Experts vs. Semantic Categories",
                    f"artifacts/switch_heatmap_{n}.png", n)
        print_top_words(sw_df, "expert_id", "Switch Expert")


if __name__ == "__main__":
    main()
