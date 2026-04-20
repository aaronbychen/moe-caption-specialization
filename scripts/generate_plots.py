import os
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.cluster import KMeans

FONT = {"title": 18, "label": 14, "tick": 12}


def make_heatmap(ct_norm, cmap, title, save_path):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(ct_norm, annot=True, fmt=".2f", cmap=cmap, ax=ax,
                annot_kws={"size": FONT["tick"]})
    ax.set_title(title, fontsize=FONT["title"], pad=14)
    ax.set_xlabel("Semantic Category", fontsize=FONT["label"])
    ax.set_ylabel(ax.get_ylabel(), fontsize=FONT["label"])
    ax.tick_params(axis="both", labelsize=FONT["tick"])
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def print_top_words(df, id_col, label):
    print(f"\n{'='*60}")
    print(f"Top 10 words per {label}")
    print("=" * 60)
    for gid in sorted(df[id_col].unique()):
        words = df.loc[df[id_col] == gid, "word"].str.lower()
        top = [w for w, _ in Counter(words).most_common(10)]
        print(f"  {label} {gid}: {', '.join(top)}")


def main():
    os.makedirs("artifacts", exist_ok=True)

    # --- T5 ---
    t5_data = torch.load("artifacts/aligned_token_table.pt")
    t5_df = pd.DataFrame([{"word": r["word"], "category": r["category"],
                            "vector": r["vector"].numpy()} for r in t5_data])
    vectors = torch.stack([r["vector"] for r in t5_data]).numpy()
    t5_df["cluster"] = KMeans(n_clusters=8, random_state=42).fit_predict(vectors)

    ct_t5 = pd.crosstab(t5_df["cluster"], t5_df["category"])
    ct_t5_norm = ct_t5.div(ct_t5.sum(axis=0), axis=1)  # column-normalize
    ct_t5_norm.index.name = "Cluster ID"

    make_heatmap(ct_t5_norm, "Blues",
                 "T5 Baseline: Latent Clusters vs. Semantic Categories",
                 "artifacts/t5_heatmap.png")
    print_top_words(t5_df, "cluster", "T5 Cluster")

    # --- Switch ---
    sw_data = torch.load("artifacts/switch_token_table.pt")
    sw_df = pd.DataFrame([{"word": r["word"], "category": r["category"],
                            "expert_id": r["expert_id"]} for r in sw_data])

    ct_sw = pd.crosstab(sw_df["expert_id"], sw_df["category"])
    ct_sw_norm = ct_sw.div(ct_sw.sum(axis=0), axis=1)
    ct_sw_norm.index.name = "Expert ID"

    make_heatmap(ct_sw_norm, "Oranges",
                 "Switch MoE: Routing Experts vs. Semantic Categories",
                 "artifacts/switch_heatmap.png")
    print_top_words(sw_df, "expert_id", "Switch Expert")


if __name__ == "__main__":
    main()
