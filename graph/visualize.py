"""Generate all three SkinGraph visualizations into output/."""
from __future__ import annotations
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns

OUTPUT_DIR = pathlib.Path("output")
DPI = 150

FST_LABELS = ["FST I", "FST II", "FST III", "FST IV", "FST V", "FST VI"]
FST_COLS = ["fst_i_pct", "fst_ii_pct", "fst_iii_pct", "fst_iv_pct", "fst_v_pct", "fst_vi_pct"]

NODE_COLORS = {
    "dataset": "#7F77DD",
    "condition": "#D85A30",
    "model": "#378ADD",
    "skin_tone": "#1D9E75",
}


def coverage_heatmap(datasets: pd.DataFrame) -> None:
    """Save FST coverage heatmap to output/coverage_heatmap.png."""
    heat = (
        datasets.set_index("dataset_name")[FST_COLS]
        .rename(columns=dict(zip(FST_COLS, FST_LABELS)))
        * 100
    )
    annot = heat.map(lambda x: f"{x:.1f}%")

    fig, ax = plt.subplots(figsize=(11, 5))
    sns.heatmap(
        heat, ax=ax, cmap="RdYlGn", vmin=0, vmax=50,
        annot=annot, fmt="", linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Coverage (%)"},
    )
    ax.set_title("FST representation across dermatology datasets", fontsize=14, pad=14)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelrotation=0, labelsize=9)

    # Red dashed border around the FST V and FST VI columns
    ax.add_patch(mpatches.Rectangle(
        (4, 0), 2, len(heat),
        fill=False, edgecolor="red", lw=2.5, linestyle="--",
        transform=ax.transData, clip_on=False,
    ))

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "coverage_heatmap.png", dpi=DPI)
    plt.close(fig)
    print("Saved coverage_heatmap.png")


def _column_layout(G: nx.DiGraph) -> dict:
    """Position nodes in four vertical columns by type: model | dataset | condition | skin_tone."""
    col_order = ["model", "dataset", "condition", "skin_tone"]
    groups: dict[str, list] = {t: [] for t in col_order}
    for n, d in G.nodes(data=True):
        t = d.get("type", "")
        if t in groups:
            groups[t].append(n)
    pos = {}
    x_positions = {t: i * 1.0 for i, t in enumerate(col_order)}
    for node_type, x in x_positions.items():
        nodes = sorted(groups[node_type])
        n = len(nodes)
        for i, node in enumerate(nodes):
            pos[node] = (x, (i - (n - 1) / 2) * 0.32)
    return pos


def graph_viz(G: nx.DiGraph) -> None:
    """Save column-layout knowledge graph to output/graph_viz.png."""
    pos = _column_layout(G)
    degrees = dict(G.degree())
    node_sizes = [400 + degrees[n] * 180 for n in G.nodes()]
    node_colors = [NODE_COLORS.get(G.nodes[n].get("type", ""), "#aaaaaa") for n in G.nodes()]

    fig, ax = plt.subplots(figsize=(16, 9))
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                           node_color=node_colors, alpha=0.92)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.35, arrows=True,
                           arrowsize=10, edge_color="#888888",
                           connectionstyle="arc3,rad=0.15")
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8,
                            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "alpha": 0.75})

    # Column header annotations
    for label, x in zip(["Models", "Datasets", "Conditions", "Skin Tones"],
                         [0.0, 1.0, 2.0, 3.0]):
        ax.text(x, 1.08, label, ha="center", va="bottom", fontsize=10,
                fontweight="bold", transform=ax.get_xaxis_transform())

    legend_handles = [
        mpatches.Patch(color=color, label=label.replace("_", " ").title())
        for label, color in NODE_COLORS.items()
    ]
    ax.legend(handles=legend_handles, loc="lower left", fontsize=10, framealpha=0.9)
    ax.set_title("SkinGraph knowledge graph", fontsize=14, pad=12)
    ax.axis("off")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "graph_viz.png", dpi=DPI)
    plt.close(fig)
    print("Saved graph_viz.png")


def model_risk_chart(G: nx.DiGraph, datasets: pd.DataFrame) -> None:
    """Save weighted FST V–VI model risk chart to output/model_risk.png."""
    ds = datasets.set_index("dataset_name")
    rows = []
    for node, data in G.nodes(data=True):
        if data.get("type") != "model":
            continue
        trained_on = [
            v for _, v, d in G.out_edges(node, data=True)
            if d.get("relationship") == "TRAINED_ON" and v in ds.index
        ]
        if not trained_on:
            continue
        total_w = sum(ds.loc[d, "total_images"] for d in trained_on)
        wtd_pct = sum(
            (ds.loc[d, "fst_v_pct"] + ds.loc[d, "fst_vi_pct"]) * ds.loc[d, "total_images"]
            for d in trained_on
        ) / total_w
        rows.append({"model": node, "fst_v_vi_pct": wtd_pct})

    df = pd.DataFrame(rows).sort_values("fst_v_vi_pct").reset_index(drop=True)
    bar_colors = ["#d62728" if v < 0.10 else "#2ca02c" for v in df["fst_v_vi_pct"]]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(df["model"], df["fst_v_vi_pct"] * 100, color=bar_colors)
    ax.axhline(10, color="red", linestyle="--", linewidth=1.5, label="10% threshold")

    for bar, (_, row) in zip(bars, df.iterrows()):
        if row["fst_v_vi_pct"] < 0.10:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.15,
                f"{row['fst_v_vi_pct'] * 100:.1f}%",
                ha="center", va="bottom", color="red", fontsize=9, fontweight="bold",
            )

    ax.set_title("Model training data: FST V–VI representation", fontsize=14, pad=12)
    ax.set_ylabel("Weighted FST V–VI coverage (%)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    ax.legend(fontsize=10)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "model_risk.png", dpi=DPI)
    plt.close(fig)
    print("Saved model_risk.png")


if __name__ == "__main__":
    try:
        from graph.build_graph import build
    except ImportError:
        from build_graph import build

    G = build()
    datasets = pd.read_csv("data/datasets.csv")

    coverage_heatmap(datasets)
    graph_viz(G)
    model_risk_chart(G, datasets)
