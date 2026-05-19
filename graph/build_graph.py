"""Build the SkinGraph NetworkX DiGraph from seed CSV data."""
from __future__ import annotations
import pathlib
from collections import Counter

import networkx as nx
import pandas as pd

DATA_DIR = pathlib.Path("data")
GRAPH_OUT = pathlib.Path("graph") / "skingraph.graphml"

FST_COLS = {
    "fst_i_pct": "FST_I",
    "fst_ii_pct": "FST_II",
    "fst_iii_pct": "FST_III",
    "fst_iv_pct": "FST_IV",
    "fst_v_pct": "FST_V",
    "fst_vi_pct": "FST_VI",
}

# FST nodes with documented elevated miss rates for conditions
HIGH_MISS_FST = ["FST_IV", "FST_V", "FST_VI"]

# Tasks that imply detection of all loaded conditions vs. melanoma only
BROAD_TASKS = {"skin_condition_classification"}
MELANOMA_TASKS = {"melanoma_detection", "melanoma_vs_nevus", "skin_lesion_classification"}


def build() -> nx.DiGraph:
    """Load data CSVs and construct the SkinGraph DiGraph."""
    G = nx.DiGraph()

    datasets = pd.read_csv(DATA_DIR / "datasets.csv")
    conditions = pd.read_csv(DATA_DIR / "conditions.csv")
    models = pd.read_csv(DATA_DIR / "models.csv")

    # Skin tone nodes
    for node_id in FST_COLS.values():
        G.add_node(node_id, type="skin_tone")

    # Dataset nodes + CONTAINS edges to skin tone nodes
    for _, row in datasets.iterrows():
        G.add_node(
            row["dataset_name"],
            type="dataset",
            total_images=int(row["total_images"]),
            year=int(row["year"]),
        )
        for col, fst_node in FST_COLS.items():
            G.add_edge(
                row["dataset_name"], fst_node,
                relationship="CONTAINS", pct=float(row[col]),
            )

    # Condition nodes + HIGH_MISS_RATE_IN edges to FST IV–VI nodes
    all_conditions = list(conditions["condition"])
    for _, row in conditions.iterrows():
        G.add_node(
            row["condition"],
            type="condition",
            body_system=row["body_system"],
            miss_rate=float(row["miss_rate_fst_iv_vi"]),
        )
        for fst_node in HIGH_MISS_FST:
            G.add_edge(
                row["condition"], fst_node,
                relationship="HIGH_MISS_RATE_IN",
                rate=float(row["miss_rate_fst_iv_vi"]),
            )

    # Model nodes + TRAINED_ON edges + DETECTS edges
    for _, row in models.iterrows():
        G.add_node(
            row["model_name"],
            type="model",
            architecture=row["architecture"],
            task=row["task"],
        )
        for dataset in row["trained_on"].split(","):
            dataset = dataset.strip()
            if G.has_node(dataset):
                G.add_edge(row["model_name"], dataset, relationship="TRAINED_ON")

        task = row["task"]
        if task in BROAD_TASKS:
            detects = all_conditions
        elif task in MELANOMA_TASKS:
            detects = ["melanoma"]
        else:
            detects = []
        for condition in detects:
            if G.has_node(condition):
                G.add_edge(row["model_name"], condition, relationship="DETECTS")

    return G


def _print_summary(G: nx.DiGraph) -> None:
    """Print node and edge counts broken down by type and relationship."""
    type_counts = Counter(d["type"] for _, d in G.nodes(data=True))
    rel_counts = Counter(d.get("relationship", "unknown") for _, _, d in G.edges(data=True))
    print(f"Nodes: {G.number_of_nodes()} total")
    for t, n in sorted(type_counts.items()):
        print(f"  {t}: {n}")
    print(f"Edges: {G.number_of_edges()} total")
    for r, n in sorted(rel_counts.items()):
        print(f"  {r}: {n}")


if __name__ == "__main__":
    G = build()
    nx.write_graphml(G, GRAPH_OUT)
    print(f"Graph saved to {GRAPH_OUT}")
    _print_summary(G)
