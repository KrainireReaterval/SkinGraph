"""Analysis functions for querying representation gaps in SkinGraph."""
from __future__ import annotations

import pandas as pd
import networkx as nx


def _load_graph() -> nx.DiGraph:
    try:
        from graph.build_graph import build
    except ImportError:
        from build_graph import build  # when run directly from graph/
    return build()


def query_gap(G: nx.DiGraph | None = None, fst: str = "FST_VI",
              threshold: float = 0.10) -> pd.DataFrame:
    """Return datasets where coverage for the given FST node is below threshold.

    Parameters
    ----------
    G:         SkinGraph DiGraph; loaded automatically if None.
    fst:       FST node ID to check, e.g. 'FST_V' or 'FST_VI'.
    threshold: Coverage fraction below which a dataset is flagged (default 0.10).

    Returns
    -------
    DataFrame with columns: dataset, fst, pct, total_images, year.
    """
    if G is None:
        G = _load_graph()
    rows = []
    for u, v, data in G.edges(data=True):
        if data.get("relationship") == "CONTAINS" and v == fst and data["pct"] < threshold:
            node = G.nodes[u]
            rows.append({
                "dataset": u,
                "fst": fst,
                "pct": data["pct"],
                "total_images": node.get("total_images"),
                "year": node.get("year"),
            })
    return pd.DataFrame(rows).sort_values("pct").reset_index(drop=True)


def models_with_gap(G: nx.DiGraph | None = None, fst: str = "FST_VI",
                    threshold: float = 0.10) -> list[str]:
    """Return model names trained on datasets underrepresented for the given FST.

    Parameters
    ----------
    G:         SkinGraph DiGraph; loaded automatically if None.
    fst:       FST node ID to check, e.g. 'FST_VI'.
    threshold: Coverage threshold passed through to query_gap.

    Returns
    -------
    Sorted list of model name strings.
    """
    if G is None:
        G = _load_graph()
    gap_datasets = set(query_gap(G, fst, threshold)["dataset"])
    models: list[str] = []
    for u, v, data in G.edges(data=True):
        if data.get("relationship") == "TRAINED_ON" and v in gap_datasets and u not in models:
            models.append(u)
    return sorted(models)


def condition_risk_summary(G: nx.DiGraph | None = None) -> pd.DataFrame:
    """Return conditions sorted by miss_rate_fst_iv_vi descending.

    Parameters
    ----------
    G: SkinGraph DiGraph; loaded automatically if None.

    Returns
    -------
    DataFrame with columns: condition, body_system, miss_rate_fst_iv_vi.
    """
    if G is None:
        G = _load_graph()
    rows = [
        {
            "condition": node,
            "body_system": data.get("body_system"),
            "miss_rate_fst_iv_vi": data.get("miss_rate"),
        }
        for node, data in G.nodes(data=True)
        if data.get("type") == "condition"
    ]
    return (
        pd.DataFrame(rows)
        .sort_values("miss_rate_fst_iv_vi", ascending=False)
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    try:
        from graph.build_graph import build
    except ImportError:
        from build_graph import build

    G = build()

    print("=== Datasets with <10% FST_VI coverage ===")
    print(query_gap(G, "FST_VI").to_string(index=False))

    print("\n=== Models trained on those datasets ===")
    print(models_with_gap(G, "FST_VI"))

    print("\n=== Condition risk summary ===")
    print(condition_risk_summary(G).to_string(index=False))
