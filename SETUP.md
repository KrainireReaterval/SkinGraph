# SkinGraph — Claude Code execution guide

You are helping set up and build SkinGraph: a Python knowledge graph that surfaces
representation gaps in dermatological AI datasets, specifically the underrepresentation
of darker skin tones (Fitzpatrick Scale Types IV–VI) in training data.

---

## Project overview

**Goal:** Build a NetworkX knowledge graph where nodes are datasets, skin tone types,
dermatological conditions, and AI models — and edges encode typed relationships like
coverage percentages, training lineage, and clinical miss rates.

**End artifact:** A Jupyter notebook demo + three visualizations (coverage heatmap,
graph network plot, model risk chart) pushed to GitHub.

---

## File structure

```
skingraph/
├── CLAUDE.md               ← you are here
├── README.md
├── requirements.txt
├── data/
│   ├── datasets.csv        ← FST distribution per dataset
│   ├── conditions.csv      ← conditions + miss-rate data
│   └── models.csv          ← AI model cards + training lineage
├── graph/
│   ├── build_graph.py      ← builds the NetworkX DiGraph
│   ├── analysis.py         ← gap detector + lineage tracer functions
│   └── skingraph.graphml   ← serialized graph output
├── notebooks/
│   └── skingraph_demo.ipynb
└── output/
    ├── coverage_heatmap.png
    ├── graph_viz.png
    └── model_risk.png
```

---

## Environment setup

**Python version:** 3.10 or higher. Check with `python --version`.

**Create and activate virtual environment:**
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Verify install:**
```bash
python -c "import networkx, pandas, matplotlib, seaborn, jupyter; print('All good')"
```

**Launch notebook:**
```bash
jupyter notebook notebooks/skingraph_demo.ipynb
```

---

## Key commands

| Task | Command |
|------|---------|
| Install deps | `pip install -r requirements.txt` |
| Build graph | `python graph/build_graph.py` |
| Run analysis | `python graph/analysis.py` |
| Launch notebook | `jupyter notebook` |
| Run all checks | `python -c "from graph.build_graph import build; G = build(); print(G)"` |

---

## Phase 1 — Environment + dependencies

### Task: scaffold the project

Create the full directory structure if it doesn't exist:

```bash
mkdir -p data graph notebooks output
touch README.md graph/__init__.py
```

### Task: write requirements.txt

Create `requirements.txt` with exactly these contents:

```
networkx==3.3
pandas==2.2.2
matplotlib==3.9.0
seaborn==0.13.2
notebook==7.2.0
jupyterlab==4.2.0
```

Then run `pip install -r requirements.txt` and confirm no errors.

---

## Phase 2 — Seed data

### Task: create data/datasets.csv

Create this file with real FST distribution data. Columns must be exactly:
`dataset_name, source_url, total_images, fst_i_pct, fst_ii_pct, fst_iii_pct, fst_iv_pct, fst_v_pct, fst_vi_pct, year, notes`

Seed rows to include (verify against cited papers):
- HAM10000 (Tschandl et al. 2018) — predominantly FST I–III, ~90%+ light skin
- ISIC Archive (isic-archive.com) — similar skew to HAM10000
- Fitzpatrick17k (Groh et al. 2021, arxiv.org/abs/2104.09957) — more diverse, ~20% FST IV–VI
- DDI (Daneshjou et al. 2022, doi.org/10.1126/sciadv.abq6147) — intentionally balanced
- PH2 (Menddonça et al. 2013) — small, predominantly light skin
- SD-198 (Sun et al. 2016) — mixed conditions, limited FST metadata

Use 0.0–1.0 decimal format for percentages (e.g. 0.03 = 3%).
If exact numbers are unavailable from the paper, note in the `notes` column and use a
conservative estimate based on the dataset's described collection context.

### Task: create data/conditions.csv

Columns: `condition, body_system, miss_rate_fst_iv_vi, source_paper`

Include these conditions at minimum:
- melanoma (miss_rate_fst_iv_vi ≈ 0.35, cite Adamson & Smith 2018 NEJM)
- eczema / atopic dermatitis
- psoriasis
- acne vulgaris
- seborrheic dermatitis
- tinea versicolor (often misdiagnosed on dark skin)

`miss_rate_fst_iv_vi` = documented or estimated diagnostic miss rate for FST IV–VI
relative to FST I–II. Use 0.0–1.0 scale.

### Task: create data/models.csv

Columns: `model_name, architecture, trained_on, task, paper_url`

`trained_on` is a comma-separated list of dataset_name values matching datasets.csv.
Include 4–6 published dermatology AI models. Examples:
- Esteva et al. 2017 (Nature) — CNN for skin cancer classification, trained on ISIC
- Han et al. 2018 — ResNet for melanoma, HAM10000-derived
- Any models from the DDI paper's benchmark section

---

## Phase 3 — Build the graph

### Task: write graph/build_graph.py

This script must:
1. Load all three CSVs from the `data/` directory using pandas
2. Create a `networkx.DiGraph()` called `G`
3. Add nodes with attributes:
   - Dataset nodes: `type="dataset"`, `total_images`, `year`
   - SkinTone nodes: `type="skin_tone"`, node IDs = "FST_I" through "FST_VI"
   - Condition nodes: `type="condition"`, `body_system`, `miss_rate`
   - Model nodes: `type="model"`, `architecture`, `task`
4. Add typed edges:
   - `(dataset) --[CONTAINS]--> (skin_tone)` with attribute `pct=float`
   - `(model) --[TRAINED_ON]--> (dataset)` (one edge per dataset in trained_on)
   - `(model) --[DETECTS]--> (condition)` (based on model's stated task)
   - `(condition) --[HIGH_MISS_RATE_IN]--> (skin_tone)` with attribute `rate=float`
5. Save graph: `nx.write_graphml(G, "graph/skingraph.graphml")`
6. Print summary: number of nodes by type, number of edges by relationship type
7. Include a `def build() -> nx.DiGraph:` function so other modules can import it

### Task: write graph/analysis.py

Must implement these three functions:

```python
def query_gap(G, fst: str, threshold: float = 0.10) -> pd.DataFrame:
    """Return datasets where FST coverage is below threshold."""

def models_with_gap(G, fst: str, threshold: float = 0.10) -> list[str]:
    """Return model names trained on datasets underrepresented for given FST."""

def condition_risk_summary(G) -> pd.DataFrame:
    """Return DataFrame of conditions sorted by miss_rate_fst_iv_vi descending."""
```

Each function must import `G` from `build_graph.build()` if no graph is passed in.
All functions return pandas DataFrames or lists — no print statements inside functions.

---

## Phase 4 — Visualizations

All charts save to the `output/` directory. Use matplotlib + seaborn. DPI = 150.

### Task: coverage heatmap (output/coverage_heatmap.png)

- Rows = datasets, Columns = FST I–VI
- Cell values = coverage percentage (0–100%)
- Colormap: `RdYlGn` (red = low coverage, green = high)
- Annotate every cell with the percentage value
- Add a red dashed line or bold border around FST V and VI columns
- Title: "FST representation across dermatology datasets"
- This is the hero visual — make it publication-quality

### Task: graph network visualization (output/graph_viz.png)

- Use `nx.spring_layout` with `seed=42` for reproducibility
- Node colors by type: dataset=purple (#7F77DD), condition=coral (#D85A30),
  model=blue (#378ADD), skin_tone=teal (#1D9E75)
- Node size proportional to degree
- Edge alpha = 0.4
- Include a legend for node types
- Title: "SkinGraph knowledge graph"

### Task: model risk chart (output/model_risk.png)

- Bar chart: X = model names, Y = weighted avg FST V–VI coverage across training datasets
- Weighted by dataset size (total_images)
- Draw a horizontal red dashed line at y=0.10 (10% threshold)
- Label bars below the threshold in red
- Title: "Model training data: FST V–VI representation"
- X-axis labels rotated 30 degrees if needed

---

## Phase 5 — Notebook and README

### Task: write notebooks/skingraph_demo.ipynb

Structure the notebook with these markdown + code cell sections:
1. **Introduction** — what SkinGraph is, why it matters (2–3 sentences)
2. **Build the graph** — `from graph.build_graph import build; G = build()`
3. **Query 1:** Which datasets have <5% FST VI coverage?
4. **Query 2:** Which models were trained on those datasets?
5. **Query 3:** Which conditions have the highest miss rates for FST IV–VI?
6. **Visualizations** — display all three output images inline

Each code section should be preceded by a markdown cell explaining what it does.

### Task: write README.md

Sections (keep total under 400 words):
1. What is SkinGraph
2. The problem: FST representation gaps
3. Graph schema (brief — entities and relationships)
4. Key findings (pull top 2–3 stats from the data)
5. Data sources and citations (properly formatted)
6. How to run (point to the environment setup above)

---

## Verification checklist

Before considering setup complete, confirm:

- [ ] `pip install -r requirements.txt` completes without errors
- [ ] `python graph/build_graph.py` prints node/edge summary without errors
- [ ] `python graph/analysis.py` runs without errors
- [ ] `graph/skingraph.graphml` exists and is non-empty
- [ ] All three PNG files exist in `output/`
- [ ] Notebook runs top-to-bottom without errors (`Kernel > Restart & Run All`)

---

## Constraints and preferences

- Python only — no JavaScript, no external databases for the MVP
- All file paths are relative to the project root
- Do not use absolute paths
- Data CSV files are the source of truth — do not hardcode values in Python scripts
- All functions must have docstrings
- Keep scripts under 150 lines each; split into helpers if needed
- Do not install packages not listed in requirements.txt without asking first
