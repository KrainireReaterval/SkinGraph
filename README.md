# SkinGraph

A Python knowledge graph that surfaces representation gaps in dermatological AI datasets,
focusing on the underrepresentation of darker skin tones (Fitzpatrick Scale Types IV–VI).

---

## The Problem: FST Representation Gaps

Most dermatology AI models are trained on datasets collected from European and Australian
hospitals, where FST I–III patients dominate. HAM10000 and ISIC Archive — the two most
widely used training sources — each contain only ~1% FST VI images. When AI models
trained on this distribution are deployed globally, they carry systematic blind spots for
darker skin tones, translating directly into higher diagnostic miss rates.

---

## Graph Schema

| Node type | Attributes |
|---|---|
| `dataset` | `total_images`, `year` |
| `skin_tone` | FST_I through FST_VI |
| `condition` | `body_system`, `miss_rate` |
| `model` | `architecture`, `task` |

| Edge type | Meaning |
|---|---|
| `CONTAINS` | Dataset → SkinTone, with `pct` (coverage fraction) |
| `TRAINED_ON` | Model → Dataset |
| `DETECTS` | Model → Condition |
| `HIGH_MISS_RATE_IN` | Condition → SkinTone, with `rate` |

---

## Key Findings

- **All 6 AI models surveyed fall below the 10% FST V–VI threshold** in weighted training
  data coverage (range: 3–8%). The best-performing model (Groh-2021) trained on
  Fitzpatrick17k still reaches only 8%.
- **Melanoma carries the highest documented miss rate for FST IV–VI patients: 35%**
  (Adamson & Smith, NEJM 2018) — the worst outcome for the condition with the highest
  mortality stakes.
- **DDI is the only dataset with > 10% FST VI coverage (11%)**, but at 656 images it is
  outweighed by HAM10000 (10,015) and ISIC Archive (25,331) in every multi-dataset
  training run.

---

## Data Sources

- Tschandl et al. 2018 — HAM10000. Harvard Dataverse. `doi:10.7910/DVN/DBW86T`
- ISIC Archive 2016. `isic-archive.com`
- Groh et al. 2021 — Fitzpatrick17k. `arXiv:2104.09957`
- Daneshjou et al. 2022 — DDI. *Science Advances*. `doi:10.1126/sciadv.abq6147`
- Mendonça et al. 2013 — PH2. ISBI 2013.
- Sun et al. 2016 — SD-198. ECCV 2016.
- Adamson & Smith 2018 — *New England Journal of Medicine*. `doi:10.1056/NEJMp1714529`

---

## How to Run

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

python graph/build_graph.py    # build graph + save .graphml
python graph/analysis.py       # run gap queries
python graph/visualize.py      # generate output/ charts

jupyter notebook notebooks/skingraph_demo.ipynb
```
