# Runs

This folder contains six **result-zip archives** — the canonical sub-experiment
CSV summaries that the paper's figures and tables are built from.

| Filename | Dataset | Mode |
|---|---|---|
| `swat_binary.zip` | SWaT | binary |
| `swat_multiclass.zip` | SWaT | multi-class (reduces to binary; see paper §3.4) |
| `hai_binary.zip` | HAI | binary |
| `hai_multiclass.zip` | HAI | multi-class (reduces to binary; see paper §3.4) |
| `wustl_binary.zip` | WUSTL-IIoT-2021 | binary |
| `wustl_multiclass.zip` | WUSTL-IIoT-2021 | multi-class (5 classes) |

## Internal structure

Each archive contains an `ot_ics_ids_llm_nebius_v1_outputs/` tree with
per-sub-experiment subfolders:

```
ot_ics_ids_llm_nebius_v1_outputs/
├── E1_cross_family/        Cross-family head-to-head (all methods × all datasets)
├── E2_kshot/               k-shot scaling sweep
├── E3_format/              Prompt-format ablation
├── E5_cost/                Cost analysis (LLM only, USD per correct)
├── E6_perclass/            Per-class confusion matrices (multi-class)
├── E7_multiseed/           Primary multi-seed evaluation (3 seeds)
├── R1_multiseed/           Cross-seed multi-LLM comparison (5 seeds)
└── R2_perclass/            Per-class multi-seed analysis (5 seeds, WUSTL only)
```

Plus a top-level `manifest_<dataset>.json` describing the run configuration.

## How these were produced

The archives are output by the six notebooks under
`../notebooks/experiments/`. Each notebook produces one archive matching
one (dataset, mode) cell of the experimental matrix.

If you re-run an experiment notebook, you can move the produced archive
into this folder under the matching canonical short name above, and the
figure-regeneration notebook in `../notebooks/figures/` will pick it up.

## How these are consumed

`../notebooks/figures/ot_ics_ids_paper_figures_v2.ipynb` unzips each
archive, parses every CSV, and writes the paper figures and tables into
`../paper_artifacts/`.

## Supplementary archives (`supplementary/`)

The `supplementary/` subfolder holds the **supplementary** result
archives. These are *not* part of the canonical six and are not read by
`ot_ics_ids_paper_figures_v2.ipynb` (which keys off an explicit filename
map); they back the ablation tables and figures added during review.

| Filename | Produced by | Contents |
|---|---|---|
| `swat_v10_temporal.zip` | `../notebooks/experiments/ot_ics_ids_llm_nebius_v10_swat.ipynb` | Full E1–E8 / R1 / R2 tree for SWaT under both `random` and stratified `chronological` splits (temporal-leakage ablation), plus E8 hybrid. Includes `manifest_swat.json`. |
| `hai_v10_temporal.zip` | `../notebooks/experiments/ot_ics_ids_llm_nebius_v10_hai.ipynb` | Same for HAI. Includes `manifest_hai.json`. |
| `xgboost_anchor_e7.zip` | `../notebooks/experiments/ot_ics_ids_xgboost_anchor_v1_3.ipynb` | XGBoost anchor under the E7 K-shot protocol: `xgb_master_summary.csv`, per-task `xgb_e7_*.csv`, `xgb_perclass_wustl.csv`, `xgb_selected_features.csv`. Feeds the new XGBoost rows in Table 2 and Table 5. |
| `anchor_maxctx_sweep.zip` | `../notebooks/experiments/ot_ics_ids_anchor_maxctx_v1_4.ipynb` | Max-context sensitivity sweep: `v14_sweep_A_kshot.csv`, `v14_sweep_B_maxctx.csv`, `v14_combined.csv`, `v14_delta.csv`, `v14_summary.csv`, `v14_perclass_wustl.csv` (paper §5.7). |
| `cost_pareto_regen.zip` | `../notebooks/figures/regenerate_cost_pareto.ipynb` | Corrected-pricing cost CSVs (`e5_cost_{swat,hai,wustl}_regen.csv`). The matching figure is written directly to `../paper_artifacts/figures/fig_cost_pareto.{pdf,png}`. |
| `ot_ics_ids_revision.zip` | `../notebooks/experiments/ot_ics_ids_revision_experiments_v7.ipynb` | Reviewer-response revision bundle (blocks REV-A…REV-F): hybrid cascade, feature-count ablation, t-SNE/UMAP embeddings, complexity profile, leave-one-attack-type-out generalisation, and the 2×2 headline figure. See *Revision archive internal layout* below. |

### v10 internal layout

The two v10 temporal archives follow the same
`ot_ics_ids_llm_nebius_v1_outputs/` tree as the canonical six, with two
additions: `E7_full_test/` now carries both `*_random_*` and
`*_chronological_*` variants of each summary / McNemar / per-seed file,
and a new `E8_hybrid/` subfolder holds the hybrid-stage results.

### Revision archive internal layout

`ot_ics_ids_revision.zip` is rooted at `ot_ics_ids_revision_outputs/`, with one
subfolder per reviewer-response block. Each block carries its own CSV(s) plus the
figure or LaTeX table it backs in the paper:

```
ot_ics_ids_revision_outputs/
├── REV_A_hybrid/          R1.1 — predicted-class router + confidence-gated cascade (WUSTL 5-class)
│   ├── reva_standalone.csv, reva_router.csv, reva_cascade.csv
│   └── reva_cascade_curve.{pdf,png}        → Fig. fgr:cascade  (drcoyz/reva_cascade_curve)
├── REV_B_featablation/    R2.8 — feature-count ablation, K_feat ∈ {6,8,10,12,16,20,all}
│   ├── revb_featablation.csv
│   └── revb_featablation.{pdf,png}         → Table t:featablation, Fig. fgr:featablation
├── REV_C_embeddings/      R2.4 — t-SNE / UMAP normal-vs-attack projections (3 datasets)
│   └── revc_embeddings.{pdf,png}           → Fig. fgr:embeddings  (drcoyz/revc_embeddings)
├── REV_D_complexity/      R2.5 — per-method computational profile
│   └── revd_complexity.csv, revd_complexity.tex   → Table t:complexity
├── REV_E_crosscampaign/   R1.2 — leave-one-attack-type-out generalisation (WUSTL)
│   ├── reve_crosscampaign.csv
│   └── reve_crosscampaign.{pdf,png}        → Table t:unseen, Fig. fgr:unseen
└── REV_F_figure1/         R1.3 — headline forest plot rebuilt as a 2×2 panel
    ├── fig1_values.csv
    └── fig_forest_headline_2x2.{pdf,png}   → Fig. fgr:headline2x2
```

REV-F is rebuilt directly from the committed canonical archives (`E7_full_test`
summaries plus `xgboost_anchor_e7.zip`), so the headline figure is reproducible
from the repository with no manual transcription. The figure/table short names in
the right-hand column are the LaTeX `\includegraphics` / `\label` targets used by
the manuscript.
