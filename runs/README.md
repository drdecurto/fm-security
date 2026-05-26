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
