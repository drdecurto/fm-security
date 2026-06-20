# LLM Security: Foundation Models for Industrial Cyber-Physical Security


We benchmark four open-source Large Language Models (Qwen3-235B-A22B, Llama-3.3-70B, Hermes-4-70B, Hermes-4-405B) against two recent tabular foundation models (TabPFN, TabICL) and a Random Forest baseline across three OT/ICS intrusion-detection datasets: SWaT, HAI, and WUSTL-IIoT-2021. Evaluation uses a multi-seed full-holdout protocol with paired McNemar tests and cross-seed Mann–Whitney comparisons.

[![Python](https://img.shields.io/badge/python-3.10+-blue)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Headline findings

| Dataset | Best detector | Margin |
|---|---|---|
| **SWaT** (water-treatment SCADA) | Qwen3-235B-A22B (LLM) | MCC = 0.700 ± 0.005, McNemar *p* < 0.002 vs every anchor |
| **HAI** (multi-process HIL) | TabICL (tabular FM) | MCC = 0.432 ± 0.013, Δaccuracy = +0.072 over best LLM, *p* < 10⁻⁴⁴ |
| **WUSTL-IIoT-2021** | TabPFN (tabular FM) | Macro F1 = 0.924 ± 0.003 on 5-class taxonomy |

On the WUSTL 5-class attack taxonomy, every evaluated LLM is 13–28 percentage points weaker than Random Forest on traffic-rich attacks (Denial-of-Service, Reconnaissance) but remains competitive on rare, subtle attack types (Backdoor, Command Injection).

Two deployment-relevant findings were added in the revision round (see *Path B++ — Reviewer-response revision experiments* below):

- **Confidence-gated cascade.** Escalating only the tabular model's low-confidence decisions to the LLM (≈ 6.3 % of samples at τ = 0.5) lifts WUSTL 5-class macro F1 to **0.945** and MCC to **0.946**, exceeding either standalone detector at a small, tunable query budget.
- **Generalisation to unseen attack families.** Under leave-one-attack-type-out, the classical anchors collapse (XGBoost mean recall 0.178, Random Forest 0.454) while the foundation models and the LLM retain it (TabPFN 0.641, TabICL 0.792, LLM 0.795) — the gap between detector families *widens* on families never seen at training time.

## Two reproducibility paths

This repository supports two complementary ways to reproduce the paper's results, with different cost and dependency profiles:

| Path | What it covers | Runtime | External services |
|---|---|---|---|
| **Scriptable package (`src/`)** | Tabular anchors (RF, TabPFN, TabICL) and baseline / few-shot / rare-attack / cross-domain experiments | Minutes on CPU | TabPFN cloud API (free tier) |
| **Notebooks (`notebooks/experiments/`)** | Full LLM evaluation against the Nebius AI Studio API, all six (dataset × mode) combinations | Hours per notebook | Nebius AI Studio API key + Kaggle credentials |
| **Figure regeneration (`notebooks/figures/`)** | Reads result CSVs and produces every figure and table in the paper | Under one minute | None |
| **Revision notebook (`notebooks/experiments/…_revision_experiments_v7.ipynb`)** | Reviewer-response additions REV-A…F: hybrid cascade, feature-count ablation, t-SNE/UMAP, complexity profile, unseen-family generalisation, 2×2 headline | Minutes (anchor-only blocks) to a few hours (REV-A LLM cascade) | Nebius API for REV-A; keyless otherwise |

If you only want to regenerate the paper figures from the pre-computed runs, jump to [Quick start: regenerate figures](#quick-start-regenerate-figures-and-tables).

## Repository layout

```
fm-security/
├── src/                              Scriptable Python package (see src/README.md)
│   ├── data/                         Dataset loaders (SWaT, HAI, WUSTL-IIoT)
│   ├── models/                       Random Forest, TabPFN, TabICL wrappers
│   ├── evaluation/                   Metrics, pipeline, few-shot, visualization
│   └── experiments/                  Executable: baseline, few-shot, rare-attack, cross-domain
├── notebooks/
│   ├── experiments/                  v8 LLM notebooks (per dataset, mode) + supplementary
│   │                                 extensions: v10 temporal-split ablation (SWaT, HAI),
│   │                                 XGBoost anchor, anchor max-context sensitivity, and the
│   │                                 reviewer-response revision notebook (revision v7, REV-A…F)
│   └── figures/                      Figure-generation notebook + cost-Pareto regeneration
├── runs/                             Six canonical result-zip archives (E1–R2)
│   └── supplementary/                supplementary archives (v10 temporal, XGBoost, max-ctx,
│                                     cost regen) plus the REV-A…F revision bundle `ot_ics_ids_revision.zip`
├── paper_artifacts/
│   ├── tables/                       LaTeX tables embedded in the paper
│   └── figures/                      Publication figures (PDF + PNG)
├── scripts/
│   └── run_all_experiments.sh        Driver for the scriptable experiments
├── config.yaml                       Global configuration
├── requirements.txt                  Python dependencies
├── LICENSE                           MIT
└── README.md                         This file
```

## Installation

Requires Python 3.10+ and approximately 10 GB of disk space (for cached Kaggle datasets and TabPFN model weights):

```bash
git clone https://github.com/drdecurto/fm-security.git
cd fm-security
pip install -r requirements.txt
```

For the notebook path, you will also need Jupyter:

```bash
pip install jupyter ipykernel
```

## Quick start: regenerate figures and tables

The fastest reproduction path uses the pre-computed result archives in `runs/`. No API keys, no Kaggle credentials, no GPU.

1. Open `notebooks/figures/ot_ics_ids_paper_figures_v2.ipynb`.
2. Run all cells. The notebook unzips the six archives in `runs/`, parses the CSV summaries, and writes every figure (`.pdf` + `.png`) and table (`.tex`) into `paper_artifacts/`. Typical runtime is under one minute on a modern laptop.

## Reproducing the experiments

### Path A — Scriptable tabular experiments (`src/`)

The four experiment scripts under `src/experiments/` cover the tabular-anchor side of the paper (RF, TabPFN, TabICL). See [`src/README.md`](src/README.md) for the full package guide.

```bash
# Experiment 1: Baseline binary comparison on each dataset
python -m src.experiments.run_baseline --dataset swat  --models all
python -m src.experiments.run_baseline --dataset hai   --models all
python -m src.experiments.run_baseline --dataset wustl --models all

# Experiment 2: Few-shot scaling (K = 5, 10, 25, 50, 100, 500)
python -m src.experiments.run_few_shot --dataset swat --shots 5 10 25 50 100 500
python -m src.experiments.run_few_shot --dataset hai  --shots 5 10 25 50 100 500

# Experiment 3: Rare-attack class detection (multi-class)
python -m src.experiments.run_rare_attack --dataset wustl --threshold 50

# Experiment 4: Cross-domain transfer
python -m src.experiments.run_cross_domain --source swat --target hai

# Or run everything at once
chmod +x scripts/run_all_experiments.sh
./scripts/run_all_experiments.sh
```

Global parameters live in `config.yaml` (random seed, number of repetitions, output directory, etc.).

### Path B — Full LLM evaluation notebooks (`notebooks/`)

The six notebooks under `notebooks/experiments/` reproduce the paper's LLM-side evaluation. Each notebook covers one combination of dataset and mode:

| Notebook | Dataset | Mode |
|---|---|---|
| `ot_ics_ids_llm_nebius_v8_swat.ipynb` | SWaT | binary |
| `ot_ics_ids_llm_nebius_v8_hai.ipynb` | HAI | binary |
| `ot_ics_ids_llm_nebius_v8_wustl.ipynb` | WUSTL-IIoT-2021 | binary |
| `ot_ics_ids_llm_nebius_multiclass_swat_v8.ipynb` | SWaT | multi-class (reduces to binary; see note below) |
| `ot_ics_ids_llm_nebius_multiclass_hai_v8.ipynb` | HAI | multi-class (reduces to binary; see note below) |
| `ot_ics_ids_llm_nebius_multiclass_wustl_v8.ipynb` | WUSTL-IIoT-2021 | multi-class (5 classes) |

Running an experiment notebook requires:

- A **Nebius AI Studio API key** (`NEBIUS_API_KEY` environment variable) for the four open-source LLMs.
- A **TabPFN client API token** (`TABPFN_API_TOKEN`); free-tier signup at <https://priorlabs.ai>.
- The **TabICL classifier checkpoint**, downloaded automatically by `tabicl` on first use.
- **Kaggle API credentials** for dataset access (the notebooks use `kagglehub`).
- **2–6 hours of wall-clock time** per notebook on a single CPU machine; the LLM inference dominates.

### Path B+ — Supplementary extensions (`notebooks/experiments/`)

Four further notebooks were added during the revision round. They reuse the v8 protocol and feed supplementary tables, figures, and ablations; their result archives live under `runs/supplementary/`.

| Notebook | What it adds | LLM key needed? | Output archive |
|---|---|---|---|
| `ot_ics_ids_llm_nebius_v10_swat.ipynb` | Temporal-split ablation on SWaT (stratified chronological 80/20) addressing temporal-leakage concerns in 1 Hz industrial time-series; `SPLIT_MODE` knob reproduces v8 (`random`) or runs the ablation (`chronological`); adds an E8 hybrid stage | Yes | `swat_v10_temporal.zip` |
| `ot_ics_ids_llm_nebius_v10_hai.ipynb` | Same temporal-split ablation on HAI | Yes | `hai_v10_temporal.zip` |
| `ot_ics_ids_xgboost_anchor_v1_3.ipynb` | **XGBoost** as a second classical anchor under the exact E7 K-shot protocol (10 per class, top-12 MI features, seeds 42/43/44); feeds new XGBoost rows into Tables 2 and 5 | No | `xgboost_anchor_e7.zip` |
| `ot_ics_ids_anchor_maxctx_v1_4.ipynb` | **Max-context sensitivity** (§5.7): each tabular/classical anchor (RF, XGBoost, TabPFN, TabICL) evaluated at both the K-shot budget and its native maximum, testing whether the K-shot ranking transfers to a data-unconstrained regime; LLMs not re-run (K=10 is their ceiling) | No (TabPFN token for its max-ctx arm) | `anchor_maxctx_sweep.zip` |

The XGBoost and max-context notebooks are anchor-only and finish in minutes. The cost-Pareto figure is regenerated separately with corrected Qwen3 pricing (\$0.20 / \$0.60 per 1M tokens) via `notebooks/figures/regenerate_cost_pareto.ipynb`; see [`paper_artifacts/README.md`](paper_artifacts/README.md).

### Path B++ — Reviewer-response revision experiments (`ot_ics_ids_revision_experiments_v7.ipynb`)

A single self-contained notebook, `notebooks/experiments/ot_ics_ids_revision_experiments_v7.ipynb` (revision v7), implements the additional experimentation requested in the *Electronics* revision round. It condenses the v8/v10 shared pipeline (identical MI feature selection, balanced K-shot split, RF/XGBoost/TabPFN/TabICL configs, role-instructed JSON prompt and label parser, and FAR/DR/MCC metrics) into **PART 1**, then runs six independent blocks in **PART 2**, each guarded by a flag in the CONFIG cell and writing its own artifacts. Run PART 1 top-to-bottom once, then run whichever blocks you need.

| Block | Reviewer ask | Needs Nebius LLM key? | Output folder | Backs in paper |
|---|---|---|---|---|
| **REV-A** | R1.1 — deployable hybrid: predicted-class router **+** confidence-gated cascade on WUSTL 5-class | Yes | `REV_A_hybrid/` | §6.4, Fig. `fgr:cascade` |
| **REV-B** | R2.8 — feature-count ablation, *K*<sub>feat</sub> ∈ {6, 8, 10, 12, 16, 20, all} across model families | Optional (anchors run keyless) | `REV_B_featablation/` | §5.9, Table `t:featablation`, Fig. `fgr:featablation` |
| **REV-C** | R2.4 — t-SNE / UMAP normal-vs-attack distribution visualisations | No | `REV_C_embeddings/` | §3, Fig. `fgr:embeddings` |
| **REV-D** | R2.5 — complexity table (params, size, inference time, memory, approx. FLOPs) | No | `REV_D_complexity/` | §5.5, Table `t:complexity` |
| **REV-E** | R1.2 — cross-campaign (leave-one-attack-type-out) generalisation on WUSTL | Optional | `REV_E_crosscampaign/` | §5.11, Table `t:unseen`, Fig. `fgr:unseen` |
| **REV-F** | R1.3 — headline Figure 1 rebuilt as a legible 2×2 panel, directly from the committed `runs/` archives | No | `REV_F_figure1/` | §5.1, Fig. `fgr:headline2x2` |

The outputs of a full run are committed as `runs/supplementary/ot_ics_ids_revision.zip` (internally rooted at `ot_ics_ids_revision_outputs/`, with one `REV_*` subfolder per block); unzip it to inspect every revision figure and table without re-execution. The paths below are relative to `ot_ics_ids_revision_outputs/` and map to the paper as follows:

| Archive path | Artifact | Paper element |
|---|---|---|
| `REV_A_hybrid/reva_standalone.csv`, `reva_router.csv`, `reva_cascade.csv` | Standalone / router / cascade operating points | §6.4 numbers |
| `REV_A_hybrid/reva_cascade_curve.{pdf,png}` | Escalation-fraction vs macro-F1 frontier | Fig. `fgr:cascade` (`drcoyz/reva_cascade_curve`) |
| `REV_B_featablation/revb_featablation.csv` + `.{pdf,png}` | MCC vs *K*<sub>feat</sub> per method/dataset | Table `t:featablation`, Fig. `fgr:featablation` (`drcoyz/revb_featablation`) |
| `REV_C_embeddings/revc_embeddings.{pdf,png}` | t-SNE / UMAP projections (3 datasets) | Fig. `fgr:embeddings` (`drcoyz/revc_embeddings`) |
| `REV_D_complexity/revd_complexity.csv` + `.tex` | Per-method computational profile | Table `t:complexity` |
| `REV_E_crosscampaign/reve_crosscampaign.csv` + `.{pdf,png}` | Leave-one-attack-type-out recall | Table `t:unseen`, Fig. `fgr:unseen` (`drcoyz/reve_crosscampaign`) |
| `REV_F_figure1/fig_forest_headline_2x2.{pdf,png}` + `fig1_values.csv` | 2×2 headline forest plot | Fig. `fgr:headline2x2` (`drcoyz/fig_forest_headline_2x2`) |

Credentials are the same as the other notebooks (Colab Secrets or env vars): `KAGGLE_USERNAME` / `KAGGLE_KEY` for all blocks, `NEBIUS_API_KEY` for REV-A and the optional LLM arms of REV-B/REV-E, and `TABPFN_TOKEN` for TabPFN. REV-F reads only the committed `runs/` archives and needs no credentials; set `REPO_ROOT` in the CONFIG cell to the repository root before running it.

### Dataset access

The three OT/ICS benchmarks are loaded from their public Kaggle mirrors:

| Dataset | Kaggle mirror |
|---|---|
| SWaT | `vishala28/swat-dataset-secure-water-treatment-system` |
| HAI | `icsdataset/hai-security-dataset` |
| WUSTL-IIoT-2021 | `annaamalaiu/wustl-iiot-2021-dataset` |

A practical note: the SWaT and HAI Kaggle mirrors ship only the binary `Normal/Attack` indicator. The per-attack-type labels present in the original iTrust and NSRI distributions were not re-uploaded. The multi-class notebooks therefore reduce to binary on those two datasets and contribute a genuine 5-class evaluation only on WUSTL.

The notebooks resolve dataset paths automatically via `kagglehub`. The scriptable package (`src/`) supports both `kagglehub` (default) and a local `data/raw/{swat,hai,wustl_iiot}/` directory; see `src/README.md`.

## Methodology summary

| Aspect | Setting |
|---|---|
| Tabular anchors | Random Forest (sklearn, 200 trees), XGBoost (supplementary second classical anchor), TabPFN (cloud API), TabICL v1.1-20250506 (local CPU) |
| Open-source LLMs | Qwen3-235B-A22B, Llama-3.3-70B, Hermes-4-70B, Hermes-4-405B (Nebius AI Studio) |
| In-context examples | K = 10 per class, role-instructed system message |
| Holdout | 80/20 stratified split; anchors see full holdout (≈ 100k rows), LLMs see n = 6,000 stratified subsample |
| Seeds | 3 for primary E7 evaluation, 5 for cross-LLM R1 and per-class R2 |
| Statistical tests | Paired McNemar (LLM vs each anchor), Mann–Whitney *U* (cross-seed) |

## Experiment taxonomy

The CSV files inside the `runs/` archives follow a fixed naming convention:

| Code | Description |
|---|---|
| **E1** | Cross-family head-to-head (single seed, all methods × all datasets) |
| **E2** | k-shot scaling sweep, K ∈ {5, 10, 25, 50, 100} |
| **E3** | Prompt-format ablation (role-instructed, raw JSON, typed JSON, natural-language narrative) |
| **E5** | Cost analysis (USD per correct prediction, LLM only) |
| **E6** | Per-class confusion matrices (multi-class only) |
| **E7** | Primary multi-seed evaluation (3 seeds, full protocol) |
| **E8** | Hybrid-stage evaluation (v10 notebooks only) |
| **R1** | Cross-seed multi-LLM comparison (5 seeds, all LLMs vs RandomForest) |
| **R2** | Per-class multi-seed analysis (5 seeds, WUSTL multi-class only) |
| **REV-A** | Deployable hybrid: predicted-class router + confidence-gated cascade (WUSTL 5-class) |
| **REV-B** | Feature-count ablation, *K*<sub>feat</sub> ∈ {6, 8, 10, 12, 16, 20, all}, across model families |
| **REV-C** | t-SNE / UMAP normal-vs-attack distribution visualisations |
| **REV-D** | Per-method complexity profile (params, size, inference time, memory, approx. FLOPs) |
| **REV-E** | Cross-campaign generalisation: leave-one-attack-type-out on WUSTL |
| **REV-F** | Headline Figure 1 rebuilt as a 2×2 panel from the committed `runs/` archives |

Codes **E1–R2** are produced by the v8/v10 notebooks and the scriptable package; **REV-A–REV-F** are produced by the revision notebook `ot_ics_ids_revision_experiments_v7.ipynb` and committed as `runs/supplementary/ot_ics_ids_revision.zip`.


## Acknowledgements

This work was supported by the **LUXEMBOURG Institute of Science and Technology** (projects *ADIALab-MAST* and *LLMs4EU*, Grant Agreement No. 101198470) and the **BARCELONA Supercomputing Center** (project *TIFON*, File MIG-20232039).

## License

This code is released under the [MIT License](LICENSE). The Kaggle dataset mirrors retain their original licensing terms; please consult the mirror pages for usage restrictions.

## Contact

For questions about the code or paper, contact Dr. J. de Curtò at <jdecurto@icai.comillas.edu>.
