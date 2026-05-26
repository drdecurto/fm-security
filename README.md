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

## Two reproducibility paths

This repository supports two complementary ways to reproduce the paper's results, with different cost and dependency profiles:

| Path | What it covers | Runtime | External services |
|---|---|---|---|
| **Scriptable package (`src/`)** | Tabular anchors (RF, TabPFN, TabICL) and baseline / few-shot / rare-attack / cross-domain experiments | Minutes on CPU | TabPFN cloud API (free tier) |
| **Notebooks (`notebooks/experiments/`)** | Full LLM evaluation against the Nebius AI Studio API, all six (dataset × mode) combinations | Hours per notebook | Nebius AI Studio API key + Kaggle credentials |
| **Figure regeneration (`notebooks/figures/`)** | Reads result CSVs and produces every figure and table in the paper | Under one minute | None |

If you only want to regenerate the paper figures from the pre-computed runs, jump to [Quick start: regenerate figures](#quick-start-regenerate-figures-and-tables).

## Repository layout

```
llm-security/
├── src/                              Scriptable Python package (see src/README.md)
│   ├── data/                         Dataset loaders (SWaT, HAI, WUSTL-IIoT)
│   ├── models/                       Random Forest, TabPFN, TabICL wrappers
│   ├── evaluation/                   Metrics, pipeline, few-shot, visualization
│   └── experiments/                  Executable: baseline, few-shot, rare-attack, cross-domain
├── notebooks/
│   ├── experiments/                  Six v8 notebooks: LLM evaluation per (dataset, mode)
│   └── figures/                      Figure-generation notebook (paper-ready output)
├── runs/                             Six result-zip archives from the paper's E1–R2 runs
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
git clone https://github.com/drdecurto/llm-security.git
cd llm-security
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
| Tabular anchors | Random Forest (sklearn, 200 trees), TabPFN (cloud API), TabICL v1.1-20250506 (local CPU) |
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
| **R1** | Cross-seed multi-LLM comparison (5 seeds, all LLMs vs RandomForest) |
| **R2** | Per-class multi-seed analysis (5 seeds, WUSTL multi-class only) |


## Acknowledgements

This work was supported by the **LUXEMBOURG Institute of Science and Technology** (projects *ADIALab-MAST* and *LLMs4EU*, Grant Agreement No. 101198470) and the **BARCELONA Supercomputing Center** (project *TIFON*, File MIG-20232039).

## License

This code is released under the [MIT License](LICENSE). The Kaggle dataset mirrors retain their original licensing terms; please consult the mirror pages for usage restrictions.

## Contact

For questions about the code or paper, contact Dr. J. de Curtò at <jdecurto@icai.comillas.edu>.
