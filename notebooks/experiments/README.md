# Notebooks: Experiments

This folder is the home for the **LLM-side reproducibility path** of the
paper. The six per-(dataset, mode) experiment notebooks live here:

| Notebook | Dataset | Mode |
|---|---|---|
| `ot_ics_ids_llm_nebius_v8_swat.ipynb` | SWaT | binary |
| `ot_ics_ids_llm_nebius_v8_hai.ipynb` | HAI | binary |
| `ot_ics_ids_llm_nebius_v8_wustl.ipynb` | WUSTL-IIoT-2021 | binary |
| `ot_ics_ids_llm_nebius_multiclass_swat_v8.ipynb` | SWaT | multi-class (reduces to binary; see paper §3.4) |
| `ot_ics_ids_llm_nebius_multiclass_hai_v8.ipynb` | HAI | multi-class (reduces to binary; see paper §3.4) |
| `ot_ics_ids_llm_nebius_multiclass_wustl_v8.ipynb` | WUSTL-IIoT-2021 | multi-class (5 classes) |

The notebooks call the Nebius AI Studio API (Qwen3-235B-A22B, Llama-3.3-70B,
Hermes-4-70B, Hermes-4-405B) and the local TabPFN / TabICL anchors,
saving per-experiment CSV summaries that the paper-figure notebook in
`../figures/` consumes.

## Supplementary extensions

Four additional notebooks. They
extend the v8 protocol with ablations and a second
classical anchor. Their outputs live under `../../runs/supplementary/`
(see that folder's README), not under the canonical six short names.

| Notebook | Role | Output archive |
|---|---|---|
| `ot_ics_ids_llm_nebius_v10_swat.ipynb` | Temporal-split ablation (stratified chronological 80/20) on SWaT, addressing temporal-leakage concerns in 1 Hz time-series. Supersedes v8/v9; adds a `SPLIT_MODE` knob (`random` reproduces v8 exactly, `chronological` runs the ablation) and an E8 hybrid stage. | `swat_v10_temporal.zip` |
| `ot_ics_ids_llm_nebius_v10_hai.ipynb` | Same temporal-split ablation on HAI. | `hai_v10_temporal.zip` |
| `ot_ics_ids_xgboost_anchor_v1_3.ipynb` | Adds **XGBoost** as a second classical anchor under the paper's exact E7 K-shot protocol (10 examples per class, top-12 MI features, seeds 42/43/44). Feeds new XGBoost rows into Table 2 and Table 5. | `xgboost_anchor_e7.zip` |
| `ot_ics_ids_anchor_maxctx_v1_4.ipynb` | **Max-context sensitivity**: evaluates every tabular/classical anchor (RF, XGBoost, TabPFN, TabICL) at both the K-shot budget and its native maximum budget, to test whether the K-shot ranking transfers to a data-unconstrained regime (paper §5.7). LLMs are not re-run (K=10 is their structural ceiling). | `anchor_maxctx_sweep.zip` |

The two v10 notebooks have the same Nebius / TabPFN / Kaggle
prerequisites as the v8 notebooks. The XGBoost and max-context anchor
notebooks are anchor-only and need **no LLM API key** — only Kaggle
credentials (plus the TabPFN cloud token if the TabPFN max-context arm is
run); they finish in minutes rather than hours.

## Prerequisites

```bash
export NEBIUS_API_KEY="your-nebius-key"
export TABPFN_API_TOKEN="your-tabpfn-token"
# Kaggle credentials in ~/.kaggle/kaggle.json or KAGGLE_USERNAME / KAGGLE_KEY
```

## Typical runtime

A single notebook takes 2–6 hours on a single CPU machine, dominated by
the LLM inference. The TabPFN and TabICL anchor evaluations on the full
holdout finish in under one minute.

## Output

Each notebook writes a result-zip archive matching the pattern
`ot_ics_ids_<dataset>_paper_<timestamp>_<mode>.zip` containing the
sub-experiment CSVs (E1, E2, E3, E5, E6, E7, R1, R2 as applicable).

For convenience, you can move the produced archive to `../../runs/`
under one of the canonical short names (`swat_binary.zip`,
`swat_multiclass.zip`, `hai_binary.zip`, etc.) so the figure-regeneration
notebook in `../figures/` picks it up.
