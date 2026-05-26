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
