# Notebooks: Figures and Tables

This folder holds the **figure-regeneration notebook** that consumes the
result-zip archives in `../../runs/` and writes every paper-ready figure
and table into `../../paper_artifacts/`.

## Single source of truth

- `ot_ics_ids_paper_figures_v2.ipynb` — paper-ready figures (PDF + PNG)
  and LaTeX tables. Reproduces every artifact embedded in the manuscript.

## Quick run

1. Make sure `../../runs/` contains the six result archives:
   `swat_binary.zip`, `swat_multiclass.zip`, `hai_binary.zip`,
   `hai_multiclass.zip`, `wustl_binary.zip`, `wustl_multiclass.zip`.
2. Open the notebook and run all cells.
3. Outputs land in `../../paper_artifacts/figures/` and
   `../../paper_artifacts/tables/` (overwriting any existing files).

Runtime is under one minute on a modern laptop. No API keys required.

## Notes

The notebook is read-only with respect to the runs: it unzips, parses
CSVs, and writes formatted artifacts. To produce new runs, see
`../experiments/`.
