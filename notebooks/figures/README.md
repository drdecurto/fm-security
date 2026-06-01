# Notebooks: Figures and Tables

This folder holds the **figure-regeneration notebook** that consumes the
result-zip archives in `../../runs/` and writes every paper-ready figure
and table into `../../paper_artifacts/`.

## Single source of truth

- `ot_ics_ids_paper_figures_v2.ipynb` — paper-ready figures (PDF + PNG)
  and LaTeX tables. Reproduces every artifact embedded in the manuscript.

## Cost-figure regeneration

- `regenerate_cost_pareto.ipynb` — standalone utility that recomputes the
  cost figures with the **corrected Qwen3-235B-A22B pricing** (Nebius
  May-2026 verified: \$0.20 / \$0.60 per 1M input/output tokens, down from
  the \$0.30 / \$0.90 used in the v10 run). It reads the three
  `e5_cost_{swat,hai,wustl}.csv` files, writes `*_regen.csv` variants, and
  regenerates `fig_cost_pareto.{pdf,png}`.

  Because the pricing correction post-dates the main figures notebook,
  `fig_cost_pareto` in `../../paper_artifacts/figures/` is the output of
  **this** notebook, not of `ot_ics_ids_paper_figures_v2.ipynb`. The
  regenerated CSVs are archived in
  `../../runs/supplementary/cost_pareto_regen.zip`.

  To run: set `INPUT_DIR` in cell 3 to a directory holding the three
  `e5_cost_*.csv` files (e.g. the unzipped E5 folders from the v10 run
  bundles), then run all cells. Outputs land in `./cost_regen_outputs/`.

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
