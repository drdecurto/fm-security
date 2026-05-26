# `src/` — Scriptable Python Package

This is the Python package that drives the **tabular-anchor side** of the paper (Random Forest, TabPFN, TabICL). It supports four experiments out of the box:

1. **Baseline cross-method comparison** (E1-equivalent)
2. **Few-shot scaling** (E2-equivalent)
3. **Rare-attack class detection** (E6-equivalent)
4. **Cross-domain transfer** (IT → OT)

For the LLM side of the paper (Qwen3, Llama-3.3, Hermes-4-{70B,405B}) and the per-seed multi-seed protocol used in the manuscript, see the Jupyter notebooks under `notebooks/experiments/` at the repository root.

## Layout

```
src/
├── data/
│   ├── _base.py                    Abstract BaseDataset (preprocessing pipeline)
│   ├── config.py                   Per-dataset paths, label maps, defaults
│   ├── preprocessing.py            Windowing, normalization, augmentation
│   └── tabular/
│       ├── swat.py                 SWaT loader
│       ├── hai.py                  HAI loader
│       └── wustl_iiot.py           WUSTL-IIoT-2021 loader
├── models/
│   ├── _base.py                    Abstract BaseModel
│   ├── config.py                   Per-model hyper-parameters
│   ├── baselines.py                Classical ML baselines (RF, LR, SVM, kNN, DT)
│   ├── tabpfn_model.py             TabPFN wrapper
│   ├── tabicl_model.py             TabICL wrapper (functional)
│   └── trainable/
│       └── tabicl.py               TabICL trainable wrapper (SklearnTrainableModel API)
├── evaluation/
│   ├── metrics.py                  Acc, MacroF1, MCC, FAR, DR, AUROC, AUPRC, per-class
│   ├── pipeline.py                 EvaluationPipeline class
│   ├── few_shot.py                 Stratified k-shot sampler + sweep utilities
│   └── visualization.py            Forest plots, confusion-matrix heatmaps, Pareto plots
└── experiments/
    ├── run_baseline.py             Exp 1 driver
    ├── run_few_shot.py             Exp 2 driver
    ├── run_rare_attack.py          Exp 3 driver
    └── run_cross_domain.py         Exp 4 driver
```

## Quick start

```python
from src.data.tabular import SWaTDataset
from src.models.tabicl_model import TabICLModel
from src.evaluation.pipeline import EvaluationPipeline

dataset = SWaTDataset(window_size=1, pca=False)
dataset.load()

X_train, X_test, y_train, y_test = dataset.train_test_split(test_size=0.2, stratify=True)

model = TabICLModel()
pipeline = EvaluationPipeline(model)
results = pipeline.evaluate(X_train, y_train, X_test, y_test)
print(results["classification_report"])
```

## Running the experiments from the command line

```bash
# Experiment 1: Baseline comparison (RF + TabPFN + TabICL on each dataset)
python -m src.experiments.run_baseline --dataset swat  --models all
python -m src.experiments.run_baseline --dataset hai   --models all
python -m src.experiments.run_baseline --dataset wustl --models all

# Experiment 2: k-shot scaling
python -m src.experiments.run_few_shot --dataset swat --shots 5 10 25 50 100 500
python -m src.experiments.run_few_shot --dataset hai  --shots 5 10 25 50 100 500

# Experiment 3: Rare-attack class detection (WUSTL 5-class)
python -m src.experiments.run_rare_attack --dataset wustl --threshold 50

# Experiment 4: Cross-domain transfer
python -m src.experiments.run_cross_domain --source swat --target hai
```

Or use `scripts/run_all_experiments.sh` at the repository root to drive all four with default arguments.

## Configuration

Two layers of configuration coexist:

1. **Global runtime parameters** in `config.yaml` at the repository root — random seed, number of repetitions, output directory, device selection.
2. **Per-component constants** in `src/data/config.py` (paths, label maps, dataset defaults) and `src/models/config.py` (per-model hyper-parameters).

To change the TabICL hyper-parameters used by every experiment, edit `TABICL_CONFIG` in `src/models/config.py`.

### TabICL 2.x parameter naming

The `TABICL_CONFIG` in `src/models/config.py` is correct for `tabicl >= 2.0`. The library renamed several parameters between 0.x and 2.x without deprecation warnings. If you see a `TypeError: __init__() got an unexpected keyword argument …`, your `tabicl` install is older than 2.0 and the constructor is rejecting the new-style parameter names. Either upgrade (`pip install -U tabicl`) or pin to the parameter dictionary used by the older release.

The current `TABICL_CONFIG` uses the v2.x naming throughout:

- `class_shuffle_method: "shift"` (was `class_shift: True` in v0.x)
- `support_many_classes: True` (was `use_hierarchical: True` in v0.x)
- `use_amp: "auto"` (was `use_amp: True` in v0.x)
- `checkpoint_version: "tabicl-classifier-v1.1-20250506.ckpt"`

## Dataset access

The package loads each dataset from either the public Kaggle mirror (default) or a local raw-CSV directory.

### Option 1 — Kaggle (default, recommended)

The three OT/ICS benchmarks are accessed via `kagglehub`. Configure your Kaggle credentials at `~/.kaggle/kaggle.json` (or via the `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables) and the loaders will download and cache the data on first use:

```python
import kagglehub
import os

# Loaders look up the mirror by ID and pull on demand
kagglehub.dataset_download("vishala28/swat-dataset-secure-water-treatment-system")
kagglehub.dataset_download("icsdataset/hai-security-dataset")
kagglehub.dataset_download("annaamalaiu/wustl-iiot-2021-dataset")
```

To activate Kaggle mode globally, set:

```bash
export OTICS_DATA_BACKEND=kagglehub
```

The loaders honour this environment variable and fall back to local-raw mode otherwise.

### Option 2 — Local raw CSV files

For sites without internet access or for the original iTrust SWaT distribution (which requires registration and is **not** on Kaggle), place files under:

```
data/raw/
├── swat/
│   ├── SWaT_Dataset_Normal_v1.csv
│   └── SWaT_Dataset_Attack_v0.csv
├── hai/
│   └── *.csv
└── wustl_iiot/
    └── *.csv
```

Paths and expected file names are configured in `src/data/config.py`.

### About the Kaggle mirrors

A practical caveat: the **SWaT and HAI Kaggle mirrors carry only the binary `Normal/Attack` indicator**. The per-attack-type labels present in the original iTrust and NSRI distributions were not re-uploaded. The multi-class experiments on those two datasets therefore reduce to binary; only WUSTL-IIoT-2021 supports a genuine 5-class evaluation through the public mirror.

## Module reference

### `src.data`

| Class | Purpose |
|---|---|
| `BaseDataset` | Abstract base providing `preprocess`, `train_test_split`, normalization, PCA, windowing |
| `SWaTDataset` | SWaT loader (binary by default; supports multi-class if labels are available) |
| `HAIDataset` | HAI loader (versions 1.0, 2.0, 3.0) |
| `WUSTLIIoTDataset` | WUSTL-IIoT-2021 loader (binary or 5-class) |

### `src.models`

| Class | Purpose |
|---|---|
| `BaseModel` | Abstract base specifying the `fit` / `predict` / `predict_proba` interface |
| `SklearnTrainableModel` | Trainable wrapper that delegates to a scikit-learn-compatible estimator |
| `TabPFNModel` | TabPFN classifier (cloud API via `tabpfn-client`) |
| `TabICLModel` | TabICL classifier (functional wrapper; uses `src/models/tabicl_model.py`) |
| `TabICLModel` (in `trainable/tabicl.py`) | TabICL classifier (trainable variant integrating with `SklearnTrainableModel`) |
| `RandomForestModel`, `LogisticRegressionModel`, `LinearSVCModel`, `KNeighborsModel`, `DecisionTreeModel` | Classical baselines |

### `src.evaluation`

| Function / Class | Purpose |
|---|---|
| `compute_metrics` | Acc, MacroF1, MCC, FAR, DR; optional AUROC / AUPRC when probabilities are available |
| `EvaluationPipeline` | Unified `evaluate(X_train, y_train, X_test, y_test)` returning a structured dict |
| `few_shot_split` | Stratified k-per-class sampler |
| `run_few_shot_sweep` | Sweeps K over a list, repeats `n_runs` times, aggregates with mean ± std |

## Dependencies

This package targets Python 3.10+ and requires:

```text
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
scipy>=1.11
matplotlib>=3.7
seaborn>=0.12
tabpfn>=2.0
tabicl>=2.0
PyYAML>=6.0
tqdm>=4.65
joblib>=1.3
kagglehub>=0.2
```

See `requirements.txt` at the repository root.

