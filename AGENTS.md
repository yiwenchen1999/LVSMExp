# Repository Guidelines

## Project Structure & Module Organization

Core training and inference entry points live at the repository root (`train.py`, `train_editor.py`, `inference.py`, and editor variants). Model components are in `model/`, dataset loaders in `data/`, and shared helpers in `utils/`. Experiment settings belong in `configs/`; prefer adding a YAML variant instead of embedding machine-specific values in Python. Data preparation and evaluation utilities are grouped under `preprocess_scripts/`, `scripts/`, and `data_postprocess/`. Reusable launch recipes are in `bash_scripts/`. Generated datasets, checkpoints, and visual outputs belong in directories such as `data_samples/`, `experiments/`, and `result_previews/`, and should not be committed unless intentionally curated. `flow_matching/` is a largely self-contained package with its own tests and tooling.

## Build, Test, and Development Commands

Create the documented Python 3.11 environment:

```bash
conda create -n LVSM python=3.11
conda activate LVSM
pip install -r requirements.txt
```

Run single-node training with `torchrun --nproc_per_node 1 train.py --config configs/LVSM_scene_encoder_decoder.yaml`; increase the process count only when GPUs and configuration support it. Run inference similarly with `torchrun --nproc_per_node 1 inference.py --config configs/LVSM_scene_encoder_decoder.yaml` plus OmegaConf overrides such as `training.dataset_path=...`. Inspect each utility's options with, for example, `python preprocess_scripts/create_evaluation_index.py --help`.

## Coding Style & Naming Conventions

Use four-space indentation and standard Python conventions: `snake_case` for modules, functions, and variables; `PascalCase` for classes; and descriptive YAML names matching the model or experiment. Keep filesystem paths and hyperparameters configurable. The `flow_matching/` package uses `ufmt` (Black plus usort), Flake8, and Google-style docstring checks; run `pre-commit run --all-files` from that directory when modifying it. Follow nearby style elsewhere, where formatting is not globally enforced.

## Testing Guidelines

For `flow_matching/`, run `pytest flow_matching/tests`. Root-level changes currently rely on targeted smoke tests: invoke the changed script with `--help`, run a minimal configuration or small sample, and validate generated metrics or images. Name new pytest files `test_<feature>.py` and keep fixtures small; avoid tests requiring full datasets or multi-GPU hardware.

## Commit & Pull Request Guidelines

Recent history uses short, experiment-oriented subjects. Prefer a concise imperative summary such as `add masked NVDiff evaluation`, and keep unrelated changes separate. Pull requests should describe the model/configuration affected, exact validation commands, required data or checkpoints, and expected metric changes. Include representative images for rendering changes and link the relevant issue or experiment. Never commit credentials from `configs/api_keys.yaml`, private dataset paths, large checkpoints, or generated result trees.
