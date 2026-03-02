# Inference Metadata Recording

## Overview

The inference pipeline now automatically records metadata about each inference run, including:
- Context (input) view indices
- Target view indices  
- Relit scene name (the scene used for relit image supervision)

## Output Files

### 1. Per-Scene Metadata (`<scene_name>/metadata.json`)

Each scene folder will contain a `metadata.json` file with the following structure:

```json
{
  "scene_name": "wooden_table_02_env_0",
  "context_view_indices": [43, 47, 49, 52],
  "target_view_indices": [62, 63, 72, 74, 79, 84, 86, 89],
  "relit_scene_name": "wooden_table_02_env_2"
}
```

**Fields:**
- `scene_name`: The name of the input scene
- `context_view_indices`: List of frame indices used as context/input views
- `target_view_indices`: List of frame indices used as target views for rendering
- `relit_scene_name`: The scene name used to load relit images for supervision (may be null if not using relit images)

### 2. Consolidated Summary (`inference_metadata_summary.json`)

A consolidated JSON file is created at the root of the output directory containing all scene metadata:

```json
{
  "wooden_table_02_env_0": {
    "scene_name": "wooden_table_02_env_0",
    "context_view_indices": [43, 47, 49, 52],
    "target_view_indices": [62, 63, 72, 74, 79, 84, 86, 89],
    "relit_scene_name": "wooden_table_02_env_2"
  },
  "wooden_table_02_env_1": {
    "scene_name": "wooden_table_02_env_1",
    "context_view_indices": [43, 47, 49, 52],
    "target_view_indices": [62, 63, 72, 74, 79, 84, 86, 89],
    "relit_scene_name": "wooden_table_02_rgb_pl_0"
  },
  ...
}
```

This file is only generated when `compute_metrics: true` in the inference config.

## Changes Made

### 1. `data/dataset_scene.py`

Added `relit_scene_name` to the returned dictionary when loading relit images:

```python
if self.use_relit_images:
    result_dict["relit_images"] = relit_images
    result_dict["relit_scene_name"] = relit_scene_name  # NEW
    ...
```

### 2. `utils/metric_utils.py`

Modified `export_results()` to:
- Extract both input and target view indices
- Extract relit scene name if available
- Save metadata JSON for each scene

### 3. `inference_editor.py`

Added post-processing step to:
- Collect all `metadata.json` files
- Create consolidated `inference_metadata_summary.json`
- Print summary statistics

## Usage

No changes needed to your inference scripts! The metadata will be automatically generated during inference.

Just run inference as usual:

```bash
bash bash_scripts/Sony_clusters/interactive_inference_polyhaven_dense.sh
```

The output directory will contain:
```
experiments/evaluation/polyhaven_dense_inference/
├── inference_metadata_summary.json          # Consolidated metadata
├── wooden_table_02_env_0/
│   ├── metadata.json                        # Per-scene metadata
│   ├── input_*.png
│   ├── render_*.png
│   └── ...
├── wooden_table_02_env_1/
│   ├── metadata.json
│   └── ...
└── ...
```

## Benefits

1. **Reproducibility**: Know exactly which views were used as input/target for each scene
2. **Analysis**: Understand which relit scenes were sampled during inference
3. **Consistency Check**: Verify that scenes with the same object prefix use the same view indices
4. **Debugging**: Trace back issues to specific view configurations
