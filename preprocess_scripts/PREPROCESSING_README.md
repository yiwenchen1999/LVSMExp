# Objaverse Data Preprocessing

This directory contains scripts to preprocess Objaverse data into the format expected by the LVSM model.

## Scripts

### 1. `preprocess_objaverse.py`

Converts Objaverse data format to re10k format.

**Usage:**
```bash
python preprocess_objaverse.py \
    --input data_samples/sample_objaverse \
    --output data_samples/objaverse_processed \
    --split test
```

**Arguments:**
- `--input, -i`: Input directory containing objaverse data (e.g., `data_samples/sample_objaverse`)
- `--output, -o`: Output directory for processed data (e.g., `data_samples/objaverse_processed`)
- `--split, -s`: Split to process (`train` or `test`, default: `test`)
- `--object-id`: Process specific object ID only (optional, processes all by default)

**What it does:**
- Reads `cameras.json` from each object's test/train folder
- Converts Blender c2w matrices to OpenCV convention (transforms rotation by `[1,0,0][0,-1,0][0,0,-1]`)
- Converts FOV to fxfycxcy intrinsic parameters
- Processes each env folder (env_0, env_1, ..., env_4, white_env_0) as a separate scene
- Creates JSON files in re10k format with scene_name and frames array
- Copies images to organized structure: `images/{scene_name}/`
- Generates `full_list.txt` listing all scene JSON files

**Output structure:**
```
objaverse_processed/
├── test/
│   ├── full_list.txt
│   ├── metadata/
│   │   ├── {object_id}_env_0.json
│   │   ├── {object_id}_env_1.json
│   │   └── ...
│   └── images/
│       ├── {object_id}_env_0/
│       │   ├── 00000.png
│       │   ├── 00001.png
│       │   └── ...
│       └── ...
```

### 2. `create_evaluation_index.py`

Creates evaluation index JSON files for inference, specifying which frames to use as input and target.

**Usage:**
```bash
python create_evaluation_index.py \
    --full-list data_samples/objaverse_processed/test/full_list.txt \
    --output data/evaluation_index_objaverse.json \
    --n-input 2 \
    --n-target 6 \
    --seed 42
```

**Arguments:**
- `--full-list, -f`: Path to full_list.txt file
- `--output, -o`: Output path for evaluation index JSON
- `--n-input`: Number of input frames (default: 2)
- `--n-target`: Number of target frames (default: 6)
- `--max-window-size`: Maximum window size for sampling (default: 4*(n_input+n_target))
- `--seed`: Random seed for reproducibility (default: 42)

**What it does:**
- Reads all scene JSON files from full_list.txt
- For each scene, randomly samples input and target frame indices
- Ensures the sampling window is smaller than `max_window_size` (default: 4*(n_input+n_target))
- Creates JSON file in format:
  ```json
  {
    "scene_name": {
      "context": [input_frame_indices],
      "target": [target_frame_indices]
    }
  }
  ```

**Output format:**
```json
{
  "01c9013483b6427fbc2f478e5e328810_env_0": {
    "context": [10, 45],
    "target": [15, 20, 25, 30, 35, 40]
  },
  ...
}
```

## Example Workflow

1. **Preprocess Objaverse data:**
   ```bash
   python preprocess_objaverse.py \
       --input data_samples/sample_objaverse \
       --output data_samples/objaverse_processed \
       --split test
   ```

2. **Create evaluation index:**
   ```bash
   python create_evaluation_index.py \
       --full-list data_samples/objaverse_processed/test/full_list.txt \
       --output data/evaluation_index_objaverse.json \
       --n-input 2 \
       --n-target 6
   ```

3. **Use in inference:**
   Update your config file to point to the new data:
   ```yaml
   training:
     dataset_path: ./data_samples/objaverse_processed/test/full_list.txt
   
   inference:
     view_idx_file_path: ./data/evaluation_index_objaverse.json
   ```

## Coordinate System Conversion

The script converts Blender camera convention to OpenCV convention:
- **Blender**: +X right, +Y forward, +Z up
- **OpenCV**: +X right, +Y down, +Z forward

The rotation matrix is transformed by multiplying from the right with:
```
[1,  0,  0]
[0, -1,  0]
[0,  0, -1]
```

The translation vector remains unchanged.

