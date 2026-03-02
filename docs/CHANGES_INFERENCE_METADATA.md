# Summary of Changes - Inference Metadata Recording

## 概述

成功添加了推理元数据记录功能，现在在推理完成后会自动生成包含以下信息的 JSON 文件：
- Context (输入) 视图索引
- Target (目标) 视图索引
- Relit scene 名称（用于监督的重光照场景）

## 修改的文件

### 1. `data/dataset_scene.py` ✅
**修改内容**：
- 第 688 行：添加 `relit_scene_name` 到返回的字典中

**作用**：
- 在数据加载时记录使用的 relit scene 名称
- 这个信息会传递到推理结果中

### 2. `utils/metric_utils.py` ✅
**修改内容**：
- 导入 `json` 模块（已存在）
- 修改 `export_results()` 函数（第 118-182 行）
  - 提取输入和目标视图索引
  - 提取 relit scene 名称
  - 为每个场景生成 `metadata.json` 文件

**生成的文件**：
```
<output_dir>/<scene_name>/metadata.json
```

**JSON 格式**：
```json
{
  "scene_name": "wooden_table_02_env_0",
  "context_view_indices": [43, 47, 49, 52],
  "target_view_indices": [62, 63, 72, 74, 79, 84, 86, 89],
  "relit_scene_name": "wooden_table_02_env_2"
}
```

### 3. `inference_editor.py` ✅
**修改内容**：
- 第 5 行：导入 `json` 模块
- 第 134-158 行：添加元数据汇总生成逻辑
  - 收集所有 `metadata.json` 文件
  - 生成合并的 `inference_metadata_summary.json`
  - 打印统计信息

**生成的文件**：
```
<output_dir>/inference_metadata_summary.json
```

**JSON 格式**：
```json
{
  "scene_name_1": {
    "scene_name": "...",
    "context_view_indices": [...],
    "target_view_indices": [...],
    "relit_scene_name": "..."
  },
  "scene_name_2": { ... },
  ...
}
```

## 新增的工具

### 4. `utils/analyze_inference_metadata.py` ✅
**功能**：
- 分析推理元数据摘要
- 检查视图索引一致性
- 统计 relit scene 分布
- 显示示例数据

**使用方法**：
```bash
python utils/analyze_inference_metadata.py experiments/evaluation/polyhaven_dense_inference/inference_metadata_summary.json
```

### 5. `docs/INFERENCE_METADATA.md` ✅
**内容**：
- 完整的功能文档
- 输出文件格式说明
- 使用示例
- 修改说明

## 使用方式

### 运行推理
```bash
# 使用现有的推理脚本，无需任何修改
bash bash_scripts/Sony_clusters/interactive_inference_polyhaven_dense.sh
```

### 查看元数据
```bash
# 查看合并的元数据摘要
cat experiments/evaluation/polyhaven_dense_inference/inference_metadata_summary.json

# 分析元数据
python utils/analyze_inference_metadata.py experiments/evaluation/polyhaven_dense_inference/inference_metadata_summary.json

# 查看单个场景的元数据
cat experiments/evaluation/polyhaven_dense_inference/wooden_table_02_env_0/metadata.json
```

## 输出结构

```
experiments/evaluation/polyhaven_dense_inference/
├── inference_metadata_summary.json          # 汇总文件（新增）
├── metrics_summary.json                      # 原有的指标摘要
├── wooden_table_02_env_0/
│   ├── metadata.json                        # 场景元数据（新增）
│   ├── input_0.png
│   ├── render_0.png
│   └── ...
├── wooden_table_02_env_1/
│   ├── metadata.json                        # 场景元数据（新增）
│   └── ...
└── ...
```

## 特性

### ✅ 自动生成
- 无需手动配置
- 推理时自动记录
- 与现有推理流程完全兼容

### ✅ 完整信息
- Context 视图索引
- Target 视图索引
- Relit scene 来源

### ✅ 一致性保证
- 相同对象的不同光照条件使用相同的视图索引
- 可通过分析工具验证一致性

### ✅ 易于分析
- JSON 格式，易于解析
- 提供分析工具
- 支持批量处理

## 测试

推理完成后，使用分析工具验证：

```bash
python utils/analyze_inference_metadata.py <path_to_inference_metadata_summary.json>
```

输出示例：
```
================================================================================
Inference Metadata Analysis
================================================================================
Total scenes: 48

Unique objects: 6

Checking view index consistency across lighting conditions...
✓ All objects have consistent view indices across lighting conditions!

Relit scene distribution:
  Scenes with relit supervision: 48
  Scenes without relit supervision: 0
  Unique relit scenes used: 24

Sample metadata (first 3 scenes):
...
================================================================================
```

## 注意事项

1. **合并摘要生成条件**：只有当 `compute_metrics: true` 时才会生成 `inference_metadata_summary.json`
2. **Per-scene 元数据**：每个场景的 `metadata.json` 始终生成
3. **向后兼容**：不影响现有功能和推理脚本

## 相关文件

- 📝 详细文档：`docs/INFERENCE_METADATA.md`
- 🔧 分析工具：`utils/analyze_inference_metadata.py`
- 🚀 推理脚本：`bash_scripts/Sony_clusters/interactive_inference_*.sh`
