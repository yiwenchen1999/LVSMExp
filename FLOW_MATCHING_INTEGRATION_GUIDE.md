# Flow Matching 集成指南

## 修改步骤

### 1. 在主模型文件中添加 Flow Matching 支持

在 `LVSM_scene_encoder_decoder_wEditor.py` 中需要修改以下部分：

#### 1.1 导入 Flow Matching 模块

```python
from .flow_matching_editor import FlowMatchingEditor, solve_ode_euler, solve_ode_rk4
```

#### 1.2 修改 `_init_transformer()` 方法

将 `transformer_editor` 的初始化改为使用 `FlowMatchingEditor`（如果启用 flow matching）：

```python
def _init_transformer(self):
    # ... 现有代码 ...
    
    # Transformer editor block
    editor_config = config.get("editor", {})
    use_flow_matching = editor_config.get("use_flow_matching", False)
    
    if use_flow_matching:
        # 使用 Flow Matching Editor
        self.transformer_editor = FlowMatchingEditor(config)
    else:
        # 使用原始的一步编辑
        editor_n_layer = editor_config.get("n_layer", 1)
        self.transformer_editor = nn.ModuleList([
            QK_Norm_TransformerBlock(
                config.d, config.d_head, use_qk_norm=use_qk_norm
            ) for _ in range(editor_n_layer)
        ])
        # ... 初始化代码 ...
```

#### 1.3 修改 `forward()` 方法中的编辑逻辑

在 `forward()` 方法中（约第 429-517 行），将编辑逻辑改为：

```python
# 3.3: Flow Matching 编辑过程
if use_flow_matching:
    # Flow Matching 训练模式
    if self.training:
        # 随机采样时间步
        t = torch.rand(b, device=latent_tokens.device)  # [b]
        
        # 获取目标 latent tokens（如果有 ground truth）
        # 注意：这需要从数据中获取编辑后的场景 latent tokens
        # 如果只有渲染图像，可以通过 reconstructor 从 relit_images 重建
        if hasattr(target, 'relit_images') and target.relit_images is not None:
            # 从 relit images 重建目标 latent tokens
            target_latent_tokens, _, _ = self.reconstructor(
                edict(image=target.relit_images, ray_o=target.ray_o, ray_d=target.ray_d)
            )
        else:
            # 如果没有 relit images，使用当前 latent tokens 作为目标（自监督）
            target_latent_tokens = latent_tokens
        
        # 线性插值：x_t = (1-t) * x_0 + t * x_1
        x_t = (1 - t.unsqueeze(-1).unsqueeze(-1)) * latent_tokens + \
              t.unsqueeze(-1).unsqueeze(-1) * target_latent_tokens
        
        # 预测速度场
        v_pred = self.transformer_editor(x_t, env_tokens, t, return_velocity=True)
        
        # 目标速度：v_target = x_1 - x_0
        v_target = target_latent_tokens - latent_tokens
        
        # Flow matching 损失
        flow_loss = F.mse_loss(v_pred, v_target)
        
        # 使用预测的速度进行一步更新（用于渲染）
        dt = 1.0 - t.unsqueeze(-1).unsqueeze(-1)
        latent_tokens = x_t + dt * v_pred
    else:
        # 推理模式：使用 ODE 求解
        flow_config = editor_config.get("flow_matching", {})
        num_ode_steps = flow_config.get("num_ode_steps", 10)
        ode_solver = flow_config.get("ode_solver", "euler")
        
        # 定义速度场函数
        def velocity_fn(x, t_val):
            # 将标量时间转换为 batch 大小
            t_batch = torch.full((b,), t_val, device=x.device, dtype=x.dtype)
            return self.transformer_editor(x, env_tokens, t_batch, return_velocity=True)
        
        # ODE 求解
        if ode_solver == "rk4":
            latent_tokens = solve_ode_rk4(velocity_fn, latent_tokens, num_steps=num_ode_steps)
        else:
            latent_tokens = solve_ode_euler(velocity_fn, latent_tokens, num_steps=num_ode_steps)
else:
    # 原始的一步编辑逻辑
    # ... 保持现有代码不变 ...
```

#### 1.4 修改损失计算

在损失计算部分（约第 532 行），添加 flow matching 损失：

```python
if has_target_image:
    # ... 现有渲染损失计算 ...
    
    # 添加 flow matching 损失（如果使用）
    if use_flow_matching and self.training:
        loss_metrics.flow_loss = flow_loss
        loss_metrics.loss = loss_metrics.loss + flow_config.get("flow_loss_weight", 1.0) * flow_loss
```

#### 1.5 修改 `edit_scene_with_env()` 方法

将 `edit_scene_with_env()` 方法改为使用 flow matching：

```python
def edit_scene_with_env(self, latent_tokens, input_with_env):
    """使用 Flow Matching 编辑场景 latent tokens"""
    edited_latent_tokens = latent_tokens
    
    if hasattr(input_with_env, 'env_ldr') and hasattr(input_with_env, 'env_hdr') and hasattr(input_with_env, 'env_dir'):
        if input_with_env.env_ldr is not None and input_with_env.env_hdr is not None and input_with_env.env_dir is not None:
            # ... 环境贴图处理代码（保持不变）...
            
            # 检查是否使用 flow matching
            editor_config = self.config.model.transformer.get("editor", {})
            use_flow_matching = editor_config.get("use_flow_matching", False)
            
            if use_flow_matching:
                # Flow Matching 推理
                flow_config = editor_config.get("flow_matching", {})
                num_ode_steps = flow_config.get("num_ode_steps", 10)
                ode_solver = flow_config.get("ode_solver", "euler")
                
                b = latent_tokens.shape[0]
                
                def velocity_fn(x, t_val):
                    t_batch = torch.full((b,), t_val, device=x.device, dtype=x.dtype)
                    return self.transformer_editor(x, env_tokens, t_batch, return_velocity=True)
                
                if ode_solver == "rk4":
                    edited_latent_tokens = solve_ode_rk4(velocity_fn, latent_tokens, num_steps=num_ode_steps)
                else:
                    edited_latent_tokens = solve_ode_euler(velocity_fn, latent_tokens, num_steps=num_ode_steps)
            else:
                # 原始的一步编辑
                # ... 保持现有代码 ...
    
    return edited_latent_tokens
```

### 2. 配置文件修改

在配置文件中添加 flow matching 相关参数：

```yaml
model:
  transformer:
    editor:
      use_flow_matching: true  # 启用 flow matching
      flow_matching:
        num_ode_steps: 10  # 推理时的 ODE 步数
        ode_solver: "euler"  # 或 "rk4"
        time_embed_dim: 768  # 时间嵌入维度
        use_ada_ln: false  # 是否使用 AdaLN（更复杂但可能更有效）
        flow_loss_weight: 1.0  # Flow matching 损失权重
```

### 3. 训练注意事项

1. **数据要求**：
   - 训练时需要成对的（原始场景，编辑后场景）
   - 如果只有渲染图像，需要通过 reconstructor 从 relit_images 重建目标 latent tokens

2. **损失平衡**：
   - Flow matching 损失和渲染损失需要平衡
   - 建议开始时 flow_loss_weight=1.0，然后根据训练效果调整

3. **ODE 步数**：
   - 训练时不需要 ODE 求解（直接使用一步更新）
   - 推理时可以使用 5-20 步，更多步数通常更准确但更慢

### 4. 渐进式迁移策略

如果担心直接切换的风险，可以采用渐进式迁移：

1. **阶段 1**：保持原始编辑器，添加 flow matching 作为可选模式
2. **阶段 2**：同时训练两种模式，比较效果
3. **阶段 3**：完全切换到 flow matching

### 5. 潜在问题和解决方案

**问题 1**：训练时没有目标 latent tokens
- **解决**：使用自监督，将当前 latent tokens 作为目标，或者从 relit images 重建

**问题 2**：ODE 求解慢
- **解决**：使用更少的步数（5-10 步），或使用更高效的求解器

**问题 3**：训练不稳定
- **解决**：降低 flow_loss_weight，或使用梯度裁剪

## 测试建议

1. 先用小数据集测试 flow matching 是否正常工作
2. 比较一步编辑和 flow matching 的渲染质量
3. 调整 ODE 步数，找到质量和速度的平衡点
4. 监控训练损失，确保 flow matching 损失正常下降


