# Flow Matching 双损失训练策略

## 问题分析

在 Flow Matching 训练中，我们需要同时优化两个损失：

1. **速度损失（Velocity Loss）**：确保模型能够准确预测速度场 `v_θ(x_t, t, env_tokens)`
2. **渲染损失（Rendering Loss）**：确保最终渲染的图像质量

这两个损失有不同的目标：
- 速度损失：在 latent 空间学习正确的流
- 渲染损失：在像素空间确保最终输出质量

## 训练策略

### 策略 1：联合训练（Joint Training）- 推荐

**核心思想**：在同一个前向传播中同时计算两个损失，加权求和。

**优点**：
- 简单直接
- 两个损失可以相互促进
- 训练稳定

**实现**：
```python
# 伪代码
total_loss = λ_velocity * velocity_loss + λ_render * render_loss
```

### 策略 2：两阶段训练（Two-Stage Training）

**阶段 1**：只训练速度损失，学习正确的流
**阶段 2**：固定速度场，fine-tune 渲染质量

**优点**：
- 可以分别优化两个目标
- 避免早期训练时渲染损失干扰流学习

**缺点**：
- 需要两个训练阶段
- 可能过度拟合速度损失

### 策略 3：交替训练（Alternating Training）

**核心思想**：交替优化两个损失

**优点**：
- 可以动态调整重点
- 避免一个损失主导另一个

**缺点**：
- 实现复杂
- 训练可能不稳定

### 策略 4：自适应权重（Adaptive Weighting）

**核心思想**：根据训练阶段或损失大小动态调整权重

**优点**：
- 自动平衡两个损失
- 适应训练过程

**缺点**：
- 需要额外的超参数
- 可能不稳定

## 推荐实现：联合训练 + 自适应权重

### 1. 损失计算流程

```
前向传播：
1. 采样时间步 t ~ U(0,1)
2. 计算 x_t = (1-t)*x_0 + t*x_1
3. 预测速度 v_pred = v_θ(x_t, t, env_tokens)
4. 计算速度损失：L_velocity = ||v_pred - (x_1 - x_0)||²
5. 使用预测速度更新：x_updated = x_t + (1-t)*v_pred
6. 渲染图像：render = renderer(x_updated)
7. 计算渲染损失：L_render = L2 + LPIPS + Perceptual
8. 总损失：L_total = λ_v * L_velocity + λ_r * L_render
```

### 2. 权重平衡策略

#### 方案 A：固定权重（简单）
```yaml
flow_matching:
  velocity_loss_weight: 1.0
  render_loss_weight: 1.0
```

#### 方案 B：损失归一化（推荐）
```python
# 归一化两个损失到相似尺度
velocity_loss_norm = velocity_loss / (velocity_loss.detach() + 1e-8)
render_loss_norm = render_loss / (render_loss.detach() + 1e-8)
total_loss = velocity_loss_norm + render_loss_norm
```

#### 方案 C：梯度平衡
```python
# 确保两个损失的梯度幅度相似
if velocity_loss.item() > 0:
    velocity_grad_scale = render_loss.item() / (velocity_loss.item() + 1e-8)
    velocity_loss = velocity_loss * velocity_grad_scale
```

#### 方案 D：学习率调度
```python
# 训练早期：更重视速度损失
# 训练后期：更重视渲染损失
if step < warmup_steps:
    velocity_weight = 2.0
    render_weight = 0.5
else:
    velocity_weight = 0.5
    render_weight = 2.0
```

### 3. 关键实现细节

#### 3.1 共享梯度流

两个损失应该共享同一个前向传播，确保梯度能够正确回传：

```python
# 正确：共享计算图
x_t = (1-t)*x_0 + t*x_1
v_pred = model(x_t, t, env_tokens)
velocity_loss = ||v_pred - (x_1 - x_0)||²
x_updated = x_t + (1-t)*v_pred  # 使用预测的速度
render = renderer(x_updated)
render_loss = compute_render_loss(render, target)
total_loss = velocity_loss + render_loss
```

#### 3.2 分离计算图（可选）

如果速度损失和渲染损失需要不同的梯度流：

```python
# 速度损失：只更新 editor
v_pred = model(x_t, t, env_tokens)
velocity_loss = ||v_pred - (x_1 - x_0)||²

# 渲染损失：更新整个 pipeline
with torch.no_grad():
    x_updated = x_t + (1-t)*v_pred.detach()
render = renderer(x_updated)
render_loss = compute_render_loss(render, target)

# 分别反向传播
(velocity_loss * λ_v).backward(retain_graph=True)
(render_loss * λ_r).backward()
```

#### 3.3 时间步采样策略

不同的时间步采样策略会影响训练：

```python
# 均匀采样（简单）
t = torch.rand(batch_size)

# 偏向早期时间步（更关注初始编辑）
t = torch.rand(batch_size) ** 2

# 偏向后期时间步（更关注最终质量）
t = 1 - (1 - torch.rand(batch_size)) ** 2

# 混合采样（推荐）
if random.random() < 0.5:
    t = torch.rand(batch_size)  # 均匀
else:
    t = torch.rand(batch_size) ** 2  # 偏向早期
```

## 代码实现示例

### 完整的 forward 方法修改

```python
def forward(self, data_batch, has_target_image=True):
    # ... 前面的代码保持不变 ...
    
    # Step 2: Reconstructor
    latent_tokens, n_patches, d = self.reconstructor(input)
    
    # Step 3: Flow Matching 编辑
    flow_loss = None
    editor_config = self.config.model.transformer.get("editor", {})
    use_flow_matching = editor_config.get("use_flow_matching", False)
    
    if use_flow_matching and hasattr(input, 'env_ldr'):
        # ... 环境贴图处理 ...
        
        if self.training:
            # 训练模式：同时计算速度和渲染损失
            
            # 1. 获取目标 latent tokens
            if hasattr(target, 'relit_images') and target.relit_images is not None:
                target_input = edict(
                    image=target.relit_images,
                    ray_o=target.ray_o,
                    ray_d=target.ray_d
                )
                target_latent_tokens, _, _ = self.reconstructor(target_input)
            else:
                # 自监督：使用当前 latent tokens
                target_latent_tokens = latent_tokens
            
            # 2. 采样时间步
            b = latent_tokens.shape[0]
            t = torch.rand(b, device=latent_tokens.device)  # [b]
            
            # 3. 线性插值
            t_expanded = t.unsqueeze(-1).unsqueeze(-1)  # [b, 1, 1]
            x_t = (1 - t_expanded) * latent_tokens + t_expanded * target_latent_tokens
            
            # 4. 预测速度场
            v_pred = self.transformer_editor(x_t, env_tokens, t, return_velocity=True)
            
            # 5. 计算速度损失
            v_target = target_latent_tokens - latent_tokens  # [b, n_latent, d]
            flow_loss = F.mse_loss(v_pred, v_target)
            
            # 6. 使用预测速度更新 latent tokens（用于渲染）
            dt = (1 - t).unsqueeze(-1).unsqueeze(-1)  # 剩余时间
            latent_tokens = x_t + dt * v_pred
            
        else:
            # 推理模式：ODE 求解
            # ... ODE 求解代码 ...
    
    # Step 4: Renderer
    rendered_images = self.renderer(latent_tokens, target, n_patches, d)
    
    # Step 5: 计算损失
    if has_target_image:
        # 渲染损失
        if hasattr(target, 'relit_images') and target.relit_images is not None:
            target_images = target.relit_images
        else:
            target_images = target.image
        
        loss_metrics = self.loss_computer(rendered_images, target_images)
        
        # 添加 flow matching 损失
        if use_flow_matching and flow_loss is not None:
            flow_config = editor_config.get("flow_matching", {})
            velocity_weight = flow_config.get("velocity_loss_weight", 1.0)
            
            # 损失归一化（可选）
            if flow_config.get("normalize_losses", False):
                # 归一化到相似尺度
                with torch.no_grad():
                    render_loss_val = loss_metrics.loss.item()
                    flow_loss_val = flow_loss.item()
                    if flow_loss_val > 1e-8:
                        velocity_weight = velocity_weight * (render_loss_val / flow_loss_val)
            
            # 总损失
            loss_metrics.flow_loss = flow_loss
            loss_metrics.velocity_loss = flow_loss
            loss_metrics.loss = loss_metrics.loss + velocity_weight * flow_loss
    
    return result
```

## 超参数建议

### 初始配置
```yaml
flow_matching:
  velocity_loss_weight: 1.0
  render_loss_weight: 1.0
  normalize_losses: true  # 启用损失归一化
  time_sampling: "uniform"  # 或 "early_bias", "late_bias", "mixed"
```

### 调优策略

1. **如果渲染质量差**：
   - 增加 `render_loss_weight`
   - 减少 `velocity_loss_weight`
   - 增加渲染损失的权重（l2_loss_weight, perceptual_loss_weight）

2. **如果编辑效果不明显**：
   - 增加 `velocity_loss_weight`
   - 检查目标 latent tokens 是否正确

3. **如果训练不稳定**：
   - 启用 `normalize_losses`
   - 使用梯度裁剪
   - 降低学习率

## 监控指标

训练时应该监控：
- `velocity_loss`: 速度场预测误差
- `render_loss`: 渲染损失
- `flow_loss`: Flow matching 损失（如果单独计算）
- `total_loss`: 总损失
- `psnr`: 渲染质量指标

## 实验建议

1. **基线实验**：先只用速度损失训练，观察流是否学习正确
2. **联合训练**：添加渲染损失，观察质量提升
3. **权重调优**：尝试不同的权重组合
4. **时间步采样**：尝试不同的采样策略

