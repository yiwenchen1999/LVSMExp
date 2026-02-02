# Flow Matching 双损失训练总结

## 核心问题

如何在训练时**同时优化渲染损失和速度损失**？

## 解决方案：联合训练（推荐）

### 关键设计原则

1. **共享计算图**：两个损失使用同一个前向传播，确保梯度正确回传
2. **顺序计算**：先计算速度损失，再用预测速度更新 latent tokens 用于渲染
3. **权重平衡**：通过归一化或固定权重平衡两个损失

### 实现流程

```
前向传播流程：
┌─────────────────────────────────────────┐
│ 1. Reconstructor: x_0 = encode(input)  │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ 2. 采样时间步: t ~ U(0,1)               │
│ 3. 线性插值: x_t = (1-t)*x_0 + t*x_1   │
│ 4. 预测速度: v_pred = v_θ(x_t, t, env) │
│ 5. 速度损失: L_v = ||v_pred - (x_1-x_0)||² │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ 6. 更新 latent: x_updated = x_t + (1-t)*v_pred │
│ 7. 渲染图像: render = renderer(x_updated) │
│ 8. 渲染损失: L_r = L2 + LPIPS + Perceptual │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ 9. 总损失: L_total = λ_v*L_v + λ_r*L_r │
│ 10. 反向传播: backward(L_total)        │
└─────────────────────────────────────────┘
```

### 代码关键点

```python
# 训练模式
if self.training:
    # 1. 获取目标 latent tokens
    target_latent_tokens = self.reconstructor(target_input)
    
    # 2. 采样时间步并插值
    t = torch.rand(batch_size)
    x_t = (1-t)*latent_tokens + t*target_latent_tokens
    
    # 3. 预测速度场
    v_pred = self.transformer_editor(x_t, env_tokens, t)
    
    # 4. 计算速度损失
    v_target = target_latent_tokens - latent_tokens
    velocity_loss = F.mse_loss(v_pred, v_target)
    
    # 5. 使用预测速度更新（关键：保持梯度）
    dt = (1 - t).unsqueeze(-1).unsqueeze(-1)
    latent_tokens = x_t + dt * v_pred  # 这里 v_pred 有梯度！
    
# 6. 渲染（使用更新后的 latent tokens）
rendered_images = self.renderer(latent_tokens, target)

# 7. 计算渲染损失
render_loss = self.loss_computer(rendered_images, target_images)

# 8. 组合损失
total_loss = render_loss.loss + velocity_weight * velocity_loss
```

## 权重平衡策略

### 策略 1：固定权重（最简单）

```yaml
flow_matching:
  velocity_loss_weight: 1.0
```

```python
total_loss = render_loss + 1.0 * velocity_loss
```

### 策略 2：损失归一化（推荐）

```python
# 归一化到相似尺度
with torch.no_grad():
    norm_factor = render_loss.item() / (velocity_loss.item() + 1e-8)
    velocity_weight = 1.0 * norm_factor

total_loss = render_loss + velocity_weight * velocity_loss
```

### 策略 3：学习率调度

```python
# 训练早期：更重视速度损失
# 训练后期：更重视渲染损失
if step < warmup_steps:
    velocity_weight = 2.0
else:
    velocity_weight = 0.5
```

## 配置示例

```yaml
model:
  transformer:
    editor:
      use_flow_matching: true
      flow_matching:
        velocity_loss_weight: 1.0
        normalize_losses: true  # 启用损失归一化
        time_sampling: "uniform"  # 时间步采样策略
```

## 监控指标

训练时应该记录：
- `velocity_loss`: 速度场预测误差
- `render_loss`: 渲染损失（L2 + LPIPS + Perceptual）
- `total_loss`: 总损失
- `velocity_loss_ratio`: 速度损失占总损失的比例
- `psnr`: 渲染质量

## 常见问题

### Q1: 两个损失应该使用相同的权重吗？

**A**: 不一定。建议：
- 开始时使用 `velocity_loss_weight = 1.0`
- 如果渲染质量差，增加渲染损失的权重
- 如果编辑效果不明显，增加速度损失的权重
- 使用 `normalize_losses: true` 自动平衡

### Q2: 速度损失和渲染损失会冲突吗？

**A**: 理论上可能，但实践中通常相互促进：
- 速度损失确保学习正确的流
- 渲染损失确保最终输出质量
- 两者共享同一个计算图，梯度会协调优化

### Q3: 需要分别反向传播吗？

**A**: 不需要。使用联合训练时，一次 `backward()` 即可：
```python
total_loss = render_loss + velocity_weight * velocity_loss
total_loss.backward()  # 自动处理两个损失的梯度
```

### Q4: 如何判断权重是否合适？

**A**: 观察：
- 两个损失的数值应该在相似量级（归一化后）
- `velocity_loss_ratio` 应该在 0.2-0.8 之间
- 如果某个损失始终很小，可能需要调整权重

## 实施建议

1. **第一步**：实现基本的联合训练（固定权重 1.0）
2. **第二步**：添加损失归一化，观察训练稳定性
3. **第三步**：根据训练效果调整权重
4. **第四步**：尝试不同的时间步采样策略

## 参考文件

- `FLOW_MATCHING_LOSS_STRATEGY.md`: 详细的训练策略
- `model/flow_matching_forward_example.py`: 完整的代码实现示例
- `model/flow_matching_editor.py`: Flow Matching 编辑器实现




