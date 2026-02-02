"""
Flow Matching 双损失训练的 forward 方法实现示例
展示如何同时计算速度损失和渲染损失
"""

import torch
import torch.nn.functional as F
from easydict import EasyDict as edict


def forward_with_flow_matching(self, data_batch, has_target_image=True):
    """
    修改后的 forward 方法，支持 Flow Matching 双损失训练
    
    关键点：
    1. 在训练时同时计算速度损失和渲染损失
    2. 使用预测的速度更新 latent tokens 用于渲染
    3. 两个损失共享同一个计算图，确保梯度正确回传
    4. 支持损失归一化和权重平衡
    """
    #& Step 1: Data preprocessing
    input, target = self.process_data(
        data_batch, 
        has_target_image=has_target_image, 
        target_has_input=self.config.training.target_has_input, 
        compute_rays=True
    )
    
    #& Step 2: Reconstructor - Get scene latent_tokens from input images
    latent_tokens, n_patches, d = self.reconstructor(input)
    
    #& Step 3: Flow Matching Editor
    flow_loss = None
    velocity_loss = None
    editor_config = self.config.model.transformer.get("editor", {})
    use_flow_matching = editor_config.get("use_flow_matching", False)
    flow_config = editor_config.get("flow_matching", {})
    
    if use_flow_matching and hasattr(input, 'env_ldr') and hasattr(input, 'env_hdr') and hasattr(input, 'env_dir'):
        if input.env_ldr is not None and input.env_hdr is not None and input.env_dir is not None:
            # ... 环境贴图处理代码（与原来相同）...
            b, v_input = input.env_dir.shape[:2]
            env_h, env_w = input.env_dir.shape[3], input.env_dir.shape[4]
            
            # 处理环境贴图（简化版，实际需要完整的处理逻辑）
            # ... 环境贴图 resize 和 concatenation ...
            directional_env = torch.cat([env_ldr_resized, env_hdr_resized, env_dir], dim=2)
            env_tokens = self.env_tokenizer(directional_env)
            _, n_env_patches, _ = env_tokens.size()
            env_tokens = env_tokens.reshape(b, v_input * n_env_patches, d)
            
            if self.training:
                # ========== 训练模式：同时计算速度和渲染损失 ==========
                
                # 1. 获取目标 latent tokens
                # 选项 A：从 relit images 重建（如果有）
                if hasattr(target, 'relit_images') and target.relit_images is not None:
                    target_input = edict(
                        image=target.relit_images,
                        ray_o=target.ray_o,
                        ray_d=target.ray_d
                    )
                    target_latent_tokens, _, _ = self.reconstructor(target_input)
                else:
                    # 选项 B：自监督 - 使用当前 latent tokens 作为目标
                    # 这会让模型学习保持 identity，适合 fine-tuning
                    target_latent_tokens = latent_tokens.clone()
                
                # 2. 采样时间步
                time_sampling = flow_config.get("time_sampling", "uniform")
                if time_sampling == "uniform":
                    t = torch.rand(b, device=latent_tokens.device)
                elif time_sampling == "early_bias":
                    # 偏向早期时间步（更关注初始编辑阶段）
                    t = torch.rand(b, device=latent_tokens.device) ** 2
                elif time_sampling == "late_bias":
                    # 偏向后期时间步（更关注最终质量）
                    t = 1 - (1 - torch.rand(b, device=latent_tokens.device)) ** 2
                elif time_sampling == "mixed":
                    # 混合采样
                    if torch.rand(1).item() < 0.5:
                        t = torch.rand(b, device=latent_tokens.device)
                    else:
                        t = torch.rand(b, device=latent_tokens.device) ** 2
                else:
                    t = torch.rand(b, device=latent_tokens.device)
                
                # 3. 线性插值：x_t = (1-t) * x_0 + t * x_1
                t_expanded = t.unsqueeze(-1).unsqueeze(-1)  # [b, 1, 1]
                x_t = (1 - t_expanded) * latent_tokens + t_expanded * target_latent_tokens
                
                # 4. 预测速度场
                # 注意：这里假设 transformer_editor 是 FlowMatchingEditor
                # 如果是原始编辑器，需要修改为支持时间条件
                v_pred = self.transformer_editor(x_t, env_tokens, t, return_velocity=True)
                
                # 5. 计算速度损失
                v_target = target_latent_tokens - latent_tokens  # [b, n_latent, d]
                velocity_loss = F.mse_loss(v_pred, v_target)
                flow_loss = velocity_loss  # 用于记录
                
                # 6. 使用预测速度更新 latent tokens（用于渲染）
                # 关键：这里使用预测的速度，确保渲染损失能够影响速度场学习
                dt = (1 - t).unsqueeze(-1).unsqueeze(-1)  # 剩余时间 [b, 1, 1]
                latent_tokens = x_t + dt * v_pred
                
            else:
                # ========== 推理模式：ODE 求解 ==========
                from .flow_matching_editor import solve_ode_euler, solve_ode_rk4
                
                num_ode_steps = flow_config.get("num_ode_steps", 10)
                ode_solver = flow_config.get("ode_solver", "euler")
                
                def velocity_fn(x, t_val):
                    """ODE 速度场函数"""
                    t_batch = torch.full((b,), t_val, device=x.device, dtype=x.dtype)
                    return self.transformer_editor(x, env_tokens, t_batch, return_velocity=True)
                
                if ode_solver == "rk4":
                    latent_tokens = solve_ode_rk4(
                        velocity_fn, latent_tokens, 
                        t_start=0.0, t_end=1.0, num_steps=num_ode_steps
                    )
                else:
                    latent_tokens = solve_ode_euler(
                        velocity_fn, latent_tokens,
                        t_start=0.0, t_end=1.0, num_steps=num_ode_steps
                    )
    
    #& Step 4: Renderer - Decode results from target ray maps
    rendered_images = self.renderer(latent_tokens, target, n_patches, d)
    
    #& Step 5: Compute losses
    if has_target_image:
        # 5.1 计算渲染损失
        if hasattr(target, 'relit_images') and target.relit_images is not None:
            target_images = target.relit_images
        else:
            target_images = target.image
        
        loss_metrics = self.loss_computer(rendered_images, target_images)
        
        # 5.2 添加 Flow Matching 损失（如果使用）
        if use_flow_matching and flow_loss is not None:
            velocity_weight = flow_config.get("velocity_loss_weight", 1.0)
            normalize_losses = flow_config.get("normalize_losses", False)
            
            if normalize_losses:
                # 损失归一化：确保两个损失在相似尺度
                with torch.no_grad():
                    render_loss_val = loss_metrics.loss.item()
                    flow_loss_val = flow_loss.item()
                    
                    if flow_loss_val > 1e-8:
                        # 计算归一化因子
                        norm_factor = render_loss_val / flow_loss_val
                        # 应用归一化（保持梯度）
                        velocity_weight = velocity_weight * norm_factor
            
            # 记录各个损失
            loss_metrics.flow_loss = flow_loss
            loss_metrics.velocity_loss = velocity_loss if velocity_loss is not None else flow_loss
            
            # 总损失 = 渲染损失 + 速度损失
            loss_metrics.loss = loss_metrics.loss + velocity_weight * flow_loss
            
            # 记录损失比例（用于监控）
            with torch.no_grad():
                loss_metrics.velocity_loss_ratio = (velocity_weight * flow_loss).item() / loss_metrics.loss.item()
                loss_metrics.render_loss_ratio = loss_metrics.loss.item() - (velocity_weight * flow_loss).item()
    else:
        loss_metrics = None
    
    #& Step 6: Return results
    result = edict(
        input=input,
        target=target,
        loss_metrics=loss_metrics,
        render=rendered_images
    )
    
    return result


# ========== 替代方案：分离梯度流 ==========

def forward_with_separate_gradients(self, data_batch, has_target_image=True):
    """
    替代方案：如果希望速度损失和渲染损失有不同的梯度流
    
    注意：这种方法需要两次前向传播，计算成本更高
    """
    # ... 前面的代码相同 ...
    
    if self.training and use_flow_matching:
        # 第一次前向：计算速度损失（只更新 editor）
        x_t = (1 - t_expanded) * latent_tokens + t_expanded * target_latent_tokens
        v_pred = self.transformer_editor(x_t, env_tokens, t, return_velocity=True)
        v_target = target_latent_tokens - latent_tokens
        velocity_loss = F.mse_loss(v_pred, v_target)
        
        # 第二次前向：计算渲染损失（更新整个 pipeline）
        # 使用 detach 来分离计算图
        with torch.no_grad():
            dt = (1 - t).unsqueeze(-1).unsqueeze(-1)
            x_updated = x_t + dt * v_pred.detach()
        
        # 重新计算用于渲染（这次需要梯度）
        latent_tokens_for_render = x_t + dt * v_pred
        rendered_images = self.renderer(latent_tokens_for_render, target, n_patches, d)
        
        # 计算损失
        loss_metrics = self.loss_computer(rendered_images, target_images)
        
        # 分别反向传播（需要在训练循环中处理）
        # 这里只返回损失，实际的反向传播在训练循环中
        loss_metrics.velocity_loss = velocity_loss
        loss_metrics.render_loss = loss_metrics.loss
        loss_metrics.loss = None  # 需要在训练循环中组合
    
    return result


# ========== 训练循环中的使用示例 ==========

def training_step_example(model, data_batch, optimizer, scaler):
    """
    训练循环示例：展示如何处理双损失
    """
    # 前向传播
    result = model(data_batch, has_target_image=True)
    loss_metrics = result.loss_metrics
    
    # 获取损失
    total_loss = loss_metrics.loss
    
    # 反向传播（自动处理两个损失的梯度）
    scaler.scale(total_loss).backward()
    
    # 可选：分别监控两个损失
    if hasattr(loss_metrics, 'velocity_loss'):
        print(f"Velocity Loss: {loss_metrics.velocity_loss.item():.4f}")
        print(f"Render Loss: {loss_metrics.render_loss.item():.4f}")
        print(f"Velocity Ratio: {loss_metrics.velocity_loss_ratio:.4f}")
    
    # 更新参数
    scaler.step(optimizer)
    scaler.update()
    
    return loss_metrics




