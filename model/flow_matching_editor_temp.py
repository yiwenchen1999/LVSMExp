"""
Flow Matching Editor Implementation
将环境贴图条件编辑从一步修改改为流匹配过程
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class TimeEmbedding(nn.Module):
    """时间嵌入层，将标量时间 t 映射到特征空间"""
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        
        # 创建频率维度
        half_dim = dim // 2
        emb = torch.log(torch.tensor(max_period)) / (half_dim - 1)
        self.register_buffer('freq', torch.exp(-emb * torch.arange(half_dim)))
        
        # 可学习的投影层
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, t):
        """
        Args:
            t: [b] 或 [b, 1] - 时间步，范围 [0, 1]
        Returns:
            [b, dim] - 时间嵌入
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        
        # 正弦位置编码
        t = t * self.max_period
        emb = t * self.freq.unsqueeze(0)  # [b, half_dim]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # [b, dim]
        
        # 可学习投影
        emb = self.proj(emb)
        return emb


class ConditionalTransformerBlock(nn.Module):
    """支持条件注入的 Transformer Block"""
    def __init__(self, dim, head_dim, use_qk_norm=True, use_ada_ln=False):
        super().__init__()
        from .transformer import QK_Norm_SelfAttention, MLP
        
        self.use_ada_ln = use_ada_ln
        self.norm1 = nn.LayerNorm(dim, bias=False)
        self.attn = QK_Norm_SelfAttention(
            dim=dim,
            head_dim=head_dim,
            use_qk_norm=use_qk_norm,
        )
        
        self.norm2 = nn.LayerNorm(dim, bias=False)
        self.mlp = MLP(dim=dim, mlp_ratio=4, bias=False)
        
        # 如果使用 AdaLN，添加条件投影
        if use_ada_ln:
            self.ada_ln1 = nn.Sequential(
                nn.SiLU(),
                nn.Linear(dim, dim * 2)  # 输出 scale 和 shift
            )
            self.ada_ln2 = nn.Sequential(
                nn.SiLU(),
                nn.Linear(dim, dim * 2)
            )
    
    def forward(self, x, condition=None):
        """
        Args:
            x: [b, n, d] - 输入 tokens
            condition: [b, d] - 条件嵌入（时间 + 环境条件）
        """
        # Self-attention
        if self.use_ada_ln and condition is not None:
            # AdaLN: 使用条件来调制 LayerNorm
            ada_params = self.ada_ln1(condition)  # [b, 2*d]
            scale, shift = ada_params.chunk(2, dim=-1)  # [b, d], [b, d]
            x_norm = self.norm1(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        else:
            x_norm = self.norm1(x)
            if condition is not None:
                # 简单加法注入条件
                x_norm = x_norm + condition.unsqueeze(1)
        
        x = x + self.attn(x_norm)
        
        # MLP
        if self.use_ada_ln and condition is not None:
            ada_params = self.ada_ln2(condition)
            scale, shift = ada_params.chunk(2, dim=-1)
            x_norm = self.norm2(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        else:
            x_norm = self.norm2(x)
            if condition is not None:
                x_norm = x_norm + condition.unsqueeze(1)
        
        x = x + self.mlp(x_norm)
        return x


class FlowMatchingEditor(nn.Module):
    """基于 Flow Matching 的编辑器"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        d = config.model.transformer.d
        d_head = config.model.transformer.d_head
        use_qk_norm = config.model.transformer.get("use_qk_norm", False)
        
        editor_config = config.model.transformer.get("editor", {})
        flow_config = editor_config.get("flow_matching", {})
        
        # 时间嵌入
        time_embed_dim = flow_config.get("time_embed_dim", d)
        self.time_embed = TimeEmbedding(time_embed_dim)
        
        # 条件融合（时间 + 环境条件）
        self.condition_proj = nn.Linear(d + time_embed_dim, d)
        
        # Transformer blocks（支持条件注入）
        use_ada_ln = flow_config.get("use_ada_ln", False)
        n_layer = editor_config.get("n_layer", 1)
        self.transformer_blocks = nn.ModuleList([
            ConditionalTransformerBlock(
                dim=d,
                head_dim=d_head,
                use_qk_norm=use_qk_norm,
                use_ada_ln=use_ada_ln
            ) for _ in range(n_layer)
        ])
        
        # 输出投影（预测速度场）
        self.output_proj = nn.Linear(d, d)
    
    def forward(self, latent_tokens, env_tokens, t, return_velocity=True):
        """
        Flow Matching 前向传播
        
        Args:
            latent_tokens: [b, n_latent, d] - 当前状态的 latent tokens
            env_tokens: [b, n_env, d] - 环境条件 tokens
            t: [b] - 时间步，范围 [0, 1]
            return_velocity: bool - 如果 True，返回速度场；如果 False，返回编辑后的 tokens
            
        Returns:
            如果 return_velocity=True: [b, n_latent, d] - 预测的速度场
            如果 return_velocity=False: [b, n_latent, d] - 编辑后的 tokens
        """
        b, n_latent, d = latent_tokens.shape
        
        # 时间嵌入
        time_emb = self.time_embed(t)  # [b, time_embed_dim]
        
        # 环境条件聚合（平均池化）
        env_cond = env_tokens.mean(dim=1)  # [b, d]
        
        # 融合时间和环境条件
        condition = torch.cat([time_emb, env_cond], dim=-1)  # [b, d + time_embed_dim]
        condition = self.condition_proj(condition)  # [b, d]
        
        # 拼接 latent 和 env tokens
        input_tokens = torch.cat([latent_tokens, env_tokens], dim=1)  # [b, n_latent + n_env, d]
        
        # 通过 transformer blocks
        for block in self.transformer_blocks:
            input_tokens = block(input_tokens, condition=condition)
        
        # 提取 latent tokens 部分
        latent_output = input_tokens[:, :n_latent, :]  # [b, n_latent, d]
        
        # 输出投影
        output = self.output_proj(latent_output)  # [b, n_latent, d]
        
        if return_velocity:
            # 返回速度场（用于 flow matching 训练）
            return output
        else:
            # 返回编辑后的 tokens（用于一步推理，不推荐）
            return latent_tokens + output


def solve_ode_euler(velocity_fn, x0, t_start=0.0, t_end=1.0, num_steps=10):
    """
    使用 Euler 方法求解 ODE: dx/dt = v(x(t), t)
    
    Args:
        velocity_fn: 函数，接受 (x, t) 返回速度 v(x, t)
        x0: [b, n, d] - 初始状态
        t_start: 起始时间
        t_end: 结束时间
        num_steps: ODE 步数
        
    Returns:
        [b, n, d] - 最终状态
    """
    dt = (t_end - t_start) / num_steps
    x = x0
    t = t_start
    
    for _ in range(num_steps):
        v = velocity_fn(x, t)
        x = x + dt * v
        t = t + dt
    
    return x


def solve_ode_rk4(velocity_fn, x0, t_start=0.0, t_end=1.0, num_steps=10):
    """
    使用 Runge-Kutta 4 方法求解 ODE（更精确但更慢）
    """
    dt = (t_end - t_start) / num_steps
    x = x0
    t = t_start
    
    for _ in range(num_steps):
        k1 = velocity_fn(x, t)
        k2 = velocity_fn(x + dt * k1 / 2, t + dt / 2)
        k3 = velocity_fn(x + dt * k2 / 2, t + dt / 2)
        k4 = velocity_fn(x + dt * k3, t + dt)
        
        x = x + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
        t = t + dt
    
    return x

