# Flow Matching 实现方案

## 核心思想

将当前的一步编辑过程改为流匹配（Flow Matching）过程：

**当前实现**：
- 输入：`latent_tokens` + `env_tokens`
- 输出：编辑后的 `latent_tokens`
- 过程：单次 transformer_editor 前向传播

**Flow Matching 实现**：
- 定义从 `x_0`（原始 latent_tokens）到 `x_1`（编辑后的 latent_tokens）的连续流
- 学习速度场 `v_θ(x_t, t, env_tokens)`，预测在时间 t 和条件 env_tokens 下的速度
- 通过 ODE 积分从 t=0 到 t=1 生成编辑后的结果

## 实现步骤

### 1. 添加时间嵌入（Time Embedding）

需要为 transformer_editor 添加时间条件输入。

### 2. 修改 Transformer Editor 架构

- 添加时间嵌入层
- 修改 transformer blocks 以融合时间信息（可以通过 AdaLN 或简单的加法）

### 3. 实现 Flow Matching 核心逻辑

#### 训练阶段：
- 随机采样时间步 `t ~ U(0,1)`
- 线性插值：`x_t = (1-t) * x_0 + t * x_1`
- 计算目标速度：`v_target = x_1 - x_0`
- 预测速度：`v_pred = v_θ(x_t, t, env_tokens)`
- 损失：`L = ||v_pred - v_target||^2`

#### 推理阶段：
- 使用 ODE 求解器从 t=0 积分到 t=1
- 初始条件：`x(0) = x_0`
- ODE：`dx/dt = v_θ(x(t), t, env_tokens)`
- 结果：`x(1)` 即为编辑后的 latent_tokens

### 4. 修改损失函数

在训练时，需要同时计算：
- Flow matching 损失（速度场预测）
- 渲染损失（最终渲染结果）

## 具体实现建议

### 方案 A：简单实现（推荐开始）

1. **时间嵌入**：使用简单的 MLP 将标量时间 t 映射到与 latent tokens 相同的维度
2. **速度场网络**：修改 transformer_editor，使其输出速度场而不是直接输出编辑后的 tokens
3. **训练**：使用简单的 flow matching 损失
4. **推理**：使用 Euler 方法进行 ODE 积分（简单但有效）

### 方案 B：完整实现

1. **时间嵌入**：使用正弦位置编码或更复杂的嵌入
2. **条件注入**：通过 AdaLN（Adaptive Layer Normalization）将时间和环境条件注入到每个 transformer block
3. **训练**：使用条件流匹配（Conditional Flow Matching）
4. **推理**：使用更高级的 ODE 求解器（如 Runge-Kutta 4 或 Dormand-Prince）

## 代码修改点

1. `_init_transformer()`: 添加时间嵌入层
2. `transformer_editor`: 修改为接受时间条件
3. `forward()`: 实现 flow matching 训练逻辑
4. `edit_scene_with_env()`: 实现 flow matching 推理逻辑（ODE 积分）
5. 损失计算：添加 flow matching 损失项

## 配置参数

需要在 config 中添加：
```yaml
editor:
  use_flow_matching: true
  flow_matching:
    num_ode_steps: 10  # 推理时的 ODE 步数
    ode_solver: "euler"  # 或 "rk4"
    time_embed_dim: 768  # 时间嵌入维度
```




