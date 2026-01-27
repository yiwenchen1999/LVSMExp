def rectified_flow_loss(model, z_a, z_b, cond):
    """
    z_a: Latent tokens for Scene A (batch, seq_len, dim)
    z_b: Latent tokens for Scene B (batch, seq_len, dim)
    cond: Conditioning signal (c)
    """
    # 1. Sample random time t in [0, 1]
    t = torch.rand((z_a.shape[0], 1, 1), device=z_a.device)
    
    # 2. Create the linear interpolation (The Path)
    # z_t = (1 - t) * z_a + t * z_b
    z_t = (1 - t) * z_a + t * z_b
    
    # 3. The target velocity is the straight line direction
    target_velocity = z_b - z_a
    
    # 4. Model predicts velocity given current position, time, and condition
    # model(z_t, t, cond)
    pred_velocity = model(z_t, t.squeeze(), cond)
    
    # 5. Standard L2 Loss
    loss = F.mse_loss(pred_velocity, target_velocity)
    
    return loss

class TimeEmbedding(nn.Module):
    def __init__(self, model_dim):
        super().__init__()
        self.model_dim = model_dim
        
        # A small MLP to "project" the frequencies into the model's feature space
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim)
        )

    def forward(self, t):
        # t: (batch,) or (batch, 1) tensor in range [0, 1]
        half_dim = self.model_dim // 2
        
        # 1. Create sinusoidal frequencies (Standard Transformer approach)
        # Scaling by 1000 is common, but some use 10000.
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        
        # 2. Concat sine and cosine
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        
        # 3. Project through MLP
        return self.mlp(emb)
