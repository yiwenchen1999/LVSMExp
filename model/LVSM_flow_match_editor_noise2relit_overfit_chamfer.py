# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
from einops import repeat, rearrange

from .LVSM_scene_encoder_decoder_wEditor import LatentSceneEditor
from .transformer import QK_Norm_SelfAttention, MLP, init_weights
from .loss import LossComputer
from utils import camera_utils, data_utils

def chamfer_distance(v_gt, v_pred, chunk_size=256):
    """
    Compute Chamfer distance between two sets of vectors (memory-efficient version with double chunking).
    
    Args:
        v_gt: Ground truth vectors, shape [batch, n_vectors, d]
        v_pred: Predicted vectors, shape [batch, n_vectors, d]
        chunk_size: Size of chunks for memory-efficient computation (default 256 for large tensors)
    
    Returns:
        Chamfer distance, scalar tensor
    """
    batch_size, n_gt, d = v_gt.shape
    n_pred = v_pred.shape[1]
    
    # Compute chamfer_gt_to_pred: for each point in v_gt, find minimum distance to v_pred
    # Use double chunking: chunk both v_gt and v_pred to minimize memory
    min_dist_gt_to_pred_list = []
    for i in range(0, n_gt, chunk_size):
        chunk_end_gt = min(i + chunk_size, n_gt)
        v_gt_chunk = v_gt[:, i:chunk_end_gt, :]  # [batch, chunk_size_gt, d]
        
        # For each chunk of v_gt, compute min distance to all chunks of v_pred
        min_dists_for_gt_chunk = []
        for j in range(0, n_pred, chunk_size):
            chunk_end_pred = min(j + chunk_size, n_pred)
            v_pred_chunk = v_pred[:, j:chunk_end_pred, :]  # [batch, chunk_size_pred, d]
            
            # Compute distances: [batch, chunk_size_gt, chunk_size_pred, d]
            diff = v_gt_chunk.unsqueeze(2) - v_pred_chunk.unsqueeze(1)
            dist_chunk = torch.norm(diff, p=2, dim=-1)  # [batch, chunk_size_gt, chunk_size_pred]
            min_dists_for_gt_chunk.append(dist_chunk.min(dim=2)[0])  # [batch, chunk_size_gt]
        
        # Take minimum across all v_pred chunks for each v_gt point
        min_dist_chunk = torch.stack(min_dists_for_gt_chunk, dim=0).min(dim=0)[0]  # [batch, chunk_size_gt]
        min_dist_gt_to_pred_list.append(min_dist_chunk)
    
    min_dist_gt_to_pred = torch.cat(min_dist_gt_to_pred_list, dim=1)  # [batch, n_gt]
    chamfer_gt_to_pred = min_dist_gt_to_pred.mean()
    
    # Compute chamfer_pred_to_gt: for each point in v_pred, find minimum distance to v_gt
    # Use double chunking: chunk both v_pred and v_gt to minimize memory
    min_dist_pred_to_gt_list = []
    for i in range(0, n_pred, chunk_size):
        chunk_end_pred = min(i + chunk_size, n_pred)
        v_pred_chunk = v_pred[:, i:chunk_end_pred, :]  # [batch, chunk_size_pred, d]
        
        # For each chunk of v_pred, compute min distance to all chunks of v_gt
        min_dists_for_pred_chunk = []
        for j in range(0, n_gt, chunk_size):
            chunk_end_gt = min(j + chunk_size, n_gt)
            v_gt_chunk = v_gt[:, j:chunk_end_gt, :]  # [batch, chunk_size_gt, d]
            
            # Compute distances: [batch, chunk_size_pred, chunk_size_gt, d]
            diff = v_pred_chunk.unsqueeze(2) - v_gt_chunk.unsqueeze(1)
            dist_chunk = torch.norm(diff, p=2, dim=-1)  # [batch, chunk_size_pred, chunk_size_gt]
            min_dists_for_pred_chunk.append(dist_chunk.min(dim=2)[0])  # [batch, chunk_size_pred]
        
        # Take minimum across all v_gt chunks for each v_pred point
        min_dist_chunk = torch.stack(min_dists_for_pred_chunk, dim=0).min(dim=0)[0]  # [batch, chunk_size_pred]
        min_dist_pred_to_gt_list.append(min_dist_chunk)
    
    min_dist_pred_to_gt = torch.cat(min_dist_pred_to_gt_list, dim=1)  # [batch, n_pred]
    chamfer_pred_to_gt = min_dist_pred_to_gt.mean()
    
    # Symmetric Chamfer distance
    chamfer_loss = chamfer_gt_to_pred + chamfer_pred_to_gt
    
    return chamfer_loss

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
        
        # Handle shape of t
        if t.dim() == 1:
            t = t.unsqueeze(1)
            
        emb = t * emb[None, :]
        
        # 2. Concat sine and cosine
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        
        # Pad if model_dim is odd
        if self.model_dim % 2 == 1:
            emb = F.pad(emb, (0, 1, 0, 0))
            
        # 3. Project through MLP
        return self.mlp(emb)

class AdaLN_TransformerBlock(nn.Module):
    """
    Transformer block with AdaLN-Zero conditioning.
    Modulates LayerNorm parameters based on conditioning vector (time).
    """

    def __init__(
        self,
        dim,
        head_dim,
        attn_qkv_bias=False,
        attn_dropout=0.0,
        attn_fc_bias=False,
        attn_fc_dropout=0.0,
        mlp_ratio=4,
        mlp_bias=False,
        mlp_dropout=0.0,
        use_qk_norm=True,
    ):
        super().__init__()
        self.attn = QK_Norm_SelfAttention(
            dim=dim,
            head_dim=head_dim,
            qkv_bias=attn_qkv_bias,
            fc_bias=attn_fc_bias,
            attn_dropout=attn_dropout,
            fc_dropout=attn_fc_dropout,
            use_qk_norm=use_qk_norm,
        )

        self.mlp = MLP(
            dim=dim,
            mlp_ratio=mlp_ratio,
            bias=mlp_bias,
            dropout=mlp_dropout,
        )
        
        # AdaLN components
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        
        # Prediction of scale and shift from condition
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )
        
        # Zero initialization for the last linear layer
        with torch.no_grad():
            self.adaLN_modulation[1].weight.zero_()
            self.adaLN_modulation[1].bias.zero_()

    def forward(self, x, c):
        # c: conditioning vector (batch, dim)
        
        # Predict modulation parameters
        # shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
        params = self.adaLN_modulation(c).chunk(6, dim=1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = [p.unsqueeze(1) for p in params]
        
        # Attention block
        # x = x + gate_msa * attn(norm1(x) * (1 + scale_msa) + shift_msa)
        norm1_x = self.norm1(x)
        modulated_norm1 = norm1_x * (1 + scale_msa) + shift_msa
        x = x + gate_msa * self.attn(modulated_norm1)
        
        # MLP block
        # x = x + gate_mlp * mlp(norm2(x) * (1 + scale_mlp) + shift_mlp)
        norm2_x = self.norm2(x)
        modulated_norm2 = norm2_x * (1 + scale_mlp) + shift_mlp
        x = x + gate_mlp * self.mlp(modulated_norm2)
        
        return x

class FlowMatchEditor(LatentSceneEditor):
    def __init__(self, config):
        # Initialize parent class (will call _init_transformer)
        super().__init__(config)
        
        # Initialize time embedding
        self.time_embedder = TimeEmbedding(config.model.transformer.d)
        
        # Initialize loss computer for reconstruction loss
        self.loss_computer = LossComputer(config)
        
        # Re-initialize transformer_editor with AdaLN blocks
        self._reinit_editor_with_adaln()

        # Add EMA for noise scale
        self.register_buffer('noise_scale_ema', torch.tensor(1.0))
        
    def _reinit_editor_with_adaln(self):
        config = self.config.model.transformer
        use_qk_norm = config.get("use_qk_norm", False)
        
        editor_config = config.get("editor", {})
        editor_n_layer = editor_config.get("n_layer", 1)
        
        # Replace transformer_editor with AdaLN blocks
        self.transformer_editor = nn.ModuleList([
            AdaLN_TransformerBlock(
                dim=config.d,
                head_dim=config.d_head,
                use_qk_norm=use_qk_norm
            ) for _ in range(editor_n_layer)
        ])

    def init_from_singleStepEditor(self, editor_checkpoint_path):
        """
        Initialize from a trained single-step LatentSceneEditor checkpoint.
        This loads:
        1. Reconstructor & Renderer weights (standard load)
        2. Env Tokenizer weights
        3. Transfers Attention & MLP weights from standard Transformer blocks to AdaLN blocks in the editor
        """
        print(f"Loading single-step editor checkpoint from {editor_checkpoint_path}")
        
        # 1. Load the checkpoint
        if os.path.isdir(editor_checkpoint_path):
            ckpt_names = sorted([f for f in os.listdir(editor_checkpoint_path) if f.endswith(".pt")])
            if not ckpt_names:
                print(f"No .pt files found in {editor_checkpoint_path}")
                return None
            ckpt_path = os.path.join(editor_checkpoint_path, ckpt_names[-1])
        else:
            ckpt_path = editor_checkpoint_path
            
        try:
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            return None

        # 2. Load compatible weights (Reconstructor, Renderer, Env Tokenizer)
        # We filter out transformer_editor weights because keys won't match or shapes/logic differ
        my_state_dict = self.state_dict()
        compatible_state_dict = {}
        
        # Store editor weights from checkpoint for manual transfer
        editor_source_weights = {}
        
        for k, v in state_dict.items():
            if k.startswith("transformer_editor"):
                editor_source_weights[k] = v
            elif k in my_state_dict and my_state_dict[k].shape == v.shape:
                compatible_state_dict[k] = v
                
        # Load non-editor parts
        missing, unexpected = self.load_state_dict(compatible_state_dict, strict=False)
        print(f"Loaded base components. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
        
        # 3. Manual transfer of Editor weights (Attn + MLP)
        # source keys look like: transformer_editor.0.attn.to_qkv.weight
        # target structure is same for attn/mlp, just wrapped in AdaLN block
        
        print("Transferring Editor weights from Standard to AdaLN blocks...")
        success_count = 0
        
        for i, block in enumerate(self.transformer_editor):
            prefix = f"transformer_editor.{i}."
            
            # Helper to load submodule
            def load_submodule(submodule, sub_name):
                # sub_name e.g., "attn" or "mlp"
                sub_prefix = prefix + sub_name + "."
                sub_state = {}
                for k, v in editor_source_weights.items():
                    if k.startswith(sub_prefix):
                        # Remove prefix to get local key
                        local_key = k[len(sub_prefix):]
                        sub_state[local_key] = v
                
                if sub_state:
                    try:
                        submodule.load_state_dict(sub_state, strict=False)
                        return True
                    except Exception as e:
                        print(f"Error loading {sub_name} for block {i}: {e}")
                return False

            # Load Attn
            if load_submodule(block.attn, "attn"):
                pass
            
            # Load MLP
            if load_submodule(block.mlp, "mlp"):
                pass
                
            success_count += 1
            
        print(f"Transferred weights for {success_count}/{len(self.transformer_editor)} editor blocks.")
        return 0

    def init_from_LVSM(self, lvsm_checkpoint_path):
        """
        Initialize from Images2LatentScene (Reconstructor+Renderer only).
        Editor is initialized from scratch.
        """
        return super().init_from_LVSM(lvsm_checkpoint_path)

    def pass_editor_layers(self, input_tokens, time_emb, gradient_checkpoint=False, checkpoint_every=1):
        """
        Pass input tokens through editor blocks with AdaLN conditioning.
        """
        num_layers = len(self.transformer_editor)
        
        if not gradient_checkpoint:
            for layer in self.transformer_editor:
                input_tokens = layer(input_tokens, time_emb)
            return input_tokens
            
        def _process_layer_group(tokens, t_emb, start_idx, end_idx):
            for idx in range(start_idx, end_idx):
                tokens = self.transformer_editor[idx](tokens, t_emb)
            return tokens
            
        for start_idx in range(0, num_layers, checkpoint_every):
            end_idx = min(start_idx + checkpoint_every, num_layers)
            input_tokens = torch.utils.checkpoint.checkpoint(
                _process_layer_group,
                input_tokens,
                time_emb,
                start_idx,
                end_idx,
                use_reentrant=False
            )
            
        return input_tokens

    def forward(self, data_batch, timestep=None, skip_renderer=True):
        #& Step 1: Data processing
        input, target = self.process_data(data_batch, has_target_image=True, target_has_input=self.config.training.target_has_input, compute_rays=True)
        
        #& Step 2: Get Scene A latent tokens (Input)
        z_A, n_patches, d = self.reconstructor(input)
        
        #& Step 3: Get Scene B latent tokens (Target) - only needed for training
        # If we are in inference mode, we might not have target relit images, but flow matching training requires pairs.
        # Assuming we have paired data for training.
        if hasattr(input, 'relit_images') and input.relit_images is not None:
            # Create a target input dict for reconstructor
            target_input = edict(
                image=input.relit_images,
                ray_o=input.ray_o, # Same view poses for A and B
                ray_d=input.ray_d
            )
            
            with torch.no_grad():
                z_B, _, _ = self.reconstructor(target_input)
                z_B = z_B.detach() # Treat z_B as ground truth destination
        else:
            # Fallback or error if not available during training
            raise AssertionError("Target relit_images not found in batch during training; this must not happen in proper training. Stopping process.")
            
        #& Step 4: Sample Time t
        # Controlled environment: always use t=0 for training
        fixed_t = self.config.training.get("fixed_t", 0.0)
        if timestep is None:
            # Training: use fixed t (default 0.0) instead of random sampling
            t = torch.ones((z_A.shape[0],), device=z_A.device) * fixed_t
        else:
            # Inference: use provided t
            t = torch.ones((z_A.shape[0],), device=z_A.device) * timestep
            
        #& Step 5: Interpolate z_t
        # Flow Matching from Gaussian Noise (z_0) to Data (z_B)
        # z_t = (1 - t) * z_0 + t * z_B
        
        # Update EMA of noise scale using z_B (target distribution)
        # if self.training:
        #     with torch.no_grad():
        #         current_scale = z_B.std()
        #         ema_momentum = 0.99
        #         self.noise_scale_ema.mul_(ema_momentum).add_(current_scale * (1 - ema_momentum))
        latent_scale = 0.136 

        # Scale your inputs and targets
        z_A = z_A * latent_scale
        z_B = z_B.detach() * latent_scale

        # Use Standard Gaussian Noise with fixed seed for controlled environment
        noise_seed = self.config.training.get("noise_seed", 42)
        # Set random seed for reproducible noise generation
        generator = torch.Generator(device=z_A.device).manual_seed(noise_seed)
        z_0 = torch.randn(z_A.shape, device=z_A.device, dtype=z_A.dtype, generator=generator) # Mean 0, Std 1, fixed seed
        z_1 = z_B 
            
        # Broadcast t for interpolation
        t_expand = t.view(-1, 1, 1)
        
        # Interpolate z_t
        z_t = (1 - t_expand) * z_0 + t_expand * z_1
        # Add this check in forward:
        print(f"--- Latent Statistics ---")
        print(f"z_A (Source) | Mean: {z_A.mean().item():.4f} | Std: {z_A.std().item():.4f} | Max: {z_A.max().item():.4f}")
        print(f"z_B (Target) | Mean: {z_B.mean().item():.4f} | Std: {z_B.std().item():.4f} | Max: {z_B.max().item():.4f}")                
        #& Step 6: Prepare Env Conditioning
        # Process environment maps (LDR, HDR, Dir)
        
        # ... Reuse logic from edit_scene_with_env to get env_tokens ...
        # Copied and adapted logic:
        if hasattr(input, 'env_ldr') and hasattr(input, 'env_hdr') and hasattr(input, 'env_dir'):
             if input.env_ldr is not None:
                b, v_input = input.env_dir.shape[:2]
                env_h, env_w = input.env_dir.shape[3], input.env_dir.shape[4]
                
                single_env_map = self.config.training.get("single_env_map", False)
                if single_env_map:
                    if self.training:
                        view_idx = torch.randint(0, v_input, (b,), device=input.env_dir.device)
                    else:
                        view_idx = torch.zeros(b, dtype=torch.long, device=input.env_dir.device)
                    batch_indices = torch.arange(b, device=input.env_dir.device)
                    env_ldr = input.env_ldr[batch_indices, view_idx].unsqueeze(1)
                    env_hdr = input.env_hdr[batch_indices, view_idx].unsqueeze(1)
                    env_dir = input.env_dir[batch_indices, view_idx].unsqueeze(1)
                    v_input = 1
                else:
                    env_ldr = input.env_ldr
                    env_hdr = input.env_hdr
                    env_dir = input.env_dir

                # Resize if needed (omitted for brevity, assume correct or add if needed)
                # ... (Resize logic from original) ...
                if env_ldr.shape[3] != env_h or env_ldr.shape[4] != env_w:
                    env_ldr = F.interpolate(env_ldr.reshape(b*v_input, 3, env_ldr.shape[3], env_ldr.shape[4]), size=(env_h, env_w), mode='bilinear', align_corners=False).reshape(b, v_input, 3, env_h, env_w)
                if env_hdr.shape[3] != env_h or env_hdr.shape[4] != env_w:
                    env_hdr = F.interpolate(env_hdr.reshape(b*v_input, 3, env_hdr.shape[3], env_hdr.shape[4]), size=(env_h, env_w), mode='bilinear', align_corners=False).reshape(b, v_input, 3, env_h, env_w)
                
                directional_env = torch.cat([env_ldr, env_hdr, env_dir], dim=2)
                env_tokens = self.env_tokenizer(directional_env)
                _, n_env_patches, _ = env_tokens.size()
                env_tokens = env_tokens.reshape(b, v_input * n_env_patches, d)
        else:
            # Handle case where env is missing?
            env_tokens = torch.zeros(z_A.shape[0], 0, d, device=z_A.device)
            print("Warning: Env tokens are missing. Using zeros.")
            raise AssertionError("Env tokens are missing. Stopping process.")

        #& Step 7: Predict Velocity
        # Concatenate: [z_t, z_A, env_tokens]
        # Condition on both current state z_t and source latent z_A (plus environment)
        editor_input_tokens = torch.cat([z_t, z_A, env_tokens], dim=1)
        
        # Time Embedding
        t_emb = self.time_embedder(t) # [b, d]
        
        # Pass through editor
        checkpoint_every = self.config.training.grad_checkpoint_every
        editor_output_tokens = self.pass_editor_layers(
            editor_input_tokens, 
            t_emb, 
            gradient_checkpoint=True, 
            checkpoint_every=checkpoint_every
        )
        
        # Extract predicted velocity (corresponding to z_t part)
        n_latent_vectors = self.config.model.transformer.n_latent_vectors
        pred_velocity = editor_output_tokens[:, :n_latent_vectors, :]
        
        #& Step 8: Compute Loss
        # target_velocity = z_1 - z_0 = z_B - z_0
        # flow matching objective: v_t = z_1 - z_0
        target_velocity = z_1 - z_0
        # print("rms v*", rms(target_velocity), "max|v*|", target_velocity.abs().max().item())
        # print("rms pred_v", rms(pred_velocity))

        # Compute flow loss (MSE)
        loss_flow = F.mse_loss(pred_velocity, target_velocity)
        
        # Compute chamfer flow loss
        loss_chamfer_flow = chamfer_distance(target_velocity, pred_velocity)
        
        target_v_norm = target_velocity.norm(p=2, dim=-1).mean().item()
        pred_v_norm = pred_velocity.norm(p=2, dim=-1).mean().item()
        print(f"Target V Norm: {target_v_norm:.4f} | Pred V Norm: {pred_v_norm:.4f}")
        
        # Initialize loss metrics with flow losses
        loss_metrics = edict({
            'flow_loss': loss_flow,
            'chamfer_flow_loss': loss_chamfer_flow
        })

        # Render for visualization/metrics and compute reconstruction loss
        rendered_images = None
        reconstruction_loss_metrics = None
        
        # Controlled environment: training mode selection
        # Options: 'chamfer_flow_only', 'flow_only', 'render_only'
        training_mode = self.config.training.get("training_mode", "flow_only")
        use_chamfer_flow_loss = training_mode == "chamfer_flow_only"
        use_flow_loss = training_mode == "flow_only"
        use_render_loss = training_mode == "render_only"
        
        # Always compute render loss for metrics, even if not used for training
        if not skip_renderer:
            # Compute z_B_pred from z_t using predicted velocity
            # z_t = (1-t) * z_0 + t * z_B
            # z_B = z_1 = z_0 + v
            # z_t = z_1 - (1-t)v
            # z_B_pred = z_t + (1-t) * pred_velocity
            t_expand_for_pred = t.view(-1, 1, 1)
            z_B_pred = z_t + (1.0 - t_expand_for_pred) * pred_velocity
            
            # Scale back to original space before rendering (renderer expects unscaled latents)
            z_B_pred = z_B_pred / latent_scale
            
            # Render z_B_pred
            rendered_images = self.renderer(z_B_pred, target, n_patches, d)
            
            # Compute reconstruction loss using the same loss computer as LVSM_scene_encoder_decoder
            if hasattr(target, 'relit_images'):
                reconstruction_loss_metrics = self.loss_computer(
                    rendering=rendered_images,
                    target=target.relit_images
                )
                
                # Always store all metrics
                loss_metrics.reconstruction_loss = reconstruction_loss_metrics.loss
                loss_metrics.l2_loss = reconstruction_loss_metrics.l2_loss
                loss_metrics.lpips_loss = reconstruction_loss_metrics.lpips_loss
                loss_metrics.perceptual_loss = reconstruction_loss_metrics.perceptual_loss
                loss_metrics.psnr = reconstruction_loss_metrics.psnr
            else:
                # If no relit_images, set render loss to zero
                loss_metrics.reconstruction_loss = torch.tensor(0.0, device=loss_flow.device)
                loss_metrics.l2_loss = torch.tensor(0.0, device=loss_flow.device)
                loss_metrics.lpips_loss = torch.tensor(0.0, device=loss_flow.device)
                loss_metrics.perceptual_loss = torch.tensor(0.0, device=loss_flow.device)
                loss_metrics.psnr = torch.tensor(0.0, device=loss_flow.device)
        else:
            # Return dummy render if skipped to satisfy pipeline
            if hasattr(target, 'image'):
                rendered_images = torch.zeros_like(target.image)
            else:
                rendered_images = None
            # Set render loss to zero if skipped
            loss_metrics.reconstruction_loss = torch.tensor(0.0, device=loss_flow.device)
            loss_metrics.l2_loss = torch.tensor(0.0, device=loss_flow.device)
            loss_metrics.lpips_loss = torch.tensor(0.0, device=loss_flow.device)
            loss_metrics.perceptual_loss = torch.tensor(0.0, device=loss_flow.device)
            loss_metrics.psnr = torch.tensor(0.0, device=loss_flow.device)
        
        # Compute total loss based on training mode
        # All metrics are computed, but only the selected one is used for training
        if use_chamfer_flow_loss:
            total_loss = loss_chamfer_flow
        elif use_flow_loss:
            total_loss = loss_flow
        elif use_render_loss:
            total_loss = loss_metrics.reconstruction_loss
        else:
            # Default to flow_loss if training_mode is invalid
            total_loss = loss_flow
        
        loss_metrics.loss = total_loss

        result = edict(
            input=input,
            target=target,
            loss_metrics=loss_metrics,
            render=rendered_images,
            t=t,
            z_A=z_A,
            z_B=z_B
        )
        return result

    @torch.no_grad()
    def flow_match_inference(self, data_batch, steps=8, method='heun', input=None, target=None):
        """
        Perform flow matching inference (ODE integration).
        Note: This method should be called with model.eval() to avoid DDP issues.
        
        Args:
            data_batch: Raw data batch
            steps: Number of integration steps (ignored if single_step_inference=True)
            method: Integration method ('euler' or 'heun')
            input: Optional pre-processed input dict (to reuse views from training step)
            target: Optional pre-processed target dict (to reuse views from training step)
        """
        
        # Use provided input/target or process from batch
        if input is None or target is None:
            input, target = self.process_data(data_batch, has_target_image=False, target_has_input=self.config.training.target_has_input, compute_rays=True)
        
        # Apply same latent scaling as in training
        latent_scale = 0.136
            
        # During inference, we need to avoid gradient checkpointing to prevent DDP issues
        # Temporarily store original checkpoint setting and disable it
        original_checkpoint_every = self.config.training.grad_checkpoint_every
        self.config.training.grad_checkpoint_every = 999999  # Effectively disable checkpointing
        try:
            z_A, n_patches, d = self.reconstructor(input) 
            # Scale z_A to match training
            z_A = z_A * latent_scale
            # Start at random noise (standard Gaussian, will be in scaled space after interpolation)
            # Use fixed seed for controlled environment
            noise_seed = self.config.training.get("noise_seed", 42)
            generator = torch.Generator(device=z_A.device).manual_seed(noise_seed)
            z = torch.randn(z_A.shape, device=z_A.device, dtype=z_A.dtype, generator=generator)  # Standard Gaussian noise with fixed seed
        finally:
            # Restore original setting
            self.config.training.grad_checkpoint_every = original_checkpoint_every
        
        # Prepare Env Tokens (Fixed for all steps)
        # ... (Same env processing as forward) ...
        # reusing helper if possible or copy code
        if hasattr(input, 'env_ldr'):
             # ... copy paste env processing ...
             # For brevity, assuming self.get_env_tokens(input) exists or inlining
             # INLINING for now to be safe:
            b, v_input = input.env_dir.shape[:2]
            env_h, env_w = input.env_dir.shape[3], input.env_dir.shape[4]
            single_env_map = self.config.training.get("single_env_map", False)
            if single_env_map:
                view_idx = torch.zeros(b, dtype=torch.long, device=input.env_dir.device)
                batch_indices = torch.arange(b, device=input.env_dir.device)
                env_ldr = input.env_ldr[batch_indices, view_idx].unsqueeze(1)
                env_hdr = input.env_hdr[batch_indices, view_idx].unsqueeze(1)
                env_dir = input.env_dir[batch_indices, view_idx].unsqueeze(1)
                v_input = 1
            else:
                env_ldr = input.env_ldr
                env_hdr = input.env_hdr
                env_dir = input.env_dir

            if env_ldr.shape[3] != env_h:
                env_ldr = F.interpolate(env_ldr.reshape(b*v_input, 3, env_ldr.shape[3], env_ldr.shape[4]), size=(env_h, env_w), mode='bilinear', align_corners=False).reshape(b, v_input, 3, env_h, env_w)
            if env_hdr.shape[3] != env_h:
                env_hdr = F.interpolate(env_hdr.reshape(b*v_input, 3, env_hdr.shape[3], env_hdr.shape[4]), size=(env_h, env_w), mode='bilinear', align_corners=False).reshape(b, v_input, 3, env_h, env_w)
            
            directional_env = torch.cat([env_ldr, env_hdr, env_dir], dim=2)
            env_tokens = self.env_tokenizer(directional_env)
            _, n_env_patches, _ = env_tokens.size()
            env_tokens = env_tokens.reshape(b, v_input * n_env_patches, d)
        
        # Controlled environment: single step inference (t=0)
        single_step_inference = self.config.training.get("single_step_inference", False)
        
        if single_step_inference:
            # Single step inference at t=0
            t_val = 0.0
            t = torch.ones((z.shape[0],), device=z.device) * t_val
            
            # Predict velocity
            # Input: [z_t, z_A, env_tokens]
            editor_input = torch.cat([z, z_A, env_tokens], dim=1)
            t_emb = self.time_embedder(t)
            
            out = self.pass_editor_layers(editor_input, t_emb)
            v_pred = out[:, :self.config.model.transformer.n_latent_vectors, :]
            
            # Single step: z_1 = z_0 + v_0 (since t=0, we go from z_0 to z_1 directly)
            z = z + v_pred
        else:
            # ODE Integration (original multi-step)
            dt = 1.0 / steps
            for i in range(steps):
                t_val = i / steps
                t = torch.ones((z.shape[0],), device=z.device) * t_val
                
                # Predict velocity
                # Input: [z_t, z_A, env_tokens]
                editor_input = torch.cat([z, z_A, env_tokens], dim=1)
                t_emb = self.time_embedder(t)
                
                out = self.pass_editor_layers(editor_input, t_emb)
                v_pred = out[:, :self.config.model.transformer.n_latent_vectors, :]
                
                if method == 'euler':
                    z = z + v_pred * dt
                elif method == 'rk4':
                    # Implement RK4 if needed, stick to Euler for simplicity or Heun
                    pass
                elif method == 'heun':
                    # v1 = f(z, t)
                    # z_pred = z + v1 * dt
                    # v2 = f(z_pred, t + dt)
                    # z = z + 0.5 * dt * (v1 + v2)
                    
                    z_guess = z + v_pred * dt
                    
                    t_next = torch.ones_like(t) * (t_val + dt)
                    editor_input_next = torch.cat([z_guess, z_A, env_tokens], dim=1)
                    t_emb_next = self.time_embedder(t_next)
                    out_next = self.pass_editor_layers(editor_input_next, t_emb_next)
                    v_pred_next = out_next[:, :self.config.model.transformer.n_latent_vectors, :]
                    
                    z = z + 0.5 * dt * (v_pred + v_pred_next)

        # Scale back to original space before rendering (renderer expects unscaled latents)
        z = z / latent_scale

        # Render final z
        rendered_images = self.renderer(z, target, n_patches, d)
        
        return edict(render=rendered_images)

