# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).

import os
import torch
import torch.nn as nn
from easydict import EasyDict as edict
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
import traceback
from utils import camera_utils, data_utils
from .transformer import QK_Norm_TransformerBlock, init_weights
from .loss import LossComputer


class LatentSceneEditor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.process_data = data_utils.ProcessData(config)

        # Initialize both input tokenizers, and output de-tokenizer
        self._init_tokenizers()
        
        # Initialize transformer blocks
        self._init_transformer()
        
        # Initialize loss computer
        self.loss_computer = LossComputer(config)

    def _create_tokenizer(self, in_channels, patch_size, d_model):
        """Helper function to create a tokenizer with given config"""
        tokenizer = nn.Sequential(
            Rearrange(
                "b v c (hh ph) (ww pw) -> (b v) (hh ww) (ph pw c)",
                ph=patch_size,
                pw=patch_size,
            ),
            nn.Linear(
                in_channels * (patch_size**2),
                d_model,
                bias=False,
            ),
        )
        tokenizer.apply(init_weights)

        return tokenizer

    def _create_linear_tokenizer(self, in_channels, d_model):
        """Tokenizer for token-wise features [B, T, C] -> [B, T, d_model]."""
        tokenizer = nn.Sequential(
            nn.Linear(in_channels, d_model, bias=False),
        )
        tokenizer.apply(init_weights)
        return tokenizer

    def _init_tokenizers(self):
        """Initialize the image and target pose tokenizers, and image token decoder"""
        # Image tokenizer
        self.image_tokenizer = self._create_tokenizer(
            in_channels = self.config.model.image_tokenizer.in_channels,
            patch_size = self.config.model.image_tokenizer.patch_size,
            d_model = self.config.model.transformer.d
        )
        
        # Target pose tokenizer
        self.target_pose_tokenizer = self._create_tokenizer(
            in_channels = self.config.model.target_pose_tokenizer.in_channels,
            patch_size = self.config.model.target_pose_tokenizer.patch_size,
            d_model = self.config.model.transformer.d
        )
        
        # Resolve which lighting tokenizers to create based on relight_signals
        signal_list = self.config.training.get("relight_signals", ["envmap"])
        if isinstance(signal_list, str):
            signal_list = [signal_list]
        use_envmap = "envmap" in signal_list
        use_point_light = "point_light" in signal_list
        if not use_envmap and not use_point_light:
            use_envmap = True  # fallback

        # Environment tokenizer (RGB + ray directions) - only if envmap in relight_signals
        if use_envmap:
            env_config = self.config.model.get("env_tokenizer", {})
            if env_config:
                self.env_tokenizer = self._create_tokenizer(
                    in_channels=env_config.get("in_channels", 9),
                    patch_size=env_config.get("patch_size", 8),
                    d_model=self.config.model.transformer.d
                )
            else:
                self.env_tokenizer = self._create_tokenizer(
                    in_channels=6,
                    patch_size=8,
                    d_model=self.config.model.transformer.d
                )
            print(f"Initialized env_tokenizer (relight_signals includes envmap)")
        else:
            self.env_tokenizer = None

        # Point-light tokenizer - only if point_light in relight_signals
        if use_point_light:
            pl_config = self.config.model.get("point_light_tokenizer", {})
            self.point_light_tokenizer = self._create_linear_tokenizer(
                in_channels=pl_config.get("in_channels", 10),
                d_model=self.config.model.transformer.d,
            )
            print(f"Initialized point_light_tokenizer (relight_signals includes point_light)")
        else:
            self.point_light_tokenizer = None
        
        # Image token decoder (decode image tokens into pixels)
        self.image_token_decoder = nn.Sequential(
            nn.LayerNorm(self.config.model.transformer.d, bias=False),
            nn.Linear(
                self.config.model.transformer.d,
                (self.config.model.target_pose_tokenizer.patch_size**2) * 3,
                bias=False,
            ),
            nn.Sigmoid()
        )
        self.image_token_decoder.apply(init_weights)


    def _init_transformer(self):
        """Initialize transformer blocks"""
        config = self.config.model.transformer
        use_qk_norm = config.get("use_qk_norm", False)

        # latent vectors for LVSM encoder-decoder
        self.n_light_field_latent = nn.Parameter(
            torch.randn(
                config.n_latent_vectors,
                config.d,
            )
        )
        nn.init.trunc_normal_(self.n_light_field_latent, std=0.02)

        # Create transformer blocks
        self.transformer_encoder = [
            QK_Norm_TransformerBlock(
                config.d, config.d_head, use_qk_norm=use_qk_norm
            ) for _ in range(config.encoder_n_layer)
        ]

        self.transformer_decoder = [
            QK_Norm_TransformerBlock(
                config.d, config.d_head, use_qk_norm=use_qk_norm
            ) for _ in range(config.decoder_n_layer)
        ]
        
        # Initialize albedo decoder if configured
        self.use_albedo_decoder = config.get("use_albedo_decoder", False)
        print(f"use_albedo_decoder: {self.use_albedo_decoder}")
        if self.use_albedo_decoder:
            self.transformer_decoder_albedo = [
                QK_Norm_TransformerBlock(
                    config.d, config.d_head, use_qk_norm=use_qk_norm
                ) for _ in range(config.decoder_n_layer)
            ]
        
        # Apply special initialization if configured
        if config.get("special_init", False):
            # Encoder
            for idx, block in enumerate(self.transformer_encoder):
                if config.get("depth_init", False):
                    weight_init_std = 0.02 / (2 * (idx + 1)) ** 0.5
                else:
                    weight_init_std = 0.02 / (2 * config.encoder_n_layer) ** 0.5
                block.apply(lambda module: init_weights(module, weight_init_std))

            # Decoder
            for idx, block in enumerate(self.transformer_decoder):
                if config.get("depth_init", False):
                    weight_init_std = 0.02 / (2 * (idx + 1)) ** 0.5
                else:
                    weight_init_std = 0.02 / (2 * config.decoder_n_layer) ** 0.5
                block.apply(lambda module: init_weights(module, weight_init_std))
            
            # Albedo decoder (if enabled)
            if self.use_albedo_decoder:
                for idx, block in enumerate(self.transformer_decoder_albedo):
                    if config.get("depth_init", False):
                        weight_init_std = 0.02 / (2 * (idx + 1)) ** 0.5
                    else:
                        weight_init_std = 0.02 / (2 * config.decoder_n_layer) ** 0.5
                    block.apply(lambda module: init_weights(module, weight_init_std))
        else:
            # Encoder
            for block in self.transformer_encoder:
                block.apply(init_weights)

            # Decoder
            for block in self.transformer_decoder:
                block.apply(init_weights)
            
            # Albedo decoder (if enabled)
            if self.use_albedo_decoder:
                for block in self.transformer_decoder_albedo:
                    block.apply(init_weights)

                
        self.transformer_encoder = nn.ModuleList(self.transformer_encoder)
        self.transformer_decoder = nn.ModuleList(self.transformer_decoder)
        if self.use_albedo_decoder:
            self.transformer_decoder_albedo = nn.ModuleList(self.transformer_decoder_albedo)
        self.transformer_input_layernorm_decoder = nn.LayerNorm(config.d, bias=False)
        
        # Transformer editor block (for editing latent tokens)
        editor_config = config.get("editor", {})
        editor_n_layer = editor_config.get("n_layer", 1)
        self.transformer_editor = nn.ModuleList([
            QK_Norm_TransformerBlock(
                config.d, config.d_head, use_qk_norm=use_qk_norm
            ) for _ in range(editor_n_layer)
        ])
        # Initialize editor blocks with smaller weights to start near identity
        # This prevents the editor from disrupting pre-trained latent tokens initially
        editor_init_scale = editor_config.get("init_scale", 0.1)  # Default: 0.1x of normal init
        if config.get("special_init", False):
            for idx, block in enumerate(self.transformer_editor):
                if config.get("depth_init", False):
                    # Use smaller initialization for editor (scaled by editor_init_scale)
                    weight_init_std = (0.02 / (2 * (idx + 1)) ** 0.5) * editor_init_scale
                else:
                    weight_init_std = (0.02 / (2 * editor_n_layer) ** 0.5) * editor_init_scale
                block.apply(lambda module: init_weights(module, weight_init_std))
        else:
            # Use smaller initialization for editor (scaled by editor_init_scale)
            editor_std = 0.02 * editor_init_scale
            for block in self.transformer_editor:
                block.apply(lambda module: init_weights(module, editor_std))


    def train(self, mode=True):
        """Override the train method to keep the loss computer in eval mode"""
        super().train(mode)
        self.loss_computer.eval()
    
    def freeze_reconstructor_and_renderer(self):
        """
        Freeze all layers used in reconstructor and renderer, 
        only keep editor layers (transformer_editor and lighting tokenizers) trainable.
        
        Reconstructor layers:
        - image_tokenizer
        - n_light_field_latent
        - transformer_encoder
        
        Renderer layers:
        - target_pose_tokenizer
        - transformer_input_layernorm_decoder
        - transformer_decoder
        - image_token_decoder
        
        Editor layers (kept trainable):
        - env_tokenizer
        - point_light_tokenizer
        - transformer_editor
        """
        # Freeze reconstructor layers
        print("**********Freezing reconstructor layers...**********")
        for param in self.image_tokenizer.parameters():
            param.requires_grad = False
        self.n_light_field_latent.requires_grad = False
        for param in self.transformer_encoder.parameters():
            param.requires_grad = False
        
        # Freeze renderer layers
        for param in self.target_pose_tokenizer.parameters():
            param.requires_grad = False
        for param in self.transformer_input_layernorm_decoder.parameters():
            param.requires_grad = False
        for param in self.transformer_decoder.parameters():
            param.requires_grad = False
        for param in self.image_token_decoder.parameters():
            param.requires_grad = False
        
        # Freeze albedo decoder if it exists
        if self.use_albedo_decoder:
            for param in self.transformer_decoder_albedo.parameters():
                param.requires_grad = False
        
        # Keep editor layers trainable (only those that exist)
        trainable_parts = []
        if self.env_tokenizer is not None:
            for param in self.env_tokenizer.parameters():
                param.requires_grad = True
            trainable_parts.append("env_tokenizer")
        if self.point_light_tokenizer is not None:
            for param in self.point_light_tokenizer.parameters():
                param.requires_grad = True
            trainable_parts.append("point_light_tokenizer")
        for param in self.transformer_editor.parameters():
            param.requires_grad = True
        trainable_parts.append("transformer_editor")
        print(f"Frozen reconstructor and renderer layers. Only editor layers ({', '.join(trainable_parts)}) are trainable.")
    
    def unfreeze_all(self):
        """Unfreeze all layers for full model training"""
        for param in self.parameters():
            param.requires_grad = True
        print("Unfrozen all layers.")


    
    def pass_layers(self, transformer_blocks, input_tokens, gradient_checkpoint=False, checkpoint_every=1):
        """
        Helper function to pass input tokens through all transformer blocks with optional gradient checkpointing.
        
        Args:
            input_tokens: Tensor of shape [batch_size, num_views * num_patches, hidden_dim]
                The input tokens to process through the transformer blocks.
            gradient_checkpoint: bool, default False
                Whether to use gradient checkpointing to save memory during training.
            checkpoint_every: int, default 1 
                Number of transformer layers to group together for gradient checkpointing.
                Only used when gradient_checkpoint=True.
                
        Returns:
            Tensor of shape [batch_size, num_views * num_patches, hidden_dim]
                The processed tokens after passing through all transformer blocks.
        """
        num_layers = len(transformer_blocks)
        
        if not gradient_checkpoint:
            # Standard forward pass through all layers
            for layer in transformer_blocks:
                input_tokens = layer(input_tokens)
            return input_tokens
            
        # Gradient checkpointing enabled - process layers in groups
        def _process_layer_group(tokens, start_idx, end_idx):
            """Helper to process a group of consecutive layers."""
            for idx in range(start_idx, end_idx):
                tokens = transformer_blocks[idx](tokens)
            return tokens
            
        # Process layer groups with gradient checkpointing
        for start_idx in range(0, num_layers, checkpoint_every):
            end_idx = min(start_idx + checkpoint_every, num_layers)
            input_tokens = torch.utils.checkpoint.checkpoint(
                _process_layer_group,
                input_tokens,
                start_idx,
                end_idx,
                use_reentrant=False
            )
            
        return input_tokens
            

    def get_posed_input(self, images=None, ray_o=None, ray_d=None, method="default_plucker"):
        '''
        Args:
            images: [b, v, c, h, w]
            ray_o: [b, v, 3, h, w]
            ray_d: [b, v, 3, h, w]
            method: Method for creating pose conditioning
        Returns:
            posed_images: [b, v, c+6, h, w] or [b, v, 6, h, w] if images is None
        '''

        if method == "custom_plucker":
            o_dot_d = torch.sum(-ray_o * ray_d, dim=2, keepdim=True)
            nearest_pts = ray_o + o_dot_d * ray_d
            pose_cond = torch.cat([ray_d, nearest_pts], dim=2)
            
        elif method == "aug_plucker":
            o_dot_d = torch.sum(-ray_o * ray_d, dim=2, keepdim=True)
            nearest_pts = ray_o + o_dot_d * ray_d
            o_cross_d = torch.cross(ray_o, ray_d, dim=2)
            pose_cond = torch.cat([o_cross_d, ray_d, nearest_pts], dim=2)
            
        else:  # default_plucker
            o_cross_d = torch.cross(ray_o, ray_d, dim=2)
            pose_cond = torch.cat([o_cross_d, ray_d], dim=2)

        if images is None:
            return pose_cond
        else:
            return torch.cat([images * 2.0 - 1.0, pose_cond], dim=2)

    def _point_rays_to_pose_cond(self, ray_o, ray_d, method="default_plucker"):
        """
        Args:
            ray_o: [b, v, n_rays, 3]
            ray_d: [b, v, n_rays, 3]
        Returns:
            pose_cond: [b, v, n_rays, 6] (default/custom) or [b, v, n_rays, 9] (aug)
        """
        if method == "custom_plucker":
            o_dot_d = torch.sum(-ray_o * ray_d, dim=-1, keepdim=True)
            nearest_pts = ray_o + o_dot_d * ray_d
            pose_cond = torch.cat([ray_d, nearest_pts], dim=-1)
        elif method == "aug_plucker":
            o_dot_d = torch.sum(-ray_o * ray_d, dim=-1, keepdim=True)
            nearest_pts = ray_o + o_dot_d * ray_d
            o_cross_d = torch.cross(ray_o, ray_d, dim=-1)
            pose_cond = torch.cat([o_cross_d, ray_d, nearest_pts], dim=-1)
        else:
            o_cross_d = torch.cross(ray_o, ray_d, dim=-1)
            pose_cond = torch.cat([o_cross_d, ray_d], dim=-1)
        return pose_cond

    def _build_env_tokens(self, input_dict, token_dim):
        if self.env_tokenizer is None:
            return None
        if not (hasattr(input_dict, 'env_ldr') and hasattr(input_dict, 'env_hdr') and hasattr(input_dict, 'env_dir')):
            return None
        if input_dict.env_ldr is None or input_dict.env_hdr is None or input_dict.env_dir is None:
            return None

        b, v_input = input_dict.env_dir.shape[:2]
        env_h, env_w = input_dict.env_dir.shape[3], input_dict.env_dir.shape[4]

        single_env_map = self.config.training.get("single_env_map", False)
        if single_env_map:
            if self.training:
                view_idx = torch.randint(0, v_input, (b,), device=input_dict.env_dir.device)
            else:
                view_idx = torch.zeros(b, dtype=torch.long, device=input_dict.env_dir.device)
            batch_indices = torch.arange(b, device=input_dict.env_dir.device)
            env_ldr = input_dict.env_ldr[batch_indices, view_idx].unsqueeze(1)
            env_hdr = input_dict.env_hdr[batch_indices, view_idx].unsqueeze(1)
            env_dir = input_dict.env_dir[batch_indices, view_idx].unsqueeze(1)
            v_input = 1
        else:
            env_ldr = input_dict.env_ldr
            env_hdr = input_dict.env_hdr
            env_dir = input_dict.env_dir

        if env_ldr.shape[3] != env_h or env_ldr.shape[4] != env_w:
            env_ldr = torch.nn.functional.interpolate(
                env_ldr.reshape(b * v_input, 3, env_ldr.shape[3], env_ldr.shape[4]),
                size=(env_h, env_w),
                mode='bilinear',
                align_corners=False
            ).reshape(b, v_input, 3, env_h, env_w)

        if env_hdr.shape[3] != env_h or env_hdr.shape[4] != env_w:
            env_hdr = torch.nn.functional.interpolate(
                env_hdr.reshape(b * v_input, 3, env_hdr.shape[3], env_hdr.shape[4]),
                size=(env_h, env_w),
                mode='bilinear',
                align_corners=False
            ).reshape(b, v_input, 3, env_h, env_w)

        directional_env = torch.cat([env_ldr, env_hdr, env_dir], dim=2)  # [b, v, 9, env_h, env_w]
        env_tokens = self.env_tokenizer(directional_env)  # [b*v, n_patch, d]
        _, n_env_patches, _ = env_tokens.size()
        env_tokens = env_tokens.reshape(b, v_input * n_env_patches, token_dim)
        return env_tokens

    def _build_point_light_tokens(self, input_dict, token_dim):
        if self.point_light_tokenizer is None:
            return None
        if not hasattr(input_dict, "point_light_rays"):
            return None
        if input_dict.point_light_rays is None:
            return None

        # [b, v, n_rays, 10] = intensity(1), color(3), ray_o(3), ray_d(3)
        point_rays = input_dict.point_light_rays
        b, v_input, n_rays, _ = point_rays.shape

        # Optional single-view point-light conditioning (similar to single_env_map)
        single_point_light = self.config.training.get("single_point_light", False)
        if single_point_light:
            if self.training:
                view_idx = torch.randint(0, v_input, (b,), device=point_rays.device)
            else:
                view_idx = torch.zeros(b, dtype=torch.long, device=point_rays.device)
            batch_indices = torch.arange(b, device=point_rays.device)
            point_rays = point_rays[batch_indices, view_idx].unsqueeze(1)  # [b,1,n,10]
            v_input = 1

        intensity_and_color = point_rays[..., :4]  # [b, v, n, 4]
        ray_o = point_rays[..., 4:7]
        ray_d = point_rays[..., 7:10]
        ray_d = ray_d / (torch.norm(ray_d, dim=-1, keepdim=True) + 1e-8)

        plucker_method = self.config.training.get("point_light_plucker_method", "default_plucker")
        ray_pose_cond = self._point_rays_to_pose_cond(ray_o, ray_d, method=plucker_method)
        point_light_features = torch.cat([intensity_and_color, ray_pose_cond], dim=-1)  # default: 4+6=10

        # Tokenize each ray as one token
        point_light_features = point_light_features.reshape(b, v_input * n_rays, point_light_features.shape[-1])
        point_tokens = self.point_light_tokenizer(point_light_features)  # [b, v*n_rays, d]
        return point_tokens

    def _build_editor_condition_tokens(self, input_dict, token_dim):
        print('building conditional tokens...')
        signal_list = self.config.training.get("relight_signals", ["envmap"])
        if isinstance(signal_list, str):
            signal_list = [signal_list]
        print("signal_list:", signal_list)

        cond_tokens = []
        if "envmap" in signal_list:
            env_tokens = self._build_env_tokens(input_dict, token_dim)
            if env_tokens is not None:
                cond_tokens.append(env_tokens)

        if "point_light" in signal_list:
            point_tokens = self._build_point_light_tokens(input_dict, token_dim)
            if point_tokens is not None:
                cond_tokens.append(point_tokens)

        if len(cond_tokens) == 0:
            return None
        if len(cond_tokens) == 1:
            return cond_tokens[0]
        return torch.cat(cond_tokens, dim=1)
    
    
    def reconstructor(self, input, checkpoint_every=None):
        """
        从输入图像重建场景的 latent_tokens
        
        Args:
            input: 包含 input.image, input.ray_o, input.ray_d 的 edict
            checkpoint_every: 梯度检查点间隔，如果为 None 则使用 config 中的值
            
        Returns:
            latent_tokens: [b, n_latent_vectors, d] - 编码后的场景潜在表示
            n_patches: int - 每个视图的 patch 数量
            d: int - token 维度
        """
        if checkpoint_every is None:
            checkpoint_every = self.config.training.grad_checkpoint_every
        
        #& Step 1: Build input image pose conditioning - Concatenate RGB images with ray info
        posed_input_images = self.get_posed_input(
            images=input.image, ray_o=input.ray_o, ray_d=input.ray_d
        )
        b, v_input, c, h, w = posed_input_images.size()

        #& Step 2: Image tokenization - Split images into patches and convert to tokens
        input_img_tokens = self.image_tokenizer(posed_input_images)  # [b*v, n_patches, d]

        _, n_patches, d = input_img_tokens.size()  # [b*v, n_patches, d]
        #*Reshape input tokens - Merge batch and view dimensions
        input_img_tokens = input_img_tokens.reshape(b, v_input * n_patches, d)  # [b, v*n_patches, d]
        
        #& Step 3: Initialize latent vectors - Learnable scene representation
        #* latent_vector_tokens: [b, 3072, 768] - Learnable scene latent tokens
        #* this corresponds to the scene latents in rayZer
        latent_vector_tokens = self.n_light_field_latent.expand(b, -1, -1) # [b, n_latent_vectors, d]
        encoder_input_tokens = torch.cat((latent_vector_tokens, input_img_tokens), dim=1) # [b, n_latent_vectors + v*n_patches, d]

        #& Step 4: Transformer encoder - Process through 12 transformer blocks
        #* intermediate_tokens is the concatenation of the latent tokens and the input image tokens
        intermediate_tokens = self.pass_layers(self.transformer_encoder, encoder_input_tokens, gradient_checkpoint=True, checkpoint_every=checkpoint_every)

        latent_tokens, input_img_tokens = intermediate_tokens.split(
            [self.config.model.transformer.n_latent_vectors, v_input * n_patches], dim=1
        ) # [b, n_latent_vectors, d], [b, v*n_patches, d]
        
        return latent_tokens, n_patches, d

    def renderer(self, latent_tokens, target, n_patches, d, checkpoint_every=None):
        """
        从场景 latent_tokens 和目标 ray maps 解码渲染结果
        
        Args:
            latent_tokens: [b, n_latent_vectors, d] - 场景潜在表示
            target: 包含 target.ray_o, target.ray_d, target.image_h_w 的 edict
            n_patches: int - 每个视图的 patch 数量
            d: int - token 维度
            checkpoint_every: 梯度检查点间隔，如果为 None 则使用 config 中的值
            
        Returns:
            rendered_images: [b, v_target, 3, h, w] - 渲染的 RGB 图像
        """
        if checkpoint_every is None:
            checkpoint_every = self.config.training.grad_checkpoint_every
        n_latent_vectors = self.config.model.transformer.n_latent_vectors
        
        #& Step 1: Build target pose conditioning - Only use pose info (no RGB image)
        target_pose_cond= self.get_posed_input(ray_o=target.ray_o, ray_d=target.ray_d)
        b, v_target, c, h, w = target_pose_cond.size()
        
        #& Step 2: Replicate latent tokens for each target view
        repeated_latent_tokens = repeat(
                                latent_tokens,
                                'b nl d -> (b v_target) nl d', 
                                v_target=v_target) 

        #& Step 3: Target pose tokenization - Convert target pose conditioning to tokens
        target_pose_tokens = self.target_pose_tokenizer(target_pose_cond) # [b*v_target, n_patches, d]
        
        #& Step 5: Put target pose tokens and latent tokens through decoder transformer
        decoder_input_tokens = torch.cat((target_pose_tokens, repeated_latent_tokens), dim=1) # [b*v_target, n_latent_vectors + n_patches, d]
        decoder_input_tokens = self.transformer_input_layernorm_decoder(decoder_input_tokens)

        transformer_output_tokens = self.pass_layers(self.transformer_decoder, decoder_input_tokens, gradient_checkpoint=True, checkpoint_every=checkpoint_every)
        target_image_tokens, _ = transformer_output_tokens.split(
            [n_patches, n_latent_vectors], dim=1
        ) # [b*v_target, n_patches, d], [b*v_target, n_latent_vectors, d]

        #& Step 6: Image token decoding - Decode tokens to RGB pixel values
        rendered_images = self.image_token_decoder(target_image_tokens)
        
        height, width = target.image_h_w

        patch_size = self.config.model.target_pose_tokenizer.patch_size
        rendered_images = rearrange(
            rendered_images, "(b v) (h w) (p1 p2 c) -> b v c (h p1) (w p2)",
            v=v_target,
            h=height // patch_size, 
            w=width // patch_size, 
            p1=patch_size, 
            p2=patch_size, 
            c=3
        )
        
        return rendered_images

    def renderer_albedo(self, latent_tokens, target, n_patches, d, checkpoint_every=None):
        """
        从场景 latent_tokens 和目标 ray maps 解码渲染 albedo 结果
        
        Args:
            latent_tokens: [b, n_latent_vectors, d] - 场景潜在表示
            target: 包含 target.ray_o, target.ray_d, target.image_h_w 的 edict
            n_patches: int - 每个视图的 patch 数量
            d: int - token 维度
            checkpoint_every: 梯度检查点间隔，如果为 None 则使用 config 中的值
            
        Returns:
            rendered_albedos: [b, v_target, 3, h, w] - 渲染的 albedo 图像
        """
        if checkpoint_every is None:
            checkpoint_every = self.config.training.grad_checkpoint_every
        n_latent_vectors = self.config.model.transformer.n_latent_vectors
        
        #& Step 1: Build target pose conditioning - Only use pose info (no RGB image)
        target_pose_cond= self.get_posed_input(ray_o=target.ray_o, ray_d=target.ray_d)
        b, v_target, c, h, w = target_pose_cond.size()
        
        #& Step 2: Replicate latent tokens for each target view
        repeated_latent_tokens = repeat(
                                latent_tokens,
                                'b nl d -> (b v_target) nl d', 
                                v_target=v_target) 

        #& Step 3: Target pose tokenization - Convert target pose conditioning to tokens
        target_pose_tokens = self.target_pose_tokenizer(target_pose_cond) # [b*v_target, n_patches, d]
        
        #& Step 4: Put target pose tokens and latent tokens through albedo decoder transformer
        decoder_input_tokens = torch.cat((target_pose_tokens, repeated_latent_tokens), dim=1) # [b*v_target, n_latent_vectors + n_patches, d]
        decoder_input_tokens = self.transformer_input_layernorm_decoder(decoder_input_tokens)

        transformer_albedo_output_tokens = self.pass_layers(self.transformer_decoder_albedo, decoder_input_tokens, gradient_checkpoint=True, checkpoint_every=checkpoint_every)
        target_albedo_tokens, _ = transformer_albedo_output_tokens.split(
            [n_patches, n_latent_vectors], dim=1
        ) # [b*v_target, n_patches, d], [b*v_target, n_latent_vectors, d]

        #& Step 5: Image token decoding - Decode tokens to RGB pixel values (albedo)
        rendered_albedos = self.image_token_decoder(target_albedo_tokens)
        
        height, width = target.image_h_w

        patch_size = self.config.model.target_pose_tokenizer.patch_size
        rendered_albedos = rearrange(
            rendered_albedos, "(b v) (h w) (p1 p2 c) -> b v c (h p1) (w p2)",
            v=v_target,
            h=height // patch_size, 
            w=width // patch_size, 
            p1=patch_size, 
            p2=patch_size, 
            c=3
        )
        
        return rendered_albedos

    def forward(self, data_batch, has_target_image=True):
        #& Step 1: Data preprocessing - Extract input and target data from data_batch, compute rays
        # input.image: [b, v_input, 3, h, w] - Input RGB images (v_input=2 views)
        # input.ray_o: [b, v_input, 3, h, w] - Ray origins
        # input.ray_d: [b, v_input, 3, h, w] - Ray directions
        # target.image: [b, v_target, 3, h, w] - Target RGB images (v_target=6 views)
        # target.ray_o: [b, v_target, 3, h, w] - Target ray origins
        # target.ray_d: [b, v_target, 3, h, w] - Target ray directions
        input, target = self.process_data(data_batch, has_target_image=has_target_image, target_has_input = self.config.training.target_has_input, compute_rays=True)
        
        #& Step 2: Reconstructor - Get scene latent_tokens from input images
        latent_tokens, n_patches, d = self.reconstructor(input)

        #& Step 3: Editor - edit latent tokens with configured lighting signals (envmap / point_light / both)
        condition_tokens = self._build_editor_condition_tokens(input, d)
        print("condition_tokens:", condition_tokens.shape)
        if condition_tokens is not None:
            editor_input_tokens = torch.cat([latent_tokens, condition_tokens], dim=1)
            checkpoint_every = self.config.training.grad_checkpoint_every
            editor_output_tokens = self.pass_layers(
                self.transformer_editor,
                editor_input_tokens,
                gradient_checkpoint=True,
                checkpoint_every=checkpoint_every
            )
            n_latent_vectors = self.config.model.transformer.n_latent_vectors
            latent_tokens = editor_output_tokens[:, :n_latent_vectors, :]
        
        #& Step 4: Renderer - Decode results from target ray maps
        rendered_images = self.renderer(latent_tokens, target, n_patches, d)
        
        #& Step 4.5: Process albedo decoder if enabled
        rendered_albedos = None
        if self.use_albedo_decoder:
            rendered_albedos = self.renderer_albedo(latent_tokens, target, n_patches, d)
        
        
        #& Step 5: Compute loss (if target images are provided)
        # loss_metrics contains: L2 loss, LPIPS loss, perceptual loss, etc.
        # Use relit_images target split if available, otherwise fall back to original image
        if has_target_image:
            # Check if relit_images are available in target (from relit scene)
            if hasattr(target, 'relit_images') and target.relit_images is not None:
                target_images = target.relit_images
            else:
                target_images = target.image
            
            # Compute image loss
            image_loss_metrics = self.loss_computer(
                rendered_images,
                target_images
            )
            
            # Compute albedo loss if albedo decoder is enabled and target albedos are available
            if self.use_albedo_decoder and hasattr(target, 'albedos') and target.albedos is not None:
                albedo_loss_metrics = self.loss_computer(
                    rendered_albedos,
                    target.albedos
                )
                
                # Combine losses: add image loss and albedo loss
                # Sum all loss values
                combined_loss = image_loss_metrics.loss + albedo_loss_metrics.loss
                
                # Create combined loss metrics with both image and albedo losses
                loss_metrics = edict(
                    loss=combined_loss,
                    image_loss=image_loss_metrics.loss,
                    albedo_loss=albedo_loss_metrics.loss,
                    l2_loss=image_loss_metrics.l2_loss + albedo_loss_metrics.l2_loss,
                    lpips_loss=image_loss_metrics.lpips_loss + albedo_loss_metrics.lpips_loss,
                    perceptual_loss=image_loss_metrics.perceptual_loss + albedo_loss_metrics.perceptual_loss,
                    psnr=image_loss_metrics.psnr,  # Keep image PSNR
                    norm_perceptual_loss=image_loss_metrics.norm_perceptual_loss + albedo_loss_metrics.norm_perceptual_loss,
                    norm_lpips_loss=image_loss_metrics.norm_lpips_loss + albedo_loss_metrics.norm_lpips_loss,
                )
            else:
                # If albedo decoder is enabled but no target albedos, still use image loss only
                # (albedo decoder will still generate albedos, but no loss is computed)
                loss_metrics = image_loss_metrics
        else:
            loss_metrics = None

        #& Step 5: Return results
        result = edict(
            input=input,
            target=target,
            loss_metrics=loss_metrics,
            render=rendered_images        
            )
        
        if rendered_albedos is not None:
            result.render_albedo = rendered_albedos
        
        return result

    def edit_scene_with_env(self, latent_tokens, input_with_env):
        """
        Edit scene latent tokens using configured lighting conditions from input.
        
        Args:
            latent_tokens: [b, n_latent_vectors, d] - Scene latent representation
            input_with_env: Input dict containing env and/or point-light conditions
            
        Returns:
            edited_latent_tokens: [b, n_latent_vectors, d] - Edited scene latent representation
        """
        edited_latent_tokens = latent_tokens
        condition_tokens = self._build_editor_condition_tokens(input_with_env, latent_tokens.shape[-1])
        if condition_tokens is None:
            return edited_latent_tokens

        editor_input_tokens = torch.cat([latent_tokens, condition_tokens], dim=1)
        checkpoint_every = self.config.training.grad_checkpoint_every
        editor_output_tokens = self.pass_layers(
            self.transformer_editor,
            editor_input_tokens,
            gradient_checkpoint=True,
            checkpoint_every=checkpoint_every
        )
        n_latent_vectors = self.config.model.transformer.n_latent_vectors
        edited_latent_tokens = editor_output_tokens[:, :n_latent_vectors, :]
        return edited_latent_tokens


    @torch.no_grad()
    def render_video(self, data_batch, traj_type="interpolate", num_frames=60, loop_video=False, order_poses=False):
        """
        Render a video from the model.
        
        Args:
            result: Edict from forward pass or just data
            traj_type: Type of trajectory
            num_frames: Number of frames to render
            loop_video: Whether to loop the video
            order_poses: Whether to order poses
            
        Returns:
            result: Updated with video rendering
        """
    
        if data_batch.input is None:
            input, target = self.process_data(data_batch, has_target_image=False, target_has_input=self.config.training.target_has_input, compute_rays=True)
            data_batch = edict(input=input, target=target)
        else:
            input, target = data_batch.input, data_batch.target
        
        # Prepare input tokens; [b, v, 3+6, h, w]
        posed_images = self.get_posed_input(
            images=input.image, ray_o=input.ray_o, ray_d=input.ray_d
        )
        bs, v_input, c, h, w = posed_images.size()

        input_img_tokens = self.image_tokenizer(posed_images)  # [b*v_input, n_patches, d]
        _, n_patches, d = input_img_tokens.size()  # [b*v_input, n_patches, d]
        input_img_tokens = input_img_tokens.reshape(bs, v_input * n_patches, d)  # [b, v_input*n_patches, d]

        latent_vector_tokens = self.n_light_field_latent.expand(bs, -1, -1) # [b, n_latent_vectors, d]
        encoder_input_tokens = torch.cat((latent_vector_tokens, input_img_tokens), dim=1) # [b, n_latent_vectors + v*n_patches, d]

        # Process through encoder
        intermediate_tokens = self.pass_layers(self.transformer_encoder, encoder_input_tokens, gradient_checkpoint=False)
        latent_tokens, _ = intermediate_tokens.split(
            [self.config.model.transformer.n_latent_vectors, v_input * n_patches], dim=1
        ) # [b, n_latent_vectors, d]

        if traj_type == "interpolate":
            c2ws = input.c2w # [b, v, 4, 4]
            fxfycxcy = input.fxfycxcy #  [b, v, 4]
            device = input.c2w.device

            # Create intrinsics from fxfycxcy
            intrinsics = torch.zeros((c2ws.shape[0], c2ws.shape[1], 3, 3), device=device) # [b, v, 3, 3]
            intrinsics[:, :,  0, 0] = fxfycxcy[:, :, 0]
            intrinsics[:, :,  1, 1] = fxfycxcy[:, :, 1]
            intrinsics[:, :,  0, 2] = fxfycxcy[:, :, 2]
            intrinsics[:, :,  1, 2] = fxfycxcy[:, :, 3]

            # Loop video if requested
            if loop_video:
                c2ws = torch.cat([c2ws, c2ws[:, [0], :]], dim=1)
                intrinsics = torch.cat([intrinsics, intrinsics[:, [0], :]], dim=1)

            # Interpolate camera poses
            all_c2ws, all_intrinsics = [], []
            for b in range(input.image.size(0)):
                cur_c2ws, cur_intrinsics = camera_utils.get_interpolated_poses_many(
                    c2ws[b, :, :3, :4], intrinsics[b], num_frames, order_poses=order_poses
                )
                all_c2ws.append(cur_c2ws.to(device))
                all_intrinsics.append(cur_intrinsics.to(device))

            all_c2ws = torch.stack(all_c2ws, dim=0) # [b, num_frames, 3, 4]
            all_intrinsics = torch.stack(all_intrinsics, dim=0) # [b, num_frames, 3, 3]

            # Add homogeneous row to c2ws
            homogeneous_row = torch.tensor([[[0, 0, 0, 1]]], device=device).expand(all_c2ws.shape[0], all_c2ws.shape[1], -1, -1)
            all_c2ws = torch.cat([all_c2ws, homogeneous_row], dim=2)

            # Convert intrinsics to fxfycxcy format
            all_fxfycxcy = torch.zeros((all_intrinsics.shape[0], all_intrinsics.shape[1], 4), device=device)
            all_fxfycxcy[:, :, 0] = all_intrinsics[:, :, 0, 0]  # fx
            all_fxfycxcy[:, :, 1] = all_intrinsics[:, :, 1, 1]  # fy
            all_fxfycxcy[:, :, 2] = all_intrinsics[:, :, 0, 2]  # cx
            all_fxfycxcy[:, :, 3] = all_intrinsics[:, :, 1, 2]  # cy

        # Compute rays for rendering
        rendering_ray_o, rendering_ray_d = self.process_data.compute_rays(
            fxfycxcy=all_fxfycxcy, c2w=all_c2ws, h=h, w=w, device=device
        )

        # Get pose conditioning for target views
        target_pose_cond = self.get_posed_input(
            ray_o=rendering_ray_o.to(input.image.device), 
            ray_d=rendering_ray_d.to(input.image.device)
        )
                
        _, num_views, c, h, w = target_pose_cond.size()
    
        target_pose_tokens = self.target_pose_tokenizer(target_pose_cond) # [bs*v_target, n_patches, d]
        _, n_patches, d = target_pose_tokens.size()  # [b*v_target, n_patches, d]
        target_pose_tokens = target_pose_tokens.reshape(bs, num_views * n_patches, d)  # [b, v_target*n_patches, d]

        view_chunk_size = 4
        video_rendering_list = []
        
        for cur_chunk in range(0, num_views, view_chunk_size):
            cur_view_chunk_size = min(view_chunk_size, num_views - cur_chunk)
            
            # Get current chunk of target pose tokens
            start_idx, end_idx = cur_chunk * n_patches, (cur_chunk + cur_view_chunk_size) * n_patches
            cur_target_pose_tokens = rearrange(target_pose_tokens[:, start_idx:end_idx,: ], 
                                               "b (v_chunk p) d -> (b v_chunk) p d", 
                                               v_chunk=cur_view_chunk_size, p=n_patches)

            cur_repeated_latent_tokens = repeat(
                latent_tokens,
                'b nl d -> (b v_chunk) nl d', 
                v_chunk=cur_view_chunk_size
                )

            decoder_input_tokens = torch.cat((cur_target_pose_tokens, cur_repeated_latent_tokens), dim=1)
            decoder_input_tokens = self.transformer_input_layernorm_decoder(decoder_input_tokens)

            transformer_output_tokens = self.pass_layers(
                self.transformer_decoder, 
                decoder_input_tokens, 
                gradient_checkpoint=False
            )

            target_image_tokens, _ = transformer_output_tokens.split(
                [n_patches, self.config.model.transformer.n_latent_vectors], dim=1
            )

            # Decode to images
            height, width = target.image_h_w
            patch_size = self.config.model.target_pose_tokenizer.patch_size
            
            video_rendering = self.image_token_decoder(target_image_tokens)
            video_rendering = rearrange(
                video_rendering, "(b v) (h w) (p1 p2 c) -> b v c (h p1) (w p2)",
                v=cur_view_chunk_size,
                h=height // patch_size, 
                w=width // patch_size, 
                p1=patch_size, 
                p2=patch_size, 
                c=3
            ).cpu()

            video_rendering_list.append(video_rendering)

        # Combine all chunks
        video_rendering = torch.cat(video_rendering_list, dim=1)
        data_batch.video_rendering = video_rendering

        return data_batch

    @torch.no_grad()
    def load_ckpt(self, load_path):
        if os.path.isdir(load_path):
            ckpt_names = [file_name for file_name in os.listdir(load_path) if file_name.endswith(".pt")]
            ckpt_names = sorted(ckpt_names, key=lambda x: x)
            ckpt_paths = [os.path.join(load_path, ckpt_name) for ckpt_name in ckpt_names]
        else:
            if os.path.exists(load_path) and load_path.endswith(".pt"):
                ckpt_paths = [load_path]
            else:
                ckpt_paths = []
        
        # Check if checkpoint files exist
        if len(ckpt_paths) == 0:
            print(f"No checkpoint files found in {load_path}")
            return None
        
        try:
            checkpoint = torch.load(ckpt_paths[-1], map_location="cpu", weights_only=True)
        except Exception as e:
            traceback.print_exc()
            print(f"Failed to load {ckpt_paths[-1]}: {e}")
            return None
        
        checkpoint_state_dict = checkpoint["model"]
        
        # Check if transformer_decoder_albedo exists in checkpoint
        has_albedo_decoder_in_ckpt = any(
            key.startswith("transformer_decoder_albedo") for key in checkpoint_state_dict.keys()
        )
        
        # If we need albedo decoder but it doesn't exist in checkpoint, initialize from transformer_decoder
        if self.use_albedo_decoder and not has_albedo_decoder_in_ckpt:
            print("!!!transformer_decoder_albedo not found in checkpoint, initializing from transformer_decoder weights")
            # Copy weights from transformer_decoder to transformer_decoder_albedo
            # First collect all keys to copy (to avoid mutating dict during iteration)
            keys_to_copy = {}
            for i in range(len(self.transformer_decoder)):
                # Copy each transformer block
                src_prefix = f"transformer_decoder.{i}."
                tgt_prefix = f"transformer_decoder_albedo.{i}."
                
                for key, value in checkpoint_state_dict.items():
                    if key.startswith(src_prefix):
                        new_key = key.replace(src_prefix, tgt_prefix)
                        keys_to_copy[new_key] = value.clone()
            
            # Now add all collected keys to the state dict
            checkpoint_state_dict.update(keys_to_copy)
        
        self.load_state_dict(checkpoint_state_dict, strict=False)
        return 0

    @torch.no_grad()
    def init_from_LVSM(self, lvsm_checkpoint_path):
        """
        从 Images2LatentScene 模型初始化 LatentSceneEditor
        
        Args:
            lvsm_checkpoint_path: Images2LatentScene 模型的 checkpoint 路径
            
        Returns:
            0 if successful, None if failed
        """
        # Import Images2LatentScene
        from .LVSM_scene_encoder_decoder import Images2LatentScene
        
        # Load checkpoint
        if os.path.isdir(lvsm_checkpoint_path):
            ckpt_names = [file_name for file_name in os.listdir(lvsm_checkpoint_path) if file_name.endswith(".pt")]
            ckpt_names = sorted(ckpt_names, key=lambda x: x)
            ckpt_paths = [os.path.join(lvsm_checkpoint_path, ckpt_name) for ckpt_name in ckpt_names]
        else:
            ckpt_paths = [lvsm_checkpoint_path]
        
        try:
            checkpoint = torch.load(ckpt_paths[-1], map_location="cpu", weights_only=True)
        except:
            traceback.print_exc()
            print(f"Failed to load LVSM checkpoint from {ckpt_paths[-1]}")
            return None
        
        # Get state dicts directly from checkpoint
        checkpoint_state_dict = checkpoint["model"]
        editor_state_dict = self.state_dict()
        
        # Copy matching weights from checkpoint to LatentSceneEditor
        matched_keys = []
        unmatched_keys = []
        ignored_keys = []
        
        # Handle albedo decoder weights from checkpoint
        # If we don't use albedo decoder but checkpoint has it, ignore those weights
        # If we use albedo decoder but checkpoint doesn't have it, we'll initialize from decoder later
        if not self.use_albedo_decoder:
            # Ignore albedo decoder weights if they exist in checkpoint
            for key in list(checkpoint_state_dict.keys()):
                if key.startswith("transformer_decoder_albedo"):
                    ignored_keys.append(key)
        else:
            # We use albedo decoder - check if checkpoint has it
            has_albedo_decoder_in_ckpt = any(
                key.startswith("transformer_decoder_albedo") for key in checkpoint_state_dict.keys()
            )
            
            if not has_albedo_decoder_in_ckpt:
                # Checkpoint doesn't have albedo decoder, initialize from transformer_decoder
                print("transformer_decoder_albedo not found in checkpoint, initializing from transformer_decoder weights")
                # Copy weights from transformer_decoder to transformer_decoder_albedo
                keys_to_copy = {}
                for i in range(len(self.transformer_decoder)):
                    # Copy each transformer block
                    src_prefix = f"transformer_decoder.{i}."
                    tgt_prefix = f"transformer_decoder_albedo.{i}."
                    
                    for key, value in checkpoint_state_dict.items():
                        if key.startswith(src_prefix):
                            new_key = key.replace(src_prefix, tgt_prefix)
                            keys_to_copy[new_key] = value.clone()
                
                # Add copied keys to checkpoint_state_dict for matching
                checkpoint_state_dict.update(keys_to_copy)
        
        # Now match all keys
        for key in editor_state_dict.keys():
            if key in checkpoint_state_dict:
                # Check if shapes match
                if editor_state_dict[key].shape == checkpoint_state_dict[key].shape:
                    editor_state_dict[key] = checkpoint_state_dict[key]
                    matched_keys.append(key)
                else:
                    print(f"Warning: Shape mismatch for {key}: {editor_state_dict[key].shape} vs {checkpoint_state_dict[key].shape}")
                    unmatched_keys.append(key)
            else:
                unmatched_keys.append(key)
        
        # Optionally initialize editor from decoder weights
        editor_config = self.config.model.transformer.get("editor", {})
        init_editor_from_decoder = editor_config.get("init_from_decoder", False)
        
        if init_editor_from_decoder:
            # Copy weights from decoder to editor (useful since both process latent tokens)
            editor_n_layer = len(self.transformer_editor)
            decoder_n_layer = len(self.transformer_decoder)
            
            # Copy decoder weights to editor (use last N layers if editor has fewer layers)
            copy_start_idx = max(0, decoder_n_layer - editor_n_layer)
            copied_layers = 0
            
            for editor_idx in range(editor_n_layer):
                decoder_idx = copy_start_idx + editor_idx
                if decoder_idx < decoder_n_layer:
                    # Copy weights from decoder layer to editor layer
                    editor_block = self.transformer_editor[editor_idx]
                    decoder_block = self.transformer_decoder[decoder_idx]
                    
                    editor_block.load_state_dict(decoder_block.state_dict(), strict=True)
                    copied_layers += 1
            
            print(f"  - Initialized {copied_layers}/{editor_n_layer} editor layers from decoder")
        
        # Load the combined state dict
        self.load_state_dict(editor_state_dict, strict=False)
        
        print(f"Initialized LatentSceneEditor from Images2LatentScene:")
        print(f"  - Matched keys: {len(matched_keys)}")
        print(f"  - Unmatched keys (using new weights): {len(unmatched_keys)}")
        if ignored_keys:
            print(f"  - Ignored keys (not used in this model): {len(ignored_keys)}")
        if unmatched_keys:
            print(f"  - New components: {', '.join(unmatched_keys)}")
        
        return 0


