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
        
        # Environment tokenizer (RGB + ray directions)
        env_config = self.config.model.get("env_tokenizer", {})
        if env_config:
            self.env_tokenizer = self._create_tokenizer(
                in_channels = env_config.get("in_channels", 6),  # 3 RGB + 3 ray_d
                patch_size = env_config.get("patch_size", 8),
                d_model = self.config.model.transformer.d
            )
        else:
            # Default: 3 RGB + 3 ray_d = 6 channels
            self.env_tokenizer = self._create_tokenizer(
                in_channels = 6,
                patch_size = 8,
                d_model = self.config.model.transformer.d
            )
        
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
        else:
            # Encoder
            for block in self.transformer_encoder:
                block.apply(init_weights)

            # Decoder
            for block in self.transformer_decoder:
                block.apply(init_weights)

                
        self.transformer_encoder = nn.ModuleList(self.transformer_encoder)
        self.transformer_decoder = nn.ModuleList(self.transformer_decoder)
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

        # todo: step3: Editor - Edit the scene latent_tokens(placeholder for now, add layer later)
        # latent_tokens = self.editor(latent_tokens)
        
        #& Step 4: Renderer - Decode results from target ray maps
        rendered_images = self.renderer(latent_tokens, target, n_patches, d)
        
        #& Step 5: Compute loss (if target images are provided)
        # loss_metrics contains: L2 loss, LPIPS loss, perceptual loss, etc.
        if has_target_image:
            loss_metrics = self.loss_computer(
                rendered_images,
                target.image
            )
        else:
            loss_metrics = None

        #& Step 5: Return results
        result = edict(
            input=input,
            target=target,
            loss_metrics=loss_metrics,
            render=rendered_images        
            )
        
        return result


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
        
        self.load_state_dict(checkpoint["model"], strict=False)
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
        
        # Create Images2LatentScene model and load weights
        lvsm_model = Images2LatentScene(self.config)
        lvsm_model.load_state_dict(checkpoint["model"], strict=False)
        
        # Get state dicts
        lvsm_state_dict = lvsm_model.state_dict()
        editor_state_dict = self.state_dict()
        
        # Copy matching weights from Images2LatentScene to LatentSceneEditor
        matched_keys = []
        unmatched_keys = []
        for key in editor_state_dict.keys():
            if key in lvsm_state_dict:
                # Check if shapes match
                if editor_state_dict[key].shape == lvsm_state_dict[key].shape:
                    editor_state_dict[key] = lvsm_state_dict[key]
                    matched_keys.append(key)
                else:
                    print(f"Warning: Shape mismatch for {key}: {editor_state_dict[key].shape} vs {lvsm_state_dict[key].shape}")
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
        if unmatched_keys:
            print(f"  - New components: {', '.join(unmatched_keys)}")
        
        return 0


