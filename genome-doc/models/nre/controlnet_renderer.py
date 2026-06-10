"""
NRE — Neural Re-Rendering Engine (ControlNet + SD 1.5)

Generates photorealistic clean document images conditioned on:
1. A spatial skeleton map (rendered from the Document Genome)
2. A style embedding (from SIR) injected via cross-attention

Architecture:
    skeleton_map + style_embedding → ControlNet → SD 1.5 U-Net → clean document

Key features:
- Stable Diffusion 1.5 base (frozen)
- ControlNet adapter for spatial conditioning
- Style embedding projection (512 → 768) injected via cross-attention
- LoRA on cross-attention layers for efficiency
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


class StyleProjector(nn.Module):
    """
    Projects SIR style embeddings (512-dim) to the text encoder
    hidden dimension (768 for SD 1.5) for cross-attention injection.
    """

    def __init__(
        self,
        style_dim: int = 512,
        target_dim: int = 768,
        num_tokens: int = 4,
    ):
        """
        Args:
            style_dim: Input style embedding dimension (from SIR).
            target_dim: SD cross-attention dimension (768 for SD 1.5).
            num_tokens: Number of pseudo-tokens to generate from the style vector.
                       More tokens = finer style control but higher compute.
        """
        super().__init__()
        self.num_tokens = num_tokens
        self.target_dim = target_dim

        # Project style to multiple pseudo-tokens
        self.projection = nn.Sequential(
            nn.Linear(style_dim, target_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(target_dim * 2, target_dim * num_tokens),
        )

        self.layer_norm = nn.LayerNorm(target_dim)

    def forward(self, style_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            style_embedding: (B, style_dim) style vectors from SIR

        Returns:
            (B, num_tokens, target_dim) pseudo-token embeddings for
            cross-attention conditioning
        """
        B = style_embedding.shape[0]
        projected = self.projection(style_embedding)               # (B, target*N)
        tokens = projected.reshape(B, self.num_tokens, self.target_dim)  # (B, N, target)
        tokens = self.layer_norm(tokens)
        return tokens


class NREControlNet(nn.Module):
    """
    Neural Re-Rendering Engine using ControlNet + Stable Diffusion 1.5.

    The model takes:
    - A skeleton map (rendered from Document Genome) as spatial conditioning
    - A style embedding (from SIR) as cross-attention conditioning

    And generates a photorealistic clean document image.
    """

    def __init__(
        self,
        sd_model_id: str = "runwayml/stable-diffusion-v1-5",
        controlnet_model_id: str = "lllyasviel/sd-controlnet-canny",
        style_dim: int = 512,
        num_style_tokens: int = 4,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        use_lora: bool = True,
    ):
        """
        Args:
            sd_model_id: HuggingFace model ID for Stable Diffusion 1.5.
            controlnet_model_id: HuggingFace model ID for ControlNet.
            style_dim: SIR style embedding dimension.
            num_style_tokens: Number of pseudo-tokens for style injection.
            lora_rank: LoRA adapter rank.
            lora_alpha: LoRA alpha scaling.
            use_lora: Whether to apply LoRA adapters.
        """
        super().__init__()

        self.sd_model_id = sd_model_id
        self.controlnet_model_id = controlnet_model_id
        self.style_dim = style_dim
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha

        # Style projector
        self.style_projector = StyleProjector(
            style_dim=style_dim,
            target_dim=768,  # SD 1.5 cross-attn dim
            num_tokens=num_style_tokens,
        )

        # Pipeline components (lazy-loaded)
        self._pipeline = None
        self._controlnet = None
        self._unet = None
        self._vae = None
        self._text_encoder = None
        self._tokenizer = None
        self._scheduler = None
        self._initialized = False

    def _sync_pipeline_components(self) -> None:
        """Keep the Diffusers pipeline references aligned after component swaps."""
        if self._pipeline is None:
            return
        if self._controlnet is not None:
            self._pipeline.controlnet = self._controlnet
        if self._unet is not None:
            self._pipeline.unet = self._unet

    def initialize(self, device: torch.device = torch.device("cpu")) -> None:
        """
        Lazy-initialize the diffusion components.
        Call this before training or inference.
        """
        if self._initialized:
            return

        from diffusers import (
            StableDiffusionControlNetPipeline,
            ControlNetModel,
            UNet2DConditionModel,
            AutoencoderKL,
            DDPMScheduler,
        )

        print(f"Loading ControlNet from {self.controlnet_model_id}...")
        self._controlnet = ControlNetModel.from_pretrained(
            self.controlnet_model_id,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        )

        print(f"Loading SD 1.5 pipeline from {self.sd_model_id}...")
        self._pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            self.sd_model_id,
            controlnet=self._controlnet,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            safety_checker=None,
        )

        self._unet = self._pipeline.unet
        self._vae = self._pipeline.vae
        self._text_encoder = self._pipeline.text_encoder
        self._tokenizer = self._pipeline.tokenizer
        self._scheduler = DDPMScheduler.from_config(self._pipeline.scheduler.config)

        # Freeze text encoder (pretrained CLIP, no training needed)
        for param in self._text_encoder.parameters():
            param.requires_grad = False

        # Freeze base SD model
        self._freeze_base_model()

        # Apply LoRA if configured
        if self.use_lora:
            self._setup_lora()

        self._initialized = True
        self.to(device)

        # Count parameters
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"NRE params: {total:,} total, {trainable:,} trainable")

    def _freeze_base_model(self) -> None:
        """Freeze SD base model, keep ControlNet trainable."""
        # Freeze UNet
        for param in self._unet.parameters():
            param.requires_grad = False

        # Freeze VAE
        for param in self._vae.parameters():
            param.requires_grad = False

        # ControlNet stays trainable
        for param in self._controlnet.parameters():
            param.requires_grad = True

    def _setup_lora(self) -> None:
        """Apply LoRA to UNet cross-attention layers."""
        try:
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                target_modules=["to_q", "to_v", "to_k", "to_out.0"],
                lora_dropout=0.05,
            )

            self._unet = get_peft_model(self._unet, lora_config)
            print(f"LoRA applied to UNet (rank={self.lora_rank})")
            self._unet.print_trainable_parameters()

        except ImportError:
            print("WARNING: peft not installed. Skipping UNet LoRA.")

    def encode_style(self, style_embedding: torch.Tensor) -> torch.Tensor:
        """
        Encode style embedding into pseudo-tokens for cross-attention.

        Args:
            style_embedding: (B, 512) from SIR

        Returns:
            (B, num_tokens, 768) pseudo-token embeddings
        """
        return self.style_projector(style_embedding)

    def _encode_prompt(
        self,
        text_prompt: list[str],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Encode text prompts using the SD 1.5 CLIP text encoder.

        Args:
            text_prompt: List of B text strings (genome token sequences)
            device: Target device

        Returns:
            (B, 77, 768) CLIP text embeddings
        """
        tokens = self._tokenizer(
            text_prompt,
            padding="max_length",
            max_length=self._tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        text_embeddings = self._text_encoder(tokens.input_ids)[0]  # (B, 77, 768)
        return text_embeddings

    def _build_encoder_hidden_states(
        self,
        style_embedding: torch.Tensor,
        text_prompt: list[str] | None = None,
    ) -> torch.Tensor:
        """
        Build the combined encoder_hidden_states by concatenating
        CLIP text embeddings with style pseudo-tokens.

        Args:
            style_embedding: (B, 512) style vectors from SIR
            text_prompt: Optional list of B text strings. If None,
                         uses empty strings (unconditional).

        Returns:
            (B, 77 + num_style_tokens, 768) combined embeddings
        """
        B = style_embedding.shape[0]
        device = style_embedding.device

        # Text embeddings from CLIP
        if text_prompt is None:
            text_prompt = [""] * B
        text_emb = self._encode_prompt(text_prompt, device)  # (B, 77, 768)

        # Style pseudo-tokens
        style_tokens = self.encode_style(style_embedding)  # (B, N, 768)

        # Concatenate: [text_embeddings | style_tokens]
        combined = torch.cat([text_emb, style_tokens], dim=1)  # (B, 77+N, 768)
        return combined

    def prepare_skeleton_conditioning(
        self,
        skeleton: torch.Tensor,
    ) -> torch.Tensor:
        """
        Prepare skeleton image for ControlNet conditioning.

        Args:
            skeleton: (B, 3, H, W) skeleton images in [0, 1]

        Returns:
            (B, 3, H, W) conditioned for ControlNet
        """
        # ControlNet expects images in [0, 1] already
        if skeleton.shape[2:] != (512, 512):
            skeleton = F.interpolate(
                skeleton, size=(512, 512), mode="bilinear", align_corners=False
            )
        return skeleton

    def forward(
        self,
        skeleton: torch.Tensor,
        style_embedding: torch.Tensor,
        clean_image: torch.Tensor,
        text_prompt: list[str] | None = None,
        timesteps: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Training forward pass.

        During training, we:
        1. Encode clean_image to latent space via VAE encoder
        2. Add noise at random timestep
        3. Build combined text+style conditioning
        4. Predict noise using U-Net + ControlNet conditioning
        5. Compute loss between predicted and actual noise

        Args:
            skeleton: (B, 3, 512, 512) skeleton images
            style_embedding: (B, 512) style vectors
            clean_image: (B, 3, 512, 512) target clean images
            text_prompt: List of B text strings (genome token sequences)
            timesteps: (B,) optional specific timesteps (random if None)

        Returns:
            Dict with 'noise_pred', 'noise_target', 'timesteps', 'loss'
        """
        assert self._initialized, "Call initialize(device) first"

        B = skeleton.shape[0]
        device = skeleton.device

        # 1. Encode clean image to latent space
        with torch.no_grad():
            vae_input = clean_image.to(dtype=self._vae.dtype)
            if vae_input.min() >= 0.0 and vae_input.max() <= 1.0:
                vae_input = vae_input * 2.0 - 1.0
            latents = self._vae.encode(vae_input).latent_dist.sample()
            latents = latents * self._vae.config.scaling_factor

        # 2. Sample noise
        noise = torch.randn_like(latents)

        # 3. Sample timesteps
        if timesteps is None:
            timesteps = torch.randint(
                0, self._scheduler.config.num_train_timesteps,
                (B,), device=device
            ).long()

        # 4. Add noise to latents
        noisy_latents = self._scheduler.add_noise(latents, noise, timesteps)

        # 5. Build combined text + style conditioning
        encoder_hidden_states = self._build_encoder_hidden_states(
            style_embedding, text_prompt
        )  # (B, 77+N, 768)

        # 6. Prepare skeleton for ControlNet
        controlnet_input = self.prepare_skeleton_conditioning(skeleton)

        # 7. Get ControlNet output (down/mid block additional residuals)
        down_block_res, mid_block_res = self._controlnet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=controlnet_input,
            return_dict=False,
        )

        # 8. Predict noise with U-Net (conditioned by ControlNet residuals + text + style)
        noise_pred = self._unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=down_block_res,
            mid_block_additional_residual=mid_block_res,
        ).sample

        # 9. Compute MSE loss
        loss = F.mse_loss(noise_pred, noise)

        return {
            "noise_pred": noise_pred,
            "noise_target": noise,
            "timesteps": timesteps,
            "loss": loss,
        }

    @torch.no_grad()
    def generate(
        self,
        skeleton: torch.Tensor,
        style_embedding: torch.Tensor,
        text_prompt: list[str] | None = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
    ) -> torch.Tensor:
        """
        Generate clean document image from skeleton + style + text (inference).
        Supports Classifier-Free Guidance.

        Args:
            skeleton: (B, 3, 512, 512) skeleton images
            style_embedding: (B, 512) style vectors
            text_prompt: List of B text strings (genome token sequences)
            num_inference_steps: Number of diffusion denoising steps
            guidance_scale: Classifier-free guidance scale

        Returns:
            (B, 3, 512, 512) generated clean images in [0, 1]
        """
        assert self._initialized, "Call initialize(device) first"

        from diffusers import DDIMScheduler

        B = skeleton.shape[0]
        device = skeleton.device

        # Use DDIM for faster inference
        scheduler = DDIMScheduler.from_config(self._scheduler.config)
        scheduler.set_timesteps(num_inference_steps)

        # Build conditional embeddings (text + style)
        cond_hidden = self._build_encoder_hidden_states(
            style_embedding, text_prompt
        )  # (B, 77+N, 768)

        do_cfg = guidance_scale > 1.0

        if do_cfg:
            # Unconditional embeddings: empty text + zeroed style
            uncond_style = torch.zeros_like(style_embedding)
            uncond_hidden = self._build_encoder_hidden_states(
                uncond_style, None  # None → empty strings
            )  # (B, 77+N, 768)
            encoder_hidden_states = torch.cat([uncond_hidden, cond_hidden])  # (2B, 77+N, 768)
        else:
            encoder_hidden_states = cond_hidden

        # Prepare skeleton
        controlnet_input_cond = self.prepare_skeleton_conditioning(skeleton)
        if do_cfg:
            controlnet_input = torch.cat([controlnet_input_cond, controlnet_input_cond])
        else:
            controlnet_input = controlnet_input_cond

        # Start from random noise
        latents = torch.randn(
            (B, 4, 64, 64),  # SD 1.5 latent size at 512x512
            device=device,
            dtype=skeleton.dtype,
        )

        # Denoising loop
        for t in scheduler.timesteps:
            latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            t_batch = t.expand(latent_model_input.shape[0]).to(device)

            # ControlNet
            down_block_res, mid_block_res = self._controlnet(
                latent_model_input,
                t_batch,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=controlnet_input,
                return_dict=False,
            )

            # UNet prediction
            noise_pred = self._unet(
                latent_model_input,
                t_batch,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=down_block_res,
                mid_block_additional_residual=mid_block_res,
            ).sample

            # Apply classifier-free guidance
            if do_cfg:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_cond - noise_pred_uncond
                )

            # Scheduler step
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        # Decode latents to images
        latents = latents / self._vae.config.scaling_factor
        images = self._vae.decode(latents.to(self._vae.dtype)).sample

        # Clamp to [0, 1]
        images = (images / 2 + 0.5).clamp(0, 1)

        return images

    def get_trainable_parameters(self):
        """Return all trainable parameters for the optimizer."""
        params = []

        # Style projector (always trainable)
        params.extend(self.style_projector.parameters())

        # ControlNet (trainable)
        if self._controlnet is not None:
            params.extend(
                p for p in self._controlnet.parameters() if p.requires_grad
            )

        # UNet LoRA parameters (if applied)
        if self._unet is not None:
            params.extend(
                p for p in self._unet.parameters() if p.requires_grad
            )

        return params

    def save_pretrained(self, output_dir: str) -> None:
        """Save all trainable components."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save style projector
        torch.save(
            self.style_projector.state_dict(),
            output_dir / "style_projector.pt"
        )

        # Save ControlNet
        if self._controlnet is not None:
            self._controlnet.save_pretrained(output_dir / "controlnet")

        # Save UNet LoRA weights
        if self._unet is not None and self.use_lora:
            try:
                self._unet.save_pretrained(output_dir / "unet_lora")
            except Exception:
                torch.save(
                    {k: v for k, v in self._unet.state_dict().items()
                     if "lora" in k.lower()},
                    output_dir / "unet_lora_weights.pt"
                )

        print(f"NRE saved to {output_dir}")

    def load_pretrained(
        self,
        model_dir: str | Path,
        device: torch.device = torch.device("cpu"),
        is_trainable: bool = False,
    ) -> None:
        """Load trainable NRE components saved by save_pretrained."""
        model_path = Path(model_dir)
        self.initialize(device)

        projector_path = model_path / "style_projector.pt"
        if projector_path.exists():
            self.style_projector.load_state_dict(
                torch.load(str(projector_path), map_location=device, weights_only=True)
            )
            print(f"Loaded style projector from {projector_path}")
        else:
            print(f"WARNING: Missing style projector at {projector_path}")

        controlnet_path = model_path / "controlnet"
        if controlnet_path.exists() and self._controlnet is not None:
            from diffusers import ControlNetModel

            dtype = torch.float16 if device.type == "cuda" else torch.float32
            loaded_controlnet = ControlNetModel.from_pretrained(
                str(controlnet_path),
                torch_dtype=dtype,
            ).to(device)
            self._controlnet.load_state_dict(loaded_controlnet.state_dict())
            print(f"Loaded tuned ControlNet from {controlnet_path}")
        else:
            print(f"WARNING: Missing ControlNet weights at {controlnet_path}")

        unet_lora_path = model_path / "unet_lora"
        unet_lora_weights = model_path / "unet_lora_weights.pt"
        if self._unet is not None and self.use_lora:
            if (
                (unet_lora_path / "config.json").exists()
                and (
                    (unet_lora_path / "diffusion_pytorch_model.safetensors").exists()
                    or (unet_lora_path / "diffusion_pytorch_model.bin").exists()
                )
                and not (unet_lora_path / "adapter_config.json").exists()
            ):
                from diffusers import UNet2DConditionModel

                dtype = torch.float16 if device.type == "cuda" else torch.float32
                self._unet = UNet2DConditionModel.from_pretrained(
                    str(unet_lora_path),
                    torch_dtype=dtype,
                ).to(device)
                print(f"Loaded tuned UNet from {unet_lora_path}")
            elif unet_lora_path.exists():
                try:
                    loaded_into_existing_adapter = False
                    if is_trainable:
                        safetensors_path = unet_lora_path / "adapter_model.safetensors"
                        bin_path = unet_lora_path / "pytorch_model.bin"
                        if safetensors_path.exists():
                            from safetensors.torch import load_file
                            state_dict = load_file(str(safetensors_path), device=str(device))
                            self._unet.load_state_dict(state_dict, strict=False)
                            loaded_into_existing_adapter = True
                        elif bin_path.exists():
                            state_dict = torch.load(str(bin_path), map_location=device, weights_only=True)
                            self._unet.load_state_dict(state_dict, strict=False)
                            loaded_into_existing_adapter = True

                    if loaded_into_existing_adapter:
                        pass
                    elif hasattr(self._unet, "load_adapter"):
                        self._unet.load_adapter(
                            str(unet_lora_path),
                            adapter_name="trained",
                            is_trainable=is_trainable,
                        )
                        self._unet.set_adapter("trained")
                    else:
                        from peft import PeftModel
                        self._unet = PeftModel.from_pretrained(
                            self._unet,
                            str(unet_lora_path),
                            is_trainable=is_trainable,
                        )
                    print(f"Loaded UNet LoRA from {unet_lora_path}")
                except Exception as exc:
                    print(f"WARNING: Failed to load UNet LoRA from {unet_lora_path}: {exc}")
            elif unet_lora_weights.exists():
                self._unet.load_state_dict(
                    torch.load(str(unet_lora_weights), map_location=device, weights_only=True),
                    strict=False,
                )
                print(f"Loaded UNet LoRA weights from {unet_lora_weights}")
            else:
                print(f"WARNING: Missing UNet LoRA weights under {model_path}")

        self._sync_pipeline_components()
