"""
Genome-Doc Inference Pipeline

End-to-end restoration: degraded image → Document Genome → clean document.

Usage:
    python inference/restore.py --image path/to/degraded.png \
                                 --dgi-checkpoint checkpoints/dgi/best_model.pt \
                                 --sir-checkpoint checkpoints/sir/best_model.pt \
                                 --nre-dir checkpoints/nre/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genome.schema import DocumentGenome
from genome.utils import genome_to_token_sequence, token_sequence_to_genome
from genome.renderer import render_skeleton, render_skeleton_to_tensor


class GenomeDocPipeline:
    """
    End-to-end inference pipeline for Genome-Doc restoration.

    Steps:
        1. DGI: degraded image → Document Genome
        2. SIR: degraded image → style embedding
        3. Renderer: genome → skeleton image
        4. NRE: skeleton + style → clean document
    """

    def __init__(
        self,
        dgi_checkpoint: str | None = None,
        sir_checkpoint: str | None = None,
        nre_dir: str | None = None,
        device: str = "auto",
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.dgi_model = None
        self.sir_model = None
        self.nre_model = None
        self.dgi_image_size = (512, 512)

        if dgi_checkpoint:
            self._load_dgi(dgi_checkpoint)
        if sir_checkpoint:
            self._load_sir(sir_checkpoint)
        if nre_dir:
            self._load_nre(nre_dir)

    def _load_dgi(self, checkpoint_path: str) -> None:
        """Load DGI model."""
        from models.dgi.donut_genome import DonutGenomeModel

        print(f"Loading DGI from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        config = checkpoint.get("config", {})
        model_cfg = config.get("model", {})
        layout_cfg = model_cfg.get("layout_head", {})
        self.dgi_image_size = tuple(model_cfg.get("image_size", [512, 512]))

        self.dgi_model = DonutGenomeModel(
            pretrained_model=model_cfg.get("pretrained", model_cfg.get("backbone", "naver-clova-ix/donut-base")),
            max_length=model_cfg.get("max_length", 2048),
            layout_hidden_dim=layout_cfg.get("hidden_dim", model_cfg.get("layout_hidden_dim", 256)),
            layout_num_layers=layout_cfg.get("num_layers", model_cfg.get("layout_num_layers", 2)),
            max_regions=layout_cfg.get("max_regions", model_cfg.get("max_regions", 64)),
        ).to(self.device)

        lora_cfg = model_cfg.get("lora", {})
        if lora_cfg.get("enabled", True):
            self.dgi_model.setup_lora(
                rank=lora_cfg.get("rank", 16),
                alpha=lora_cfg.get("alpha", 32),
                dropout=lora_cfg.get("dropout", 0.05),
            )

        if "layout_head_state_dict" in checkpoint:
            self.dgi_model.layout_head.load_state_dict(checkpoint["layout_head_state_dict"])
        if "model_state_dict" in checkpoint:
            try:
                self.dgi_model.model.load_state_dict(checkpoint["model_state_dict"])
            except Exception as exc:
                print(f"WARNING: Could not load DGI model weights: {exc}")

        self.dgi_model.eval()
        print("DGI loaded")

    def _load_sir(self, checkpoint_path: str) -> None:
        """Load SIR model."""
        from models.sir.style_encoder import StyleEncoder

        print(f"Loading SIR from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        sir_config = checkpoint.get("config", {}).get("model", {})

        self.sir_model = StyleEncoder(
            embedding_dim=sir_config.get("embedding_dim", 512),
            backbone=sir_config.get("backbone", "resnet50"),
            pretrained=False,
        ).to(self.device)

        self.sir_model.load_state_dict(checkpoint["model_state_dict"])
        self.sir_model.eval()
        print("SIR loaded")

    def _load_nre(self, nre_dir: str) -> None:
        """Load NRE model."""
        from models.nre.controlnet_renderer import NREControlNet

        print(f"Loading NRE from {nre_dir}...")
        nre_path = Path(nre_dir)
        checkpoint_config = {}
        metadata_paths = (
            list(nre_path.glob("best_model*.pt"))
            + list(nre_path.glob("final_model*.pt"))
            + list(nre_path.glob("checkpoint_epoch*.pt"))
        )
        if metadata_paths:
            meta_path = sorted(metadata_paths)[0]
            checkpoint_config = torch.load(
                str(meta_path),
                map_location="cpu",
                weights_only=False,
            ).get("config", {})

        model_cfg = checkpoint_config.get("model", {})
        nre_cfg = model_cfg.get("nre", model_cfg)
        self.nre_model = NREControlNet(
            sd_model_id=nre_cfg.get("sd_model_id", "stable-diffusion-v1-5/stable-diffusion-v1-5"),
            controlnet_model_id=nre_cfg.get("controlnet_model_id", "lllyasviel/sd-controlnet-canny"),
            style_dim=nre_cfg.get("style_dim", 512),
            num_style_tokens=nre_cfg.get("num_style_tokens", 4),
            lora_rank=nre_cfg.get("lora_rank", 8),
            lora_alpha=nre_cfg.get("lora_alpha", 16),
            use_lora=nre_cfg.get("use_lora", True),
        )
        self.nre_model.load_pretrained(nre_dir, self.device)
        self.nre_model.eval()
        print("NRE loaded")

    def preprocess_image(self, image: Image.Image, size: tuple[int, int] = (512, 512)) -> torch.Tensor:
        """Preprocess image for model input."""
        image = image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
        arr = np.array(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)  # (1, 3, H, W)
        return tensor.to(self.device)

    @torch.no_grad()
    def extract_genome(self, image_tensor: torch.Tensor) -> Optional[DocumentGenome]:
        """Step 1: Extract Document Genome from degraded image."""
        if self.dgi_model is None:
            raise RuntimeError("DGI model not loaded")

        outputs = self.dgi_model.generate(image_tensor)
        generated_text = outputs["text"][0]

        genome = token_sequence_to_genome(generated_text)
        return genome

    @torch.no_grad()
    def extract_style(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Step 2: Extract style embedding from degraded image."""
        if self.sir_model is None:
            raise RuntimeError("SIR model not loaded")

        from data.dataset import extract_patches

        # Convert tensor to PIL for patch extraction
        img_np = (image_tensor[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)

        patches = extract_patches(img_pil, patch_size=128, num_patches=8, strategy="quality")
        patches_tensor = torch.stack(patches).unsqueeze(0).to(self.device)  # (1, K, 3, P, P)

        style_embedding = self.sir_model(patches_tensor)  # (1, 512)
        return style_embedding

    @torch.no_grad()
    def render_from_genome(
        self,
        genome: DocumentGenome,
        style_embedding: torch.Tensor,
        num_inference_steps: int = 50,
    ) -> torch.Tensor:
        """Steps 3-4: Render skeleton → NRE → clean document."""
        if self.nre_model is None:
            raise RuntimeError("NRE model not loaded")

        # Render skeleton from genome
        skeleton_np = render_skeleton_to_tensor(genome, target_size=(512, 512))
        skeleton = torch.from_numpy(skeleton_np).unsqueeze(0).to(self.device)

        # Generate clean image
        clean_image = self.nre_model.generate(
            skeleton=skeleton,
            style_embedding=style_embedding,
            text_prompt=[genome_to_token_sequence(genome)],
            num_inference_steps=num_inference_steps,
        )

        return clean_image

    def restore(
        self,
        image: Image.Image | str,
        num_inference_steps: int = 50,
        return_intermediate: bool = False,
    ) -> dict:
        """
        Full end-to-end document restoration.

        Args:
            image: Input degraded image (PIL Image or file path)
            num_inference_steps: NRE diffusion steps
            return_intermediate: If True, return genome, skeleton, style

        Returns:
            Dict with 'restored_image', 'genome', and optionally intermediate outputs
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        start_time = time.time()

        # Preprocess
        dgi_tensor = self.preprocess_image(image, self.dgi_image_size)
        style_tensor = self.preprocess_image(image, (512, 512))

        # Step 1: Extract genome
        genome = self.extract_genome(dgi_tensor)
        if genome is None:
            print("WARNING: DGI failed to extract valid genome")
            return {"restored_image": image, "genome": None, "time": 0}

        # Step 2: Extract style
        style_embedding = self.extract_style(style_tensor)

        # Steps 3-4: Render
        clean_tensor = self.render_from_genome(
            genome, style_embedding, num_inference_steps
        )

        # Convert to PIL
        clean_np = (clean_tensor[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        restored_image = Image.fromarray(clean_np)

        elapsed = time.time() - start_time

        result = {
            "restored_image": restored_image,
            "genome": genome,
            "time": elapsed,
        }

        if return_intermediate:
            skeleton = render_skeleton(genome, target_size=(512, 512))
            result["skeleton"] = skeleton
            result["style_embedding"] = style_embedding.cpu()

        return result


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Restore a degraded document")
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--dgi-checkpoint", type=str, default="checkpoints/dgi/best_model.pt")
    parser.add_argument("--sir-checkpoint", type=str, default="checkpoints/sir/best_model.pt")
    parser.add_argument("--nre-dir", type=str, default="checkpoints/nre/")
    parser.add_argument("--output", type=str, default=None, help="Output image path")
    parser.add_argument("--output-genome", type=str, default=None, help="Save genome JSON")
    parser.add_argument("--steps", type=int, default=50, help="Diffusion steps")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    pipeline = GenomeDocPipeline(
        dgi_checkpoint=args.dgi_checkpoint,
        sir_checkpoint=args.sir_checkpoint,
        nre_dir=args.nre_dir,
        device=args.device,
    )

    print(f"\nRestoring: {args.image}")
    result = pipeline.restore(
        args.image,
        num_inference_steps=args.steps,
        return_intermediate=True,
    )

    # Save outputs
    output_path = args.output or args.image.replace(".", "_restored.", 1)
    result["restored_image"].save(output_path)
    print(f"Restored image → {output_path}")

    if result["genome"]:
        genome_path = args.output_genome or output_path.replace(".png", "_genome.json")
        result["genome"].save(genome_path)
        print(f"Genome JSON → {genome_path}")

    print(f"Done in {result['time']:.2f}s")


if __name__ == "__main__":
    main()
