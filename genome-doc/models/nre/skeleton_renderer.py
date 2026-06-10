"""
NRE Skeleton Renderer — Spatial Control Map Generator

Wrapper module that converts a Document Genome JSON into a skeleton image
suitable for ControlNet conditioning. The skeleton is a flat image with text
rendered at the correct bounding box positions using a standard font.

This module re-exports and extends the core renderer from genome/renderer.py,
providing NRE-specific utilities like batch tensor generation and
ControlNet-compatible preprocessing.

Usage:
    from models.nre.skeleton_renderer import SkeletonRenderer

    renderer = SkeletonRenderer(target_size=(512, 512))
    skeleton_tensor = renderer.render_to_tensor(genome)  # (3, 512, 512)
    skeleton_batch = renderer.render_batch(genomes)       # (B, 3, 512, 512)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from genome.schema import DocumentGenome
from genome.renderer import (
    render_skeleton,
    render_skeleton_to_tensor,
    render_skeleton_batch,
    render_debug_skeleton,
)


class SkeletonRenderer:
    """
    NRE-specific skeleton renderer.

    Wraps genome/renderer.py functions with NRE-specific defaults:
    - Fixed 512×512 output size (matching SD 1.5 latent space)
    - RGB float32 tensors in [0, 1] range, channels-first (3, H, W)
    - Batch rendering with automatic stacking to (B, 3, H, W) tensors
    - ControlNet-compatible preprocessing (normalization, resizing)
    """

    def __init__(
        self,
        target_size: tuple[int, int] = (512, 512),
        font_path: Optional[str] = None,
        background_color: tuple[int, int, int] = (255, 255, 255),
    ):
        """
        Args:
            target_size: Output (width, height). Default (512, 512) for SD 1.5.
            font_path: Path to .ttf font file. Uses system default if None.
            background_color: RGB background color for the skeleton.
        """
        self.target_size = target_size
        self.font_path = font_path
        self.background_color = background_color

    def render(self, genome: DocumentGenome) -> "PIL.Image.Image":
        """
        Render a skeleton image from a Document Genome.

        Args:
            genome: The Document Genome to render.

        Returns:
            PIL.Image.Image: RGB skeleton image at target_size.
        """
        return render_skeleton(
            genome,
            target_size=self.target_size,
            font_path=self.font_path,
            background_color=self.background_color,
        )

    def render_to_numpy(self, genome: DocumentGenome) -> np.ndarray:
        """
        Render genome to a float32 numpy array.

        Returns:
            np.ndarray: Shape (3, H, W), values in [0, 1].
        """
        return render_skeleton_to_tensor(
            genome,
            target_size=self.target_size,
            font_path=self.font_path,
        )

    def render_to_tensor(self, genome: DocumentGenome) -> torch.Tensor:
        """
        Render genome to a PyTorch tensor.

        Returns:
            torch.Tensor: Shape (3, H, W), dtype float32, values in [0, 1].
        """
        arr = self.render_to_numpy(genome)
        return torch.from_numpy(arr)

    def render_batch(self, genomes: list[DocumentGenome]) -> torch.Tensor:
        """
        Render a batch of genomes to a stacked tensor.

        Args:
            genomes: List of Document Genomes.

        Returns:
            torch.Tensor: Shape (B, 3, H, W), dtype float32, values in [0, 1].
        """
        tensors = [self.render_to_tensor(g) for g in genomes]
        return torch.stack(tensors)

    def render_debug(
        self,
        genome: DocumentGenome,
        save_path: Optional[str | Path] = None,
    ) -> "PIL.Image.Image":
        """
        Render a debug skeleton with bounding boxes and type-colored text.
        Useful for visually inspecting genome extraction quality.

        Args:
            genome: The Document Genome to render.
            save_path: Optional path to save the debug image.

        Returns:
            PIL.Image.Image: Debug skeleton image.
        """
        return render_debug_skeleton(
            genome,
            target_size=self.target_size,
            font_path=self.font_path,
            save_path=save_path,
        )

    @staticmethod
    def preprocess_for_controlnet(skeleton: torch.Tensor) -> torch.Tensor:
        """
        Preprocess skeleton tensor for ControlNet input.

        ControlNet expects images in [0, 1] at 512×512. This method
        ensures the tensor meets those requirements.

        Args:
            skeleton: (B, 3, H, W) or (3, H, W) tensor

        Returns:
            Preprocessed tensor ready for ControlNet conditioning.
        """
        import torch.nn.functional as F

        # Add batch dimension if needed
        if skeleton.ndim == 3:
            skeleton = skeleton.unsqueeze(0)

        # Ensure [0, 1] range
        skeleton = skeleton.clamp(0, 1)

        # Resize to 512×512 if needed
        if skeleton.shape[2:] != (512, 512):
            skeleton = F.interpolate(
                skeleton, size=(512, 512),
                mode="bilinear", align_corners=False,
            )

        return skeleton
