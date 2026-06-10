"""
Patch Selector for SIR

Selects the cleanest (least-degraded) patches from a document image
using quality estimation based on local contrast, edge sharpness,
and noise level. These patches are used by the Style Encoder to
extract the document's visual identity.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PatchQualityEstimator(nn.Module):
    """
    Lightweight CNN that estimates the quality (cleanliness) of image patches.

    Takes a patch and outputs a scalar quality score. Higher scores indicate
    cleaner, less-degraded patches that are better for style extraction.
    """

    def __init__(self, patch_size: int = 128):
        super().__init__()
        self.patch_size = patch_size

        # Simple ConvNet for quality estimation
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64x64

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32x32

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),  # 4x4
        )

        self.quality_head = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid(),  # Output in [0, 1]
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patches: (B, 3, P, P) batch of patches

        Returns:
            (B,) quality scores in [0, 1]
        """
        features = self.features(patches)
        features = features.flatten(1)
        scores = self.quality_head(features).squeeze(-1)
        return scores


class HandcraftedQualityEstimator:
    """
    Non-learnable quality estimator using handcrafted features.
    Used as a fallback or for initial patch selection before model training.
    """

    @staticmethod
    def estimate_quality(patch: np.ndarray) -> float:
        """
        Estimate patch quality using handcrafted features.

        Args:
            patch: (H, W, 3) numpy array in [0, 255] range

        Returns:
            Quality score in [0, 1]. Higher = cleaner.
        """
        gray = np.mean(patch, axis=2).astype(np.float32)

        # 1. Local contrast (std dev) — higher is better (has content)
        contrast = np.std(gray) / 128.0  # Normalize to ~[0, 1]

        # 2. Edge sharpness (Laplacian variance) — higher means sharper
        from cv2 import Laplacian, CV_32F
        laplacian = Laplacian(gray, CV_32F)
        sharpness = min(np.var(laplacian) / 1000.0, 1.0)

        # 3. Noise estimate (high-freq energy) — lower is better
        # Simple: difference between original and blurred
        from cv2 import GaussianBlur
        blurred = GaussianBlur(gray, (5, 5), 0)
        noise = np.mean(np.abs(gray - blurred)) / 30.0
        noise_score = max(0, 1.0 - noise)

        # 4. Brightness sanity — penalize very dark or very bright
        mean_val = np.mean(gray) / 255.0
        brightness_ok = 1.0 - 2.0 * abs(mean_val - 0.5)
        brightness_ok = max(0, brightness_ok)

        # Weighted combination
        quality = (
            0.3 * contrast +
            0.3 * sharpness +
            0.25 * noise_score +
            0.15 * brightness_ok
        )
        return min(max(quality, 0.0), 1.0)


class PatchSelector(nn.Module):
    """
    Attention-based patch selector that identifies the top-K cleanest
    patches from a document image.

    Can use either a learned quality estimator or handcrafted features.
    """

    def __init__(
        self,
        patch_size: int = 128,
        top_k: int = 8,
        stride: int = 64,
        use_learned_estimator: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.top_k = top_k
        self.stride = stride
        self.use_learned_estimator = use_learned_estimator

        if use_learned_estimator:
            self.quality_estimator = PatchQualityEstimator(patch_size)
        else:
            self.handcrafted_estimator = HandcraftedQualityEstimator()

    def extract_patches(self, image: torch.Tensor) -> tuple[torch.Tensor, list[tuple[int, int]]]:
        """
        Extract all possible patches from an image using sliding window.

        Args:
            image: (3, H, W) single image tensor

        Returns:
            patches: (N, 3, P, P) all extracted patches
            positions: List of (x, y) top-left corners
        """
        _, H, W = image.shape
        P = self.patch_size
        S = self.stride

        patches = []
        positions = []

        for y in range(0, H - P + 1, S):
            for x in range(0, W - P + 1, S):
                patch = image[:, y:y + P, x:x + P]
                patches.append(patch)
                positions.append((x, y))

        if not patches:
            # Image too small — return the whole thing resized
            resized = F.interpolate(
                image.unsqueeze(0), size=(P, P), mode="bilinear", align_corners=False
            ).squeeze(0)
            return resized.unsqueeze(0), [(0, 0)]

        return torch.stack(patches), positions

    def forward(
        self,
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Select top-K cleanest patches from an image.

        Args:
            image: (3, H, W) single image tensor, values in [0, 1]

        Returns:
            selected_patches: (K, 3, P, P) top-K patches
            quality_scores: (K,) quality scores of selected patches
        """
        # Extract all candidate patches
        all_patches, positions = self.extract_patches(image)

        if len(all_patches) <= self.top_k:
            # Fewer patches than K — return all of them
            scores = torch.ones(len(all_patches), device=image.device)
            return all_patches, scores

        # Score each patch
        if self.use_learned_estimator:
            with torch.no_grad():
                scores = self.quality_estimator(all_patches)
        else:
            # Handcrafted scoring (slower, used for bootstrapping)
            scores_list = []
            for patch in all_patches:
                patch_np = (patch.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                score = self.handcrafted_estimator.estimate_quality(patch_np)
                scores_list.append(score)
            scores = torch.tensor(scores_list, device=image.device)

        # Select top-K by quality score
        top_k_indices = torch.topk(scores, min(self.top_k, len(scores))).indices
        selected_patches = all_patches[top_k_indices]
        selected_scores = scores[top_k_indices]

        return selected_patches, selected_scores

    def forward_batch(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Select top-K patches for a batch of images.

        Args:
            images: (B, 3, H, W) batch of images

        Returns:
            selected_patches: (B, K, 3, P, P) top-K patches per image
            quality_scores: (B, K) quality scores
        """
        batch_patches = []
        batch_scores = []

        for i in range(images.shape[0]):
            patches, scores = self.forward(images[i])

            # Pad if fewer than top_k
            if patches.shape[0] < self.top_k:
                pad_count = self.top_k - patches.shape[0]
                pad_patches = patches[-1:].expand(pad_count, -1, -1, -1)
                pad_scores = torch.zeros(pad_count, device=images.device)
                patches = torch.cat([patches, pad_patches])
                scores = torch.cat([scores, pad_scores])

            batch_patches.append(patches)
            batch_scores.append(scores)

        return torch.stack(batch_patches), torch.stack(batch_scores)
