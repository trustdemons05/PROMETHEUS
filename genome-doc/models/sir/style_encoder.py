"""
Style Encoder for SIR

ResNet-50 backbone that encodes document patches into a 512-dimensional
style embedding. Trained with InfoNCE contrastive loss so that patches
from the same document are close together in embedding space.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from models.sir.patch_selector import PatchSelector


class StyleEncoder(nn.Module):
    """
    ResNet-50 based style encoder that produces a 512-dim style embedding
    from a set of document patches.

    Architecture:
        patches → ResNet-50 backbone → per-patch features → attention pooling → 512-dim embedding
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        backbone: str = "resnet50",
        pretrained: bool = True,
        freeze_backbone_stages: int = 2,
    ):
        """
        Args:
            embedding_dim: Output embedding dimension.
            backbone: Backbone architecture name.
            pretrained: Whether to use ImageNet pretrained weights.
            freeze_backbone_stages: Number of early ResNet stages to freeze (0-4).
        """
        super().__init__()
        self.embedding_dim = embedding_dim

        # Load backbone
        if backbone == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            resnet = models.resnet50(weights=weights)
            backbone_dim = 2048
        elif backbone == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            resnet = models.resnet34(weights=weights)
            backbone_dim = 512
        elif backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            resnet = models.resnet18(weights=weights)
            backbone_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Remove classification head, keep feature extractor
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,  # Stage 1
            resnet.layer2,  # Stage 2
            resnet.layer3,  # Stage 3
            resnet.layer4,  # Stage 4
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # Freeze early stages
        self._freeze_stages(freeze_backbone_stages)

        # Projection head: backbone_dim → embedding_dim
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, backbone_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(backbone_dim // 2, embedding_dim),
        )

        # Attention weights for patch aggregation
        self.patch_attention = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def _freeze_stages(self, num_stages: int) -> None:
        """Freeze the first N stages of the backbone."""
        if num_stages <= 0:
            return

        # Stage mapping in nn.Sequential
        # 0-3: conv1, bn1, relu, maxpool (pre-stages)
        # 4: layer1, 5: layer2, 6: layer3, 7: layer4
        freeze_up_to = min(4 + num_stages, 8)

        for i, child in enumerate(self.backbone.children()):
            if i < freeze_up_to:
                for param in child.parameters():
                    param.requires_grad = False

    def encode_single_patch(self, patch: torch.Tensor) -> torch.Tensor:
        """
        Encode a single patch into an embedding.

        Args:
            patch: (B, 3, P, P) batch of individual patches

        Returns:
            (B, embedding_dim) embeddings
        """
        features = self.backbone(patch)            # (B, backbone_dim, h, w)
        features = self.avgpool(features)          # (B, backbone_dim, 1, 1)
        features = features.flatten(1)             # (B, backbone_dim)
        embedding = self.projection(features)      # (B, embedding_dim)
        return embedding

    def aggregate_patches(
        self,
        patch_embeddings: torch.Tensor,
        quality_scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Aggregate multiple patch embeddings into a single document embedding
        using attention-weighted mean pooling.

        Args:
            patch_embeddings: (B, K, embedding_dim) per-patch embeddings
            quality_scores: (B, K) optional quality scores to modulate attention

        Returns:
            (B, embedding_dim) aggregated document-level embedding
        """
        # Compute attention weights
        attn_logits = self.patch_attention(patch_embeddings)  # (B, K, 1)
        attn_logits = attn_logits.squeeze(-1)                # (B, K)

        # Optionally modulate by quality scores
        if quality_scores is not None:
            attn_logits = attn_logits + torch.log(quality_scores.clamp(min=1e-6))

        attn_weights = F.softmax(attn_logits, dim=-1)        # (B, K)
        attn_weights = attn_weights.unsqueeze(-1)             # (B, K, 1)

        # Weighted sum
        aggregated = (patch_embeddings * attn_weights).sum(dim=1)  # (B, embedding_dim)

        return aggregated

    def forward(
        self,
        patches: torch.Tensor,
        quality_scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode a set of patches into a single style embedding.

        Args:
            patches: (B, K, 3, P, P) K patches per image in a batch of B
            quality_scores: (B, K) optional quality scores

        Returns:
            (B, embedding_dim) style embeddings
        """
        B, K, C, H, W = patches.shape

        # Reshape to process all patches through backbone at once
        flat_patches = patches.reshape(B * K, C, H, W)       # (B*K, 3, P, P)
        flat_embeddings = self.encode_single_patch(flat_patches)  # (B*K, embedding_dim)

        # Reshape back to (B, K, embedding_dim)
        patch_embeddings = flat_embeddings.reshape(B, K, self.embedding_dim)

        # Aggregate into single embedding per document
        style_embedding = self.aggregate_patches(patch_embeddings, quality_scores)

        # L2 normalize the final embedding
        style_embedding = F.normalize(style_embedding, dim=-1)

        return style_embedding


class SIRModule(nn.Module):
    """
    Complete SIR (Style & Identity Refiner) module.

    Combines PatchSelector + StyleEncoder into an end-to-end pipeline:
        image → select patches → encode → aggregate → 512-dim style vector
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        backbone: str = "resnet50",
        pretrained: bool = True,
        patch_size: int = 128,
        top_k: int = 8,
        use_learned_selector: bool = False,
        freeze_backbone_stages: int = 2,
    ):
        super().__init__()
        self.patch_selector = PatchSelector(
            patch_size=patch_size,
            top_k=top_k,
            use_learned_estimator=use_learned_selector,
        )
        self.style_encoder = StyleEncoder(
            embedding_dim=embedding_dim,
            backbone=backbone,
            pretrained=pretrained,
            freeze_backbone_stages=freeze_backbone_stages,
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract style embedding from a single image.

        Args:
            image: (3, H, W) single image tensor in [0, 1]

        Returns:
            (embedding_dim,) style embedding vector
        """
        patches, scores = self.patch_selector(image)     # (K, 3, P, P), (K,)
        patches = patches.unsqueeze(0)                   # (1, K, 3, P, P)
        scores = scores.unsqueeze(0)                     # (1, K)

        embedding = self.style_encoder(patches, scores)  # (1, embedding_dim)
        return embedding.squeeze(0)

    def forward_batch(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract style embeddings for a batch of images.

        Args:
            images: (B, 3, H, W) batch of images in [0, 1]

        Returns:
            (B, embedding_dim) style embeddings
        """
        # Select patches for all images
        patches, scores = self.patch_selector.forward_batch(images)  # (B,K,3,P,P),(B,K)

        # Encode
        embeddings = self.style_encoder(patches, scores)  # (B, embedding_dim)
        return embeddings

    def forward_from_patches(
        self,
        patches: torch.Tensor,
        quality_scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode pre-selected patches directly (skip patch selection).
        Used during training when patches are pre-extracted by the DataLoader.

        Args:
            patches: (B, K, 3, P, P) pre-selected patches
            quality_scores: (B, K) optional quality scores

        Returns:
            (B, embedding_dim) style embeddings
        """
        return self.style_encoder(patches, quality_scores)
