"""
Loss Functions for Genome-Doc Training

All loss functions used across the three training stages:
- Stage 1 (DGI): Genome token cross-entropy + bbox regression
- Stage 2 (SIR): InfoNCE contrastive loss
- Stage 3 (NRE): Pixel L1 + LPIPS perceptual + style consistency
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Stage 1: DGI Losses
# ============================================================================

class GenomeLoss(nn.Module):
    """
    Combined loss for Document Genome Inferrer.

    Combines:
    - Cross-entropy loss on genome JSON token predictions
    - L1 + GIoU loss on bounding box coordinate regression
    """

    def __init__(
        self,
        ce_weight: float = 1.0,
        bbox_l1_weight: float = 5.0,
        bbox_giou_weight: float = 2.0,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.bbox_l1_weight = bbox_l1_weight
        self.bbox_giou_weight = bbox_giou_weight
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(
        self,
        token_logits: torch.Tensor,
        token_targets: torch.Tensor,
        pred_bboxes: torch.Tensor | None = None,
        target_bboxes: torch.Tensor | None = None,
        bbox_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            token_logits: (B, seq_len, vocab_size)
            token_targets: (B, seq_len)
            pred_bboxes: (B, num_regions, 4) — predicted [x1, y1, x2, y2]
            target_bboxes: (B, num_regions, 4) — ground truth
            bbox_mask: (B, num_regions) — 1 for valid regions, 0 for padding

        Returns:
            Dict with individual losses and total loss.
        """
        # Token cross-entropy
        B, S, V = token_logits.shape
        ce = self.ce_loss(
            token_logits.reshape(B * S, V),
            token_targets.reshape(B * S),
        )

        losses = {"ce_loss": ce}
        total = self.ce_weight * ce

        # Bounding box losses (if provided)
        if pred_bboxes is not None and target_bboxes is not None:
            if bbox_mask is not None:
                # Mask out padding regions
                mask = bbox_mask.unsqueeze(-1).expand_as(pred_bboxes)
                pred_masked = pred_bboxes * mask
                target_masked = target_bboxes * mask
                num_valid = bbox_mask.sum().clamp(min=1)
            else:
                pred_masked = pred_bboxes
                target_masked = target_bboxes
                num_valid = pred_bboxes.shape[0] * pred_bboxes.shape[1]

            # L1 loss on coordinates
            bbox_l1 = F.l1_loss(pred_masked, target_masked, reduction="sum") / num_valid
            losses["bbox_l1"] = bbox_l1
            total = total + self.bbox_l1_weight * bbox_l1

            # Generalized IoU loss
            giou = self._generalized_iou_loss(pred_bboxes, target_bboxes, bbox_mask)
            losses["bbox_giou"] = giou
            total = total + self.bbox_giou_weight * giou

        losses["total"] = total
        return losses

    @staticmethod
    def _generalized_iou_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute Generalized IoU loss for bounding boxes.

        Args:
            pred: (B, N, 4) in [x1, y1, x2, y2] format
            target: (B, N, 4) in [x1, y1, x2, y2] format
            mask: (B, N) boolean mask for valid boxes
        """
        # Intersection
        inter_x1 = torch.max(pred[..., 0], target[..., 0])
        inter_y1 = torch.max(pred[..., 1], target[..., 1])
        inter_x2 = torch.min(pred[..., 2], target[..., 2])
        inter_y2 = torch.min(pred[..., 3], target[..., 3])

        inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

        # Union
        pred_area = (pred[..., 2] - pred[..., 0]).clamp(min=0) * (pred[..., 3] - pred[..., 1]).clamp(min=0)
        target_area = (target[..., 2] - target[..., 0]).clamp(min=0) * (target[..., 3] - target[..., 1]).clamp(min=0)
        union_area = pred_area + target_area - inter_area

        iou = inter_area / union_area.clamp(min=1e-6)

        # Enclosing box
        enclose_x1 = torch.min(pred[..., 0], target[..., 0])
        enclose_y1 = torch.min(pred[..., 1], target[..., 1])
        enclose_x2 = torch.max(pred[..., 2], target[..., 2])
        enclose_y2 = torch.max(pred[..., 3], target[..., 3])

        enclose_area = (enclose_x2 - enclose_x1).clamp(min=0) * (enclose_y2 - enclose_y1).clamp(min=0)

        giou = iou - (enclose_area - union_area) / enclose_area.clamp(min=1e-6)
        giou_loss = 1 - giou  # Loss form

        if mask is not None:
            giou_loss = giou_loss * mask
            return giou_loss.sum() / mask.sum().clamp(min=1)

        return giou_loss.mean()


# ============================================================================
# Stage 2: SIR Losses
# ============================================================================

class InfoNCELoss(nn.Module):
    """
    InfoNCE contrastive loss for style embedding learning.

    Given a batch of (anchor, positive) pairs, treats all other samples
    in the batch as negatives. Optionally caps negatives and mines hard ones.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        num_negatives: int = 0,
        hard_negative_ratio: float = 0.0,
    ):
        """
        Args:
            temperature: Softmax temperature for similarity scaling.
            num_negatives: Max negatives per sample (0 = use all in-batch).
            hard_negative_ratio: Fraction of negatives selected by hardness
                                 (highest similarity). 0.0 = random only.
        """
        super().__init__()
        self.temperature = temperature
        self.num_negatives = num_negatives
        self.hard_negative_ratio = hard_negative_ratio

    def forward(
        self,
        anchors: torch.Tensor,
        positives: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            anchors: (B, D) — anchor embeddings
            positives: (B, D) — positive embeddings (same document, different patch)

        Returns:
            Dict with 'contrastive_loss', 'total', and 'accuracy' keys.
        """
        # Normalize embeddings
        anchors = F.normalize(anchors, dim=-1)
        positives = F.normalize(positives, dim=-1)

        # Full similarity matrix: (B, B)
        logits = torch.matmul(anchors, positives.T) / self.temperature

        B = logits.shape[0]
        labels = torch.arange(B, device=logits.device)

        # Optionally subsample negatives per row
        if self.num_negatives > 0 and self.num_negatives < B - 1:
            logits = self._subsample_negatives(logits, labels)
            # After subsampling, positive is always at column 0
            labels = torch.zeros(B, dtype=torch.long, device=logits.device)

        # Cross-entropy loss (both directions for symmetry)
        loss_a2p = F.cross_entropy(logits, labels)
        loss_p2a = F.cross_entropy(logits.T if self.num_negatives == 0 else logits, labels)
        loss = (loss_a2p + loss_p2a) / 2.0

        # Compute top-1 accuracy for monitoring
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            accuracy = (preds == labels).float().mean()

        return {
            "contrastive_loss": loss,
            "total": loss,
            "accuracy": accuracy,
        }

    def _subsample_negatives(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Subsample negatives: mix of hard (highest sim) and random.

        Returns (B, 1 + num_negatives) logits with positive at column 0.
        """
        B = logits.shape[0]
        K = self.num_negatives
        num_hard = int(K * self.hard_negative_ratio)
        num_rand = K - num_hard

        subsampled = []
        for i in range(B):
            pos_logit = logits[i, labels[i]].unsqueeze(0)  # (1,)
            # Negative logits (all columns except the positive)
            neg_mask = torch.ones(B, dtype=torch.bool, device=logits.device)
            neg_mask[labels[i]] = False
            neg_logits = logits[i, neg_mask]  # (B-1,)

            selected = []
            # Hard negatives: pick highest similarity
            if num_hard > 0 and neg_logits.numel() > 0:
                hard_k = min(num_hard, neg_logits.numel())
                _, hard_idx = neg_logits.topk(hard_k)
                selected.append(neg_logits[hard_idx])

            # Random negatives
            if num_rand > 0 and neg_logits.numel() > 0:
                rand_k = min(num_rand, neg_logits.numel())
                perm = torch.randperm(neg_logits.numel(), device=logits.device)[:rand_k]
                selected.append(neg_logits[perm])

            neg_selected = torch.cat(selected) if selected else neg_logits[:K]
            row = torch.cat([pos_logit, neg_selected])  # (1 + K,)
            subsampled.append(row)

        return torch.stack(subsampled)  # (B, 1 + K)


# ============================================================================
# Stage 3: NRE Losses
# ============================================================================

class RenderingLoss(nn.Module):
    """
    Combined loss for Neural Re-Rendering Engine.

    Combines:
    - L1 pixel loss
    - LPIPS perceptual loss
    - Style consistency loss (MSE between SIR embeddings)
    """

    def __init__(
        self,
        pixel_weight: float = 1.0,
        perceptual_weight: float = 0.5,
        style_weight: float = 0.1,
        use_lpips: bool = True,
    ):
        super().__init__()
        self.pixel_weight = pixel_weight
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight

        # LPIPS will be initialized lazily (requires lpips package)
        self._lpips_net = None
        self._use_lpips = use_lpips

    def _get_lpips(self, device: torch.device) -> nn.Module:
        """Lazily initialize LPIPS network."""
        if self._lpips_net is None:
            try:
                import lpips
                self._lpips_net = lpips.LPIPS(net="vgg").to(device)
                self._lpips_net.eval()
                for p in self._lpips_net.parameters():
                    p.requires_grad = False
            except ImportError:
                print("WARNING: lpips not installed, using L1 as perceptual loss")
                self._use_lpips = False
        return self._lpips_net

    def forward(
        self,
        rendered: torch.Tensor,
        target: torch.Tensor,
        rendered_style: torch.Tensor | None = None,
        target_style: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            rendered: (B, 3, H, W) — rendered document image
            target: (B, 3, H, W) — ground truth clean image
            rendered_style: (B, D) — SIR embedding of rendered image (optional)
            target_style: (B, D) — SIR embedding of target image (optional)

        Returns:
            Dict with individual losses and total loss.
        """
        losses = {}

        # L1 pixel loss
        pixel_loss = F.l1_loss(rendered, target)
        losses["pixel_l1"] = pixel_loss
        total = self.pixel_weight * pixel_loss

        # LPIPS perceptual loss
        if self._use_lpips:
            lpips_net = self._get_lpips(rendered.device)
            if lpips_net is not None:
                # LPIPS expects images in [-1, 1] range
                rendered_scaled = rendered * 2.0 - 1.0
                target_scaled = target * 2.0 - 1.0
                perceptual_loss = lpips_net(rendered_scaled, target_scaled).mean()
                losses["perceptual_lpips"] = perceptual_loss
                total = total + self.perceptual_weight * perceptual_loss

        # Style consistency loss
        if rendered_style is not None and target_style is not None:
            style_loss = F.mse_loss(rendered_style, target_style)
            losses["style_mse"] = style_loss
            total = total + self.style_weight * style_loss

        losses["total"] = total
        return losses


# ============================================================================
# Anti-Hallucination Verification Loss
# ============================================================================

class VerificationLoss(nn.Module):
    """
    OCR verification loss: Compares OCR output on rendered image against
    the genome text tokens.

    This is a non-differentiable metric used as a training regularizer
    via reward-based signals or as a validation-only metric.
    """

    def __init__(self):
        super().__init__()
        # This loss is computed externally (OCR is not differentiable)
        # Used during validation to measure hallucination rate

    @staticmethod
    def compute_hallucination_rate(
        ocr_text: str,
        genome_text: str,
    ) -> dict[str, float]:
        """
        Compare OCR output on rendered image against genome text.

        Args:
            ocr_text: Text extracted by OCR from the rendered output.
            genome_text: Text from the Document Genome specification.

        Returns:
            Dict with character-level and word-level hallucination rates.
        """
        # Character-level comparison
        ocr_chars = list(ocr_text.replace(" ", "").replace("\n", "").lower())
        genome_chars = list(genome_text.replace(" ", "").replace("\n", "").lower())

        if not genome_chars:
            return {"char_hallucination_rate": 0.0, "word_hallucination_rate": 0.0}

        # Simple character-level diff
        max_len = max(len(ocr_chars), len(genome_chars))
        if max_len == 0:
            return {"char_hallucination_rate": 0.0, "word_hallucination_rate": 0.0}

        # Pad shorter sequence
        while len(ocr_chars) < max_len:
            ocr_chars.append("")
        while len(genome_chars) < max_len:
            genome_chars.append("")

        char_diffs = sum(1 for a, b in zip(ocr_chars, genome_chars) if a != b)
        char_rate = char_diffs / len(genome_chars)

        # Word-level comparison
        ocr_words = set(ocr_text.lower().split())
        genome_words = set(genome_text.lower().split())

        if not genome_words:
            word_rate = 0.0
        else:
            hallucinated = ocr_words - genome_words
            word_rate = len(hallucinated) / len(genome_words)

        return {
            "char_hallucination_rate": min(char_rate, 1.0),
            "word_hallucination_rate": min(word_rate, 1.0),
        }
