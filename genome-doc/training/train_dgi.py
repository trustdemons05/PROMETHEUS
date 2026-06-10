"""
DGI Training Script — Stage 1

Trains the Document Genome Inferrer using the Donut backbone with LoRA,
template-guided decoding, and combined genome + layout losses.

Usage:
    python training/train_dgi.py --config configs/dgi_donut.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.dgi.donut_genome import DonutGenomeModel
from training.losses import GenomeLoss
from data.dataset import GenomeDocDataset, create_dataloader


# ============================================================================
# Training Loop
# ============================================================================

class DGITrainer:
    """Trainer for the Document Genome Inferrer."""

    def __init__(
        self,
        config: dict,
        device: str = "auto",
    ):
        self.config = config

        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Build model
        model_cfg = config["model"]
        layout_cfg = model_cfg.get("layout_head", {})
        self.model = DonutGenomeModel(
            pretrained_model=model_cfg.get("pretrained", model_cfg.get("backbone", "naver-clova-ix/donut-base")),
            max_length=model_cfg.get("max_length", 2048),
            layout_hidden_dim=layout_cfg.get("hidden_dim", model_cfg.get("layout_hidden_dim", 256)),
            layout_num_layers=layout_cfg.get("num_layers", model_cfg.get("layout_num_layers", 2)),
            max_regions=layout_cfg.get("max_regions", model_cfg.get("max_regions", 64)),
        ).to(self.device)

        # Freeze encoder
        self.model.freeze_encoder()
        print("Encoder frozen — only decoder + layout head are trainable")

        # Setup LoRA if configured
        lora_cfg = model_cfg.get("lora", {})
        if lora_cfg.get("enabled", True):
            self.model.setup_lora(
                rank=lora_cfg.get("rank", 16),
                alpha=lora_cfg.get("alpha", 32),
                dropout=lora_cfg.get("dropout", 0.05),
            )

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable "
              f"({100 * trainable_params / total_params:.1f}%)")

        # Loss function
        train_cfg = config["training"]
        loss_cfg = train_cfg.get("loss_weights", {})
        self.criterion = GenomeLoss(
            ce_weight=loss_cfg.get("ce", 1.0),
            bbox_l1_weight=loss_cfg.get("bbox_l1", 5.0),
            bbox_giou_weight=loss_cfg.get("bbox_giou", 2.0),
        )

        # Optimizer
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg.get("weight_decay", 0.01),
        )

        # Scheduler
        self.max_epochs = train_cfg["max_epochs"]
        self.gradient_accumulation = train_cfg.get("gradient_accumulation", 1)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=max(1, self.max_epochs // 3),
            T_mult=1,
            eta_min=train_cfg["learning_rate"] * 0.01,
        )

        # Training config
        self.fp16 = train_cfg.get("fp16", True)
        self.gradient_clip = train_cfg.get("gradient_clip", 1.0)
        self.patience = train_cfg.get("patience", 15)
        self.log_interval = config.get("logging", {}).get("log_every_n_steps", 50)
        self.save_interval = config.get("logging", {}).get("save_every_n_epochs", 10)

        # Mixed precision
        self.scaler = torch.amp.GradScaler("cuda") if self.fp16 and self.device.type == "cuda" else None

        # Best tracking
        self.best_loss = float("inf")
        self.epochs_without_improvement = 0

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> dict:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_ce_loss = 0.0
        total_bbox_loss = 0.0
        num_batches = 0
        epoch_start = time.time()
        self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(dataloader):
            images = batch["image"].to(self.device)
            token_sequences = batch["token_sequence"]  # List of strings
            gt_bboxes = batch["bboxes"].to(self.device)
            bbox_mask = batch["bbox_mask"].to(self.device)

            # Prepare training batch (tokenize genome sequences)
            prepared = self.model.prepare_training_batch(images, token_sequences)

            if self.scaler is not None:
                with torch.amp.autocast("cuda"):
                    outputs = self.model(
                        pixel_values=prepared["pixel_values"],
                        decoder_input_ids=prepared["decoder_input_ids"],
                        decoder_attention_mask=prepared["decoder_attention_mask"],
                        labels=prepared["labels"],
                        element_positions=prepared["element_positions"],
                    )

                    # Combined loss
                    loss_dict = self._compute_loss(outputs, gt_bboxes, bbox_mask)
                    loss = loss_dict["total"]
                    scaled_loss = loss / self.gradient_accumulation

                self.scaler.scale(scaled_loss).backward()
                if (batch_idx + 1) % self.gradient_accumulation == 0:
                    if self.gradient_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.gradient_clip
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
            else:
                outputs = self.model(
                    pixel_values=prepared["pixel_values"],
                    decoder_input_ids=prepared["decoder_input_ids"],
                    decoder_attention_mask=prepared["decoder_attention_mask"],
                    labels=prepared["labels"],
                    element_positions=prepared["element_positions"],
                )

                loss_dict = self._compute_loss(outputs, gt_bboxes, bbox_mask)
                loss = loss_dict["total"]
                scaled_loss = loss / self.gradient_accumulation

                scaled_loss.backward()
                if (batch_idx + 1) % self.gradient_accumulation == 0:
                    if self.gradient_clip > 0:
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.gradient_clip
                        )
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item()
            total_ce_loss += loss_dict.get("ce_loss", 0.0)
            total_bbox_loss += loss_dict.get("bbox_loss", 0.0)
            num_batches += 1

            # Logging
            if (batch_idx + 1) % self.log_interval == 0:
                avg_loss = total_loss / num_batches
                lr = self.optimizer.param_groups[0]["lr"]
                elapsed = time.time() - epoch_start
                batches_left = len(dataloader) - (batch_idx + 1)
                eta_sec = (elapsed / (batch_idx + 1)) * batches_left
                eta_min = eta_sec / 60

                # GPU VRAM usage
                vram_str = ""
                if self.device.type == "cuda":
                    vram_used = torch.cuda.memory_allocated() / 1024**3
                    vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    vram_str = f" | VRAM: {vram_used:.1f}/{vram_total:.0f}GB"

                print(
                    f"  Epoch {epoch} [{batch_idx + 1}/{len(dataloader)}] "
                    f"Loss: {avg_loss:.4f} "
                    f"(CE: {total_ce_loss/num_batches:.4f}, "
                    f"BBox: {total_bbox_loss/num_batches:.4f}) "
                    f"| LR: {lr:.6f} | ETA: {eta_min:.1f}min{vram_str}"
                )

        if num_batches > 0 and num_batches % self.gradient_accumulation != 0:
            if self.scaler is not None:
                if self.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                if self.gradient_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        return {
            "loss": total_loss / max(num_batches, 1),
            "ce_loss": total_ce_loss / max(num_batches, 1),
            "bbox_loss": total_bbox_loss / max(num_batches, 1),
        }

    def _compute_loss(
        self,
        outputs: dict,
        gt_bboxes: torch.Tensor,
        bbox_mask: torch.Tensor,
    ) -> dict:
        """Compute combined genome + layout loss."""
        result = {}

        # Cross-entropy loss (from Donut model)
        ce_loss = outputs.get("loss", torch.tensor(0.0, device=self.device))
        result["ce_loss"] = ce_loss.item()
        loss_cfg = self.config["training"].get("loss_weights", {})
        ce_weight = loss_cfg.get("genome_ce", loss_cfg.get("ce", 1.0))

        # BBox regression loss
        bbox_loss = torch.tensor(0.0, device=self.device)
        if "pred_bboxes" in outputs:
            pred_bboxes = outputs["pred_bboxes"]  # (B, max_regions, 4)

            # Apply mask — only compute loss for real regions
            mask = bbox_mask.unsqueeze(-1)  # (B, max_regions, 1)

            # L1 loss
            l1_loss = (torch.abs(pred_bboxes - gt_bboxes) * mask).sum() / mask.sum().clamp(min=1)

            # GIoU loss (simplified)
            giou_loss = self._compute_giou_loss(pred_bboxes, gt_bboxes, bbox_mask)

            bbox_loss = (
                loss_cfg.get("bbox_l1", 5.0) * l1_loss +
                loss_cfg.get("bbox_giou", 2.0) * giou_loss
            )
            result["bbox_loss"] = bbox_loss.item()

        # Total loss
        total = ce_weight * ce_loss + bbox_loss
        result["total"] = total

        return result

    def _compute_giou_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Generalized IoU loss between predicted and target bboxes."""
        # pred, target: (B, N, 4) in [0, 1] format [x1, y1, x2, y2]
        # Ensure x2 > x1, y2 > y1
        pred_x1, pred_y1 = pred[..., 0], pred[..., 1]
        pred_x2, pred_y2 = pred[..., 2].clamp(min=pred_x1 + 1e-6), pred[..., 3].clamp(min=pred_y1 + 1e-6)

        gt_x1, gt_y1 = target[..., 0], target[..., 1]
        gt_x2, gt_y2 = target[..., 2], target[..., 3]

        # Intersection
        inter_x1 = torch.max(pred_x1, gt_x1)
        inter_y1 = torch.max(pred_y1, gt_y1)
        inter_x2 = torch.min(pred_x2, gt_x2)
        inter_y2 = torch.min(pred_y2, gt_y2)

        inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

        # Union
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
        union_area = pred_area + gt_area - inter_area + 1e-6

        iou = inter_area / union_area

        # Enclosing box
        enclose_x1 = torch.min(pred_x1, gt_x1)
        enclose_y1 = torch.min(pred_y1, gt_y1)
        enclose_x2 = torch.max(pred_x2, gt_x2)
        enclose_y2 = torch.max(pred_y2, gt_y2)
        enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1) + 1e-6

        giou = iou - (enclose_area - union_area) / enclose_area
        giou_loss = 1 - giou

        # Apply mask
        masked_loss = (giou_loss * mask).sum() / mask.sum().clamp(min=1)
        return masked_loss

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> dict:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        total_ce = 0.0
        total_bbox = 0.0
        num_batches = 0

        for batch in dataloader:
            images = batch["image"].to(self.device)
            token_sequences = batch["token_sequence"]
            gt_bboxes = batch["bboxes"].to(self.device)
            bbox_mask = batch["bbox_mask"].to(self.device)

            prepared = self.model.prepare_training_batch(images, token_sequences)

            if self.fp16 and self.device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    outputs = self.model(
                        pixel_values=prepared["pixel_values"],
                        decoder_input_ids=prepared["decoder_input_ids"],
                        decoder_attention_mask=prepared["decoder_attention_mask"],
                        labels=prepared["labels"],
                        element_positions=prepared["element_positions"],
                    )
                    loss_dict = self._compute_loss(outputs, gt_bboxes, bbox_mask)
            else:
                outputs = self.model(
                    pixel_values=prepared["pixel_values"],
                    decoder_input_ids=prepared["decoder_input_ids"],
                    decoder_attention_mask=prepared["decoder_attention_mask"],
                    labels=prepared["labels"],
                    element_positions=prepared["element_positions"],
                )
                loss_dict = self._compute_loss(outputs, gt_bboxes, bbox_mask)

            total_loss += loss_dict["total"].item()
            total_ce += loss_dict.get("ce_loss", 0.0)
            total_bbox += loss_dict.get("bbox_loss", 0.0)
            num_batches += 1

        return {
            "loss": total_loss / max(num_batches, 1),
            "ce_loss": total_ce / max(num_batches, 1),
            "bbox_loss": total_bbox / max(num_batches, 1),
        }

    def save_checkpoint(self, path: str, epoch: int, metrics: dict) -> None:
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save Donut model and layout head separately
        save_dict = {
            "epoch": epoch,
            "layout_head_state_dict": self.model.layout_head.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "config": self.config,
        }

        # If using LoRA, save adapter weights
        try:
            save_dict["model_state_dict"] = self.model.model.state_dict()
        except Exception:
            save_dict["model_state_dict"] = {
                k: v for k, v in self.model.state_dict().items()
                if "layout_head" not in k
            }

        torch.save(save_dict, path)
        print(f"  Saved checkpoint → {path}")

    def load_checkpoint(self, path: str) -> int:
        """Load from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.layout_head.load_state_dict(checkpoint["layout_head_state_dict"])

        if "model_state_dict" in checkpoint:
            try:
                self.model.model.load_state_dict(checkpoint["model_state_dict"])
            except Exception as e:
                print(f"  Warning: Could not load full model state: {e}")

        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_loss = checkpoint.get("metrics", {}).get("val_loss", float("inf"))
        print(f"  Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint["epoch"]

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        output_dir: str = "checkpoints/dgi",
        resume_from: str | None = None,
    ) -> None:
        """Full training loop."""
        output_dir = Path(output_dir)
        start_epoch = 0

        if resume_from and os.path.exists(resume_from):
            start_epoch = self.load_checkpoint(resume_from) + 1

        # Startup banner
        if self.device.type == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        else:
            gpu_name = "CPU"
            vram_total = 0

        train_cfg = self.config.get("training", {})
        print(f"\n{'='*60}")
        print(f"DGI Training — {self.max_epochs} epochs")
        print(f"{'='*60}")
        print(f"  GPU: {gpu_name} ({vram_total:.0f}GB)")
        print(f"  Batch size: {train_cfg.get('batch_size', '?')}")
        print(f"  Grad accum: {self.gradient_accumulation}")
        print(f"  Effective batch: {train_cfg.get('batch_size', '?') * self.gradient_accumulation}")
        print(f"  Train batches: {len(train_dataloader)} | Val batches: {len(val_dataloader)}")
        print(f"  FP16: {self.fp16} | Grad clip: {self.gradient_clip}")
        print(f"  Early stopping patience: {self.patience}")
        print(f"{'='*60}\n")

        training_start = time.time()

        for epoch in range(start_epoch, self.max_epochs):
            epoch_start = time.time()

            # Train
            train_metrics = self.train_epoch(train_dataloader, epoch)

            # Validate
            val_metrics = self.validate(val_dataloader)

            # Step scheduler
            self.scheduler.step()

            epoch_time = time.time() - epoch_start
            total_elapsed = time.time() - training_start
            epochs_done = epoch - start_epoch + 1
            epochs_left = self.max_epochs - epoch - 1
            eta_total = (total_elapsed / epochs_done) * epochs_left
            eta_h = int(eta_total // 3600)
            eta_m = int((eta_total % 3600) // 60)

            print(
                f"Epoch {epoch}/{self.max_epochs} ({epoch_time:.1f}s) "
                f"— Train Loss: {train_metrics['loss']:.4f} "
                f"(CE: {train_metrics['ce_loss']:.4f}, BBox: {train_metrics['bbox_loss']:.4f}) "
                f"| Val Loss: {val_metrics['loss']:.4f} "
                f"| ETA: {eta_h}h{eta_m:02d}m"
            )

            # Save best model
            if val_metrics["loss"] < self.best_loss:
                self.best_loss = val_metrics["loss"]
                self.epochs_without_improvement = 0
                self.save_checkpoint(
                    str(output_dir / "best_model.pt"),
                    epoch,
                    {"val_loss": val_metrics["loss"]},
                )
            else:
                self.epochs_without_improvement += 1

            # Periodic checkpoint
            if (epoch + 1) % self.save_interval == 0:
                self.save_checkpoint(
                    str(output_dir / f"checkpoint_epoch_{epoch}.pt"),
                    epoch,
                    {"val_loss": val_metrics["loss"]},
                )

            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {self.patience} epochs)")
                break

        # Save final
        self.save_checkpoint(
            str(output_dir / "final_model.pt"),
            epoch,
            {"val_loss": val_metrics["loss"]},
        )
        print(f"\nTraining complete! Best val loss: {self.best_loss:.4f}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train DGI Genome Inferrer")
    parser.add_argument(
        "--config", type=str, default="configs/dgi_donut.yaml",
        help="Path to DGI config YAML"
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Override data directory from config"
    )
    parser.add_argument(
        "--val-dir", type=str, default=None,
        help="Override validation data directory from config"
    )
    parser.add_argument(
        "--output-dir", type=str, default="checkpoints/dgi",
        help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device: 'auto', 'cuda', 'cpu'"
    )
    parser.add_argument(
        "--allow-clean-fallback", action="store_true",
        help="Allow DGI to train on clean images if degraded images are absent"
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Data directories
    data_cfg = config.get("data", {})
    train_dir = args.data_dir or data_cfg.get("train_dir", "data/synthetic/train")
    val_dir = args.val_dir or data_cfg.get("val_dir", "data/synthetic/val")

    if not Path(val_dir).exists():
        print(f"Val dir '{val_dir}' not found, using train dir")
        val_dir = train_dir

    train_cfg = config.get("training", {})
    model_cfg = config.get("model", {})
    image_size = tuple(model_cfg.get("image_size", data_cfg.get("image_size", [512, 512])))
    require_degraded = data_cfg.get("require_degraded", True) and not args.allow_clean_fallback

    # Create dataloaders in 'dgi' mode
    print("Loading training data...")
    train_loader = create_dataloader(
        data_dir=train_dir,
        mode="dgi",
        batch_size=train_cfg.get("batch_size", 4),
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 4),
        image_size=image_size,
        require_degraded=require_degraded,
    )

    print("Loading validation data...")
    val_loader = create_dataloader(
        data_dir=val_dir,
        mode="dgi",
        batch_size=train_cfg.get("batch_size", 4),
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 4),
        image_size=image_size,
        require_degraded=require_degraded,
    )

    # Build trainer
    trainer = DGITrainer(config=config, device=args.device)

    # Train
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        output_dir=args.output_dir,
        resume_from=args.resume,
    )


if __name__ == "__main__":
    main()
