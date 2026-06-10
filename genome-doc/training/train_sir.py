"""
SIR Training Script — Stage 2

Trains the Style & Identity Refiner using InfoNCE contrastive learning.
Patches from the same document form positive pairs; patches from different
documents form negative pairs.

Usage:
    python training/train_sir.py --config configs/sir_resnet.yaml
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

from models.sir.style_encoder import StyleEncoder
from training.losses import InfoNCELoss
from data.dataset import GenomeDocDataset, create_dataloader


# ============================================================================
# Training Loop
# ============================================================================

class SIRTrainer:
    """Trainer for the SIR Style Encoder using contrastive learning."""

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
        self.model = StyleEncoder(
            embedding_dim=model_cfg["embedding_dim"],
            backbone=model_cfg["backbone"],
            pretrained=model_cfg["pretrained"],
            freeze_backbone_stages=2,
        ).to(self.device)

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

        # Loss
        train_cfg = config["training"]
        contrastive_cfg = train_cfg.get("contrastive", {})
        self.criterion = InfoNCELoss(
            temperature=contrastive_cfg.get("temperature", 0.07),
            num_negatives=contrastive_cfg.get("num_negatives", 0),
            hard_negative_ratio=contrastive_cfg.get("hard_negative_ratio", 0.0),
        )

        # Optimizer
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
        )

        # Scheduler with warmup (clamp so cosine T_max >= 1)
        self.warmup_epochs = min(
            train_cfg.get("warmup_epochs", 5),
            train_cfg["max_epochs"] - 1,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, train_cfg["max_epochs"] - self.warmup_epochs),
            eta_min=train_cfg["learning_rate"] * 0.01,
        )
        self.warmup_scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.warmup_epochs,
        )
        self.combined_scheduler = optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[self.warmup_scheduler, self.scheduler],
            milestones=[self.warmup_epochs],
        )

        # Training config
        self.max_epochs = train_cfg["max_epochs"]
        self.fp16 = train_cfg.get("fp16", True)
        self.patience = train_cfg.get("patience", 10)
        self.grad_accum_steps = train_cfg.get("gradient_accumulation", 1)
        self.log_interval = config.get("logging", {}).get("log_every_n_steps", 100)
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
        total_acc = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            anchor_patches = batch["anchor_patches"].to(self.device, non_blocking=True)
            positive_patches = batch["positive_patches"].to(self.device, non_blocking=True)

            if self.scaler is not None:
                with torch.amp.autocast("cuda"):
                    anchor_emb = self.model(anchor_patches)       # (B, D)
                    positive_emb = self.model(positive_patches)   # (B, D)
                    losses = self.criterion(anchor_emb, positive_emb)
                    loss = losses["total"] / self.grad_accum_steps

                self.scaler.scale(loss).backward()
                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
            else:
                anchor_emb = self.model(anchor_patches)
                positive_emb = self.model(positive_patches)
                losses = self.criterion(anchor_emb, positive_emb)
                loss = losses["total"] / self.grad_accum_steps

                loss.backward()
                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * self.grad_accum_steps  # undo scaling
            total_acc += losses["accuracy"].item()
            num_batches += 1

            # Logging
            if (batch_idx + 1) % self.log_interval == 0:
                avg_loss = total_loss / num_batches
                avg_acc = total_acc / num_batches
                lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"  Epoch {epoch} [{batch_idx + 1}/{len(dataloader)}] "
                    f"Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f} | LR: {lr:.6f}"
                )

        # Flush tail micro-batches (if len(dataloader) % grad_accum_steps != 0)
        if num_batches > 0 and (batch_idx + 1) % self.grad_accum_steps != 0:
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        return {
            "loss": total_loss / max(num_batches, 1),
            "accuracy": total_acc / max(num_batches, 1),
        }

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> dict:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0

        for batch in dataloader:
            anchor_patches = batch["anchor_patches"].to(self.device)
            positive_patches = batch["positive_patches"].to(self.device)

            if self.fp16 and self.device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    anchor_emb = self.model(anchor_patches)
                    positive_emb = self.model(positive_patches)
                    losses = self.criterion(anchor_emb, positive_emb)
            else:
                anchor_emb = self.model(anchor_patches)
                positive_emb = self.model(positive_patches)
                losses = self.criterion(anchor_emb, positive_emb)

            total_loss += losses["total"].item()
            total_acc += losses["accuracy"].item()
            num_batches += 1

        return {
            "loss": total_loss / max(num_batches, 1),
            "accuracy": total_acc / max(num_batches, 1),
        }

    def save_checkpoint(self, path: str, epoch: int, metrics: dict) -> None:
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.combined_scheduler.state_dict(),
            "metrics": metrics,
            "config": self.config,
        }, path)
        print(f"  Saved checkpoint → {path}")

    def load_checkpoint(self, path: str) -> int:
        """Load from checkpoint. Returns the epoch number."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.combined_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_loss = checkpoint.get("metrics", {}).get("val_loss", float("inf"))
        print(f"  Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint["epoch"]

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        output_dir: str = "checkpoints/sir",
        resume_from: str | None = None,
    ) -> None:
        """Full training loop."""
        output_dir = Path(output_dir)
        start_epoch = 0

        if resume_from and os.path.exists(resume_from):
            start_epoch = self.load_checkpoint(resume_from) + 1

        print(f"\n{'='*60}")
        print(f"SIR Training — {self.max_epochs} epochs")
        print(f"{'='*60}")

        for epoch in range(start_epoch, self.max_epochs):
            epoch_start = time.time()

            # Train
            train_metrics = self.train_epoch(train_dataloader, epoch)

            # Validate
            val_metrics = self.validate(val_dataloader)

            # Step scheduler
            self.combined_scheduler.step()

            epoch_time = time.time() - epoch_start

            print(
                f"Epoch {epoch}/{self.max_epochs} ({epoch_time:.1f}s) "
                f"— Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.4f} "
                f"| Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f}"
            )

            # Save best model
            if val_metrics["loss"] < self.best_loss:
                self.best_loss = val_metrics["loss"]
                self.epochs_without_improvement = 0
                self.save_checkpoint(
                    str(output_dir / "best_model.pt"),
                    epoch,
                    {"val_loss": val_metrics["loss"], "val_acc": val_metrics["accuracy"]},
                )
            else:
                self.epochs_without_improvement += 1

            # Periodic checkpoint
            if (epoch + 1) % self.save_interval == 0:
                self.save_checkpoint(
                    str(output_dir / f"checkpoint_epoch_{epoch}.pt"),
                    epoch,
                    {"val_loss": val_metrics["loss"], "val_acc": val_metrics["accuracy"]},
                )

            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {self.patience} epochs)")
                break

        # Save final model
        self.save_checkpoint(
            str(output_dir / "final_model.pt"),
            epoch,
            {"val_loss": val_metrics["loss"], "val_acc": val_metrics["accuracy"]},
        )
        print(f"\nTraining complete! Best val loss: {self.best_loss:.4f}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train SIR Style Encoder")
    parser.add_argument(
        "--config", type=str, default="configs/sir_resnet.yaml",
        help="Path to SIR config YAML"
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
        "--output-dir", type=str, default="checkpoints/sir",
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
        "--test", action="store_true",
        help="Quick smoke test: 2 epochs, small batch, 100 samples"
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # --test mode: override config for fast validation
    if args.test:
        print("\n" + "!" * 60)
        print("  TEST MODE — 2 epochs, batch=4, 100 samples")
        print("!" * 60 + "\n")
        config["training"]["max_epochs"] = 2
        config["training"]["batch_size"] = 4
        config.setdefault("logging", {})["log_every_n_steps"] = 5

    # Data directories
    data_cfg = config.get("data", {})
    train_dir = args.data_dir or data_cfg.get("train_dir", "data/synthetic/train")
    val_dir = args.val_dir or data_cfg.get("val_dir", "data/synthetic/val")

    # If no separate val dir exists, use train dir (will split internally)
    if not Path(val_dir).exists():
        print(f"Val dir '{val_dir}' not found, using train dir with split")
        val_dir = train_dir

    train_cfg = config.get("training", {})
    model_cfg = config.get("model", {})
    patch_cfg = model_cfg.get("patch_selector", {})

    batch_size = train_cfg.get("batch_size", 16)

    # Create dataloaders
    print("Loading training data...")
    train_dataset = GenomeDocDataset(
        data_dir=train_dir,
        mode="sir",
        patch_size=patch_cfg.get("patch_size", 128),
        num_patches=patch_cfg.get("top_k", 8),
    )

    # In test mode, use only first 100 samples
    if args.test:
        from torch.utils.data import Subset
        subset_size = min(100, len(train_dataset))
        train_dataset = Subset(train_dataset, list(range(subset_size)))
        print(f"  TEST: Using {subset_size} samples (of {subset_size})")

    def collate_fn(batch):
        collated = {}
        for key in batch[0]:
            values = [item[key] for item in batch]
            if isinstance(values[0], torch.Tensor):
                collated[key] = torch.stack(values)
            elif isinstance(values[0], str):
                collated[key] = values
            elif isinstance(values[0], (int, float)):
                collated[key] = torch.tensor(values)
            else:
                collated[key] = values
        return collated

    num_workers = data_cfg.get("num_workers", 4)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    print("Loading validation data...")
    val_dataset = GenomeDocDataset(
        data_dir=val_dir,
        mode="sir",
        patch_size=patch_cfg.get("patch_size", 128),
        num_patches=patch_cfg.get("top_k", 8),
    )

    if args.test:
        from torch.utils.data import Subset
        val_subset = min(50, len(val_dataset))
        val_dataset = Subset(val_dataset, list(range(val_subset)))

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    # Build trainer
    trainer = SIRTrainer(config=config, device=args.device)

    # Train
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        output_dir=args.output_dir,
        resume_from=args.resume,
    )

    if args.test:
        print("\n✅ TEST PASSED — Pipeline works! Ready for full training.")


if __name__ == "__main__":
    main()
