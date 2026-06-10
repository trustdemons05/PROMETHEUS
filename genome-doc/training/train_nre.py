"""
NRE Training Script — Stage 3

Trains the Neural Re-Rendering Engine using ControlNet + SD 1.5 with
style embedding conditioning from a pretrained SIR model.

Usage:
    python training/train_nre.py --config configs/nre_controlnet.yaml \
                                  --sir-checkpoint checkpoints/sir/best_model.pt
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.nre.controlnet_renderer import NREControlNet
from models.sir.style_encoder import StyleEncoder
from training.losses import RenderingLoss
from data.dataset import GenomeDocDataset, create_dataloader


# ============================================================================
# Training Loop
# ============================================================================

class NRETrainer:
    """Trainer for the Neural Re-Rendering Engine."""

    def __init__(
        self,
        config: dict,
        sir_checkpoint: str | None = None,
        allow_random_style: bool = False,
        device: str = "auto",
    ):
        self.config = config
        self.allow_random_style = allow_random_style

        # Device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")
        if self.device.type == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB VRAM)")

        # Build NRE model
        model_cfg = config["model"]
        nre_cfg = model_cfg.get("nre", model_cfg)

        self.model = NREControlNet(
            sd_model_id=nre_cfg.get("sd_model_id", "stable-diffusion-v1-5/stable-diffusion-v1-5"),
            controlnet_model_id=nre_cfg.get("controlnet_model_id", "lllyasviel/sd-controlnet-canny"),
            style_dim=nre_cfg.get("style_dim", 512),
            num_style_tokens=nre_cfg.get("num_style_tokens", 4),
            lora_rank=nre_cfg.get("lora_rank", 8),
            lora_alpha=nre_cfg.get("lora_alpha", 16),
            use_lora=nre_cfg.get("use_lora", True),
        )

        # Initialize (loads SD and ControlNet)
        print("Initializing NRE (loading diffusion models)...")
        self.model.initialize(self.device)

        # Ensure all trainable params are FP32 (required for GradScaler)
        # LoRA/ControlNet weights may load as FP16 from the SD checkpoint
        for p in self.model.get_trainable_parameters():
            if p.dtype != torch.float32:
                p.data = p.data.float()

        # Load pretrained SIR for style embedding extraction
        self.sir_model = None
        if sir_checkpoint and os.path.exists(sir_checkpoint):
            self._load_sir(sir_checkpoint)
        elif not allow_random_style:
            raise FileNotFoundError(
                f"SIR checkpoint not found: {sir_checkpoint}. "
                "Pass --allow-random-style only for smoke tests."
            )

        # Loss function — keys aligned with RenderingLoss constructor
        train_cfg = config["training"]
        loss_cfg = train_cfg.get("loss_weights", {})
        self.rendering_loss = RenderingLoss(
            pixel_weight=loss_cfg.get("l1", 1.0),
            perceptual_weight=loss_cfg.get("lpips", 0.5),
            style_weight=loss_cfg.get("style", 0.1),
        )

        # Optimizer — only trainable parameters
        self.optimizer = optim.AdamW(
            self.model.get_trainable_parameters(),
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg.get("weight_decay", 0.01),
        )

        # Scheduler
        self.max_epochs = train_cfg["max_epochs"]
        self.gradient_accumulation = train_cfg.get("gradient_accumulation", 1)

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.max_epochs,
            eta_min=train_cfg["learning_rate"] * 0.01,
        )

        # Config
        self.fp16 = train_cfg.get("fp16", True)
        self.gradient_clip = train_cfg.get("gradient_clip", 1.0)
        self.patience = train_cfg.get("patience", 15)
        self.log_interval = config.get("logging", {}).get("log_every_n_steps", 50)
        self.save_interval = config.get("logging", {}).get("save_every_n_epochs", 5)
        self.gradient_checkpointing = train_cfg.get("gradient_checkpointing", True)

        # Mixed precision
        self.scaler = torch.amp.GradScaler("cuda") if self.fp16 and self.device.type == "cuda" else None

        # Best tracking
        self.best_loss = float("inf")
        self.epochs_without_improvement = 0
        self.epoch_times = []  # Track epoch durations for ETA

    @staticmethod
    def _vram_stats() -> str:
        """Return a formatted string of current VRAM usage."""
        if not torch.cuda.is_available():
            return "VRAM: N/A (CPU)"
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f"VRAM: {allocated:.1f}G alloc / {reserved:.1f}G reserved / {total:.1f}G total"

    def _load_sir(self, checkpoint_path: str) -> None:
        """Load pretrained SIR model for style embedding extraction."""
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

        # Freeze SIR
        for param in self.sir_model.parameters():
            param.requires_grad = False

        print("SIR loaded and frozen")

    def _extract_style(self, style_patches: torch.Tensor) -> torch.Tensor:
        """
        Extract style embeddings using the SIR model.

        Args:
            style_patches: (B, K, 3, P, P) patches

        Returns:
            (B, 512) style embeddings
        """
        if self.sir_model is not None:
            with torch.no_grad():
                return self.sir_model(style_patches)
        else:
            if not self.allow_random_style:
                raise RuntimeError("SIR model is not loaded; refusing to use random style embeddings")
            # Fallback: random embedding for smoke tests only.
            B = style_patches.shape[0]
            return torch.randn(B, 512, device=style_patches.device)

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> dict:
        """Train for one epoch with gradient accumulation support."""
        self.model.style_projector.train()
        if self.model._controlnet is not None:
            self.model._controlnet.train()

        total_loss = 0.0
        num_batches = 0
        self.optimizer.zero_grad()

        pbar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"  Train E{epoch:02d}",
            unit="batch",
            bar_format="{l_bar}{bar:30}{r_bar}",
            leave=True,
        )

        for batch_idx, batch in pbar:
            skeleton = batch["skeleton"].to(self.device)
            clean_image = batch["clean_image"].to(self.device)
            style_patches = batch["style_patches"].to(self.device)
            text_prompt = batch["token_sequence"]  # List of strings

            # Extract style embedding
            style_embedding = self._extract_style(style_patches)

            if self.scaler is not None:
                with torch.amp.autocast("cuda"):
                    # Diffusion training step
                    outputs = self.model(
                        skeleton=skeleton,
                        style_embedding=style_embedding,
                        clean_image=clean_image,
                        text_prompt=text_prompt,
                    )
                    loss = outputs["loss"] / self.gradient_accumulation

                self.scaler.scale(loss).backward()

                if (batch_idx + 1) % self.gradient_accumulation == 0:
                    if self.gradient_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(
                            self.model.get_trainable_parameters(),
                            self.gradient_clip,
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(
                    skeleton=skeleton,
                    style_embedding=style_embedding,
                    clean_image=clean_image,
                    text_prompt=text_prompt,
                )
                loss = outputs["loss"] / self.gradient_accumulation

                loss.backward()

                if (batch_idx + 1) % self.gradient_accumulation == 0:
                    if self.gradient_clip > 0:
                        nn.utils.clip_grad_norm_(
                            self.model.get_trainable_parameters(),
                            self.gradient_clip,
                        )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            total_loss += outputs["loss"].item()  # Log unscaled loss
            num_batches += 1
            avg_loss = total_loss / num_batches
            lr = self.optimizer.param_groups[0]["lr"]

            # Update progress bar every step
            pbar.set_postfix({
                "loss": f"{avg_loss:.5f}",
                "lr": f"{lr:.2e}",
            })

            # Detailed log with VRAM at intervals
            if (batch_idx + 1) % self.log_interval == 0:
                tqdm.write(
                    f"    Step [{batch_idx+1}/{len(dataloader)}] "
                    f"Loss: {avg_loss:.6f} | LR: {lr:.7f} | {self._vram_stats()}"
                )

        pbar.close()

        if num_batches > 0 and num_batches % self.gradient_accumulation != 0:
            if self.scaler is not None:
                if self.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.get_trainable_parameters(),
                        self.gradient_clip,
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                if self.gradient_clip > 0:
                    nn.utils.clip_grad_norm_(
                        self.model.get_trainable_parameters(),
                        self.gradient_clip,
                    )
                self.optimizer.step()
            self.optimizer.zero_grad()

        return {"loss": total_loss / max(num_batches, 1)}

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> dict:
        """Run validation."""
        self.model.style_projector.eval()
        if self.model._controlnet is not None:
            self.model._controlnet.eval()

        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(
            dataloader,
            desc="  Validate",
            unit="batch",
            bar_format="{l_bar}{bar:30}{r_bar}",
            leave=True,
        )

        for batch in pbar:
            skeleton = batch["skeleton"].to(self.device)
            clean_image = batch["clean_image"].to(self.device)
            style_patches = batch["style_patches"].to(self.device)
            text_prompt = batch["token_sequence"]  # List of strings

            style_embedding = self._extract_style(style_patches)

            if self.fp16 and self.device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    outputs = self.model(
                        skeleton=skeleton,
                        style_embedding=style_embedding,
                        clean_image=clean_image,
                        text_prompt=text_prompt,
                    )
            else:
                outputs = self.model(
                    skeleton=skeleton,
                    style_embedding=style_embedding,
                    clean_image=clean_image,
                    text_prompt=text_prompt,
                )

            total_loss += outputs["loss"].item()
            num_batches += 1
            pbar.set_postfix({"val_loss": f"{total_loss / num_batches:.5f}"})

        pbar.close()
        return {"loss": total_loss / max(num_batches, 1)}

    def save_checkpoint(self, path: str, epoch: int, metrics: dict) -> None:
        """Save checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save_pretrained(os.path.dirname(path))

        torch.save({
            "epoch": epoch,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "config": self.config,
        }, path)
        print(f"  Saved checkpoint → {path}")

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        output_dir: str = "checkpoints/nre",
        resume_from: str | None = None,
    ) -> None:
        """Full training loop."""
        output_dir = Path(output_dir)
        start_epoch = 0

        if resume_from and os.path.exists(resume_from):
            self.model.load_pretrained(Path(resume_from).parent, self.device, is_trainable=True)
            checkpoint = torch.load(resume_from, map_location=self.device, weights_only=False)
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            self.best_loss = checkpoint.get("metrics", {}).get("val_loss", float("inf"))

        print(f"\n{'='*60}")
        print(f"NRE Training — {self.max_epochs} epochs")
        print(f"  Batch size: {train_dataloader.batch_size}")
        print(f"  Gradient accumulation: {self.gradient_accumulation}")
        print(f"  Effective batch size: {train_dataloader.batch_size * self.gradient_accumulation}")
        print(f"  Training samples: {len(train_dataloader.dataset)}")
        print(f"  Steps per epoch: {len(train_dataloader)}")
        print(f"  Total optimizer steps: ~{len(train_dataloader) * self.max_epochs // self.gradient_accumulation}")
        print(f"  FP16: {self.fp16}")
        print(f"  Gradient clip: {self.gradient_clip}")
        print(f"  Early stopping patience: {self.patience} epochs")
        print(f"  {self._vram_stats()}")
        print(f"{'='*60}\n")

        training_start = time.time()

        for epoch in range(start_epoch, self.max_epochs):
            epoch_start = time.time()

            print(f"\n{'─'*60}")
            print(f"Epoch {epoch}/{self.max_epochs-1}")
            print(f"{'─'*60}")

            train_metrics = self.train_epoch(train_dataloader, epoch)
            val_metrics = self.validate(val_dataloader)
            self.scheduler.step()

            epoch_time = time.time() - epoch_start
            self.epoch_times.append(epoch_time)

            # ETA calculation
            avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
            remaining_epochs = self.max_epochs - (epoch + 1)
            eta_seconds = avg_epoch_time * remaining_epochs
            eta_str = str(timedelta(seconds=int(eta_seconds)))
            elapsed_str = str(timedelta(seconds=int(time.time() - training_start)))

            # Improvement indicator
            improved = val_metrics["loss"] < self.best_loss
            indicator = "⬇ NEW BEST" if improved else f"⬆ no improve ({self.epochs_without_improvement+1}/{self.patience})"

            print(f"\n  ┌─ Epoch {epoch} Summary ─────────────────────────────")
            print(f"  │ Train Loss:  {train_metrics['loss']:.6f}")
            print(f"  │ Val Loss:    {val_metrics['loss']:.6f}  {indicator}")
            print(f"  │ Best Val:    {self.best_loss:.6f}")
            print(f"  │ LR:          {self.optimizer.param_groups[0]['lr']:.7f}")
            print(f"  │ Epoch Time:  {epoch_time:.1f}s ({epoch_time/60:.1f}min)")
            print(f"  │ Elapsed:     {elapsed_str}")
            print(f"  │ ETA:         {eta_str} ({remaining_epochs} epochs left)")
            print(f"  │ {self._vram_stats()}")
            print(f"  └──────────────────────────────────────────────────")

            if improved:
                self.best_loss = val_metrics["loss"]
                self.epochs_without_improvement = 0
                self.save_checkpoint(
                    str(output_dir / "best_model.pt"), epoch,
                    {"val_loss": val_metrics["loss"]},
                )
            else:
                self.epochs_without_improvement += 1

            if (epoch + 1) % self.save_interval == 0:
                self.save_checkpoint(
                    str(output_dir / f"checkpoint_epoch_{epoch}.pt"), epoch,
                    {"val_loss": val_metrics["loss"]},
                )

            if self.epochs_without_improvement >= self.patience:
                print(f"\n⏹ Early stopping at epoch {epoch} (no improvement for {self.patience} epochs)")
                break

        total_time = time.time() - training_start
        total_str = str(timedelta(seconds=int(total_time)))

        self.save_checkpoint(
            str(output_dir / "final_model.pt"), epoch,
            {"val_loss": val_metrics["loss"]},
        )

        print(f"\n{'='*60}")
        print(f"🎉 Training complete!")
        print(f"  Total time:    {total_str}")
        print(f"  Epochs run:    {epoch - start_epoch + 1}")
        print(f"  Best val loss: {self.best_loss:.6f}")
        print(f"  Final val loss: {val_metrics['loss']:.6f}")
        print(f"  Checkpoints:   {output_dir}")
        print(f"{'='*60}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train NRE Renderer")
    parser.add_argument("--config", type=str, default="configs/nre_controlnet.yaml")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--val-dir", type=str, default=None)
    parser.add_argument("--sir-checkpoint", type=str, default="checkpoints/sir/best_model.pt")
    parser.add_argument("--output-dir", type=str, default="checkpoints/nre")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--allow-random-style",
        action="store_true",
        help="Allow random style embeddings when the SIR checkpoint is missing"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    data_cfg = config.get("data", {})
    train_dir = args.data_dir or data_cfg.get("train_dir", "data/synthetic/train")
    val_dir = args.val_dir or data_cfg.get("val_dir", "data/synthetic/val")

    if not Path(val_dir).exists():
        val_dir = train_dir

    train_cfg = config.get("training", {})
    preload = data_cfg.get("preload_to_ram", False)
    image_size = tuple(data_cfg.get("image_size", [512, 512]))

    print("Loading training data (NRE mode)...")
    if preload:
        print("  → preload_to_ram=True: caching entire dataset in RAM...")
    train_loader = create_dataloader(
        data_dir=train_dir,
        mode="nre",
        batch_size=train_cfg.get("batch_size", 4),
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 4),
        image_size=image_size,
        preload_to_ram=preload,
    )
    print(f"  Loaded {len(train_loader.dataset)} training samples")

    print("Loading validation data...")
    val_loader = create_dataloader(
        data_dir=val_dir,
        mode="nre",
        batch_size=train_cfg.get("batch_size", 4),
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 4),
        image_size=image_size,
        preload_to_ram=preload,
    )
    print(f"  Loaded {len(val_loader.dataset)} validation samples")

    trainer = NRETrainer(
        config=config,
        sir_checkpoint=args.sir_checkpoint,
        allow_random_style=args.allow_random_style,
        device=args.device,
    )

    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        output_dir=args.output_dir,
        resume_from=args.resume,
    )


if __name__ == "__main__":
    main()
