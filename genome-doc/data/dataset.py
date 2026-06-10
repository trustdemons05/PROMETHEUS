"""
PyTorch Dataset for Genome-Doc

Loads paired (degraded image, clean image, genome JSON) samples for training
the DGI, SIR, and NRE modules.

Supports three modes:
- 'dgi': Returns (degraded_image, genome_token_sequence, bboxes) for DGI training
- 'sir': Returns (clean_patches, document_ids) for SIR contrastive training
- 'nre': Returns (skeleton_image, style_patches, clean_image) for NRE training
- 'full': Returns all components for inference/evaluation
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genome.schema import BoundingBox, DocumentGenome
from genome.utils import genome_to_token_sequence
from genome.renderer import render_skeleton, render_skeleton_to_tensor


# ============================================================================
# Transforms
# ============================================================================

def image_to_tensor(img: Image.Image, size: tuple[int, int] = (512, 512)) -> torch.Tensor:
    """Convert PIL Image to normalized float32 tensor (C, H, W) in [0, 1]."""
    img = img.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1))  # HWC -> CHW


def extract_patches(
    img: Image.Image,
    patch_size: int = 128,
    num_patches: int = 8,
    strategy: str = "quality",
) -> list[torch.Tensor]:
    """
    Extract patches from an image for SIR training.

    Args:
        img: Source image.
        patch_size: Size of square patches.
        num_patches: Number of patches to extract.
        strategy: 'quality' (prefer high-contrast regions) or 'random'.

    Returns:
        List of tensor patches, each (3, patch_size, patch_size).
    """
    img = img.convert("RGB")
    w, h = img.size
    arr = np.array(img)

    if strategy == "quality":
        # Score regions by local contrast (std dev)
        gray = np.mean(arr, axis=2)
        scores = []
        positions = []

        step = max(patch_size // 2, 1)
        for y in range(0, h - patch_size, step):
            for x in range(0, w - patch_size, step):
                patch = gray[y:y + patch_size, x:x + patch_size]
                # Higher std = more content = better patch
                score = np.std(patch)
                # Penalize very dark or very bright patches
                mean_val = np.mean(patch)
                if 50 < mean_val < 240:
                    scores.append(score)
                    positions.append((x, y))

        if not positions:
            # Fallback to random
            return extract_patches(img, patch_size, num_patches, "random")

        # Select top-K by quality score
        indices = np.argsort(scores)[-num_patches * 3:]  # Over-sample then pick
        np.random.shuffle(indices)
        selected = indices[:num_patches]

        patches = []
        for idx in selected:
            x, y = positions[idx]
            patch = arr[y:y + patch_size, x:x + patch_size]
            patch_t = torch.from_numpy(patch.astype(np.float32) / 255.0).permute(2, 0, 1)
            patches.append(patch_t)

    else:
        # Random patches
        patches = []
        for _ in range(num_patches):
            x = random.randint(0, max(0, w - patch_size))
            y = random.randint(0, max(0, h - patch_size))
            patch = arr[y:y + patch_size, x:x + patch_size]
            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                patch = np.pad(
                    patch,
                    ((0, patch_size - patch.shape[0]),
                     (0, patch_size - patch.shape[1]),
                     (0, 0)),
                    mode="edge",
                )
            patch_t = torch.from_numpy(patch.astype(np.float32) / 255.0).permute(2, 0, 1)
            patches.append(patch_t)

    # Guarantee exactly num_patches by repeating existing patches if needed
    if len(patches) == 0:
        # Edge case: no patches at all — create a blank patch
        patches.append(torch.zeros(3, patch_size, patch_size))
    while len(patches) < num_patches:
        patches.append(patches[len(patches) % len(patches)].clone())
    patches = patches[:num_patches]

    return patches


# ============================================================================
# Main Dataset Class
# ============================================================================

class GenomeDocDataset(Dataset):
    """
    PyTorch Dataset for Genome-Doc training.

    Expected directory structure:
        data_dir/
            images/
                clean/          # Clean document images (PNG)
                degraded/       # Degraded document images (PNG)
            genomes/            # Ground-truth Genome JSONs
    """

    def __init__(
        self,
        data_dir: str | Path,
        mode: Literal["dgi", "sir", "nre", "full"] = "full",
        image_size: tuple[int, int] = (512, 512),
        patch_size: int = 128,
        num_patches: int = 8,
        max_seq_length: int = 2048,
        tokenizer: Optional[object] = None,
        transform: Optional[object] = None,
        preload_to_ram: bool = False,
        require_degraded: bool | None = None,
    ):
        """
        Args:
            data_dir: Path to dataset directory.
            mode: Training mode — determines what data is returned.
            image_size: Target (width, height) for image resizing.
            patch_size: Patch size for SIR training.
            num_patches: Number of patches for SIR.
            max_seq_length: Maximum token sequence length for DGI.
            tokenizer: Optional tokenizer for DGI (if None, returns raw string).
            transform: Optional additional transforms.
            preload_to_ram: If True, pre-compute and cache all samples in RAM.
                           Eliminates disk I/O during training. Requires enough
                           system RAM to hold the entire processed dataset.
            require_degraded: If True, DGI/full samples must have matching degraded
                           images. Defaults to True for DGI/full and False otherwise.
        """
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.transform = transform
        self.preload_to_ram = preload_to_ram
        self.require_degraded = mode in {"dgi", "full"} if require_degraded is None else require_degraded

        # Discover samples
        self.clean_dir = self.data_dir / "images" / "clean"
        self.degraded_dir = self.data_dir / "images" / "degraded"
        self.genome_dir = self.data_dir / "genomes"

        # Get list of sample IDs (filenames without extension)
        self.sample_ids = self._discover_samples()
        if len(self.sample_ids) == 0:
            expected = (
                f"Expected data under {self.data_dir} with images/clean"
                + (", images/degraded" if self.require_degraded else "")
                + (", and genomes" if self.mode != "sir" else "")
                + "."
            )
            raise ValueError(f"No valid {mode} samples found in {data_dir}. {expected}")

        # Preload entire dataset into RAM if requested
        self._cache = None
        if self.preload_to_ram:
            self._preload_all()

    @staticmethod
    def _image_ids(directory: Path) -> set[str]:
        if not directory.exists():
            return set()
        return {
            f.stem for f in directory.iterdir()
            if f.suffix.lower() in (".png", ".jpg", ".jpeg")
        }

    def _discover_samples(self) -> list[str]:
        """Find valid samples for the selected training mode."""
        clean_files = self._image_ids(self.clean_dir)
        degraded_files = self._image_ids(self.degraded_dir)
        if self.genome_dir.exists():
            genome_files = {
                f.stem for f in self.genome_dir.iterdir()
                if f.suffix == ".json"
            }
        else:
            genome_files = set()

        if self.mode == "sir":
            valid = clean_files
        else:
            valid = clean_files & genome_files

        if self.require_degraded:
            valid = valid & degraded_files

        return sorted(valid)

    def _preload_all(self) -> None:
        """Pre-compute and cache every sample in RAM. Eliminates disk I/O."""
        import gc
        print(f"  Preloading {len(self.sample_ids)} samples into RAM (mode={self.mode})...")
        self._cache = [None] * len(self.sample_ids)
        for i in range(len(self.sample_ids)):
            self._cache[i] = self._load_sample(i)
            if (i + 1) % 1000 == 0 or (i + 1) == len(self.sample_ids):
                print(f"    Cached {i+1}/{len(self.sample_ids)} samples")
        gc.collect()
        print(f"  Preloading complete — {len(self._cache)} samples in RAM")

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> dict:
        if self._cache is not None:
            return self._cache[idx]
        return self._load_sample(idx)

    def _load_sample(self, idx: int) -> dict:
        """Load and prepare a single sample from disk."""
        sample_id = self.sample_ids[idx]

        # SIR fast path: only needs clean image, skip genome/degraded entirely
        if self.mode == "sir":
            clean_path = self._find_image(self.clean_dir, sample_id)
            clean_img = Image.open(str(clean_path)).convert("RGB")
            if self.transform:
                clean_img = self.transform(clean_img)
            return self._prepare_sir_sample(clean_img, sample_id)

        # Load genome (only for dgi/nre/full modes)
        genome_path = self.genome_dir / f"{sample_id}.json"
        genome = DocumentGenome.from_file(genome_path)

        # Load clean image
        clean_path = self._find_image(self.clean_dir, sample_id)
        clean_img = Image.open(str(clean_path)).convert("RGB")

        # Load degraded image, if required/available.
        degraded_img = None
        if self.degraded_dir.exists():
            degraded_path = self._find_image(self.degraded_dir, sample_id)
            if degraded_path:
                degraded_img = Image.open(str(degraded_path)).convert("RGB")
        if self.require_degraded and degraded_img is None:
            raise FileNotFoundError(
                f"Missing degraded image for sample '{sample_id}' in {self.degraded_dir}"
            )

        # Apply custom transform
        if self.transform:
            clean_img = self.transform(clean_img)
            if degraded_img:
                degraded_img = self.transform(degraded_img)

        # Return data based on mode
        if self.mode == "dgi":
            return self._prepare_dgi_sample(degraded_img or clean_img, genome)
        elif self.mode == "nre":
            return self._prepare_nre_sample(genome, clean_img)
        elif self.mode == "full":
            return self._prepare_full_sample(
                degraded_img or clean_img, clean_img, genome, sample_id
            )
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _find_image(self, directory: Path, sample_id: str) -> Optional[Path]:
        """Find an image file by sample ID, trying multiple extensions."""
        for ext in [".png", ".jpg", ".jpeg"]:
            path = directory / f"{sample_id}{ext}"
            if path.exists():
                return path
        return None

    def _prepare_dgi_sample(
        self,
        degraded_img: Image.Image,
        genome: DocumentGenome,
    ) -> dict:
        """Prepare sample for DGI training."""
        # Image tensor
        img_tensor = image_to_tensor(degraded_img, self.image_size)

        # Genome token sequence
        token_seq = genome_to_token_sequence(genome)

        # Bounding boxes (normalized to [0, 1])
        bboxes = []
        for elem in genome.content:
            bb = elem.bbox
            bboxes.append([
                bb.x1 / genome.layout.page_width,
                bb.y1 / genome.layout.page_height,
                bb.x2 / genome.layout.page_width,
                bb.y2 / genome.layout.page_height,
            ])

        # Pad bboxes to fixed size
        max_regions = 64
        while len(bboxes) < max_regions:
            bboxes.append([0.0, 0.0, 0.0, 0.0])
        bboxes = bboxes[:max_regions]

        bbox_mask = [1.0] * min(len(genome.content), max_regions)
        bbox_mask += [0.0] * (max_regions - len(bbox_mask))

        result = {
            "image": img_tensor,
            "token_sequence": token_seq,
            "bboxes": torch.tensor(bboxes, dtype=torch.float32),
            "bbox_mask": torch.tensor(bbox_mask, dtype=torch.float32),
            "num_regions": len(genome.content),
        }

        # Tokenize if tokenizer provided
        if self.tokenizer:
            tokens = self.tokenizer(
                token_seq,
                max_length=self.max_seq_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            result["input_ids"] = tokens["input_ids"].squeeze(0)
            result["attention_mask"] = tokens["attention_mask"].squeeze(0)

        return result

    def _prepare_sir_sample(
        self,
        clean_img: Image.Image,
        sample_id: str,
    ) -> dict:
        """Prepare sample for SIR contrastive training."""
        # Extract two sets of patches from the same document (positive pair)
        patches_anchor = extract_patches(
            clean_img, self.patch_size, self.num_patches, "quality"
        )
        patches_positive = extract_patches(
            clean_img, self.patch_size, self.num_patches, "random"
        )

        # Safety: ensure both lists have exactly num_patches tensors
        assert len(patches_anchor) == self.num_patches, (
            f"anchor got {len(patches_anchor)}, expected {self.num_patches}"
        )
        assert len(patches_positive) == self.num_patches, (
            f"positive got {len(patches_positive)}, expected {self.num_patches}"
        )

        return {
            "anchor_patches": torch.stack(patches_anchor),     # (K, 3, P, P)
            "positive_patches": torch.stack(patches_positive), # (K, 3, P, P)
            "document_id": sample_id,
        }

    def _prepare_nre_sample(
        self,
        genome: DocumentGenome,
        clean_img: Image.Image,
    ) -> dict:
        """Prepare sample for NRE training."""
        # Skeleton image from genome
        skeleton_tensor = render_skeleton_to_tensor(
            genome, target_size=self.image_size
        )

        # Clean image tensor (target)
        clean_tensor = image_to_tensor(clean_img, self.image_size)

        # Style patches from clean image (for SIR embedding)
        style_patches = extract_patches(
            clean_img, self.patch_size, self.num_patches, "quality"
        )

        # Genome token sequence (text content for CLIP conditioning)
        token_seq = genome_to_token_sequence(genome)

        return {
            "skeleton": torch.from_numpy(skeleton_tensor),       # (3, H, W)
            "clean_image": clean_tensor,                         # (3, H, W)
            "style_patches": torch.stack(style_patches),         # (K, 3, P, P)
            "token_sequence": token_seq,                         # str
        }

    def _prepare_full_sample(
        self,
        degraded_img: Image.Image,
        clean_img: Image.Image,
        genome: DocumentGenome,
        sample_id: str,
    ) -> dict:
        """Prepare all components for evaluation/inference."""
        # Combine all mode outputs
        dgi_data = self._prepare_dgi_sample(degraded_img, genome)
        nre_data = self._prepare_nre_sample(genome, clean_img)

        return {
            "sample_id": sample_id,
            "degraded_image": dgi_data["image"],
            "clean_image": nre_data["clean_image"],
            "skeleton": nre_data["skeleton"],
            "style_patches": nre_data["style_patches"],
            "token_sequence": dgi_data["token_sequence"],
            "bboxes": dgi_data["bboxes"],
            "bbox_mask": dgi_data["bbox_mask"],
            "num_regions": dgi_data["num_regions"],
        }


# ============================================================================
# DataLoader Helpers
# ============================================================================

def create_dataloader(
    data_dir: str | Path,
    mode: Literal["dgi", "sir", "nre", "full"] = "full",
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 4,
    image_size: tuple[int, int] = (512, 512),
    **kwargs,
) -> DataLoader:
    """Convenience function to create a DataLoader."""
    dataset = GenomeDocDataset(
        data_dir=data_dir,
        mode=mode,
        image_size=image_size,
        **kwargs,
    )

    # Custom collate for string fields
    def collate_fn(batch):
        collated = {}
        for key in batch[0]:
            values = [item[key] for item in batch]
            if isinstance(values[0], torch.Tensor):
                collated[key] = torch.stack(values)
            elif isinstance(values[0], str):
                collated[key] = values  # Keep as list of strings
            elif isinstance(values[0], (int, float)):
                collated[key] = torch.tensor(values)
            else:
                collated[key] = values
        return collated

    # When preloading to RAM, data workers are unnecessary (no disk I/O)
    # Setting num_workers=0 avoids redundant memory copies across processes
    effective_workers = 0 if dataset.preload_to_ram else num_workers

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=effective_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
        persistent_workers=False if effective_workers == 0 else True,
    )


# ============================================================================
# Quick Test
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test dataset loading")
    parser.add_argument("--data-dir", type=str, default="data/synthetic/train")
    parser.add_argument("--mode", type=str, default="full",
                        choices=["dgi", "sir", "nre", "full"])
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    print(f"Loading dataset from {args.data_dir} in '{args.mode}' mode...")

    dataset = GenomeDocDataset(data_dir=args.data_dir, mode=args.mode)
    print(f"Found {len(dataset)} samples")

    # Load first sample
    sample = dataset[0]
    print(f"\nSample keys: {list(sample.keys())}")
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape} ({v.dtype})")
        elif isinstance(v, str):
            print(f"  {k}: str (len={len(v)})")
        else:
            print(f"  {k}: {type(v).__name__} = {v}")
