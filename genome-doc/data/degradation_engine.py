"""
Degradation Engine

Applies realistic synthetic degradations to clean document images to create
training pairs. Each degradation is parameterized and randomly combined for
diversity.

Supported degradations:
- Gaussian noise
- Motion blur
- JPEG compression artifacts
- Perspective warping
- Coffee/tea stain overlays
- Paper yellowing (color shift)
- Ink fading (contrast reduction)
- Crease/fold simulation
- Bleed-through from reverse side

Usage:
    python data/degradation_engine.py --input-dir data/synthetic/train/images/clean \
                                       --output-dir data/synthetic/train/images/degraded
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageFilter


# ============================================================================
# Individual Degradation Functions
# ============================================================================

def add_gaussian_noise(
    img: np.ndarray,
    mean: float = 0.0,
    sigma_range: tuple[float, float] = (5.0, 30.0),
) -> np.ndarray:
    """Add Gaussian noise to the image."""
    sigma = random.uniform(*sigma_range)
    noise = np.random.normal(mean, sigma, img.shape).astype(np.float32)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


def add_motion_blur(
    img: np.ndarray,
    kernel_size_range: tuple[int, int] = (3, 9),
    angle_range: tuple[float, float] = (0, 180),
) -> np.ndarray:
    """Apply motion blur with random direction."""
    k = random.choice(range(kernel_size_range[0], kernel_size_range[1] + 1, 2))
    angle = random.uniform(*angle_range)

    # Create motion blur kernel
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0 / k

    # Rotate the kernel
    center = (k // 2, k // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    kernel = cv2.warpAffine(kernel, M, (k, k))
    kernel = kernel / kernel.sum()

    return cv2.filter2D(img, -1, kernel)


def apply_jpeg_compression(
    img: np.ndarray,
    quality_range: tuple[int, int] = (15, 60),
) -> np.ndarray:
    """Simulate JPEG compression artifacts."""
    quality = random.randint(*quality_range)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode(".jpg", img, encode_param)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return decoded


def apply_perspective_warp(
    img: np.ndarray,
    max_displacement: float = 0.05,
) -> np.ndarray:
    """Apply slight perspective warping to simulate scanning artifacts."""
    h, w = img.shape[:2]
    d = max_displacement

    # Source corners
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    # Randomly displace corners
    dst = np.float32([
        [random.uniform(0, w * d), random.uniform(0, h * d)],
        [w - random.uniform(0, w * d), random.uniform(0, h * d)],
        [w - random.uniform(0, w * d), h - random.uniform(0, h * d)],
        [random.uniform(0, w * d), h - random.uniform(0, h * d)],
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        img, M, (w, h),
        borderMode=cv2.BORDER_REPLICATE
    )
    return warped


def add_stain_overlay(
    img: np.ndarray,
    num_stains: int = 1,
    stain_type: str = "coffee",
) -> np.ndarray:
    """Add coffee/tea stain overlays."""
    result = img.astype(np.float32)
    h, w = img.shape[:2]

    for _ in range(num_stains):
        # Random position and size
        cx = random.randint(w // 4, 3 * w // 4)
        cy = random.randint(h // 4, 3 * h // 4)
        radius = random.randint(min(h, w) // 10, min(h, w) // 4)

        # Create circular gradient mask
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(np.float32)
        mask = np.clip(1.0 - dist / radius, 0, 1) ** 2

        # Add some irregularity
        noise = cv2.GaussianBlur(
            np.random.randn(h, w).astype(np.float32) * 0.3,
            (15, 15), 5
        )
        mask = np.clip(mask + noise * mask, 0, 1)

        # Stain color
        if stain_type == "coffee":
            stain_color = np.array([
                random.randint(100, 150),
                random.randint(70, 120),
                random.randint(40, 80)
            ], dtype=np.float32)
        else:  # tea
            stain_color = np.array([
                random.randint(150, 190),
                random.randint(140, 170),
                random.randint(80, 120)
            ], dtype=np.float32)

        opacity = random.uniform(0.1, 0.35)
        mask_3d = mask[:, :, np.newaxis] * opacity

        result = result * (1 - mask_3d) + stain_color * mask_3d

    return np.clip(result, 0, 255).astype(np.uint8)


def apply_yellowing(
    img: np.ndarray,
    intensity_range: tuple[float, float] = (0.02, 0.12),
) -> np.ndarray:
    """Simulate paper yellowing / aging."""
    intensity = random.uniform(*intensity_range)
    result = img.astype(np.float32)

    # Yellow shift: increase R slightly, increase G less, decrease B
    result[:, :, 0] = np.clip(result[:, :, 0] + intensity * 40, 0, 255)   # R (BGR ordering in cv2)
    result[:, :, 1] = np.clip(result[:, :, 1] + intensity * 25, 0, 255)   # G
    result[:, :, 2] = np.clip(result[:, :, 2] - intensity * 30, 0, 255)   # B

    return result.astype(np.uint8)


def apply_ink_fading(
    img: np.ndarray,
    factor_range: tuple[float, float] = (0.55, 0.85),
) -> np.ndarray:
    """Reduce contrast to simulate faded ink."""
    factor = random.uniform(*factor_range)
    # Bring towards gray
    gray_val = 200
    result = img.astype(np.float32)
    result = gray_val + factor * (result - gray_val)
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_crease(
    img: np.ndarray,
    num_creases: int = 1,
) -> np.ndarray:
    """Simulate fold/crease lines across the document."""
    result = img.copy()
    h, w = img.shape[:2]

    for _ in range(num_creases):
        # Random line across the image
        if random.random() < 0.5:
            # Horizontal crease
            y = random.randint(h // 5, 4 * h // 5)
            thickness = random.randint(1, 3)
            shadow_width = random.randint(3, 10)

            for dy in range(-shadow_width, shadow_width + 1):
                if 0 <= y + dy < h:
                    factor = 1.0 - 0.15 * np.exp(-(dy ** 2) / (2 * (shadow_width / 2) ** 2))
                    result[y + dy, :] = np.clip(
                        result[y + dy, :].astype(np.float32) * factor, 0, 255
                    ).astype(np.uint8)

            # Dark crease line
            cv2.line(result, (0, y), (w, y), (180, 180, 180), thickness)
        else:
            # Vertical crease
            x = random.randint(w // 5, 4 * w // 5)
            thickness = random.randint(1, 3)
            shadow_width = random.randint(3, 10)

            for dx in range(-shadow_width, shadow_width + 1):
                if 0 <= x + dx < w:
                    factor = 1.0 - 0.15 * np.exp(-(dx ** 2) / (2 * (shadow_width / 2) ** 2))
                    result[:, x + dx] = np.clip(
                        result[:, x + dx].astype(np.float32) * factor, 0, 255
                    ).astype(np.uint8)

            cv2.line(result, (x, 0), (x, h), (180, 180, 180), thickness)

    return result


def apply_bleed_through(
    img: np.ndarray,
    intensity_range: tuple[float, float] = (0.05, 0.2),
) -> np.ndarray:
    """Simulate bleed-through from the reverse side of the page."""
    h, w = img.shape[:2]
    intensity = random.uniform(*intensity_range)

    # Create a fake "reverse side" by flipping and darkening
    reverse = cv2.flip(img, 1)  # Horizontal flip
    reverse = cv2.GaussianBlur(reverse, (7, 7), 3)

    # Make it ghostly
    ghost = 255 - (255 - reverse).astype(np.float32) * intensity

    # Blend
    result = np.minimum(img.astype(np.float32), ghost)
    return np.clip(result, 0, 255).astype(np.uint8)


def add_salt_pepper_noise(
    img: np.ndarray,
    amount: float = 0.005,
) -> np.ndarray:
    """Add salt and pepper noise."""
    result = img.copy()
    h, w = img.shape[:2]
    num_pixels = int(amount * h * w)

    # Salt (white) pixels
    for _ in range(num_pixels):
        y, x = random.randint(0, h - 1), random.randint(0, w - 1)
        result[y, x] = 255

    # Pepper (black) pixels
    for _ in range(num_pixels):
        y, x = random.randint(0, h - 1), random.randint(0, w - 1)
        result[y, x] = 0

    return result


# ============================================================================
# Degradation Pipeline
# ============================================================================

class DegradationPipeline:
    """
    Applies a randomized combination of degradations to clean document images.

    Each degradation has a probability of being applied, and its parameters
    are randomized within configurable ranges.
    """

    # Default probabilities for each degradation type
    DEFAULT_PROBABILITIES = {
        "gaussian_noise": 0.7,
        "motion_blur": 0.3,
        "jpeg_compression": 0.5,
        "perspective_warp": 0.3,
        "stain": 0.2,
        "yellowing": 0.5,
        "ink_fading": 0.4,
        "crease": 0.2,
        "bleed_through": 0.15,
        "salt_pepper": 0.2,
    }

    def __init__(
        self,
        probabilities: Optional[dict[str, float]] = None,
        min_degradations: int = 1,
        max_degradations: int = 5,
        severity: str = "medium",
        seed: Optional[int] = None,
    ):
        """
        Args:
            probabilities: Override default per-degradation probabilities.
            min_degradations: Minimum number of degradations to apply.
            max_degradations: Maximum number of degradations to apply.
            severity: 'light', 'medium', or 'heavy' — scales degradation intensity.
            seed: Random seed.
        """
        self.probabilities = {**self.DEFAULT_PROBABILITIES}
        if probabilities:
            self.probabilities.update(probabilities)

        self.min_degradations = min_degradations
        self.max_degradations = max_degradations
        self.severity = severity

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Severity multipliers
        self._severity_scale = {
            "light": 0.5,
            "medium": 1.0,
            "heavy": 1.5,
        }.get(severity, 1.0)

    def degrade(self, img: Image.Image) -> Image.Image:
        """
        Apply random degradations to a clean document image.

        Args:
            img: Clean PIL Image (RGB).

        Returns:
            Degraded PIL Image (RGB).
        """
        # Convert PIL -> numpy (RGB -> BGR for OpenCV)
        img_np = np.array(img)
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Select which degradations to apply
        available = []
        for name, prob in self.probabilities.items():
            if random.random() < prob:
                available.append(name)

        # Ensure minimum degradations
        if len(available) < self.min_degradations:
            all_names = list(self.probabilities.keys())
            while len(available) < self.min_degradations and all_names:
                extra = random.choice(all_names)
                if extra not in available:
                    available.append(extra)
                all_names.remove(extra)

        # Cap at max
        if len(available) > self.max_degradations:
            available = random.sample(available, self.max_degradations)

        # Shuffle order (order matters for some degradations)
        random.shuffle(available)

        # Apply each selected degradation
        s = self._severity_scale

        for name in available:
            try:
                if name == "gaussian_noise":
                    img_cv = add_gaussian_noise(img_cv, sigma_range=(5 * s, 30 * s))

                elif name == "motion_blur":
                    k_min = max(3, int(3 * s))
                    k_max = max(k_min, int(9 * s))
                    if k_min % 2 == 0:
                        k_min += 1
                    if k_max % 2 == 0:
                        k_max += 1
                    img_cv = add_motion_blur(img_cv, kernel_size_range=(k_min, k_max))

                elif name == "jpeg_compression":
                    q_min = max(5, int(15 / s))
                    q_max = max(q_min + 5, int(60 / s))
                    img_cv = apply_jpeg_compression(img_cv, quality_range=(q_min, q_max))

                elif name == "perspective_warp":
                    img_cv = apply_perspective_warp(img_cv, max_displacement=0.05 * s)

                elif name == "stain":
                    stain_type = random.choice(["coffee", "tea"])
                    num = random.randint(1, max(1, int(2 * s)))
                    img_cv = add_stain_overlay(img_cv, num_stains=num, stain_type=stain_type)

                elif name == "yellowing":
                    img_cv = apply_yellowing(img_cv, intensity_range=(0.02 * s, 0.12 * s))

                elif name == "ink_fading":
                    factor_min = max(0.3, 0.55 - 0.1 * s)
                    factor_max = min(0.95, 0.85 + 0.05 * s)
                    img_cv = apply_ink_fading(img_cv, factor_range=(factor_min, factor_max))

                elif name == "crease":
                    num = random.randint(1, max(1, int(2 * s)))
                    img_cv = apply_crease(img_cv, num_creases=num)

                elif name == "bleed_through":
                    img_cv = apply_bleed_through(img_cv, intensity_range=(0.05 * s, 0.2 * s))

                elif name == "salt_pepper":
                    img_cv = add_salt_pepper_noise(img_cv, amount=0.005 * s)

            except Exception as e:
                # Skip failed degradation silently
                pass

        # Convert back BGR -> RGB -> PIL
        result_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result_rgb)

    def degrade_batch(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        progress_interval: int = 100,
    ) -> int:
        """
        Apply degradations to all images in input_dir, save to output_dir.

        Returns:
            Number of images processed.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        image_files = sorted([
            f for f in input_dir.iterdir()
            if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".tiff", ".bmp")
        ])

        count = 0
        for img_path in image_files:
            img = Image.open(str(img_path)).convert("RGB")
            degraded = self.degrade(img)
            degraded.save(str(output_dir / img_path.name))
            count += 1

            if count % progress_interval == 0:
                print(f"  Degraded {count}/{len(image_files)} images...")

        print(f"Done! Degraded {count} images → {output_dir}")
        return count


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Apply synthetic degradations to clean document images"
    )
    parser.add_argument(
        "--input-dir", type=str, required=True,
        help="Directory containing clean document images"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory for degraded images"
    )
    parser.add_argument(
        "--severity", type=str, default="medium",
        choices=["light", "medium", "heavy"],
        help="Degradation severity level (default: medium)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    print(f"Applying {args.severity} degradations...")
    print(f"  Input:  {args.input_dir}")
    print(f"  Output: {args.output_dir}")

    pipeline = DegradationPipeline(
        severity=args.severity,
        seed=args.seed,
    )

    pipeline.degrade_batch(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
