"""
Check whether an NRE checkpoint directory has the files needed for inference.

Usage:
    python inference/check_nre_checkpoint.py --nre-dir checkpoints/nre
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def check_nre_dir(nre_dir: Path) -> int:
    if not nre_dir.exists():
        print(f"ERROR: NRE directory does not exist: {nre_dir}")
        return 2

    metadata_paths = (
        list(nre_dir.glob("best_model*.pt"))
        + list(nre_dir.glob("final_model*.pt"))
        + list(nre_dir.glob("checkpoint_epoch*.pt"))
    )

    checks = [
        (
            "metadata checkpoint",
            metadata_paths or [nre_dir / "best_model.pt", nre_dir / "final_model.pt"],
            False,
        ),
        ("style projector", [nre_dir / "style_projector.pt"], True),
        ("ControlNet config", [nre_dir / "controlnet" / "config.json"], True),
        (
            "ControlNet weights",
            [
                nre_dir / "controlnet" / "diffusion_pytorch_model.safetensors",
                nre_dir / "controlnet" / "diffusion_pytorch_model.bin",
            ],
            True,
        ),
        (
            "UNet LoRA",
            [
                nre_dir / "unet_lora" / "adapter_config.json",
                nre_dir / "unet_lora" / "adapter_model.safetensors",
                nre_dir / "unet_lora" / "pytorch_model.bin",
                nre_dir / "unet_lora" / "config.json",
                nre_dir / "unet_lora" / "diffusion_pytorch_model.safetensors",
                nre_dir / "unet_lora" / "diffusion_pytorch_model.bin",
                nre_dir / "unet_lora_weights.pt",
            ],
            False,
        ),
    ]

    missing_required = []
    for label, paths, required in checks:
        found = [p for p in paths if p.exists()]
        if found:
            print(f"OK: {label}: {found[0]}")
        elif required:
            print(f"ERROR: missing {label}")
            for p in paths:
                print(f"  expected one of: {p}")
            missing_required.append(label)
        else:
            print(f"WARN: missing optional {label}")
            for p in paths:
                print(f"  checked: {p}")

    if missing_required:
        print("\nNRE checkpoint is NOT usable for full trained inference.")
        print("The folder must include style_projector.pt and the saved controlnet/ directory.")
        return 1

    print("\nNRE checkpoint structure looks usable.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Check NRE checkpoint files")
    parser.add_argument("--nre-dir", required=True, help="Path to trained NRE checkpoint directory")
    args = parser.parse_args()
    sys.exit(check_nre_dir(Path(args.nre_dir)))


if __name__ == "__main__":
    main()
