"""
Evaluation Metrics for Genome-Doc

Implements all metrics from the evaluation plan:
- Image Quality: PSNR, SSIM, LPIPS
- OCR Accuracy: CER, WER
- Semantic Fidelity: BERTScore
- Layout Accuracy: mAP@IoU=0.5
- Hallucination Rate: genome text vs OCR on rendered output
- Genome Accuracy: token-level F1

Usage:
    python eval/metrics.py --pred-dir results/ --gt-dir data/synthetic/test/
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ============================================================================
# Image Quality Metrics
# ============================================================================

def compute_psnr(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Peak Signal-to-Noise Ratio between two images.

    Args:
        pred, target: (H, W, C) uint8 arrays

    Returns:
        PSNR in dB (higher = better)
    """
    mse = np.mean((pred.astype(np.float64) - target.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(255.0 ** 2 / mse)


def compute_ssim(
    pred: np.ndarray,
    target: np.ndarray,
    window_size: int = 11,
) -> float:
    """
    Structural Similarity Index (simplified implementation).

    Args:
        pred, target: (H, W, C) uint8 arrays

    Returns:
        SSIM score in [0, 1] (higher = better)
    """
    try:
        from skimage.metrics import structural_similarity
        # Convert to grayscale for SSIM
        pred_gray = np.mean(pred, axis=2) if pred.ndim == 3 else pred
        target_gray = np.mean(target, axis=2) if target.ndim == 3 else target
        return structural_similarity(
            pred_gray, target_gray,
            data_range=255, win_size=min(window_size, min(pred_gray.shape[:2]))
        )
    except ImportError:
        # Fallback: simplified SSIM
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        pred_f = pred.astype(np.float64)
        target_f = target.astype(np.float64)

        mu_pred = np.mean(pred_f)
        mu_target = np.mean(target_f)
        sigma_pred = np.std(pred_f)
        sigma_target = np.std(target_f)
        sigma_cross = np.mean((pred_f - mu_pred) * (target_f - mu_target))

        ssim = ((2 * mu_pred * mu_target + C1) * (2 * sigma_cross + C2)) / \
               ((mu_pred ** 2 + mu_target ** 2 + C1) * (sigma_pred ** 2 + sigma_target ** 2 + C2))
        return float(ssim)


def compute_lpips(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> float:
    """
    LPIPS perceptual distance (requires lpips package).

    Args:
        pred, target: (B, 3, H, W) tensors in [0, 1]

    Returns:
        LPIPS distance (lower = better)
    """
    try:
        import lpips
        loss_fn = lpips.LPIPS(net="alex", verbose=False)
        if pred.is_cuda:
            loss_fn = loss_fn.cuda()
        with torch.no_grad():
            # LPIPS expects [-1, 1] range
            pred_scaled = pred * 2 - 1
            target_scaled = target * 2 - 1
            return loss_fn(pred_scaled, target_scaled).mean().item()
    except ImportError:
        print("WARNING: lpips not installed. Skipping LPIPS.")
        return 0.0


# ============================================================================
# OCR Accuracy Metrics
# ============================================================================

def compute_cer(predicted_text: str, reference_text: str) -> float:
    """
    Character Error Rate using edit distance.

    Returns:
        CER in [0, inf) (lower = better). 0 = perfect match.
    """
    try:
        from jiwer import cer
        return cer(reference_text, predicted_text)
    except ImportError:
        # Fallback: simple Levenshtein implementation
        return _levenshtein_distance(predicted_text, reference_text) / max(len(reference_text), 1)


def compute_wer(predicted_text: str, reference_text: str) -> float:
    """
    Word Error Rate.

    Returns:
        WER in [0, inf) (lower = better).
    """
    try:
        from jiwer import wer
        return wer(reference_text, predicted_text)
    except ImportError:
        pred_words = predicted_text.split()
        ref_words = reference_text.split()
        return _levenshtein_distance(
            " ".join(pred_words), " ".join(ref_words)
        ) / max(len(ref_words), 1)


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr_row.append(min(
                curr_row[-1] + 1,       # insertion
                prev_row[j + 1] + 1,    # deletion
                prev_row[j] + cost,     # substitution
            ))
        prev_row = curr_row

    return prev_row[-1]


# ============================================================================
# Semantic Fidelity
# ============================================================================

def compute_bert_score(
    predicted_texts: list[str],
    reference_texts: list[str],
) -> dict[str, float]:
    """
    BERTScore for semantic similarity.

    Returns:
        Dict with 'precision', 'recall', 'f1' (higher = better)
    """
    try:
        from bert_score import score as bert_score
        P, R, F1 = bert_score(
            predicted_texts, reference_texts,
            lang="en", verbose=False
        )
        return {
            "precision": P.mean().item(),
            "recall": R.mean().item(),
            "f1": F1.mean().item(),
        }
    except ImportError:
        print("WARNING: bert-score not installed. Skipping BERTScore.")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}


# ============================================================================
# Layout Accuracy
# ============================================================================

def compute_layout_map(
    pred_bboxes: list[list[float]],
    gt_bboxes: list[list[float]],
    iou_threshold: float = 0.5,
) -> float:
    """
    Mean Average Precision for layout detection at given IoU threshold.

    Args:
        pred_bboxes: List of [x1, y1, x2, y2] predicted boxes
        gt_bboxes: List of [x1, y1, x2, y2] ground truth boxes
        iou_threshold: IoU threshold for matching

    Returns:
        mAP score in [0, 1]
    """
    if not gt_bboxes:
        return 1.0 if not pred_bboxes else 0.0
    if not pred_bboxes:
        return 0.0

    # Compute IoU matrix
    num_pred = len(pred_bboxes)
    num_gt = len(gt_bboxes)

    tp = 0
    matched_gt = set()

    for pred in pred_bboxes:
        best_iou = 0.0
        best_gt_idx = -1

        for j, gt in enumerate(gt_bboxes):
            if j in matched_gt:
                continue

            iou = _compute_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            matched_gt.add(best_gt_idx)

    precision = tp / num_pred if num_pred > 0 else 0.0
    recall = tp / num_gt if num_gt > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return f1  # Using F1 as a proxy for mAP in single-class case


def _compute_iou(box1: list[float], box2: list[float]) -> float:
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


# ============================================================================
# Hallucination Rate
# ============================================================================

def compute_hallucination_rate(
    genome_text: str,
    rendered_image: Image.Image | np.ndarray,
    ocr_engine: str = "easyocr",
) -> dict[str, float]:
    """
    Hallucination rate: compare genome text vs OCR of rendered output.

    Args:
        genome_text: Text from the Document Genome
        rendered_image: The NRE-rendered output image
        ocr_engine: Which OCR engine to use

    Returns:
        Dict with 'char_hallucination_rate', 'word_hallucination_rate'
    """
    if isinstance(rendered_image, np.ndarray):
        rendered_image = Image.fromarray(rendered_image)

    # Run OCR on rendered image
    ocr_text = _run_ocr(rendered_image, engine=ocr_engine)

    # Compute error rates
    char_rate = compute_cer(ocr_text, genome_text)
    word_rate = compute_wer(ocr_text, genome_text)

    return {
        "char_hallucination_rate": char_rate,
        "word_hallucination_rate": word_rate,
        "ocr_text": ocr_text,
    }


def _run_ocr(image: Image.Image, engine: str = "easyocr") -> str:
    """Run OCR on an image and return extracted text."""
    if engine == "easyocr":
        try:
            import easyocr
            reader = easyocr.Reader(["en"], verbose=False)
            results = reader.readtext(np.array(image))
            return " ".join([r[1] for r in results])
        except ImportError:
            pass

    if engine == "tesseract" or engine == "easyocr":
        try:
            import pytesseract
            return pytesseract.image_to_string(image).strip()
        except ImportError:
            pass

    # Fallback
    print("WARNING: No OCR engine available. Install easyocr or pytesseract.")
    return ""


# ============================================================================
# Full Evaluation Suite
# ============================================================================

class GenomeDocEvaluator:
    """
    Comprehensive evaluator for Genome-Doc results.
    Computes all metrics from the evaluation plan.
    """

    def __init__(self, use_lpips: bool = True, ocr_engine: str = "easyocr"):
        self.use_lpips = use_lpips
        self.ocr_engine = ocr_engine

    def evaluate_sample(
        self,
        pred_image: np.ndarray | Image.Image,
        gt_image: np.ndarray | Image.Image,
        pred_genome: Optional[object] = None,
        gt_genome: Optional[object] = None,
    ) -> dict[str, float]:
        """Evaluate a single restored document."""
        if isinstance(pred_image, Image.Image):
            pred_image = np.array(pred_image)
        if isinstance(gt_image, Image.Image):
            gt_image = np.array(gt_image)

        results = {}

        # Image quality
        results["psnr"] = compute_psnr(pred_image, gt_image)
        results["ssim"] = compute_ssim(pred_image, gt_image)

        # Genome accuracy
        if pred_genome and gt_genome:
            from genome.utils import compute_genome_token_f1
            f1_results = compute_genome_token_f1(pred_genome, gt_genome)
            results["genome_f1"] = f1_results["f1"]

            # Layout accuracy
            pred_bboxes = [[e.bbox.x1, e.bbox.y1, e.bbox.x2, e.bbox.y2]
                           for e in pred_genome.content]
            gt_bboxes = [[e.bbox.x1, e.bbox.y1, e.bbox.x2, e.bbox.y2]
                         for e in gt_genome.content]
            results["layout_map"] = compute_layout_map(pred_bboxes, gt_bboxes)

            # Hallucination rate
            pred_text = " ".join(e.text for e in pred_genome.content)
            gt_text = " ".join(e.text for e in gt_genome.content)
            results["cer"] = compute_cer(pred_text, gt_text)
            results["wer"] = compute_wer(pred_text, gt_text)

        return results

    def evaluate_batch(
        self,
        pred_images: list[np.ndarray],
        gt_images: list[np.ndarray],
        pred_genomes: list | None = None,
        gt_genomes: list | None = None,
    ) -> dict[str, float]:
        """Evaluate a batch and return averaged metrics."""
        all_results = []

        for i in range(len(pred_images)):
            pg = pred_genomes[i] if pred_genomes else None
            gg = gt_genomes[i] if gt_genomes else None
            r = self.evaluate_sample(pred_images[i], gt_images[i], pg, gg)
            all_results.append(r)

        # Average all metrics
        avg = {}
        for key in all_results[0]:
            vals = [r[key] for r in all_results if key in r]
            avg[key] = sum(vals) / len(vals) if vals else 0.0

        return avg

    def print_results(self, results: dict[str, float]) -> None:
        """Pretty-print evaluation results."""
        print(f"\n{'='*50}")
        print("Genome-Doc Evaluation Results")
        print(f"{'='*50}")

        metric_groups = {
            "Image Quality": ["psnr", "ssim", "lpips"],
            "Text Accuracy": ["cer", "wer"],
            "Genome Quality": ["genome_f1", "layout_map"],
            "Hallucination": ["char_hallucination_rate", "word_hallucination_rate"],
        }

        for group_name, metrics in metric_groups.items():
            present = {m: results[m] for m in metrics if m in results}
            if present:
                print(f"\n{group_name}:")
                for m, v in present.items():
                    direction = "↑" if m in ["psnr", "ssim", "genome_f1", "layout_map"] else "↓"
                    print(f"  {m:30s} {v:.4f} {direction}")
