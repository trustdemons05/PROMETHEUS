"""
Genome Utilities

Helper functions for manipulating, comparing, and transforming Document Genomes.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import numpy as np

from genome.schema import (
    BoundingBox,
    ContentElement,
    DocumentGenome,
    LayoutInfo,
    StyleInfo,
    TextRegionType,
)


# ============================================================================
# Genome Comparison
# ============================================================================

def compute_genome_token_f1(
    predicted: DocumentGenome,
    ground_truth: DocumentGenome,
) -> dict[str, float]:
    """
    Compute token-level F1 score between predicted and ground-truth genomes.

    Returns:
        Dict with 'precision', 'recall', 'f1' keys.
    """
    pred_tokens = _tokenize_genome_text(predicted)
    gt_tokens = _tokenize_genome_text(ground_truth)

    if not gt_tokens and not pred_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not gt_tokens or not pred_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    pred_set = set(pred_tokens)
    gt_set = set(gt_tokens)

    tp = len(pred_set & gt_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gt_set) if gt_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def compute_bbox_iou_matrix(
    predicted: DocumentGenome,
    ground_truth: DocumentGenome,
) -> np.ndarray:
    """
    Compute IoU matrix between predicted and ground-truth bounding boxes.

    Returns:
        np.ndarray of shape (num_predicted, num_ground_truth).
    """
    n_pred = len(predicted.content)
    n_gt = len(ground_truth.content)
    iou_matrix = np.zeros((n_pred, n_gt), dtype=np.float32)

    for i, pred_elem in enumerate(predicted.content):
        for j, gt_elem in enumerate(ground_truth.content):
            iou_matrix[i, j] = pred_elem.bbox.iou(gt_elem.bbox)

    return iou_matrix


def match_regions(
    predicted: DocumentGenome,
    ground_truth: DocumentGenome,
    iou_threshold: float = 0.5,
) -> list[tuple[int, int, float]]:
    """
    Match predicted regions to ground-truth regions using greedy IoU matching.

    Returns:
        List of (pred_idx, gt_idx, iou_score) tuples for matched pairs.
    """
    iou_matrix = compute_bbox_iou_matrix(predicted, ground_truth)
    matches = []
    used_gt = set()

    # Greedy matching: for each predicted box, find best unmatched GT box
    for i in range(iou_matrix.shape[0]):
        best_j = -1
        best_iou = iou_threshold
        for j in range(iou_matrix.shape[1]):
            if j not in used_gt and iou_matrix[i, j] > best_iou:
                best_iou = iou_matrix[i, j]
                best_j = j
        if best_j >= 0:
            matches.append((i, best_j, best_iou))
            used_gt.add(best_j)

    return matches


# ============================================================================
# Genome Text Manipulation
# ============================================================================

def _tokenize_genome_text(genome: DocumentGenome) -> list[str]:
    """Tokenize all text in a genome into lowercase word tokens."""
    all_text = " ".join(elem.text for elem in genome.content)
    tokens = re.findall(r'\b\w+\b', all_text.lower())
    return tokens


def extract_text_by_type(
    genome: DocumentGenome,
    region_type: TextRegionType,
) -> str:
    """Extract concatenated text from all regions of a specific type."""
    regions = genome.get_regions_by_type(region_type)
    return "\n".join(r.text for r in regions)


def reorder_content_by_position(genome: DocumentGenome) -> DocumentGenome:
    """
    Re-sort content elements top-to-bottom, left-to-right.
    Useful when reading order is not provided.
    """
    sorted_content = sorted(
        genome.content,
        key=lambda e: (e.bbox.y1, e.bbox.x1),
    )
    new_order = list(range(len(sorted_content)))

    return DocumentGenome(
        content=sorted_content,
        layout=LayoutInfo(
            page_width=genome.layout.page_width,
            page_height=genome.layout.page_height,
            columns=genome.layout.columns,
            margins=genome.layout.margins,
            reading_order=new_order,
        ),
        style=genome.style,
    )


# ============================================================================
# Genome ↔ Flat Token Sequence (for DGI decoder)
# ============================================================================

# Template tokens for the DGI decoder
GENOME_TEMPLATE_TOKENS = {
    "bos": "<genome>",
    "eos": "</genome>",
    "content_start": "<content>",
    "content_end": "</content>",
    "element_start": "<elem>",
    "element_end": "</elem>",
    "text_start": "<text>",
    "text_end": "</text>",
    "bbox_start": "<bbox>",
    "bbox_end": "</bbox>",
    "type_start": "<type>",
    "type_end": "</type>",
    "layout_start": "<layout>",
    "layout_end": "</layout>",
    "style_start": "<style>",
    "style_end": "</style>",
    "sep": "<sep>",
}


def genome_to_token_sequence(genome: DocumentGenome) -> str:
    """
    Convert a DocumentGenome to a flat token sequence for DGI decoder training.

    This is the template-guided format (lean version) that the Donut decoder
    learns to generate. Example output:

        <genome><content><elem><type>heading</type><text>Chapter 1</text>
        <bbox>120 50 500 90</bbox></elem>...</content>
        <layout>2480 3508 1 200 200 200 250</layout>
        <style>academic_paper serif modern_digital white_bond</style></genome>
    """
    t = GENOME_TEMPLATE_TOKENS
    parts = [t["bos"], t["content_start"]]

    for elem in genome.content:
        parts.append(t["element_start"])
        parts.append(f'{t["type_start"]}{elem.type.value}{t["type_end"]}')
        parts.append(f'{t["text_start"]}{elem.text}{t["text_end"]}')
        bb = elem.bbox
        parts.append(f'{t["bbox_start"]}{bb.x1} {bb.y1} {bb.x2} {bb.y2}{t["bbox_end"]}')
        parts.append(t["element_end"])

    parts.append(t["content_end"])

    # Layout as space-separated values
    layout = genome.layout
    layout_str = f"{layout.page_width} {layout.page_height} {layout.columns}"
    layout_str += " " + " ".join(str(m) for m in layout.margins)
    parts.append(f'{t["layout_start"]}{layout_str}{t["layout_end"]}')

    # Style as space-separated enum values
    style = genome.style
    style_str = (
        f"{style.document_class.value} {style.primary_font_class.value} "
        f"{style.estimated_era.value} {style.paper_type.value}"
    )
    parts.append(f'{t["style_start"]}{style_str}{t["style_end"]}')

    parts.append(t["eos"])
    return "".join(parts)


def token_sequence_to_genome(sequence: str) -> Optional[DocumentGenome]:
    """
    Parse a flat token sequence back into a DocumentGenome.

    Returns None if parsing fails (malformed output from decoder).
    """
    t = GENOME_TEMPLATE_TOKENS

    try:
        # Extract content elements
        content_match = re.search(
            rf'{re.escape(t["content_start"])}(.*?){re.escape(t["content_end"])}',
            sequence, re.DOTALL
        )
        if not content_match:
            return None

        content_str = content_match.group(1)
        elements = []

        elem_pattern = rf'{re.escape(t["element_start"])}(.*?){re.escape(t["element_end"])}'
        for elem_match in re.finditer(elem_pattern, content_str, re.DOTALL):
            elem_str = elem_match.group(1)

            # Extract type
            type_match = re.search(
                rf'{re.escape(t["type_start"])}(.*?){re.escape(t["type_end"])}',
                elem_str
            )
            # Extract text
            text_match = re.search(
                rf'{re.escape(t["text_start"])}(.*?){re.escape(t["text_end"])}',
                elem_str, re.DOTALL
            )
            # Extract bbox
            bbox_match = re.search(
                rf'{re.escape(t["bbox_start"])}(.*?){re.escape(t["bbox_end"])}',
                elem_str
            )

            if not all([type_match, text_match, bbox_match]):
                continue

            region_type = TextRegionType(type_match.group(1).strip())
            text = text_match.group(1).strip()
            bbox_vals = [int(v) for v in bbox_match.group(1).strip().split()]

            elements.append(ContentElement(
                text=text,
                confidence=1.0,  # Confidence is assigned post-hoc
                bbox=BoundingBox.from_list(bbox_vals),
                type=region_type,
            ))

        # Extract layout
        layout_match = re.search(
            rf'{re.escape(t["layout_start"])}(.*?){re.escape(t["layout_end"])}',
            sequence
        )
        if not layout_match:
            return None

        layout_vals = layout_match.group(1).strip().split()
        layout = LayoutInfo(
            page_width=int(layout_vals[0]),
            page_height=int(layout_vals[1]),
            columns=int(layout_vals[2]),
            margins=[int(v) for v in layout_vals[3:7]],
        )

        # Extract style
        style_match = re.search(
            rf'{re.escape(t["style_start"])}(.*?){re.escape(t["style_end"])}',
            sequence
        )
        if style_match:
            style_vals = style_match.group(1).strip().split()
            style = StyleInfo(
                document_class=style_vals[0] if len(style_vals) > 0 else "general",
                primary_font_class=style_vals[1] if len(style_vals) > 1 else "unknown",
                estimated_era=style_vals[2] if len(style_vals) > 2 else "unknown",
                paper_type=style_vals[3] if len(style_vals) > 3 else "unknown",
            )
        else:
            style = StyleInfo()

        return DocumentGenome(
            content=elements,
            layout=layout,
            style=style,
        )

    except (ValueError, IndexError, KeyError) as e:
        # Parsing failed — return None (decoder produced malformed output)
        return None


# ============================================================================
# Genome Statistics
# ============================================================================

def genome_statistics(genome: DocumentGenome) -> dict:
    """Compute summary statistics for a Document Genome."""
    if not genome.content:
        return {
            "num_regions": 0,
            "total_chars": 0,
            "total_words": 0,
            "avg_confidence": 0.0,
            "type_distribution": {},
            "avg_region_area": 0.0,
            "page_coverage": 0.0,
        }

    total_chars = sum(len(e.text) for e in genome.content)
    total_words = sum(len(e.text.split()) for e in genome.content)
    type_dist = {}
    for e in genome.content:
        type_dist[e.type.value] = type_dist.get(e.type.value, 0) + 1

    total_area = sum(e.bbox.area for e in genome.content)
    page_area = genome.layout.page_width * genome.layout.page_height

    return {
        "num_regions": len(genome.content),
        "total_chars": total_chars,
        "total_words": total_words,
        "avg_confidence": genome.avg_confidence,
        "type_distribution": type_dist,
        "avg_region_area": total_area / len(genome.content),
        "page_coverage": total_area / page_area if page_area > 0 else 0.0,
    }
