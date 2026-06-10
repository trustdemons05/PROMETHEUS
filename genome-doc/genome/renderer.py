"""
Skeleton Renderer

Renders a "skeleton" image from a Document Genome JSON — a flat, visually simple
image with text placed at the correct bounding box positions using a standard font.
This skeleton image serves as the spatial control map for the NRE ControlNet.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from genome.schema import (
    BoundingBox,
    ContentElement,
    DocumentGenome,
    TextRegionType,
)


# ============================================================================
# Default Configuration
# ============================================================================

# Default font — DejaVu Sans is available on most Linux systems
# Falls back to Pillow's built-in font if not found
_DEFAULT_FONT_PATH = None  # Set to a .ttf path if you have a specific font
_FALLBACK_FONT_SIZE = 16

# Color scheme for the skeleton (can be customized)
SKELETON_COLORS = {
    "background": (255, 255, 255),     # White
    "text": (0, 0, 0),                 # Black
    "bbox_outline": (200, 200, 200),   # Light gray (optional debug overlay)
}

# Type-specific colors for debug visualization
TYPE_COLORS = {
    TextRegionType.HEADING: (0, 0, 180),       # Blue
    TextRegionType.PARAGRAPH: (0, 0, 0),        # Black
    TextRegionType.CAPTION: (100, 100, 100),    # Gray
    TextRegionType.TABLE_CELL: (0, 120, 0),     # Green
    TextRegionType.LIST_ITEM: (80, 80, 80),     # Dark gray
    TextRegionType.FOOTER: (150, 150, 150),     # Light gray
    TextRegionType.HEADER: (150, 150, 150),     # Light gray
    TextRegionType.PAGE_NUMBER: (180, 180, 180),# Very light gray
}


# ============================================================================
# Font Management
# ============================================================================

def _get_font(size: int, font_path: Optional[str] = None) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """
    Load a TrueType font at the given size, with fallback to Pillow's default.
    """
    if font_path:
        try:
            return ImageFont.truetype(font_path, size=size)
        except (IOError, OSError):
            pass

    # Try common system font paths
    common_fonts = [
        "DejaVuSans.ttf",
        "arial.ttf",
        "Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for fpath in common_fonts:
        try:
            return ImageFont.truetype(fpath, size=size)
        except (IOError, OSError):
            continue

    # Ultimate fallback
    return ImageFont.load_default()


def _estimate_font_size(bbox: BoundingBox, text: str, font_path: Optional[str] = None) -> int:
    """
    Estimate the best font size to fit text within a bounding box.
    Uses binary search to find the largest font size that fits.
    """
    if not text.strip():
        return _FALLBACK_FONT_SIZE

    bbox_height = bbox.height
    bbox_width = bbox.width

    # Start with height-based estimate
    estimated_size = max(8, int(bbox_height * 0.75))

    # Binary search for best fit
    low, high = 6, min(estimated_size + 10, 120)
    best_size = low

    for _ in range(10):  # Max iterations
        mid = (low + high) // 2
        font = _get_font(mid, font_path)

        # Estimate text width
        try:
            text_bbox = font.getbbox(text)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
        except Exception:
            text_width = mid * len(text) * 0.6
            text_height = mid

        if text_width <= bbox_width and text_height <= bbox_height:
            best_size = mid
            low = mid + 1
        else:
            high = mid - 1

        if low > high:
            break

    return max(6, best_size)


# ============================================================================
# Core Skeleton Rendering
# ============================================================================

def render_skeleton(
    genome: DocumentGenome,
    target_size: Optional[tuple[int, int]] = None,
    font_path: Optional[str] = None,
    draw_bboxes: bool = False,
    use_type_colors: bool = False,
    background_color: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """
    Render a skeleton image from a Document Genome.

    The skeleton image has text drawn at the correct bounding box positions
    using a standard font. This is used as the spatial control map for ControlNet.

    Args:
        genome: The Document Genome to render.
        target_size: Optional (width, height) to resize the output. If None,
                     uses the page dimensions from the genome.
        font_path: Path to a .ttf font file. Uses system default if None.
        draw_bboxes: If True, draw bounding box outlines (for debugging).
        use_type_colors: If True, color text by region type (for debugging).
        background_color: RGB background color.

    Returns:
        PIL.Image.Image: The rendered skeleton image (RGB).
    """
    page_w = genome.layout.page_width
    page_h = genome.layout.page_height

    # Create blank canvas
    img = Image.new("RGB", (page_w, page_h), background_color)
    draw = ImageDraw.Draw(img)

    # Render each content element
    for elem in genome.content:
        _render_element(
            draw=draw,
            element=elem,
            font_path=font_path,
            draw_bbox=draw_bboxes,
            use_type_color=use_type_colors,
        )

    # Resize to target if specified
    if target_size is not None and target_size != (page_w, page_h):
        img = img.resize(target_size, Image.Resampling.LANCZOS)

    return img


def _render_element(
    draw: ImageDraw.ImageDraw,
    element: ContentElement,
    font_path: Optional[str] = None,
    draw_bbox: bool = False,
    use_type_color: bool = False,
) -> None:
    """Render a single content element onto the image."""
    bb = element.bbox
    text = element.text.strip()

    if not text:
        return

    # Determine text color
    if use_type_color:
        color = TYPE_COLORS.get(element.type, (0, 0, 0))
    else:
        color = SKELETON_COLORS["text"]

    # Draw bounding box outline (debug mode)
    if draw_bbox:
        draw.rectangle(
            [bb.x1, bb.y1, bb.x2, bb.y2],
            outline=SKELETON_COLORS["bbox_outline"],
            width=1,
        )

    # Estimate font size to fit bounding box
    font_size = _estimate_font_size(bb, text, font_path)
    font = _get_font(font_size, font_path)

    # Handle text wrapping for long text
    wrapped_text = _wrap_text(text, font, bb.width)

    # Draw text within bounding box
    draw.text(
        (bb.x1 + 2, bb.y1 + 1),  # Small padding from top-left
        wrapped_text,
        fill=color,
        font=font,
    )


def _wrap_text(text: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont, max_width: int) -> str:
    """
    Wrap text to fit within a maximum pixel width.
    """
    words = text.split()
    if not words:
        return ""

    lines = []
    current_line = words[0]

    for word in words[1:]:
        test_line = current_line + " " + word
        try:
            text_bbox = font.getbbox(test_line)
            line_width = text_bbox[2] - text_bbox[0]
        except Exception:
            line_width = len(test_line) * 8  # Rough estimate

        if line_width <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word

    lines.append(current_line)
    return "\n".join(lines)


# ============================================================================
# Batch Rendering
# ============================================================================

def render_skeleton_batch(
    genomes: list[DocumentGenome],
    target_size: tuple[int, int] = (512, 512),
    font_path: Optional[str] = None,
) -> list[Image.Image]:
    """Render skeletons for a batch of genomes."""
    return [
        render_skeleton(g, target_size=target_size, font_path=font_path)
        for g in genomes
    ]


def render_skeleton_to_tensor(
    genome: DocumentGenome,
    target_size: tuple[int, int] = (512, 512),
    font_path: Optional[str] = None,
) -> np.ndarray:
    """
    Render skeleton and return as a float32 numpy array in [0, 1] range.
    Shape: (3, H, W) — channels first for PyTorch.
    """
    img = render_skeleton(genome, target_size=target_size, font_path=font_path)
    arr = np.array(img, dtype=np.float32) / 255.0
    # HWC -> CHW
    arr = arr.transpose(2, 0, 1)
    return arr


# ============================================================================
# Debug Visualization
# ============================================================================

def render_debug_skeleton(
    genome: DocumentGenome,
    target_size: Optional[tuple[int, int]] = None,
    font_path: Optional[str] = None,
    save_path: Optional[str | Path] = None,
) -> Image.Image:
    """
    Render a debug skeleton with bounding boxes and type-colored text.
    Useful for visually inspecting genome extraction quality.
    """
    img = render_skeleton(
        genome,
        target_size=target_size,
        font_path=font_path,
        draw_bboxes=True,
        use_type_colors=True,
        background_color=(245, 245, 245),  # Slightly off-white
    )

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(save_path))

    return img
