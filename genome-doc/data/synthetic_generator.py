"""
Synthetic Document Generator

Generates clean document images paired with ground-truth Document Genome JSONs.
Documents are created with diverse fonts, layouts, and text content to provide
training data for the DGI and NRE modules.

Usage:
    python data/synthetic_generator.py --num-samples 30000 --output-dir data/synthetic
"""

from __future__ import annotations

import argparse
import json
import os
import random
import string
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genome.schema import (
    BoundingBox,
    ContentElement,
    DocumentClass,
    DocumentEra,
    DocumentGenome,
    FontClass,
    LayoutInfo,
    PaperType,
    StyleInfo,
    TextRegionType,
)


# ============================================================================
# Constants
# ============================================================================

# Default rendering resolution
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512

# Text content sources (fallback lorem ipsum when corpora unavailable)
LOREM_IPSUM = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
    "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
    "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris. "
    "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum. "
    "Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia."
)

SAMPLE_TITLES = [
    "Introduction to Machine Learning",
    "Chapter 1: The Beginning",
    "Results and Discussion",
    "Abstract",
    "Methodology",
    "Experimental Setup",
    "Conclusion",
    "Literature Review",
    "Data Analysis",
    "System Architecture",
    "Performance Evaluation",
    "Future Work",
    "Acknowledgments",
    "References",
    "Technical Report",
    "Annual Financial Statement",
    "Meeting Minutes",
    "Project Proposal",
    "Research Summary",
    "Clinical Trial Results",
]

SAMPLE_PARAGRAPHS = [
    "The rapid advancement of artificial intelligence has transformed numerous industries, "
    "enabling automated decision-making systems that were previously thought impossible.",
    "In this paper, we propose a novel approach to document restoration that leverages "
    "symbolic inference rather than direct pixel-level translation.",
    "Our experimental results demonstrate significant improvements over existing baselines "
    "across all evaluation metrics, particularly in terms of text accuracy.",
    "The dataset comprises 30,000 synthetic document images spanning multiple document types "
    "including academic papers, letters, invoices, and historical manuscripts.",
    "We evaluate our method on standard benchmarks including DIBCO and HDR28K, achieving "
    "state-of-the-art results on both character error rate and visual quality metrics.",
    "The proposed architecture consists of three tightly integrated modules that operate "
    "in a single forward pass without iterative refinement.",
    "Statistical analysis reveals that the treatment group showed a significant improvement "
    "compared to the control group with a confidence level of 95 percent.",
    "The board of directors approved the annual budget of twelve million dollars for the "
    "upcoming fiscal year beginning January first.",
    "Patient records indicate that the administered dosage of 500mg was effective in reducing "
    "symptoms within the first 48 hours of treatment.",
    "The structural integrity of the bridge was assessed using finite element analysis, "
    "confirming compliance with current safety standards.",
    "Revenue for the third quarter exceeded projections by 15 percent, driven primarily "
    "by strong performance in the digital services division.",
    "The algorithm processes each input frame in approximately 33 milliseconds, enabling "
    "real-time performance on standard consumer hardware.",
]

SAMPLE_CAPTIONS = [
    "Figure 1: System architecture overview.",
    "Table 2: Comparison of baseline methods.",
    "Fig. 3: Qualitative results on test set.",
    "Table 1: Dataset statistics.",
    "Figure 4: Training loss convergence.",
    "Fig. 2: Distribution of document types.",
]

SAMPLE_LIST_ITEMS = [
    "First, preprocess the raw input data.",
    "Apply normalization to all feature channels.",
    "Train the model for 100 epochs with early stopping.",
    "Evaluate on the held-out test set.",
    "Report mean and standard deviation across 3 runs.",
    "Compare against published baselines.",
    "Submit source code for reproducibility.",
]

SAMPLE_FOOTERS = [
    "Page 1 of 12",
    "Confidential - Do Not Distribute",
    "Draft Version 2.3",
    "Copyright 2024 All Rights Reserved",
    "Internal Use Only",
    "Prepared by the Department of Research",
]

SAMPLE_HEADERS = [
    "Technical Report TR-2024-001",
    "University of Science and Technology",
    "Department of Computer Science",
    "Quarterly Performance Review",
    "Board Meeting - March 2024",
]


# ============================================================================
# Layout Templates
# ============================================================================

class LayoutTemplate:
    """Defines a document layout template with region placement rules."""

    def __init__(
        self,
        name: str,
        doc_class: DocumentClass,
        columns: int = 1,
        has_header: bool = False,
        has_footer: bool = False,
        num_headings: tuple[int, int] = (1, 3),
        num_paragraphs: tuple[int, int] = (2, 6),
        num_captions: tuple[int, int] = (0, 2),
        num_list_items: tuple[int, int] = (0, 4),
        margins: tuple[int, int, int, int] = (40, 30, 40, 30),
    ):
        self.name = name
        self.doc_class = doc_class
        self.columns = columns
        self.has_header = has_header
        self.has_footer = has_footer
        self.num_headings = num_headings
        self.num_paragraphs = num_paragraphs
        self.num_captions = num_captions
        self.num_list_items = num_list_items
        self.margins = margins  # top, right, bottom, left


# Predefined layout templates
LAYOUT_TEMPLATES = [
    LayoutTemplate("academic_single", DocumentClass.ACADEMIC_PAPER,
                   columns=1, has_header=True, has_footer=True,
                   num_headings=(1, 3), num_paragraphs=(3, 6)),
    LayoutTemplate("academic_double", DocumentClass.ACADEMIC_PAPER,
                   columns=2, has_header=True, has_footer=True,
                   num_headings=(1, 2), num_paragraphs=(2, 4)),
    LayoutTemplate("book_page", DocumentClass.BOOK_PAGE,
                   columns=1, has_header=True, has_footer=True,
                   num_headings=(0, 1), num_paragraphs=(3, 7)),
    LayoutTemplate("letter", DocumentClass.LETTER,
                   columns=1, has_header=False, has_footer=False,
                   num_headings=(1, 1), num_paragraphs=(2, 5)),
    LayoutTemplate("form", DocumentClass.FORM,
                   columns=1, has_header=True, has_footer=False,
                   num_headings=(1, 2), num_paragraphs=(1, 3),
                   num_list_items=(3, 8)),
    LayoutTemplate("invoice", DocumentClass.INVOICE,
                   columns=1, has_header=True, has_footer=True,
                   num_headings=(1, 2), num_paragraphs=(1, 3)),
    LayoutTemplate("general", DocumentClass.GENERAL,
                   columns=1, has_header=False, has_footer=False,
                   num_headings=(1, 3), num_paragraphs=(2, 5)),
]


# ============================================================================
# Font Management
# ============================================================================

def get_available_fonts() -> list[str]:
    """
    Find available TrueType fonts on the system.
    Returns a list of font file paths.
    """
    font_dirs = []

    # Windows
    if os.name == "nt":
        windir = os.environ.get("WINDIR", r"C:\Windows")
        font_dirs.append(os.path.join(windir, "Fonts"))

    # Linux
    font_dirs.extend([
        "/usr/share/fonts",
        "/usr/local/share/fonts",
        os.path.expanduser("~/.fonts"),
        os.path.expanduser("~/.local/share/fonts"),
    ])

    # macOS
    font_dirs.extend([
        "/Library/Fonts",
        "/System/Library/Fonts",
        os.path.expanduser("~/Library/Fonts"),
    ])

    fonts = []
    for font_dir in font_dirs:
        if os.path.isdir(font_dir):
            for root, _, files in os.walk(font_dir):
                for f in files:
                    if f.lower().endswith((".ttf", ".otf")):
                        fonts.append(os.path.join(root, f))

    return fonts if fonts else [None]  # None = use Pillow default


def classify_font(font_path: Optional[str]) -> FontClass:
    """Heuristic font classification based on filename."""
    if font_path is None:
        return FontClass.UNKNOWN

    name = os.path.basename(font_path).lower()

    if any(k in name for k in ["mono", "courier", "consola", "fira_code", "source_code"]):
        return FontClass.MONOSPACE
    elif any(k in name for k in ["comic", "script", "handwrit", "caveat", "dancing"]):
        return FontClass.HANDWRITTEN
    elif any(k in name for k in ["impact", "display", "decorat", "lobster"]):
        return FontClass.DECORATIVE
    elif any(k in name for k in ["arial", "helveti", "roboto", "open_sans", "lato",
                                   "noto_sans", "ubuntu", "poppins", "inter", "calibri"]):
        return FontClass.SANS_SERIF
    elif any(k in name for k in ["times", "georgia", "garamond", "palatino", "serif",
                                   "bookman", "cambria", "noto_serif"]):
        return FontClass.SERIF
    else:
        return random.choice([FontClass.SERIF, FontClass.SANS_SERIF])


def load_font(font_path: Optional[str], size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a font with fallback to Pillow default."""
    if font_path:
        try:
            return ImageFont.truetype(font_path, size=size)
        except (IOError, OSError):
            pass
    return ImageFont.load_default()


# ============================================================================
# Paper Background Generation
# ============================================================================

def generate_paper_background(
    width: int,
    height: int,
    paper_type: PaperType,
) -> Image.Image:
    """Generate a realistic paper background texture."""
    if paper_type == PaperType.WHITE_BOND:
        # Clean white with very subtle noise
        base = np.full((height, width, 3), 252, dtype=np.uint8)
        noise = np.random.normal(0, 2, (height, width, 3)).astype(np.int16)
        img_arr = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    elif paper_type == PaperType.YELLOWED:
        # Aged yellowish paper
        r = np.random.randint(235, 245)
        g = np.random.randint(220, 235)
        b = np.random.randint(190, 210)
        base = np.full((height, width, 3), [r, g, b], dtype=np.uint8)
        noise = np.random.normal(0, 4, (height, width, 3)).astype(np.int16)
        img_arr = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    elif paper_type == PaperType.PARCHMENT:
        # Old parchment look
        r = np.random.randint(220, 235)
        g = np.random.randint(200, 215)
        b = np.random.randint(170, 190)
        base = np.full((height, width, 3), [r, g, b], dtype=np.uint8)
        noise = np.random.normal(0, 6, (height, width, 3)).astype(np.int16)
        img_arr = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    elif paper_type == PaperType.NEWSPRINT:
        # Grayish newsprint
        gray = np.random.randint(225, 240)
        base = np.full((height, width, 3), [gray, gray, gray - 5], dtype=np.uint8)
        noise = np.random.normal(0, 5, (height, width, 3)).astype(np.int16)
        img_arr = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    elif paper_type == PaperType.LINED:
        base = np.full((height, width, 3), 250, dtype=np.uint8)
        line_spacing = random.randint(20, 35)
        for y in range(line_spacing, height, line_spacing):
            base[y:y+1, :] = [180, 200, 230]
        noise = np.random.normal(0, 2, (height, width, 3)).astype(np.int16)
        img_arr = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    elif paper_type == PaperType.GRID:
        base = np.full((height, width, 3), 250, dtype=np.uint8)
        grid_spacing = random.randint(15, 25)
        for y in range(0, height, grid_spacing):
            base[y:y+1, :] = [210, 220, 235]
        for x in range(0, width, grid_spacing):
            base[:, x:x+1] = [210, 220, 235]
        noise = np.random.normal(0, 2, (height, width, 3)).astype(np.int16)
        img_arr = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    else:
        # Default white
        base = np.full((height, width, 3), 250, dtype=np.uint8)
        noise = np.random.normal(0, 2, (height, width, 3)).astype(np.int16)
        img_arr = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(img_arr, "RGB")


# ============================================================================
# Text Content Generation
# ============================================================================

def generate_text_content(region_type: TextRegionType) -> str:
    """Generate random text content appropriate for the region type."""
    if region_type == TextRegionType.HEADING:
        return random.choice(SAMPLE_TITLES)
    elif region_type == TextRegionType.PARAGRAPH:
        para = random.choice(SAMPLE_PARAGRAPHS)
        # Sometimes combine two paragraphs
        if random.random() < 0.3:
            para += " " + random.choice(SAMPLE_PARAGRAPHS)
        return para
    elif region_type == TextRegionType.CAPTION:
        return random.choice(SAMPLE_CAPTIONS)
    elif region_type == TextRegionType.LIST_ITEM:
        return random.choice(SAMPLE_LIST_ITEMS)
    elif region_type == TextRegionType.FOOTER:
        return random.choice(SAMPLE_FOOTERS)
    elif region_type == TextRegionType.HEADER:
        return random.choice(SAMPLE_HEADERS)
    elif region_type == TextRegionType.PAGE_NUMBER:
        return str(random.randint(1, 200))
    elif region_type == TextRegionType.TABLE_CELL:
        # Simple table-like content
        if random.random() < 0.5:
            return f"{random.uniform(0, 100):.2f}"
        else:
            words = LOREM_IPSUM.split()
            return " ".join(random.choices(words, k=random.randint(1, 4)))
    else:
        return random.choice(SAMPLE_PARAGRAPHS)


# ============================================================================
# Core Document Generator
# ============================================================================

class SyntheticDocumentGenerator:
    """
    Generates synthetic document images with corresponding Document Genome JSONs.

    Each generated sample consists of:
    - A clean document image (PIL Image)
    - The ground-truth Document Genome (DocumentGenome)
    """

    def __init__(
        self,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        font_paths: Optional[list[str]] = None,
        seed: Optional[int] = None,
    ):
        self.width = width
        self.height = height

        if font_paths is None:
            self.font_paths = get_available_fonts()
        else:
            self.font_paths = font_paths

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def generate(self) -> tuple[Image.Image, DocumentGenome]:
        """
        Generate a single synthetic document.

        Returns:
            (clean_image, genome): A clean document image and its genome.
        """
        # 1. Choose layout template
        template = random.choice(LAYOUT_TEMPLATES)

        # 2. Choose font
        font_path = random.choice(self.font_paths)
        font_class = classify_font(font_path)

        # 3. Choose paper and style
        paper_type = random.choice(list(PaperType))
        era = random.choice([DocumentEra.MODERN_DIGITAL, DocumentEra.EARLY_DIGITAL,
                             DocumentEra.TYPEWRITTEN])

        # 4. Choose ink color
        ink_r = random.randint(0, 40)
        ink_g = random.randint(0, 40)
        ink_b = random.randint(0, 40)
        ink_color = (ink_r, ink_g, ink_b)

        # 5. Generate paper background
        img = generate_paper_background(self.width, self.height, paper_type)
        draw = ImageDraw.Draw(img)

        # 6. Plan regions
        margin_top, margin_right, margin_bottom, margin_left = template.margins
        content_width = self.width - margin_left - margin_right
        content_height = self.height - margin_top - margin_bottom

        content_elements = []
        current_y = margin_top

        # -- Header --
        if template.has_header and current_y < self.height - margin_bottom - 20:
            header_text = generate_text_content(TextRegionType.HEADER)
            font_size = random.randint(8, 11)
            font = load_font(font_path, font_size)

            text_height = font_size + 4
            bbox = BoundingBox(
                x1=margin_left, y1=current_y,
                x2=margin_left + content_width, y2=current_y + text_height
            )
            draw.text((bbox.x1 + 2, bbox.y1), header_text, fill=ink_color, font=font)

            content_elements.append(ContentElement(
                text=header_text, confidence=1.0, bbox=bbox,
                type=TextRegionType.HEADER,
            ))
            current_y += text_height + random.randint(8, 15)

        # -- Headings --
        num_headings = random.randint(*template.num_headings)
        heading_indices = []
        for _ in range(num_headings):
            if current_y >= self.height - margin_bottom - 20:
                break

            heading_text = generate_text_content(TextRegionType.HEADING)
            font_size = random.randint(14, 22)
            font = load_font(font_path, font_size)

            # Measure text
            try:
                text_bbox = font.getbbox(heading_text)
                text_w = text_bbox[2] - text_bbox[0]
                text_h = text_bbox[3] - text_bbox[1]
            except Exception:
                text_w = font_size * len(heading_text) * 0.6
                text_h = font_size

            text_h = max(text_h, font_size)
            bbox = BoundingBox(
                x1=margin_left, y1=current_y,
                x2=min(margin_left + int(text_w) + 4, self.width - margin_right),
                y2=current_y + int(text_h) + 4
            )
            draw.text((bbox.x1 + 2, bbox.y1), heading_text, fill=ink_color, font=font)

            heading_idx = len(content_elements)
            heading_indices.append(heading_idx)
            content_elements.append(ContentElement(
                text=heading_text, confidence=1.0, bbox=bbox,
                type=TextRegionType.HEADING,
            ))
            current_y += int(text_h) + random.randint(10, 20)

            # Add paragraphs after heading
            num_paras = random.randint(*template.num_paragraphs)
            for _ in range(num_paras):
                if current_y >= self.height - margin_bottom - 30:
                    break

                para_text = generate_text_content(TextRegionType.PARAGRAPH)
                font_size = random.randint(9, 13)
                font = load_font(font_path, font_size)

                # Calculate wrapped text height
                line_height = font_size + 3
                chars_per_line = max(1, content_width // max(1, int(font_size * 0.55)))
                num_lines = max(1, len(para_text) // chars_per_line + 1)
                block_height = num_lines * line_height

                # Clamp to available space
                max_height = self.height - margin_bottom - current_y
                block_height = min(block_height, max_height)

                bbox = BoundingBox(
                    x1=margin_left, y1=current_y,
                    x2=margin_left + content_width,
                    y2=current_y + int(block_height)
                )

                # Draw wrapped text
                self._draw_wrapped_text(
                    draw, para_text, font, bbox, ink_color, line_height
                )

                content_elements.append(ContentElement(
                    text=para_text, confidence=1.0, bbox=bbox,
                    type=TextRegionType.PARAGRAPH,
                ))
                current_y += int(block_height) + random.randint(6, 14)

        # -- List items --
        num_list = random.randint(*template.num_list_items)
        for i in range(num_list):
            if current_y >= self.height - margin_bottom - 20:
                break

            item_text = generate_text_content(TextRegionType.LIST_ITEM)
            bullet_text = f"  {i+1}. {item_text}" if random.random() < 0.5 else f"  • {item_text}"
            font_size = random.randint(9, 12)
            font = load_font(font_path, font_size)

            text_h = font_size + 4
            bbox = BoundingBox(
                x1=margin_left + 10, y1=current_y,
                x2=margin_left + content_width - 10,
                y2=current_y + text_h
            )
            draw.text((bbox.x1, bbox.y1), bullet_text, fill=ink_color, font=font)

            content_elements.append(ContentElement(
                text=bullet_text.strip(), confidence=1.0, bbox=bbox,
                type=TextRegionType.LIST_ITEM,
            ))
            current_y += text_h + random.randint(2, 6)

        # -- Captions --
        num_captions = random.randint(*template.num_captions)
        for _ in range(num_captions):
            if current_y >= self.height - margin_bottom - 20:
                break

            caption_text = generate_text_content(TextRegionType.CAPTION)
            font_size = random.randint(8, 10)
            font = load_font(font_path, font_size)

            text_h = font_size + 4
            bbox = BoundingBox(
                x1=margin_left + 20, y1=current_y,
                x2=margin_left + content_width - 20,
                y2=current_y + text_h
            )
            draw.text((bbox.x1, bbox.y1), caption_text, fill=ink_color, font=font)

            content_elements.append(ContentElement(
                text=caption_text, confidence=1.0, bbox=bbox,
                type=TextRegionType.CAPTION,
            ))
            current_y += text_h + random.randint(8, 15)

        # -- Footer --
        if template.has_footer and len(content_elements) > 0:
            footer_text = generate_text_content(TextRegionType.FOOTER)
            font_size = random.randint(7, 10)
            font = load_font(font_path, font_size)

            footer_y = self.height - margin_bottom + 5
            text_h = font_size + 4
            bbox = BoundingBox(
                x1=margin_left, y1=footer_y,
                x2=margin_left + content_width, y2=footer_y + text_h
            )
            if bbox.y2 <= self.height:
                draw.text((bbox.x1 + 2, bbox.y1), footer_text, fill=ink_color, font=font)
                content_elements.append(ContentElement(
                    text=footer_text, confidence=1.0, bbox=bbox,
                    type=TextRegionType.FOOTER,
                ))

        # 7. Build the Document Genome
        reading_order = list(range(len(content_elements)))

        style = StyleInfo(
            document_class=template.doc_class,
            primary_font_class=font_class,
            estimated_era=era,
            paper_type=paper_type,
        )

        layout = LayoutInfo(
            page_width=self.width,
            page_height=self.height,
            columns=template.columns,
            margins=list(template.margins),
            reading_order=reading_order,
        )

        genome = DocumentGenome(
            content=content_elements,
            layout=layout,
            style=style,
        )

        return img, genome

    def _draw_wrapped_text(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
        bbox: BoundingBox,
        color: tuple[int, int, int],
        line_height: int,
    ) -> None:
        """Draw text wrapped within a bounding box."""
        words = text.split()
        if not words:
            return

        x, y = bbox.x1 + 2, bbox.y1
        max_x = bbox.x2 - 2
        max_y = bbox.y2

        current_line = words[0]
        for word in words[1:]:
            test_line = current_line + " " + word
            try:
                tw = font.getbbox(test_line)[2] - font.getbbox(test_line)[0]
            except Exception:
                tw = len(test_line) * 6

            if x + tw <= max_x:
                current_line = test_line
            else:
                if y + line_height > max_y:
                    break
                draw.text((x, y), current_line, fill=color, font=font)
                y += line_height
                current_line = word

        if y + line_height <= max_y:
            draw.text((x, y), current_line, fill=color, font=font)

    def generate_batch(
        self,
        num_samples: int,
        output_dir: str | Path,
        start_idx: int = 0,
        progress_interval: int = 100,
    ) -> None:
        """
        Generate a batch of synthetic documents and save to disk.

        Directory structure:
            output_dir/
                images/
                    clean/      # Clean document images
                    degraded/   # Degraded versions (generated separately)
                genomes/        # Ground-truth Genome JSON files
        """
        output_dir = Path(output_dir)
        clean_dir = output_dir / "images" / "clean"
        genome_dir = output_dir / "genomes"

        clean_dir.mkdir(parents=True, exist_ok=True)
        genome_dir.mkdir(parents=True, exist_ok=True)

        for i in range(num_samples):
            idx = start_idx + i
            img, genome = self.generate()

            # Save clean image
            img_path = clean_dir / f"doc_{idx:06d}.png"
            img.save(str(img_path))

            # Save genome JSON
            genome_path = genome_dir / f"doc_{idx:06d}.json"
            genome.save(str(genome_path))

            if (i + 1) % progress_interval == 0:
                print(f"  Generated {i + 1}/{num_samples} documents...")

        print(f"Done! Generated {num_samples} documents in {output_dir}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic document training data"
    )
    parser.add_argument(
        "--num-samples", type=int, default=100,
        help="Number of document pairs to generate (default: 100)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/synthetic/train",
        help="Output directory for generated data"
    )
    parser.add_argument(
        "--width", type=int, default=DEFAULT_WIDTH,
        help=f"Image width in pixels (default: {DEFAULT_WIDTH})"
    )
    parser.add_argument(
        "--height", type=int, default=DEFAULT_HEIGHT,
        help=f"Image height in pixels (default: {DEFAULT_HEIGHT})"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    args = parser.parse_args()

    print(f"Generating {args.num_samples} synthetic documents...")
    print(f"  Resolution: {args.width}x{args.height}")
    print(f"  Output: {args.output_dir}")
    print(f"  Seed: {args.seed}")

    generator = SyntheticDocumentGenerator(
        width=args.width,
        height=args.height,
        seed=args.seed,
    )

    generator.generate_batch(
        num_samples=args.num_samples,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
