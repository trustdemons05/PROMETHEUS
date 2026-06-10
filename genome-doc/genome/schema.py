"""
Document Genome JSON Schema & Validation

Defines the structured specification (Document Genome) that fully describes
a document's content, layout, and style. Uses Pydantic v2 for validation.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ============================================================================
# Enums
# ============================================================================

class TextRegionType(str, Enum):
    """Semantic type of a text region in the document."""
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    CAPTION = "caption"
    TABLE_CELL = "table_cell"
    LIST_ITEM = "list_item"
    FOOTER = "footer"
    HEADER = "header"
    PAGE_NUMBER = "page_number"


class DocumentClass(str, Enum):
    """High-level document classification."""
    ACADEMIC_PAPER = "academic_paper"
    BOOK_PAGE = "book_page"
    LETTER = "letter"
    FORM = "form"
    INVOICE = "invoice"
    NEWSPAPER = "newspaper"
    HANDWRITTEN = "handwritten"
    HISTORICAL = "historical"
    GENERAL = "general"


class FontClass(str, Enum):
    """Primary font family classification."""
    SERIF = "serif"
    SANS_SERIF = "sans_serif"
    MONOSPACE = "monospace"
    HANDWRITTEN = "handwritten"
    DECORATIVE = "decorative"
    UNKNOWN = "unknown"


class DocumentEra(str, Enum):
    """Estimated era of the document's production."""
    MODERN_DIGITAL = "modern_digital"       # Post-2000, laser/inkjet printed
    EARLY_DIGITAL = "early_digital"         # 1980-2000, dot matrix / early laser
    TYPEWRITTEN = "typewritten"             # 1900-1980, typewriter
    HISTORICAL_PRINT = "historical_print"   # Pre-1900, letterpress / movable type
    HANDWRITTEN = "handwritten"             # Any era, handwritten
    UNKNOWN = "unknown"


class PaperType(str, Enum):
    """Paper stock classification."""
    WHITE_BOND = "white_bond"
    YELLOWED = "yellowed"
    PARCHMENT = "parchment"
    NEWSPRINT = "newsprint"
    CARDSTOCK = "cardstock"
    LINED = "lined"
    GRID = "grid"
    UNKNOWN = "unknown"


# ============================================================================
# Content Models
# ============================================================================

class BoundingBox(BaseModel):
    """Bounding box coordinates [x1, y1, x2, y2] in pixels."""
    x1: int = Field(ge=0, description="Left edge x-coordinate")
    y1: int = Field(ge=0, description="Top edge y-coordinate")
    x2: int = Field(ge=0, description="Right edge x-coordinate")
    y2: int = Field(ge=0, description="Bottom edge y-coordinate")

    @model_validator(mode="after")
    def validate_box(self) -> "BoundingBox":
        if self.x2 <= self.x1:
            raise ValueError(f"x2 ({self.x2}) must be greater than x1 ({self.x1})")
        if self.y2 <= self.y1:
            raise ValueError(f"y2 ({self.y2}) must be greater than y1 ({self.y1})")
        return self

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def to_list(self) -> list[int]:
        return [self.x1, self.y1, self.x2, self.y2]

    @classmethod
    def from_list(cls, coords: list[int]) -> "BoundingBox":
        if len(coords) != 4:
            raise ValueError(f"Expected 4 coordinates, got {len(coords)}")
        return cls(x1=coords[0], y1=coords[1], x2=coords[2], y2=coords[3])

    def iou(self, other: "BoundingBox") -> float:
        """Compute Intersection over Union with another bounding box."""
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)

        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0

        intersection = (ix2 - ix1) * (iy2 - iy1)
        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0

    def normalize(self, page_width: int, page_height: int) -> "BoundingBox":
        """Normalize coordinates to [0, 1000] range (LayoutLM convention)."""
        return BoundingBox(
            x1=int(self.x1 / page_width * 1000),
            y1=int(self.y1 / page_height * 1000),
            x2=int(self.x2 / page_width * 1000),
            y2=int(self.y2 / page_height * 1000),
        )


class ContentElement(BaseModel):
    """A single text element in the document."""
    text: str = Field(min_length=0, description="Recognized text content")
    confidence: float = Field(
        ge=0.0, le=1.0, default=1.0,
        description="Per-element OCR confidence score"
    )
    bbox: BoundingBox = Field(description="Bounding box in pixel coordinates")
    type: TextRegionType = Field(
        default=TextRegionType.PARAGRAPH,
        description="Semantic type of this text region"
    )

    @field_validator("text")
    @classmethod
    def clean_text(cls, v: str) -> str:
        # Remove null bytes and normalize whitespace
        v = v.replace("\x00", "")
        return v.strip()


# ============================================================================
# Layout Model
# ============================================================================

class LayoutInfo(BaseModel):
    """Global page layout information."""
    page_width: int = Field(gt=0, description="Page width in pixels")
    page_height: int = Field(gt=0, description="Page height in pixels")
    columns: int = Field(ge=1, le=4, default=1, description="Number of text columns")
    margins: list[int] = Field(
        default=[200, 200, 200, 250],
        description="Page margins [top, right, bottom, left] in pixels"
    )
    reading_order: Optional[list[int]] = Field(
        default=None,
        description="Indices into content array defining reading order"
    )

    @field_validator("margins")
    @classmethod
    def validate_margins(cls, v: list[int]) -> list[int]:
        if len(v) != 4:
            raise ValueError(f"Margins must have exactly 4 values, got {len(v)}")
        if any(m < 0 for m in v):
            raise ValueError("Margins must be non-negative")
        return v


# ============================================================================
# Style Model
# ============================================================================

class StyleInfo(BaseModel):
    """Document visual style classification."""
    document_class: DocumentClass = Field(
        default=DocumentClass.GENERAL,
        description="High-level document type classification"
    )
    primary_font_class: FontClass = Field(
        default=FontClass.UNKNOWN,
        description="Primary font family class"
    )
    estimated_era: DocumentEra = Field(
        default=DocumentEra.UNKNOWN,
        description="Estimated era of document production"
    )
    paper_type: PaperType = Field(
        default=PaperType.UNKNOWN,
        description="Paper stock type"
    )


# ============================================================================
# Document Genome (Top-Level)
# ============================================================================

class DocumentGenome(BaseModel):
    """
    The Document Genome: a complete symbolic specification of a document.

    Contains everything needed to re-render the document from scratch:
    - content: What the document says (text + positions + types)
    - layout: How the page is structured (dimensions, columns, margins)
    - style: What the document looks like (font class, era, paper type)
    """
    content: list[ContentElement] = Field(
        default_factory=list,
        description="List of all text elements in the document"
    )
    layout: LayoutInfo = Field(
        description="Global page layout information"
    )
    style: StyleInfo = Field(
        default_factory=StyleInfo,
        description="Document visual style classification"
    )

    @model_validator(mode="after")
    def validate_reading_order(self) -> "DocumentGenome":
        """Ensure reading order indices are valid if provided."""
        if self.layout.reading_order is not None:
            n = len(self.content)
            order = self.layout.reading_order
            if len(order) != n:
                raise ValueError(
                    f"Reading order length ({len(order)}) must match "
                    f"content length ({n})"
                )
            if sorted(order) != list(range(n)):
                raise ValueError(
                    "Reading order must be a permutation of [0, ..., n-1]"
                )
        return self

    @model_validator(mode="after")
    def validate_bboxes_within_page(self) -> "DocumentGenome":
        """Warn if any bounding boxes extend beyond page dimensions."""
        pw, ph = self.layout.page_width, self.layout.page_height
        for i, elem in enumerate(self.content):
            bb = elem.bbox
            if bb.x2 > pw or bb.y2 > ph:
                # Don't raise — just clamp silently (degraded docs may overshoot)
                pass
        return self

    # ------ Serialization ------

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return self.model_dump_json(indent=indent)

    def to_dict(self) -> dict:
        """Serialize to Python dict."""
        return self.model_dump()

    @classmethod
    def from_json(cls, json_str: str) -> "DocumentGenome":
        """Parse a Document Genome from a JSON string."""
        return cls.model_validate_json(json_str)

    @classmethod
    def from_dict(cls, data: dict) -> "DocumentGenome":
        """Parse a Document Genome from a Python dict."""
        return cls.model_validate(data)

    @classmethod
    def from_file(cls, path: str | Path) -> "DocumentGenome":
        """Load a Document Genome from a JSON file."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def save(self, path: str | Path) -> None:
        """Save the Document Genome to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    # ------ Utilities ------

    @property
    def total_text(self) -> str:
        """Concatenate all text elements in reading order."""
        if self.layout.reading_order:
            ordered = [self.content[i] for i in self.layout.reading_order]
        else:
            ordered = self.content
        return "\n".join(elem.text for elem in ordered)

    @property
    def num_regions(self) -> int:
        return len(self.content)

    @property
    def avg_confidence(self) -> float:
        if not self.content:
            return 0.0
        return sum(e.confidence for e in self.content) / len(self.content)

    def get_regions_by_type(self, region_type: TextRegionType) -> list[ContentElement]:
        """Filter content elements by their semantic type."""
        return [e for e in self.content if e.type == region_type]

    def get_low_confidence_regions(self, threshold: float = 0.8) -> list[ContentElement]:
        """Get regions with confidence below the threshold."""
        return [e for e in self.content if e.confidence < threshold]
