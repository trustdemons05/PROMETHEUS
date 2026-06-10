"""
DGI — Document Genome Inferrer (Donut Backbone)

Fine-tunes the Donut (Document Understanding Transformer) model to extract
structured Document Genome specifications from degraded document images.

Uses template-guided decoding: the decoder fills predefined slot tokens
(<text>, <bbox>, <type>) in a fixed template structure, ensuring valid
output by construction.

Key components:
- Donut visual encoder (Swin Transformer)
- Template-guided text decoder with special genome tokens
- Parallel Layout Head (MLP for bbox regression)
- LoRA adapters on decoder attention layers
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
    VisionEncoderDecoderConfig,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from genome.utils import GENOME_TEMPLATE_TOKENS


# ============================================================================
# Special Tokens for Genome Template
# ============================================================================

SPECIAL_TOKENS = list(GENOME_TEMPLATE_TOKENS.values())


# ============================================================================
# Layout Head — Parallel BBox Regression
# ============================================================================

class LayoutHead(nn.Module):
    """
    Parallel MLP head that regresses bounding box coordinates from
    decoder hidden states. Operates on the hidden states at <elem> positions
    to predict [x1, y1, x2, y2] for each detected text region.
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        layout_hidden_dim: int = 256,
        num_layers: int = 2,
        max_regions: int = 64,
    ):
        super().__init__()
        self.max_regions = max_regions

        layers = []
        in_dim = hidden_dim
        for i in range(num_layers):
            out_dim = layout_hidden_dim if i < num_layers - 1 else layout_hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(0.1))
            in_dim = out_dim

        # Final projection to 4 bbox coordinates (normalized [0, 1])
        layers.append(nn.Linear(layout_hidden_dim, 4))
        layers.append(nn.Sigmoid())  # Bbox coords in [0, 1]

        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        hidden_states: torch.Tensor,
        element_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, seq_len, hidden_dim) decoder hidden states
            element_positions: (B, max_regions) indices of <elem> tokens.
                               If None, uses first max_regions positions.

        Returns:
            bboxes: (B, max_regions, 4) predicted bounding boxes in [0, 1]
        """
        B = hidden_states.shape[0]

        if element_positions is not None:
            # Gather hidden states at element positions
            gathered = []
            for b in range(B):
                positions = element_positions[b]
                valid_pos = positions[positions >= 0]  # -1 = padding

                if len(valid_pos) > 0:
                    elem_hidden = hidden_states[b, valid_pos]  # (N, hidden_dim)
                    # Pad to max_regions
                    if elem_hidden.shape[0] < self.max_regions:
                        pad = torch.zeros(
                            self.max_regions - elem_hidden.shape[0],
                            elem_hidden.shape[1],
                            device=hidden_states.device,
                        )
                        elem_hidden = torch.cat([elem_hidden, pad], dim=0)
                    else:
                        elem_hidden = elem_hidden[:self.max_regions]
                else:
                    elem_hidden = torch.zeros(
                        self.max_regions, hidden_states.shape[2],
                        device=hidden_states.device,
                    )

                gathered.append(elem_hidden)

            gathered = torch.stack(gathered)  # (B, max_regions, hidden_dim)
        else:
            # Use first max_regions positions
            gathered = hidden_states[:, :self.max_regions, :]
            if gathered.shape[1] < self.max_regions:
                pad = torch.zeros(
                    B, self.max_regions - gathered.shape[1], hidden_states.shape[2],
                    device=hidden_states.device,
                )
                gathered = torch.cat([gathered, pad], dim=1)

        bboxes = self.mlp(gathered)  # (B, max_regions, 4)
        return bboxes


# ============================================================================
# Donut Genome Model
# ============================================================================

class DonutGenomeModel(nn.Module):
    """
    Fine-tuned Donut model for Document Genome extraction.

    Architecture:
        - Visual Encoder: Swin Transformer (from pretrained Donut)
        - Text Decoder: BART decoder with template-guided generation
        - Layout Head: Parallel MLP for bbox regression
        - LoRA: Applied to decoder attention layers

    The decoder generates a flat token sequence in the genome template format:
        <genome><content><elem><type>heading</type><text>Chapter 1</text>
        <bbox>120 50 500 90</bbox></elem>...</content>
        <layout>...</layout><style>...</style></genome>
    """

    def __init__(
        self,
        pretrained_model: str = "naver-clova-ix/donut-base",
        max_length: int = 2048,
        layout_hidden_dim: int = 256,
        layout_num_layers: int = 2,
        max_regions: int = 64,
    ):
        super().__init__()
        self.max_length = max_length
        self.max_regions = max_regions

        # Load pretrained Donut
        self.processor = DonutProcessor.from_pretrained(pretrained_model)
        self.model = VisionEncoderDecoderModel.from_pretrained(pretrained_model)

        # Add special genome tokens to tokenizer
        self._add_special_tokens()

        # Resize token embeddings to account for new tokens
        vocab_size = len(self.processor.tokenizer)
        self.model.decoder.resize_token_embeddings(vocab_size)
        self.model.decoder.config.vocab_size = vocab_size
        self.model.config.vocab_size = vocab_size

        # Layout head for parallel bbox regression
        decoder_hidden_dim = self.model.decoder.config.d_model
        self.layout_head = LayoutHead(
            hidden_dim=decoder_hidden_dim,
            layout_hidden_dim=layout_hidden_dim,
            num_layers=layout_num_layers,
            max_regions=max_regions,
        )

        # Store special token IDs
        self.special_token_ids = {
            name: self.processor.tokenizer.convert_tokens_to_ids(token)
            for name, token in GENOME_TEMPLATE_TOKENS.items()
        }

    def _add_special_tokens(self) -> None:
        """Add genome template tokens to the tokenizer."""
        new_tokens = []
        for token in SPECIAL_TOKENS:
            if token not in self.processor.tokenizer.get_vocab():
                new_tokens.append(token)

        if new_tokens:
            self.processor.tokenizer.add_special_tokens({
                "additional_special_tokens": new_tokens
            })

    def get_tokenizer(self):
        """Return the tokenizer for external use."""
        return self.processor.tokenizer

    def freeze_encoder(self) -> None:
        """Freeze the visual encoder (Swin Transformer)."""
        for param in self.model.encoder.parameters():
            param.requires_grad = False

    def setup_lora(self, rank: int = 16, alpha: int = 32, dropout: float = 0.05) -> None:
        """
        Apply LoRA adapters to decoder attention layers.

        Requires the `peft` library.
        """
        try:
            from peft import LoraConfig, get_peft_model, TaskType

            lora_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                r=rank,
                lora_alpha=alpha,
                lora_dropout=dropout,
                target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
                modules_to_save=["layout_head"],
            )

            # Apply LoRA to the decoder only
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

        except ImportError:
            print("WARNING: peft not installed. Training without LoRA (full fine-tuning).")
            print("Install with: pip install peft")

    def preprocess_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        Preprocess images for the Donut encoder.
        Expects images in [0, 1] range with shape (B, 3, H, W).

        Dataset code handles resizing. This normalizes tensors with the
        Donut image processor statistics before passing them to the encoder.
        """
        if images.dtype != torch.float32:
            images = images.float()
        if images.max() > 2.0:
            images = images / 255.0

        # Already-normalized tensors generally contain negative values.
        if images.min() < 0.0:
            return images

        image_processor = self.processor.image_processor
        mean = getattr(image_processor, "image_mean", [0.5, 0.5, 0.5])
        std = getattr(image_processor, "image_std", [0.5, 0.5, 0.5])
        mean_t = torch.tensor(mean, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
        std_t = torch.tensor(std, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
        return (images - mean_t) / std_t

    def forward(
        self,
        pixel_values: torch.Tensor,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        element_positions: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass for training.

        Args:
            pixel_values: (B, 3, H, W) preprocessed images
            decoder_input_ids: (B, seq_len) target token IDs (teacher forcing)
            decoder_attention_mask: (B, seq_len) attention mask
            labels: (B, seq_len) target labels for cross-entropy (-100 for padding)
            element_positions: (B, max_regions) positions of <elem> tokens

        Returns:
            Dict with 'logits', 'loss' (if labels provided), 'hidden_states',
            and 'pred_bboxes' from the layout head.
        """
        pixel_values = self.preprocess_image(pixel_values)

        # Forward through Donut (encoder + decoder)
        outputs = self.model(
            pixel_values=pixel_values,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
        )

        result = {
            "logits": outputs.logits,  # (B, seq_len, vocab_size)
        }

        if outputs.loss is not None:
            result["loss"] = outputs.loss

        # Extract last hidden states for layout head
        if hasattr(outputs, "decoder_hidden_states") and outputs.decoder_hidden_states is not None:
            last_hidden = outputs.decoder_hidden_states[-1]
        else:
            # Fallback: run decoder hidden states manually
            last_hidden = None

        # Layout head for bbox prediction
        if last_hidden is not None:
            pred_bboxes = self.layout_head(last_hidden, element_positions)
            result["pred_bboxes"] = pred_bboxes

        return result

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        max_length: Optional[int] = None,
        num_beams: int = 1,
        temperature: float = 1.0,
    ) -> dict:
        """
        Generate Document Genome from an image (inference mode).

        Args:
            pixel_values: (B, 3, H, W) preprocessed images
            max_length: Maximum generation length
            num_beams: Beam search width
            temperature: Sampling temperature

        Returns:
            Dict with 'token_ids', 'text', 'hidden_states'
        """
        if max_length is None:
            max_length = self.max_length

        # Create decoder start token
        bos_token = GENOME_TEMPLATE_TOKENS["bos"]
        decoder_input_ids = self.processor.tokenizer(
            bos_token,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.to(pixel_values.device)

        # Expand for batch
        B = pixel_values.shape[0]
        decoder_input_ids = decoder_input_ids.expand(B, -1)

        pixel_values = self.preprocess_image(pixel_values)

        # Generate
        outputs = self.model.generate(
            pixel_values=pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=max_length,
            num_beams=num_beams,
            temperature=temperature,
            eos_token_id=self.special_token_ids.get("eos"),
            pad_token_id=self.processor.tokenizer.pad_token_id,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

        # Decode tokens to text
        generated_ids = outputs.sequences
        generated_text = self.processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=False
        )

        return {
            "token_ids": generated_ids,
            "text": generated_text,
        }

    def tokenize_genome_sequence(
        self,
        token_sequence: str | list[str],
        max_length: Optional[int] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Tokenize a genome token sequence string into model input IDs.

        Args:
            token_sequence: Single string or list of genome token sequences
            max_length: Max sequence length (uses self.max_length if None)

        Returns:
            Dict with 'input_ids' and 'attention_mask' tensors
        """
        if max_length is None:
            max_length = self.max_length

        if isinstance(token_sequence, str):
            token_sequence = [token_sequence]

        encoded = self.processor.tokenizer(
            token_sequence,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }

    def find_element_positions(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Find positions of <elem> tokens in the input sequence.

        Args:
            input_ids: (B, seq_len) token IDs

        Returns:
            (B, max_regions) tensor of positions (-1 for padding)
        """
        elem_token_id = self.special_token_ids.get("element_start")
        if elem_token_id is None:
            return torch.full(
                (input_ids.shape[0], self.max_regions), -1,
                dtype=torch.long, device=input_ids.device,
            )

        B = input_ids.shape[0]
        result = torch.full(
            (B, self.max_regions), -1,
            dtype=torch.long, device=input_ids.device,
        )

        for b in range(B):
            positions = (input_ids[b] == elem_token_id).nonzero(as_tuple=True)[0]
            n = min(len(positions), self.max_regions)
            result[b, :n] = positions[:n]

        return result

    def prepare_training_batch(
        self,
        images: torch.Tensor,
        token_sequences: list[str],
    ) -> dict[str, torch.Tensor]:
        """
        Prepare a complete training batch from raw images and genome sequences.

        Args:
            images: (B, 3, H, W) input images
            token_sequences: List of genome token sequence strings

        Returns:
            Dict with all tensors needed for forward pass
        """
        # Process images through Donut processor
        # Note: images should already be preprocessed
        pixel_values = images

        # Tokenize genome sequences
        tokenized = self.tokenize_genome_sequence(token_sequences)
        input_ids = tokenized["input_ids"].to(images.device)
        attention_mask = tokenized["attention_mask"].to(images.device)

        # Create labels (shift right, mask padding with -100)
        labels = input_ids.clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        # Decoder input IDs (shift right)
        decoder_input_ids = torch.zeros_like(input_ids)
        decoder_input_ids[:, 1:] = input_ids[:, :-1]
        decoder_input_ids[:, 0] = self.processor.tokenizer.bos_token_id or 0

        # Find element positions for layout head
        element_positions = self.find_element_positions(input_ids)

        return {
            "pixel_values": pixel_values,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": attention_mask,
            "labels": labels,
            "element_positions": element_positions,
        }
