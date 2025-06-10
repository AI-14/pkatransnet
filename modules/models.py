import torch
import torch.nn as nn
from .blocks import (
    RMSNorm,
    VisualFeatureExtractor,
    TagEmbedding,
    ImpressionEmbedding,
    TextInputEmbedding,
    DecoderLayer,
)
from einops import rearrange
import argparse


class FgModel(nn.Module):
    def __init__(self, args: argparse.Namespace, vocab_size: int) -> None:
        """Initializes FgModel class.

        Args:
            args (argparse.Namespace): Arguments.
            vocab_size (int): Vocab size.
        """

        super().__init__()
        self.vis_feats = VisualFeatureExtractor()
        self.vis_linear = nn.Sequential(
            nn.Linear(2048, args.d_model), nn.SiLU(), nn.Dropout(args.prob)
        )

        self.tag_embs = TagEmbedding(args.tags_encoder_model_name)
        self.tag_linear = nn.Sequential(
            nn.Linear(768, args.d_model), nn.SiLU(), nn.Dropout(args.prob)
        )

        self.imp_embs = ImpressionEmbedding(args.impression_encoder_model_name)
        self.imp_linear = nn.Sequential(
            nn.Linear(768, args.d_model), nn.SiLU(), nn.Dropout(args.prob)
        )

        self.rep_emb = TextInputEmbedding(
            vocab_size, args.d_model, args.seq_len, args.prob
        )

        self.decoder_layers = DecoderLayer(
            args.d_model, args.d_ff, args.num_heads, args.prob, args.num_layers
        )

        self.final_rms_norm = RMSNorm(args.d_model)

        self.logits = nn.Linear(args.d_model, vocab_size)

    @staticmethod
    def create_causal_mask(x: torch.Tensor) -> torch.Tensor:
        """Creates causal mask.

        Args:
            x (torch.Tensor): Shape of [B, seq_len].

        Returns:
            torch.Tensor: Shape of [1, seq_len, seq_len].
        """

        seq_len = x.size(1)
        full_mask = torch.full(
            (seq_len, seq_len), fill_value=1, device=x.device
        )  # [seq_len, seq_len]
        return torch.tril(full_mask).unsqueeze(0)  # [1, seq_len, seq_len]

    @staticmethod
    def create_encoder_mask_for_cross_attention(
        seq_len: int,
        img_seq_len: int,
        tag_attention_mask: torch.Tensor,
        impression_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Creates encoder mask for self attention.

        Args:
            seq_len (int): Maximum sequence length.
            img_seq_len (int): Sequence length of image.
            tag_attention_mask (torch.Tensor): Shape of [B, tag_seq_len].
            impression_attention_mask (torch.Tensor): Shape of [B, impression_seq_len].

        Returns:
            torch.Tensor: Shape of [B, seq_len, total_seq_len] where total_seq_len=img_seq_len + tag_seq_len + impression_seq_len.
        """

        device = tag_attention_mask.device
        batch_size = tag_attention_mask.size()[0]

        img_mask = torch.ones(
            batch_size, img_seq_len, device=device
        )  # [B, img_seq_len]
        combined_mask = torch.cat(
            [img_mask, tag_attention_mask, impression_attention_mask], dim=1
        )  # [B, total_seq_len]
        combined_mask = combined_mask.unsqueeze(1).repeat(
            1, seq_len, 1
        )  # [B, seq_len, total_seq_len]
        return combined_mask

    def forward(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
        tags_input_ids: torch.Tensor,
        tags_attention_mask: torch.Tensor,
        impression_input_ids: torch.Tensor,
        impression_attention_mask: torch.Tensor,
        report_input_ids: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Applies fullmodel network.

        Args:
            image1 (torch.Tensor): Shape of [B, C, H, W].
            image2 (torch.Tensor): Shape of [B, C, H, W].
            tags_input_ids (torch.Tensor): Shape of [B, tag_seq_len].
            tags_attention_mask (torch.Tensor): Shape of [B, tag_seq_len].
            impression_input_ids (torch.Tensor): Shape of [B, impression_seq_len].
            impression_attention_mask (torch.Tensor): Shape of [B, impression_seq_len].
            report_input_ids (torch.Tensor): Shape of [B, seq_len].
            pad_mask (torch.Tensor): Shape of [B, 1, seq_len].

        Returns:
            torch.Tensor: Shape of [B, seq_len, vocab_size].
        """

        img1_feats = self.vis_feats(image1)  # [B, 1024, 8, 8]
        img2_feats = self.vis_feats(image2)  # [B, 1024, 8, 8]
        img_vis_feats = torch.cat([img1_feats, img2_feats], dim=1)  # [B, 2048, 8, 8]
        img_vis_feats = rearrange(
            img_vis_feats, "b c h w -> b (h w) c"
        )  # [B, 64, 2048]
        img_vis_feats = self.vis_linear(img_vis_feats)  # [B, 64, d_model]

        imp_embs = self.imp_embs(
            impression_input_ids, impression_attention_mask
        )  # [B, impression_seq_len 768]
        imp_embs = self.imp_linear(imp_embs)  # [B, impression_seq_len, d_model]

        tag_embs = self.tag_embs(
            tags_input_ids, tags_attention_mask
        )  # [B, tag_seq_len, 768]
        tag_embs = self.tag_linear(tag_embs)  # [B, tag_seq_len, d_model]

        memory = torch.cat(
            [img_vis_feats, tag_embs, imp_embs], dim=1
        )  # [B, total_seq_len, d_model]

        assert pad_mask.size(2) == report_input_ids.size(
            1
        ), "pad_mask and report seq_len must be equal."
        causal_pad_mask = (
            FgModel.create_causal_mask(report_input_ids) & pad_mask
        )  # [B, seq_len, seq_len]
        cross_mask = FgModel.create_encoder_mask_for_cross_attention(
            report_input_ids.size()[1],
            img_vis_feats.size()[1],
            tags_attention_mask,
            impression_attention_mask,
        ) # [B, seq_len, total_seq_len]
        rep_embs = self.rep_emb(report_input_ids)  # [B, seq_len, d_model]
        dec_out = self.decoder_layers(
            rep_embs, memory, causal_pad_mask, cross_mask
        )  # [B, seq_len, d_model]

        logits = self.logits(self.final_rms_norm(dec_out))  # [B, seq_len, vocab_size]

        return logits
