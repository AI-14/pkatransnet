import torch
import torch.nn as nn
from transformers import AutoModel
import math
from torchvision.models import swin_v2_b, Swin_V2_B_Weights
from einops import rearrange


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, prob: float) -> None:
        """Initializes PositionalEmbedding class.

        Args:
            d_model (int): Hidden dimension.
            seq_len (int): Maximum sequence length.
            prob (float): Dropout probability.
        """

        super().__init__()
        self.seq_len = seq_len
        self.dropout = nn.Dropout(prob)

        pe = torch.zeros(seq_len, d_model)  # [seq_len, d_model]

        pos = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # [seq_len, 1]
        dim_pair = torch.arange(0, d_model, 2)  # [d_model // 2]
        div_term = torch.exp(
            dim_pair * (-math.log(10000.0) / d_model)
        )  # [d_model // 2]

        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0)  # [1, seq_len, d_model]

        self.register_buffer(
            "pe", pe
        )  # Registering positional encodings as a non-learnable parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies PE.

        Args:
            x (torch.Tensor): Shape of [B, seq_len, d_model].

        Returns:
            torch.Tensor: Shape of [B, seq_len, d_model].
        """

        seq_len = x.size(1)
        out = x + self.pe[:, :seq_len, :]  # [B, seq_len, d_model]
        out = self.dropout(out)  # [B, seq_len, d_model]
        return out


class TextInputEmbedding(nn.Module):
    def __init__(
        self, vocab_size: int, d_model: int, seq_len: int, prob: float
    ) -> None:
        """Initializes TextInputEmbedding class.

        Args:
            vocab_size (int): Vocab size.
            d_model (int): Hidden dimension.
            seq_len (int): Maximum sequence length.
            prob (float): Dropout probability.
        """

        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEmbedding(d_model, seq_len, prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies embedding.

        Args:
            x (torch.Tensor): Shape of [B, seq_len].

        Returns:
            torch.Tensor: Shape of [B, seq_len, d_model].
        """

        out = self.embedding(x) * math.sqrt(self.d_model)  # [B, seq_len, d_model]
        out = self.pe(out)  # [B, seq_len, d_model]
        return out


class VisualFeatureExtractor(nn.Module):
    def __init__(self) -> None:
        """Initializes VisualFEatureExtractor class."""

        super().__init__()
        model = swin_v2_b(weights=Swin_V2_B_Weights.DEFAULT)
        layers = list(model.children())[:-3]
        self.model = nn.Sequential(*layers)
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts visual features.

        Args:
            x (torch.Tensor): Shape of [B, C, H, W].

        Returns:
            torch.Tensor: Shape of [B, 1024, 8, 8].
        """

        out = self.model(x)  # [B, 1024, 8, 8]
        return out


class TagEmbedding(nn.Module):
    def __init__(self, tags_encoder_model_name: str) -> None:
        """Initializes TagEmbedding class.

        Args:
            tags_encoder_model_name (str): Name of tag encoder model.
        """

        super().__init__()
        self.model = AutoModel.from_pretrained(tags_encoder_model_name)
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(
        self, input_ids: torch.tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Extracts tag embeddings.

        Args:
            input_ids (torch.tensor): Shape of [B, tag_seq_len].
            attention_mask (torch.Tensor): Shape of [B, tag_seq_len].

        Returns:
            torch.Tensor: Shape of [B, tag_seq_len, 768].
        """

        out = self.model(input_ids, attention_mask)
        out = out.last_hidden_state  # [B, tag_seq_len, 768]
        return out


class ImpressionEmbedding(nn.Module):
    def __init__(
        self,
        impression_encoder_model_name: str,
    ) -> None:
        """Initializes ImpressionEmbedding class.

        Args:
            impression_encoder_model_name (str): Name of impression encoder model.
        """

        super().__init__()
        self.model = AutoModel.from_pretrained(impression_encoder_model_name)
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Extracts impression embeddings.

        Args:
            input_ids (torch.Tensor): Shape of [B, impression_seq_len].
            attention_mask (torch.Tensor): Shape of [B, impression_seq_len].

        Returns:
            torch.Tensor: [B, impression_seq_len, 768].
        """

        out = self.model(input_ids, attention_mask)
        out = out.last_hidden_state  # [B, impression_seq_len, 768]
        return out


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        """
        Initialize RMSNorm class.

        Args:
            d_model (int): Hidden dimension.
            eps (float, optional): A small value added to the denominator for numerical stability. Defaults to 1e-6.
        """

        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): shape of [B, seq_len, d_model].

        Returns:
            torch.Tensor: Shape of [B, seq_len, d_model].

        """

        out = x * torch.rsqrt(
            x.pow(2).mean(-1, keepdim=True) + self.eps
        )  # [B, seq_len, d_model]
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies RMSNorm layer.

        Args:
            x (torch.Tensor): Shape of [B, seq_len, d_model].

        Returns:
            torch.Tensor: Shape of [B, seq_len, d_model].

        """

        out = self._norm(x.float()).type_as(x) * self.weight  # [B, seq_len, d_model]
        return out


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, prob: float) -> None:
        """Initializes FeedForward class.

        Args:
            d_model (int): Hidden dimension.
            d_ff (int): Hidden dimension of feedforward.
            prob (float): Dropout probability.
        """

        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.w3 = nn.Linear(d_model, d_ff)
        self.silu = nn.SiLU()
        self.dropout = nn.Dropout(prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies feedforward network.

        Args:
            x (torch.Tensor): Shape of [B, seq_len, d_model].

        Returns:
            torch.Tensor: Shape of [B, seq_len, d_model].
        """

        swish = self.silu(self.w1(x))  # [B, seq_len, d_ff]
        x_v = self.w3(x)  # [B, seq_len, d_ff]
        out = swish * x_v  # [B, seq_len, d_ff]
        out = self.dropout(self.w2(out))  # [B, seq_len, d_model]
        return out


class MultiheadAttention(nn.Module):
    def __init__(self, num_heads: int, d_model: int, prob: float) -> None:
        """Initializes MultiheadAttention class.

        Args:
            num_heads (int): Number of heads.
            d_model (int): Hidden dimension.
            prob (float): Dropout probability.
        """

        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads."

        self.num_heads = num_heads
        self.dk = d_model // num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(prob)
        self.attention_weights: torch.Tensor | None = None

    def compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Computes attention mechanism.

        Args:
            q (torch.Tensor): Shape of [B, num_heads, seq_len, dk].
            k (torch.Tensor): Shape of [B, num_heads, seq_len, dk].
            v (torch.Tensor): Shape of [B, num_heads, seq_len, dk].
            mask (torch.Tensor | None, optional): Shape of [B, seq_len, seq_len]. Defaults to None.
                                                  Note that the seq_len can be different or same.

        Returns:
            torch.Tensor: Shape of [B, num_heads, seq_len, dk].
        """

        attention_scores = torch.einsum(
            "b n i d, b n j d -> b n i j", q, k
        ) / math.sqrt(self.dk)  # [B, num_heads, seq_len, seq_len]

        if mask is not None:
            assert (
                mask.size(1) == q.size(2)
            ), "Sequence length of mask must be equal to the sequence length of the input."
            mask = mask.unsqueeze(1)  # [B, 1, seq_len, seq_len]
            attention_scores.masked_fill_(mask == 0, value=-1e9)

        attention_scores = attention_scores.softmax(
            dim=-1
        )  # [B, num_heads, seq_len, seq_len]
        attention_scores = self.dropout(attention_scores)

        self.attention_weights = attention_scores
        out = torch.einsum(
            "b n i j, b n j d -> b n i d", attention_scores, v
        )  # [B, num_heads, seq_len, dk]
        return out

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Applies MHA.

        Args:
            q (torch.Tensor): Shape of [B, seq_len, d_model].
            k (torch.Tensor): Shape of [B, seq_len, d_model].
            v (torch.Tensor): Shape of [B, seq_len, d_model].
            mask (torch.Tensor | None, optional): Shape of [B, seq_len, seq_len]. Defaults to None.
                                                  Note that the seq_len can be different or same.

        Returns:
            torch.Tensor: Shape of [B, seq_len, d_model].
        """

        q = self.wq(q)  # [B, seq_len, d_model]
        k = self.wk(k)  # [B, seq_len, d_model]
        v = self.wv(v)  # [B, seq_len, d_model]

        # Split into num_heads
        q = rearrange(
            q, "b s (dk h) -> b h s dk", h=self.num_heads
        )  # [B, num_heads, seq_len, dk]
        k = rearrange(
            k, "b s (dk h) -> b h s dk", h=self.num_heads
        )  # [B, num_heads, seq_len, dk]
        v = rearrange(
            v, "b s (dk h) -> b h s dk", h=self.num_heads
        )  # [B, num_heads, seq_len, dk]

        attention_scores = self.compute_attention(
            q, k, v, mask
        )  # [B, num_heads, seq_len, dk]

        # Combine all the num_heads together
        attention_scores = rearrange(
            attention_scores, "b h s dk -> b s (h dk)", h=self.num_heads
        )  # [B, seq_len, d_model]

        out = self.wo(attention_scores)  # [B, seq_len, d_model]
        return out


class Decoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_heads: int,
        prob: float,
    ) -> None:
        """Initializes Decoder class.

        Args:
            d_model (int): Hidden dimension.
            d_ff (int): Hidden dimension of feedforward.
            num_heads (int): Number of heads.
            prob (float): Dropout probability.
        """

        super().__init__()
        self.rms_norm1 = RMSNorm(d_model)
        self.causal_attention = MultiheadAttention(num_heads, d_model, prob)

        self.rms_norm2 = RMSNorm(d_model)
        self.cross_attention = MultiheadAttention(num_heads, d_model, prob)

        self.rms_norm3 = RMSNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, prob)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        causal_mask: torch.Tensor,
        cross_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Applies decoder network.

        Args:
            x (torch.Tensor): Shape of [B, seq_len, d_model].
            memory (torch.Tensor): Shape of [B, memory_seq_len, d_model].
            causal_mask (torch.Tensor): Shape of [B, seq_len, seq_len].
            cross_mask (torch.Tensor | None, optional): Shape of [B, seq_len, memory_seq_len]. Defaults to None.

        Returns:
            torch.Tensor: Shape of [B, seq_len, d_model].
        """

        rms_out1 = self.rms_norm1(x)  # [B, seq_len, d_model]
        out1 = x + self.causal_attention(
            rms_out1, rms_out1, rms_out1, causal_mask
        )  # [B, seq_len, d_model]

        rms_out2 = self.rms_norm2(out1)  # [B, seq_len, d_model]
        out2 = out1 + self.cross_attention(
            rms_out2, memory, memory, cross_mask
        )  # [B, seq_len, d_model]

        rms_out3 = self.rms_norm3(out2)  # [B, seq_len, d_model]
        out3 = out2 + self.ff(rms_out3)  # [B, seq_len, d_model]

        return out3


class DecoderLayer(nn.Module):
    def __init__(
        self, d_model: int, d_ff: int, num_heads: int, prob: float, num_layers: int
    ) -> None:
        """Initializes DecoderLayer class.

        Args:
            d_model (int): Hidden dimension.
            d_ff (int): Hidden dimension of feedforward.
            num_heads (int): Number of heads.
            prob (float): Dropout probability.
            num_layers (int): Number of decoder layers.
        """

        super().__init__()
        self.layers = nn.ModuleList(
            [Decoder(d_model, d_ff, num_heads, prob) for _ in range(num_layers)]
        )

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        causal_mask: torch.Tensor,
        cross_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Applies decoder layers network.

        Args:
            x (torch.Tensor): Shape of [B, seq_len, d_model].
            memory (torch.Tensor): Shape of [B, memory_seq_len, d_model].
            causal_mask (torch.Tensor): Shape of [B, seq_len, seq_len].
            cross_mask (torch.Tensor | None, optional): Shape of [B, seq_len, memory_seq_len]. Defaults to None.

        Returns:
            torch.Tensor: Shape of [B, seq_len, d_model].
        """

        out = x
        for layer in self.layers:
            out = layer(out, memory, causal_mask, cross_mask)  # [B, seq_len, d_model]
        return out
