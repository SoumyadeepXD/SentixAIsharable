"""BiLSTM and TextCNN architectures for SentixAI."""

from __future__ import annotations

import inspect
from typing import Sequence

import torch
from torch import nn


class AdditiveAttention(nn.Module):
    """Masked additive attention that returns both context and token weights."""

    def __init__(self, feature_size: int, attention_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(feature_size, attention_size)
        self.score = nn.Linear(attention_size, 1, bias=False)

    def forward(
        self, sequence_features: torch.Tensor, padding_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # sequence_features: [batch, tokens, features]
        attention_scores = self.score(torch.tanh(self.projection(sequence_features))).squeeze(-1)
        attention_scores = attention_scores.masked_fill(~padding_mask, -1e9)
        attention_weights = torch.softmax(attention_scores, dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), sequence_features).squeeze(1)
        return context, attention_weights


class BiLSTMAttentionClassifier(nn.Module):
    """Tokens -> Embedding -> BiLSTM -> Attention -> Dropout -> Dense logits."""

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.35,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.embedding_dropout = nn.Dropout(dropout)
        self.bilstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = AdditiveAttention(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)
        self.padding_idx = padding_idx

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        padding_mask = token_ids.ne(self.padding_idx)
        embeddings = self.embedding_dropout(self.embedding(token_ids))
        sequence_features, _ = self.bilstm(embeddings)
        context, attention_weights = self.attention(sequence_features, padding_mask)
        logits = self.classifier(self.dropout(context))
        return logits, attention_weights


class TextCNNClassifier(nn.Module):
    """Tokens -> Embedding -> Parallel Conv1d (length-preserving) -> Concat -> Pooling -> Dropout -> Dense logits."""

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embedding_dim: int = 128,
        num_filters: int = 128,
        kernel_sizes: Sequence[int] | str = (3, 4, 5),
        dropout: float = 0.35,
        padding_idx: int = 0,
        pooling: str = "attention",
        attention_dim: int = 128,
    ) -> None:
        super().__init__()
        if isinstance(kernel_sizes, str):
            kernel_sizes = tuple(int(k.strip()) for k in kernel_sizes.split(",") if k.strip())
        else:
            kernel_sizes = tuple(int(k) for k in kernel_sizes)
        if not kernel_sizes:
            raise ValueError("kernel_sizes must contain at least one kernel size.")

        pooling_clean = pooling.lower().strip()
        if pooling_clean not in {"attention", "max"}:
            raise ValueError(f"Unsupported pooling mode '{pooling}'; expected 'attention' or 'max'.")

        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes
        self.dropout_rate = dropout
        self.padding_idx = padding_idx
        self.pooling = pooling_clean
        self.attention_dim = attention_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.embedding_dropout = nn.Dropout(dropout)

        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=k,
                padding="same",
            )
            for k in kernel_sizes
        ])

        total_filters = num_filters * len(kernel_sizes)
        if self.pooling == "attention":
            self.attention = AdditiveAttention(total_filters, attention_dim)
        else:
            self.attention = None

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(total_filters, num_classes)

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # token_ids: [batch, tokens]
        padding_mask = token_ids.ne(self.padding_idx)
        embeddings = self.embedding_dropout(self.embedding(token_ids))  # [B, T, D]
        x = embeddings.transpose(1, 2)  # [B, D, T]

        conv_outputs = [torch.relu(conv(x)) for conv in self.convs]
        features = torch.cat(conv_outputs, dim=1)  # [B, total_filters, T]

        b, c, t = features.shape
        has_tokens = padding_mask.any(dim=1, keepdim=True)

        if self.pooling == "attention":
            seq_features = features.transpose(1, 2)  # [B, T, total_filters]
            context, token_weights = self.attention(seq_features, padding_mask)
            context = torch.where(has_tokens, context, torch.zeros_like(context))
            token_weights = torch.where(has_tokens, token_weights, torch.zeros_like(token_weights))
        else:  # "max"
            masked_features = features.masked_fill(~padding_mask.unsqueeze(1), -1e9)
            max_vals, argmax_indices = torch.max(masked_features, dim=2)  # [B, total_filters]
            context = torch.where(has_tokens, max_vals, torch.zeros_like(max_vals))

            # Channel argmax share: proportion of filter channels whose max landed on token t
            one_hot = torch.zeros(b, c, t, device=features.device, dtype=torch.float)
            one_hot.scatter_(2, argmax_indices.unsqueeze(-1), 1.0)
            token_weights = one_hot.mean(dim=1)  # [B, T]
            token_weights = torch.where(has_tokens, token_weights, torch.zeros_like(token_weights))

        logits = self.classifier(self.dropout(context))
        return logits, token_weights


def build_model(
    model_type: str,
    vocab_size: int | None = None,
    num_classes: int | None = None,
    **kwargs,
) -> nn.Module:
    """Build a classification model by type name and keyword arguments."""
    normalized = model_type.lower().strip()
    kwargs = dict(kwargs)
    if vocab_size is None:
        vocab_size = kwargs.pop("vocab_size", None)
    else:
        kwargs.pop("vocab_size", None)
    if num_classes is None:
        num_classes = kwargs.pop("num_classes", None)
    else:
        kwargs.pop("num_classes", None)

    if vocab_size is None or num_classes is None:
        raise ValueError("build_model requires both vocab_size and num_classes.")

    if normalized in {"bilstm_attention", "bilstm"}:
        target_cls = BiLSTMAttentionClassifier
    elif normalized in {"textcnn", "cnn"}:
        target_cls = TextCNNClassifier
        if "pooling" not in kwargs:
            kwargs["pooling"] = "attention"
        if "num_filters" not in kwargs and "hidden_dim" in kwargs:
            kwargs["num_filters"] = kwargs["hidden_dim"]
        if "attention_dim" not in kwargs and "hidden_dim" in kwargs:
            kwargs["attention_dim"] = kwargs["hidden_dim"]
    elif normalized in {"textcnn_max", "cnn_max"}:
        target_cls = TextCNNClassifier
        kwargs["pooling"] = "max"
        if "num_filters" not in kwargs and "hidden_dim" in kwargs:
            kwargs["num_filters"] = kwargs["hidden_dim"]
        if "attention_dim" not in kwargs and "hidden_dim" in kwargs:
            kwargs["attention_dim"] = kwargs["hidden_dim"]
    else:
        raise ValueError(
            f"Unsupported model_type: '{model_type}'. Expected one of 'bilstm_attention', 'textcnn', 'textcnn_max'."
        )

    sig = inspect.signature(target_cls.__init__)
    valid_params = set(sig.parameters.keys()) - {"self", "vocab_size", "num_classes"}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return target_cls(vocab_size, num_classes, **filtered_kwargs)
