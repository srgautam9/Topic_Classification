"""
Custom architectures trained from scratch -- no pretrained weights are loaded
anywhere in this file. All models consume hashed token ids (feature-hashing
trick, see src/utils.py) instead of an explicit vocabulary, so there is no
dependency on any pretrained tokenizer or embedding table either.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FastTextScratch(nn.Module):
    """FastText-style architecture (Joulin et al., 2016) used only as an
    architectural *reference* -- reimplemented here from scratch in PyTorch,
    not loaded from the `fasttext` library or any pretrained checkpoint.

    Hashed n-gram embeddings, mean-pooled (EmbeddingBag), -> linear
    classifier. Parameter count ~= num_buckets * embed_dim, so num_buckets is
    the main lever to stay under the 5B parameter budget.
    """

    def __init__(self, num_buckets, embed_dim, num_classes):
        super().__init__()
        self.embedding = nn.EmbeddingBag(num_buckets, embed_dim, mode="mean")
        self.fc = nn.Linear(embed_dim, num_classes)
        nn.init.uniform_(self.embedding.weight, -0.5, 0.5)
        nn.init.zeros_(self.fc.bias)

    def forward(self, ids, offsets):
        x = self.embedding(ids, offsets)
        return self.fc(x)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


class TextCNNScratch(nn.Module):
    """Kim (2014)-style CNN over hashed token embeddings, built from scratch."""

    def __init__(self, num_buckets, embed_dim, num_classes,
                 kernel_sizes=(2, 3, 4, 5), num_filters=128, dropout=0.3, pad_id=0):
        super().__init__()
        # Sequence token ids are hash_token(...) + 1 (0 reserved for padding),
        # so valid ids run 0..num_buckets inclusive -> table needs num_buckets+1 rows.
        self.embedding = nn.Embedding(num_buckets + 1, embed_dim, padding_idx=pad_id)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, ids):
        # ids: (batch, seq_len)
        x = self.embedding(ids).transpose(1, 2)  # (batch, embed_dim, seq_len)
        feats = []
        for conv in self.convs:
            if x.size(-1) < conv.kernel_size[0]:
                # sequence shorter than kernel: pad on the fly
                pad_amt = conv.kernel_size[0] - x.size(-1)
                x_conv_in = F.pad(x, (0, pad_amt))
            else:
                x_conv_in = x
            feats.append(F.relu(conv(x_conv_in)).max(dim=2).values)
        x = torch.cat(feats, dim=1)
        return self.fc(self.dropout(x))

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


class BiLSTMScratch(nn.Module):
    """Bidirectional LSTM classifier over hashed token embeddings."""

    def __init__(self, num_buckets, embed_dim, hidden_dim, num_classes,
                 num_layers=1, dropout=0.3, pad_id=0):
        super().__init__()
        # Sequence token ids are hash_token(...) + 1 (0 reserved for padding),
        # so valid ids run 0..num_buckets inclusive -> table needs num_buckets+1 rows.
        self.embedding = nn.Embedding(num_buckets + 1, embed_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                             batch_first=True, bidirectional=True,
                             dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, ids, lengths):
        x = self.embedding(ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        h = torch.cat([h[-2], h[-1]], dim=1)  # concat final fwd/bwd hidden states
        return self.fc(self.dropout(h))

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


def build_model(name, num_buckets, num_classes, **kwargs):
    if name == "fasttext":
        return FastTextScratch(num_buckets, kwargs.get("embed_dim", 100), num_classes)
    if name == "cnn":
        return TextCNNScratch(num_buckets, kwargs.get("embed_dim", 128), num_classes,
                               kernel_sizes=kwargs.get("kernel_sizes", (2, 3, 4, 5)),
                               num_filters=kwargs.get("num_filters", 128),
                               dropout=kwargs.get("dropout", 0.3))
    if name == "bilstm":
        return BiLSTMScratch(num_buckets, kwargs.get("embed_dim", 128),
                              kwargs.get("hidden_dim", 128), num_classes,
                              num_layers=kwargs.get("num_layers", 1),
                              dropout=kwargs.get("dropout", 0.3))
    raise ValueError(f"unknown model {name}")
