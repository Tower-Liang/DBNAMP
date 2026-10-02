"""Pure PyTorch molecular graph network used by the NC-AMP prioritizer."""
from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn


class MessagePassingLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int) -> None:
        super().__init__()
        self.src = nn.Linear(hidden_dim, hidden_dim)
        self.edge = nn.Linear(edge_dim, hidden_dim)
        self.update = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, node_state: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        source, target = edge_index
        messages = self.src(node_state[source]) + self.edge(edge_attr)
        aggregate = torch.zeros_like(node_state)
        aggregate.index_add_(0, target, messages)
        degree = torch.zeros(node_state.shape[0], device=node_state.device, dtype=node_state.dtype)
        degree.index_add_(0, target, torch.ones_like(target, dtype=node_state.dtype))
        aggregate = aggregate / degree.clamp_min(1.0).unsqueeze(-1)
        return self.norm(node_state + self.update(torch.cat([node_state, aggregate], dim=-1)))


class SequenceEncoder(nn.Module):
    """Token encoder for the residue/sequence view of the same SMILES input."""
    def __init__(self, vocab_size: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.gru(self.dropout(self.embedding(tokens)))
        weights = mask.to(encoded.dtype).unsqueeze(-1)
        pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.dropout(self.projection(pooled))


class SMILESAmpClassifier(nn.Module):
    """Five-step message passing GNN with mean pooling and gate fusion."""
    def __init__(
        self,
        atom_dim: int,
        edge_dim: int,
        vocab_size: int,
        hidden_dim: int = 300,
        depth: int = 5,
        num_layer: int = 5,
        classifier_dropout: float = 0.4,
        sequence_dropout: float = 0.0,
        use_sequence: bool = True,
    ) -> None:
        super().__init__()
        if depth != num_layer:
            raise ValueError("depth and num_layer are required to match for reproducibility")
        self.config = {
            "atom_dim": atom_dim,
            "edge_dim": edge_dim,
            "vocab_size": vocab_size,
            "hidden_dim": hidden_dim,
            "depth": depth,
            "num_layer": num_layer,
            "classifier_dropout": classifier_dropout,
            "sequence_dropout": sequence_dropout,
            "use_sequence": use_sequence,
        }
        self.atom_projection = nn.Sequential(nn.Linear(atom_dim, hidden_dim), nn.ReLU())
        self.layers = nn.ModuleList([MessagePassingLayer(hidden_dim, edge_dim) for _ in range(depth)])
        self.sequence = SequenceEncoder(vocab_size, hidden_dim, sequence_dropout)
        self.use_sequence = use_sequence
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        node_state = self.atom_projection(batch["x"])
        for layer in self.layers:
            node_state = layer(node_state, batch["edge_index"], batch["edge_attr"])
        graph_sum = torch.zeros(batch["num_graphs"], node_state.shape[1], device=node_state.device)
        graph_sum.index_add_(0, batch["graph_index"], node_state)
        graph_count = torch.bincount(batch["graph_index"], minlength=batch["num_graphs"]).to(node_state.dtype).clamp_min(1).unsqueeze(-1)
        graph_repr = graph_sum / graph_count
        if self.use_sequence:
            sequence_repr = self.sequence(batch["seq_tokens"], batch["seq_mask"])
        else:
            sequence_repr = graph_repr
        gate = torch.sigmoid(self.gate(torch.cat([graph_repr, sequence_repr], dim=-1)))
        fused = gate * graph_repr + (1.0 - gate) * sequence_repr
        return self.classifier(fused).squeeze(-1)


def collate_graphs(records: List[Dict[str, object]], vocab: Dict[str, int]) -> Dict[str, torch.Tensor]:
    """Batch variable-size RDKit graphs without a third-party graph library."""
    xs, edges, edge_attrs, graph_ids, token_rows, labels = [], [], [], [], [], []
    node_offset = 0
    max_tokens = max(len(record["tokens"]) for record in records)
    for graph_id, record in enumerate(records):
        x = torch.as_tensor(record["x"], dtype=torch.float32)
        edge_index = torch.as_tensor(record["edge_index"], dtype=torch.long) + node_offset
        xs.append(x)
        edges.append(edge_index)
        edge_attrs.append(torch.as_tensor(record["edge_attr"], dtype=torch.float32))
        graph_ids.append(torch.full((x.shape[0],), graph_id, dtype=torch.long))
        row = [vocab.get(token, vocab["<unk>"]) for token in record["tokens"]]
        row.extend([vocab["<pad>"]] * (max_tokens - len(row)))
        token_rows.append(row)
        labels.append(float(record.get("label", 0)))
        node_offset += x.shape[0]
    tokens = torch.as_tensor(token_rows, dtype=torch.long)
    return {
        "x": torch.cat(xs),
        "edge_index": torch.cat(edges, dim=1),
        "edge_attr": torch.cat(edge_attrs),
        "graph_index": torch.cat(graph_ids),
        "num_graphs": len(records),
        "seq_tokens": tokens,
        "seq_mask": tokens != vocab["<pad>"],
        "labels": torch.as_tensor(labels, dtype=torch.float32),
    }
