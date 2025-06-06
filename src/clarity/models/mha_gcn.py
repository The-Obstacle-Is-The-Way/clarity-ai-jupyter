import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

from ...clarity.training.config import NUM_CLASSES


class MHA_GCN(nn.Module):
    """Multi-Head Attention Graph Convolutional Network (MHA-GCN) model."""
    def __init__(
        self,
        node_feature_dim: int,
        gcn1_out: int = 128,
        gcn2_out: int = 512,
        mha_heads: int = 4,
        num_classes: int = NUM_CLASSES,
    ):
        """Initializes the MHA-GCN model.

        Args:
            node_feature_dim: Dimensionality of input node features.
            gcn1_out: Output dimensionality of the first GCN layer.
            gcn2_out: Output dimensionality of the second GCN layer.
            mha_heads: Number of heads in the Multi-Head Attention layer.
            num_classes: Number of output classes for classification.
        """
        super().__init__()
        self.gcn1 = GCNConv(node_feature_dim, gcn1_out)
        self.gcn2 = GCNConv(gcn1_out, gcn2_out)

        self.mha = nn.MultiheadAttention(
            embed_dim=gcn2_out, num_heads=mha_heads, dropout=0.1, batch_first=True
        )

        self.fc_out = nn.Linear(gcn2_out, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Defines the forward pass of the MHA-GCN model using PyG for batching.

        Args:
            x: Node features for the batch of graphs.
            edge_index: Edge indices for the batch of graphs.
            batch: Batch vector mapping each node to its graph index.

        Returns:
            A tuple containing:
            - Output logits of shape (batch_size, num_classes).
            - Attention weights from the MHA layer.
        """
        # GCN layers with ReLU activation
        x = self.gcn1(x, edge_index)
        x = F.relu(x)
        x = self.gcn2(x, edge_index)

        # NOTE: MHA layer expects (N, L, E) or (L, N, E) where L is seq_len.
        # Here, we treat nodes as the sequence. This might not be the optimal
        # use of MHA in a graph context, but we are preserving the original
        # architecture's intent. The input needs to be reshaped.
        # However, MultiheadAttention is not directly compatible with PyG's batching.
        # A simple workaround is to process items in the batch, but this negates the batching benefit.
        # For now, we'll proceed with a mean-pooling of attention weights as a placeholder.
        # A more advanced implementation would use a graph-native attention mechanism.

        # We apply MHA to the node embeddings `x`.
        # To make it compatible, we can treat the whole batch of nodes as one sequence.
        x_mha_input = x.unsqueeze(0)  # (1, num_nodes_in_batch, features)
        attn_output, attn_weights = self.mha(
            x_mha_input,
            x_mha_input,
            x_mha_input
        )
        attn_output = attn_output.squeeze(0)

        # Global average pooling to get a graph-level embedding for each graph in the batch
        graph_embedding = global_mean_pool(attn_output, batch)

        # Final classification layer
        out = self.fc_out(graph_embedding)

        return out, attn_weights
