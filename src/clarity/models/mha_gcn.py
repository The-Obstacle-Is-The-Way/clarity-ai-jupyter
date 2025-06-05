import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleGCNConv(nn.Module):
    """A simple GCN layer implementation."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Performs a GCN convolution.

        Args:
            x: Node features (N, in_channels), N is number of nodes.
            adj: Adjacency matrix (N, N).

        Returns:
            Output node features (N, out_channels).
        """
        # Add self-loops to adjacency matrix (A + I)
        N = adj.size(0)
        identity = torch.eye(N, device=adj.device)
        A_hat = adj + identity

        # Compute degree matrix D
        D_hat_diag = torch.sum(A_hat, dim=1)

        # Compute D^(-1/2) with numerical stability safeguards
        # Add small epsilon to prevent division by zero
        epsilon = 1e-10
        D_hat_diag = torch.clamp(D_hat_diag, min=epsilon)  # Ensure positive values
        D_hat_inv_sqrt = torch.pow(D_hat_diag, -0.5)

        # Safety check for infinity values
        D_hat_inv_sqrt[torch.isinf(D_hat_inv_sqrt)] = 0.0
        D_hat_inv_sqrt[torch.isnan(D_hat_inv_sqrt)] = 0.0

        # Create diagonal matrix
        D_inv_sqrt = torch.diag(D_hat_inv_sqrt)

        # Normalized adjacency: D^(-1/2) A D^(-1/2)
        norm_adj = D_inv_sqrt @ A_hat @ D_inv_sqrt

        # Check and clean norm_adj for numerical issues
        norm_adj[torch.isnan(norm_adj)] = 0.0
        norm_adj[torch.isinf(norm_adj)] = 0.0

        # Message passing
        support = norm_adj @ x

        # Linear transformation
        output = self.linear(support)
        return output


class MHA_GCN(nn.Module):
    """Multi-Head Attention Graph Convolutional Network (MHA-GCN) model."""
    def __init__(
        self,
        node_feature_dim: int,
        gcn1_out: int = 128,
        gcn2_out: int = 512,
        mha_heads: int = 4,
        num_classes: int = 2,
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
        self.gcn1 = SimpleGCNConv(node_feature_dim, gcn1_out)
        self.gcn2 = SimpleGCNConv(gcn1_out, gcn2_out)

        self.mha = nn.MultiheadAttention(
            embed_dim=gcn2_out, num_heads=mha_heads, dropout=0.1, batch_first=True
        )

        self.fc_out = nn.Linear(gcn2_out, num_classes)

    def forward(self, node_features: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass of the MHA-GCN model.

        This implementation assumes processing one graph at a time.
        Batch processing should be handled by a wrapper or an outer loop.

        Args:
            node_features: Input node features for the graph.
            adj_matrix: Adjacency matrix for the graph.

        Returns:
            Output tensor of shape (num_classes,).
        """
        # This implementation assumes batch processing is handled outside or by a wrapper.
        # It processes one graph at a time.

        # First GCN layer with ReLU activation
        x = self.gcn1(node_features, adj_matrix)

        # Safety check for numerical stability after first GCN
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        x = F.relu(x)  # Apply ReLU after numerical cleaning

        # Second GCN layer
        x = self.gcn2(x, adj_matrix)

        # Safety check for numerical stability after second GCN
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

        # Prepare for multi-head attention (add batch dimension)
        x_mha_input = x.unsqueeze(0)

        # Apply multi-head attention
        attn_output, _ = self.mha(x_mha_input, x_mha_input, x_mha_input)

        # Handle any numerical instability in attention output
        attn_output = torch.nan_to_num(attn_output, nan=0.0, posinf=1.0, neginf=-1.0)

        # Global pooling to get graph-level embedding
        graph_embedding = torch.mean(attn_output, dim=1)

        # Final classification layer
        out = self.fc_out(graph_embedding.squeeze(0))

        # Final safety check
        out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
        return out
