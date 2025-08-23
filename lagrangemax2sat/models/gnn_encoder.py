""" Architecture of the GNN from encoder to decoder. """

import torch
import torch.nn.functional as F
from torch import nn


# usefull only if per_layer_out is used
def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class GNNLayer(nn.Module):
  """Configurable GNN Layer
  Implements the Gated Graph ConvNet layer:
      h_i = ReLU ( U*h_i + Aggr.( sigma_ij, V*h_j) ),
      sigma_ij = sigmoid( A*h_i + B*h_j + C*e_ij ),
      e_ij = ReLU ( A*h_i + B*h_j + C*e_ij ),
      where Aggr. is an aggregation function: sum/mean/max.
  References:
      - X. Bresson and T. Laurent. An experimental study of neural networks for variable graphs. In International Conference on Learning Representations, 2018.
      - V. P. Dwivedi, C. K. Joshi, T. Laurent, Y. Bengio, and X. Bresson. Benchmarking graph neural networks. arXiv preprint arXiv:2003.00982, 2020.
  """

  def __init__(self, hidden_dim, aggregation="sum", norm="batch", learn_norm=True, track_norm=False, gated=True):
    """
    Args:
        hidden_dim: Hidden dimension size (int)
        aggregation: Neighborhood aggregation scheme ("sum"/"mean"/"max")
        norm: Feature normalization scheme ("layer"/"batch"/None)
        learn_norm: Whether the normalizer has learnable affine parameters (True/False)
        track_norm: Whether batch statistics are used to compute normalization mean/std (True/False)
        gated: Whether to use edge gating (True/False)
    """
    super(GNNLayer, self).__init__()
    self.hidden_dim = hidden_dim
    self.aggregation = aggregation
    self.norm = norm
    self.learn_norm = learn_norm
    self.track_norm = track_norm
    self.gated = gated
    assert self.gated, "Use gating with GCN, pass the `--gated` flag"

    self.U = nn.Linear(hidden_dim, hidden_dim, bias=True)
    self.V = nn.Linear(hidden_dim, hidden_dim, bias=True)
    self.A = nn.Linear(hidden_dim, hidden_dim, bias=True)
    self.B = nn.Linear(hidden_dim, hidden_dim, bias=True)
    self.C = nn.Linear(hidden_dim, hidden_dim, bias=True)

    self.norm_h = {
        "layer": nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
        "batch": nn.BatchNorm1d(hidden_dim, affine=learn_norm, track_running_stats=track_norm)
    }.get(self.norm, None)

    self.norm_e = {
        "layer": nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
        "batch": nn.BatchNorm1d(hidden_dim, affine=learn_norm, track_running_stats=track_norm)
    }.get(self.norm, None)

  def forward(self, h, e, mode="residual"):
    """
    Args:
        In Dense version:
          h: Embedded input node features (B x V x H)
          e: Embedded input edge features (B x V x V x H)
          mode: str
    Returns:
        Updated node and edge features
    """
    batch_size, num_nodes, hidden_dim = h.shape
    h_in = h
    e_in = e

    # Linear transformations for node update
    Uh = self.U(h)  # B x V x H

    Vh = self.V(h).unsqueeze(1).expand(-1, num_nodes, -1, -1)  # B x V x V x H

    # Linear transformations for edge update and gating
    Ah = self.A(h)  # B x V x H, source
    Bh = self.B(h)  # B x V x H, target
    Ce = self.C(e)  # B x V x V x H / E x H

    # Update edge features and compute edge gates
    e = Ah.unsqueeze(1) + Bh.unsqueeze(2) + Ce  # B x V x V x H

    gates = torch.sigmoid(e)  # B x V x V x H / E x H

    # Update node features
    h = Uh + self.aggregate(Vh, gates)  # B x V x H

    # Normalize node features
    h = self.norm_h(
      h.view(batch_size * num_nodes, hidden_dim)
    ).view(batch_size, num_nodes, hidden_dim) if self.norm_h else h

    # Normalize edge features
    e = self.norm_e(
      e.view(batch_size * num_nodes * num_nodes, hidden_dim)
    ).view(batch_size, num_nodes, num_nodes, hidden_dim) if self.norm_e else e

    # Apply non-linearity
    h = F.relu(h)
    e = F.relu(e)

    # Make residual connection
    if mode == "residual":
      h = h_in + h
      e = e_in + e

    return h, e

  def aggregate(self, Vh, gates, mode=None):
    """
    Args:
        In Dense version:
          Vh: Neighborhood features (B x V x V x H)
          gates: Edge gates (B x V x V x H)
          mode: str
    Returns:
        Aggregated neighborhood features (B x V x H)
    """
    # Perform feature-wise gating mechanism
    Vh = gates * Vh  # B x V x V x H

    # Aggregate neighborhood features
    if (mode or self.aggregation) == "max":
      return torch.max(Vh, dim=2)[0]
    else:
      return torch.sum(Vh, dim=2)


class GNNEncoder(nn.Module):
  """Configurable GNN Encoder
  """

  def __init__(self, n_layers, hidden_dim, n_nodes_features=2, n_edges_features=4,
               aggregation="sum", norm="layer",
               learn_norm=True, track_norm=False, gated=True,
               *args, **kwargs):
    super(GNNEncoder, self).__init__()
    self.hidden_dim = hidden_dim
    self.n_edges_features = n_edges_features

    # Nodes encoding
    self.node_embed = nn.Linear(n_nodes_features, hidden_dim)

    # Edges encoding
    self.edge_embed = nn.Linear(n_edges_features, hidden_dim)

    
    # Node decoder
    self.node_out = nn.Sequential(
        nn.LayerNorm(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1)
    )

    # Edges decoder
    self.edge_out = nn.Sequential(
        nn.LayerNorm(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 2)
    )

    # Layers
    self.layers = nn.ModuleList([
        GNNLayer(hidden_dim, aggregation, norm, learn_norm, track_norm, gated)
        for _ in range(n_layers)
    ])

    # Edges output after each layer
    self.per_layer_out = nn.ModuleList([
        nn.Sequential(
          nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
          nn.SiLU(),
          zero_module(
              nn.Linear(hidden_dim, hidden_dim)
          ),
        ) for _ in range(n_layers)
    ])

  def forward(self, x, graph):
    """
    Args:
        x: Input node features (B x V x 2)
        graph: Graph adjacency matrices (B x V x V x 2 x 2)
    Returns:
        Updated nodes features (B x V)
        Updated edges features (B x V x V x 2)
    """

    B, N= graph.shape[0], graph.shape[1]
    x = self.node_embed(x)
    e = self.edge_embed(graph.float().reshape(B, N, N, 4))

    for layer, out_layer in zip(self.layers, self.per_layer_out):
      x_in, e_in = x, e
      x, e = layer(x, e, mode="direct")
      x = x_in + x
      e = e_in + out_layer(e)

    x = self.node_out(x)
    x = x.squeeze(-1)
    e = self.edge_out(e)
    return x, e
