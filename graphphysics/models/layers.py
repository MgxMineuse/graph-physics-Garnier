import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    This module applies RMS normalization over the last dimension of the input tensor.
    """

    def __init__(self, d: int, p: float = -1.0, eps: float = 1e-8, bias: bool = False):
        """
        Initializes the RMSNorm module.

        Args:
            d (int): The dimension of the input tensor.
            p (float, optional): Partial RMSNorm. Valid values are in [0, 1].
                Default is -1.0 (disabled).
            eps (float, optional): A small value to avoid division by zero.
                Default is 1e-8.
            bias (bool, optional): Whether to include a bias term. Default is False.
        """
        super().__init__()

        self.d = d
        self.p = p
        self.eps = eps
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of RMSNorm.

        Args:
            x (torch.Tensor): Input tensor of shape (..., d).

        Returns:
            torch.Tensor: Normalized tensor of the same shape as input.
        """
        if self.p < 0.0 or self.p > 1.0:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)
            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x / math.sqrt(d_x)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed


ACTIVATION = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
}


def build_mlp(
    in_size: int,
    hidden_size: int,
    out_size: int,
    nb_of_layers: int = 4,
    layer_norm: bool = True,
    act: str = "relu",
) -> nn.Module:
    """
    Builds a Multilayer Perceptron.

    Args:
        in_size (int): Size of the input features.
        hidden_size (int): Size of the hidden layers.
        out_size (int): Size of the output features.
        nb_of_layers (int, optional): Total number of linear layers in the MLP.
            Must be at least 2. Defaults to 4.
        layer_norm (bool, optional): Whether to apply RMS normalization to the
            output layer. Defaults to True.
        act (str, optional): Activation function to use ('relu' or 'gelu'). Defaults to 'relu'.

    Returns:
        nn.Module: The constructed MLP model.
    """
    assert nb_of_layers >= 2, "The MLP must have at least 2 layers (input and output)."

    if act not in ACTIVATION:
        raise NotImplementedError(f"Activation '{act}' not supported.")
    activation = ACTIVATION[act]

    layers = [nn.Linear(in_size, hidden_size)]
    layers.append(activation())
    # layers.extend([nn.Dropout(p=0.2)])

    # Add hidden layers
    for _ in range(nb_of_layers - 2):
        layers.extend([nn.Linear(hidden_size, hidden_size)])
        layers.append(activation())
        # layers.extend([nn.Dropout(p=0.2)])

    # Add output layer
    layers.append(nn.Linear(hidden_size, out_size))

    if layer_norm:
        layers.append(nn.LayerNorm(out_size))

    return nn.Sequential(*layers)


class GatedMLP(nn.Module):
    """
    A Gated Multilayer Perceptron.

    This layer applies a gated activation to the input features.
    """

    def __init__(self, in_size: int, hidden_size: int, expansion_factor: int):
        """
        Initializes the GatedMLP layer.

        Args:
            in_size (int): Size of the input features.
            hidden_size (int): Size of the hidden layer.
            expansion_factor (int): Expansion factor for the hidden layer size.
        """
        super().__init__()

        self.linear1 = nn.Linear(in_size, expansion_factor * hidden_size)
        self.linear2 = nn.Linear(in_size, expansion_factor * hidden_size)

        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GatedMLP layer.

        Args:
            x (torch.Tensor): Input tensor of shape (..., in_size).

        Returns:
            torch.Tensor: Output tensor of shape (..., expansion_factor * hidden_size).
        """
        left = self.activation(self.linear1(x))
        right = self.linear2(x)
        return left * right


def build_gated_mlp(
    in_size: int,
    hidden_size: int,
    out_size: int,
    expansion_factor: int = 3,
) -> nn.Module:
    """
    Builds a Gated MLP.

    Args:
        in_size (int): Size of the input features.
        hidden_size (int): Size of the hidden layer.
        out_size (int): Size of the output features.
        expansion_factor (int, optional): Expansion factor for the hidden layer size.
            Defaults to 3.

    Returns:
        nn.Module: The constructed Gated MLP model.
    """
    layers = [
        RMSNorm(in_size),
        GatedMLP(
            in_size=in_size, hidden_size=hidden_size, expansion_factor=expansion_factor
        ),
        nn.Linear(hidden_size * expansion_factor, out_size),
    ]
    return nn.Sequential(*layers)


class Normalizer(nn.Module):
    """
    A module for normalizing data during training.

    This module maintains running statistics to normalize input data.
    """

    def __init__(
        self,
        size: int,
        max_accumulations: int = 10**5,
        std_epsilon: float = 1e-8,
        name: str = "Normalizer",
        device: Optional[Union[str, torch.device]] = "cuda",
    ):
        """
        Initializes the Normalizer module.

        Args:
            size (int): Size of the input data.
            max_accumulations (int, optional): Maximum number of accumulations allowed.
                Defaults to 1e5.
            std_epsilon (float, optional): Epsilon value to avoid division by zero in
                standard deviation. Defaults to 1e-8.
            name (str, optional): Name of the Normalizer. Defaults to "Normalizer".
            device (str or torch.device, optional): Device to run the Normalizer on.
                Defaults to "cuda".
        """
        super().__init__()
        self.name = name
        # self.device = device
        self._max_accumulations = max_accumulations
        self.std_epsilon = torch.tensor(
            std_epsilon,
            dtype=torch.float32,
            requires_grad=False,
        )
        self.register_buffer("_std_epsilon", self.std_epsilon)
        self.register_buffer(
            "_acc_count",
            torch.tensor(
                0.0,
            ),
        )
        self.register_buffer(
            "_num_accumulations",
            torch.tensor(
                0.0,
            ),
        )
        self.register_buffer(
            "_acc_sum",
            torch.zeros(
                (1, size),
                dtype=torch.float32,
                requires_grad=False,
            ),
        )
        self.register_buffer(
            "_acc_sum_squared",
            torch.zeros(
                (1, size),
                dtype=torch.float32,
                requires_grad=False,
            ),
        )

    def forward(
        self, batched_data: torch.Tensor, accumulate: bool = True
    ) -> torch.Tensor:
        """
        Normalizes input data and accumulates statistics.

        Args:
            batched_data (torch.Tensor): Input data of shape (batch_size, size).
            accumulate (bool, optional): Whether to accumulate statistics.
                Defaults to True.

        Returns:
            torch.Tensor: Normalized data of the same shape as input.
        """
        if accumulate:
            # Stop accumulating after reaching max_accumulations to prevent numerical issues
            if self._num_accumulations < self._max_accumulations:
                self._accumulate(batched_data.detach())
        return (batched_data - self._mean()) / self._std_with_epsilon()

    def inverse(self, normalized_batch_data: torch.Tensor) -> torch.Tensor:
        """
        Inverse transformation of the normalizer.

        Args:
            normalized_batch_data (torch.Tensor): Normalized data.

        Returns:
            torch.Tensor: Denormalized data.
        """
        return normalized_batch_data * self._std_with_epsilon() + self._mean()

    def _accumulate(self, batched_data: torch.Tensor):
        """
        Accumulates the statistics of the batched data.

        Args:
            batched_data (torch.Tensor): Input data of shape (batch_size, size).
        """
        count = batched_data.shape[0]
        data_sum = torch.sum(batched_data, dim=0, keepdim=True)
        squared_data_sum = torch.sum(batched_data**2, dim=0, keepdim=True)

        self._acc_sum += data_sum
        self._acc_sum_squared += squared_data_sum
        self._acc_count += count
        self._num_accumulations += 1

    def _mean(self) -> torch.Tensor:
        safe_count = torch.max(self._acc_count, torch.tensor(1.0))
        return self._acc_sum / safe_count

    def _std_with_epsilon(self) -> torch.Tensor:
        safe_count = torch.max(self._acc_count, torch.tensor(1.0))
        variance = self._acc_sum_squared / safe_count - self._mean() ** 2
        std = torch.sqrt(torch.clamp(variance, min=0.0))
        return torch.max(std, self._std_epsilon)

    def get_variable(self) -> Dict[str, Any]:
        """
        Returns the internal variables of the normalizer.

        Returns:
            Dict[str, Any]: A dictionary containing the normalizer's variables.
        """
        return {
            "_max_accumulations": self._max_accumulations,
            "_std_epsilon": self._std_epsilon,
            "_acc_count": self._acc_count,
            "_num_accumulations": self._num_accumulations,
            "_acc_sum": self._acc_sum,
            "_acc_sum_squared": self._acc_sum_squared,
            "name": self.name,
        }


class GraphNetBlock(MessagePassing):
    """
    Graph Network Block implementing the message passing mechanism.
    This block updates both node and edge features.
    """

    def __init__(
        self, hidden_size: int, nb_of_layers: int = 4, layer_norm: bool = True
    ):
        """
        Initializes the GraphNetBlock.

        Args:
            hidden_size (int): The size of the hidden representations.
            nb_of_layers (int, optional): The number of layers in the MLPs.
                Defaults to 4.
            layer_norm (bool, optional): Whether to use layer normalization in the MLPs.
                Defaults to True.
        """
        super().__init__(aggr="add", flow="source_to_target")
        edge_input_dim = 3 * hidden_size
        node_input_dim = 2 * hidden_size
        self.edge_block = build_mlp(
            in_size=edge_input_dim,
            hidden_size=hidden_size,
            out_size=hidden_size,
            nb_of_layers=nb_of_layers,
            layer_norm=layer_norm,
        )
        self.node_block = build_mlp(
            in_size=node_input_dim,
            hidden_size=hidden_size,
            out_size=hidden_size,
            nb_of_layers=nb_of_layers,
            layer_norm=layer_norm,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        size: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the GraphNetBlock.

        Args:
            x (torch.Tensor): Node features of shape [num_nodes, hidden_size].
            edge_index (torch.Tensor): Edge indices of shape [2, num_edges].
            edge_attr (torch.Tensor): Edge features of shape [num_edges, hidden_size].
            size (Size, optional): The size of the source and target nodes.
                Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Updated node features and edge features.
        """
        # Update edge attributes
        row, col = edge_index
        x_i = x[col]  # Target node features
        x_j = x[row]  # Source node features
        edge_attr_ = self.edge_update(edge_attr, x_i, x_j)

        # Perform message passing and update node features
        x_ = self.propagate(
            edge_index, x=x, edge_attr=edge_attr_, size=(x.size(0), x.size(0))
        )

        edge_attr = edge_attr + edge_attr_
        x = x + x_

        return x, edge_attr

    def edge_update(
        self, edge_attr: torch.Tensor, x_i: torch.Tensor, x_j: torch.Tensor
    ) -> torch.Tensor:
        """
        Updates edge features.

        Args:
            edge_attr (torch.Tensor): Edge features [num_edges, hidden_size].
            x_i (torch.Tensor): Target node features [num_edges, hidden_size].
            x_j (torch.Tensor): Source node features [num_edges, hidden_size].

        Returns:
            torch.Tensor: Updated edge features [num_edges, hidden_size].
        """
        edge_input = torch.cat([edge_attr, x_i, x_j], dim=-1)
        edge_attr = self.edge_block(edge_input)
        return edge_attr

    def message(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Constructs messages to be aggregated.

        Args:
            edge_attr (torch.Tensor): Edge features [num_edges, hidden_size].

        Returns:
            torch.Tensor: Messages [num_edges, hidden_size].
        """
        return edge_attr

    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Updates node features after aggregation.

        Args:
            aggr_out (torch.Tensor): Aggregated messages [num_nodes, hidden_size].
            x (torch.Tensor): Node features [num_nodes, hidden_size].

        Returns:
            torch.Tensor: Updated node features [num_nodes, hidden_size].
        """
        node_input = torch.cat([x, aggr_out], dim=-1)
        x = self.node_block(node_input)
        return x


class GraphDiffusionKernelLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batched_data: Data) -> torch.Tensor:
        h_global_list = []

        identifiants = torch.unique(batched_data.id)

        for i in identifiants:
            mask_id = batched_data.id == i
            dist_sq = (
                torch.linalg.norm(
                    batched_data.pos[mask_id].unsqueeze(1)
                    - batched_data.pos[mask_id].unsqueeze(0),
                    dim=2,
                )
                ** 2
            )
            kernel = torch.exp(-dist_sq)

            mask_kernel = kernel < 0.99
            mask_kernel = torch.logical_and(mask_kernel, kernel > 0.95)

            kernel = kernel * mask_kernel.float()
            h_global = torch.matmul(kernel, batched_data.x[mask_id])
            h_global_list.append(h_global)
        return torch.concat(h_global_list)


class GraphNeuralKernelLayer(nn.Module):
    def __init__(self, hidden_dim: int = 16, n_layers: int = 2):
        super().__init__()
        self.neural_kernel = build_mlp(
            in_size=2,
            hidden_size=hidden_dim,
            out_size=1,
            nb_of_layers=n_layers,
            layer_norm=False,
        )

    def forward(self, batched_data: Data) -> torch.Tensor:
        h_global_list = []

        identifiants = torch.unique(batched_data.id)

        for i in identifiants:

            mask_id = batched_data.id == i
            dist = batched_data.pos[mask_id].unsqueeze(1) - batched_data.pos[
                mask_id
            ].unsqueeze(0)
            with torch.no_grad():
                dist_sq = torch.linalg.norm(dist, dim=2)
                mask_kernel = torch.logical_and(dist_sq > 0.08, dist_sq < 0.18)

            kernel = self.neural_kernel(dist).squeeze(-1)
            kernel = torch.softmax(kernel, dim=1)
            kernel = kernel * mask_kernel.float()
            h_global = torch.matmul(kernel, batched_data.x[mask_id])
            h_global_list.append(h_global)
        return torch.concat(h_global_list)
