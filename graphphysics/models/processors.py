import torch
import torch.nn as nn
from torch_geometric.data import Data

from graphphysics.models.layers import (
    GraphNetBlock,
    build_mlp,
)


class EncodeProcessDecode(nn.Module):
    """
    Encode-Process-Decode model for graph neural networks.

    This model architecture is designed to process graph-structured data through three main steps:
    encoding, processing, and decoding. The encoder maps input node and edge features to a latent space,
    the processor performs message passing to update node and edge representations, and the decoder
    generates the final output from the processed graph.

    Parameters
    ----------
    message_passing_num : int
        Number of message passing steps (i.e., number of GraphNetBlock layers).
    node_input_size : int
        Size of the input node features.
    edge_input_size : int
        Size of the input edge features.
    output_size : int
        Size of the output features.
    hidden_size : int, optional
        Size of the hidden representations in all layers. Default is 128.
    only_processor : bool, optional
        If True, only the processor is used (no encoding or decoding steps). Default is False.
    nb_iterations : int, optional
        Number of iterations for message passing. Default is 1.

    Attributes
    ----------
    only_processor : bool
        Whether only the processor is used.
    hidden_size : int
        Size of the hidden representations.
    d : int
        Size of the output features (alias for `output_size`).
    nb_iterations : int
        Number of iterations for message passing.
    nodes_encoder : torch.nn.Module
        MLP for encoding node features. Only initialized if `only_processor` is False.
    edges_encoder : torch.nn.Module
        MLP for encoding edge features. Only initialized if `only_processor` is False.
    decode_module : torch.nn.Module
        MLP for decoding hidden representations to output. Only initialized if `only_processor` is False.
    processor_list : torch.nn.ModuleList
        List of GraphNetBlock modules for message passing.
    """

    def __init__(
        self,
        message_passing_num: int,
        node_input_size: int,
        edge_input_size: int,
        output_size: int,
        hidden_size: int = 128,
        only_processor: bool = False,
        nb_iterations: int = 1,
    ):
        """
        Initialize the EncodeProcessDecode model.

        Parameters
        ----------
        message_passing_num : int
            Number of message passing steps.
        node_input_size : int
            Size of the input node features.
        edge_input_size : int
            Size of the input edge features.
        output_size : int
            Size of the output features.
        hidden_size : int, optional
            Size of the hidden representations. Default is 128.
        only_processor : bool, optional
            If True, only the processor is used (no encoding or decoding). Default is False.
        nb_iterations : int, optional
            Number of iterations for message passing. Default is 1.
        """
        super().__init__()
        self.only_processor = only_processor
        self.hidden_size = hidden_size
        self.d = output_size
        self.nb_iterations = nb_iterations

        if not self.only_processor:
            self.nodes_encoder = build_mlp(
                in_size=node_input_size,
                hidden_size=hidden_size,
                out_size=hidden_size,
                nb_of_layers=2,
            )

            self.edges_encoder = build_mlp(
                in_size=edge_input_size,
                hidden_size=hidden_size,
                out_size=hidden_size,
                nb_of_layers=2,
            )

            self.decode_module = build_mlp(
                in_size=hidden_size,
                hidden_size=hidden_size,
                out_size=output_size,
                layer_norm=False,
                nb_of_layers=2,
            )

        self.processor_list = nn.ModuleList(
            [
                GraphNetBlock(hidden_size=hidden_size, nb_of_layers=2)
                for _ in range(message_passing_num)
            ]
        )

    def forward(self, graph: Data) -> torch.Tensor:
        """
        Forward pass of the EncodeProcessDecode model.

        Parameters
        ----------
        graph : Data
            Input graph data from `torch_geometric.data.Data`, containing:
            - `x` : Node features, shape (num_nodes, node_input_size).
            - `edge_index` : Graph connectivity, shape (2, num_edges).
            - `edge_attr` : Edge features, shape (num_edges, edge_input_size).

        Returns
        -------
        torch.Tensor
            Output node features after processing and decoding, shape (num_nodes, output_size).
            If `only_processor` is True, returns the processed node features without decoding.
        """
        edge_index = graph.edge_index

        if self.only_processor:
            x, edge_attr = graph.x, graph.edge_attr
        else:
            x = self.nodes_encoder(graph.x)
            edge_attr = self.edges_encoder(graph.edge_attr)

        for _ in range(self.nb_iterations):
            for block in self.processor_list:
                x, edge_attr = block(x, edge_index, edge_attr)

        if self.only_processor:
            return x
        else:
            x_decoded = self.decode_module(x)
            return x_decoded
