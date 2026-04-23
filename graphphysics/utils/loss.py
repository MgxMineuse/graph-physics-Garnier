import torch
from torch.nn.modules.loss import _Loss
from torch_geometric.data import Batch

from graphphysics.utils.nodetype import NodeType


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def divergence(batch, network_output, mask):
    row,col = batch.edge_index
    pos = batch.pos

    dx = pos[col]-pos[row]
    du = network_output[col]-network_output[row]
    du = du

    eps = 1e-8

    dudx = du[:, 0]/(dx[:, 0]+eps)
    dudy = du[:, 1]/(dx[:, 1]+eps)

    div_edge = dudx + dudy
    div_node = torch.zeros_like(pos[:, 0])
    div_node = div_node.index_add(0, row, div_edge)
    return torch.mean(div_node[mask]**2)

def _prepare_mask_for_loss(
    network_output: torch.Tensor,
    node_type: torch.Tensor,
    masks: list[NodeType],
    selected_indexes: torch.Tensor = None,
):
    mask = node_type == masks[0]
    for i in range(1, len(masks)):
        mask = torch.logical_or(mask, node_type == masks[i])

    if selected_indexes is not None:
        n, _ = network_output.shape
        nodes_mask = ~torch.isin(torch.arange(n), selected_indexes).to(device)
        mask = torch.logical_and(nodes_mask, mask)

    return mask


class L2Loss_physic(_Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def __name__(self):
        return "MSE"

    def forward(
        self,
        target: torch.Tensor,
        network_output: torch.Tensor,
        node_type: torch.Tensor,
        masks: list[NodeType],
        selected_indexes: torch.Tensor = None,
        batch: Batch = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Computes L2 loss for nodes of specific types with a physic term.

        Args:
            target (torch.Tensor): The target values.
            network_output (torch.Tensor): The predicted values from the network.
            node_type (torch.Tensor): Tensor containing the type of each node.
            masks (list[NodeType]): List of NodeTypes to include in the loss calculation.
            selected_indexes (torch.Tensor, optional): Indexes of nodes to exclude from the loss calculation.

        Returns:
            torch.Tensor: The mean squared error for the specified node types.

        Note:
            This method calculates the L2 loss only for nodes of the types specified in 'masks'.
            If 'selected_indexes' is provided, those nodes are excluded from the loss calculation.
        """
        mask = _prepare_mask_for_loss(
            network_output, node_type, masks, selected_indexes
        )
        errors = ((network_output - target) ** 2)[mask]

        lambda_loss = 2e-3
        physic_loss = divergence(batch, network_output, mask)

        return torch.mean(errors) + lambda_loss*physic_loss


class L2Loss(_Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def __name__(self):
        return "MSE"

    def forward(
        self,
        target: torch.Tensor,
        network_output: torch.Tensor,
        node_type: torch.Tensor,
        masks: list[NodeType],
        selected_indexes: torch.Tensor = None,
        batch: Batch = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Computes L2 loss for nodes of specific types.

        Args:
            target (torch.Tensor): The target values.
            network_output (torch.Tensor): The predicted values from the network.
            node_type (torch.Tensor): Tensor containing the type of each node.
            masks (list[NodeType]): List of NodeTypes to include in the loss calculation.
            selected_indexes (torch.Tensor, optional): Indexes of nodes to exclude from the loss calculation.

        Returns:
            torch.Tensor: The mean squared error for the specified node types.

        Note:
            This method calculates the L2 loss only for nodes of the types specified in 'masks'.
            If 'selected_indexes' is provided, those nodes are excluded from the loss calculation.
        """
        mask = _prepare_mask_for_loss(
            network_output, node_type, masks, selected_indexes
        )
        errors = ((network_output - target) ** 2)[mask]
        return torch.mean(errors)