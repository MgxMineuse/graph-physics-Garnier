import torch
from torch.nn.modules.loss import _Loss
from torch_geometric.data import Batch

from graphphysics.utils.nodetype import NodeType

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def divergence(batch, network_output):
    row, col = batch.edge_index
    pos = batch.pos

    dx = pos[col] - pos[row]
    du = network_output[col] - network_output[row]

    for i in range(dx.shape[0]):
        for j in range(dx.shape[1]):
            if dx[i, j] == 0:
                dx[i, j] = 1e-8

    dudx = du[:, 0] / (dx[:, 0])
    dudy = du[:, 1] / (dx[:, 1])

    div_edge = dudx + dudy
    return torch.mean(torch.abs(div_edge))


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
        nodes_mask = ~torch.isin(torch.arange(n), selected_indexes)
        mask = torch.logical_and(nodes_mask, mask)

    return mask


class L2Loss_physic(_Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def __name__(self):
        return "MSE and physics"

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
        """
        mask = _prepare_mask_for_loss(
            network_output, node_type, masks, selected_indexes
        )
        errors = ((network_output - target) ** 2)[mask]

        lambda_loss = 1e-4
        physic_loss = divergence(batch, network_output)

        return torch.mean(errors) + lambda_loss * physic_loss


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

        mask_obstacle = node_type == NodeType.OBSTACLE
        mask_obstacle = torch.logical_or(
            mask_obstacle, node_type == NodeType.WALL_BOUNDARY
        )
        errors_obstacle = ((network_output - target) ** 2)[mask_obstacle]

        beta = 1.5
        return torch.mean(errors) + beta * torch.mean(errors_obstacle)


class L2Loss_sillage(_Loss):
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

        error_general = (network_output - target) ** 2

        # loss sillage
        mask_cylinder = _prepare_mask_for_loss(
            network_output,
            node_type,
            [NodeType.OBSTACLE],
            selected_indexes,
        )
        error_cylinder = error_general[mask_cylinder]

        xmin = torch.min(kwargs["graph"].pos[:, 0][mask_cylinder])
        ymin = torch.min(kwargs["graph"].pos[:, 1][mask_cylinder])
        ymax = torch.max(kwargs["graph"].pos[:, 1][mask_cylinder])

        mask_sillage = node_type == NodeType.NORMAL
        conditions = [
            kwargs["graph"].pos[:, 0] >= xmin,
            kwargs["graph"].pos[:, 1] >= ymin,
            kwargs["graph"].pos[:, 1] <= ymax,
        ]
        for c in conditions:
            mask_sillage = torch.logical_and(mask_sillage, c)

        error_sillage = error_general[mask_sillage]

        # loss general
        anti_mask = torch.logical_not(mask_sillage)
        anti_mask = torch.logical_and(node_type == NodeType.NORMAL, anti_mask)

        error_out = error_general[anti_mask]

        return (
            3 * torch.mean(error_cylinder)
            + torch.mean(error_sillage)
            + 0.5 * torch.mean(error_out)
        )
