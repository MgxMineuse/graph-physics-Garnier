import torch
from torch.nn.modules.loss import _Loss
from torch_geometric.data import Batch
from graphphysics.utils.nodetype import NodeType
from graphphysics.utils.vectorial_operators import compute_divergence

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    def forward(self, network_output: torch.Tensor, graph: Batch) -> torch.Tensor:
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
        predict_divergence = compute_divergence(
            graph, network_output[:, :2], device=network_output.device
        )
        physic_loss = torch.mean(torch.abs(predict_divergence))

        return physic_loss


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

        # mask_obstacle = _prepare_mask_for_loss(
        #     network_output,
        #     node_type,
        #     [NodeType.OBSTACLE],
        #     selected_indexes,
        # )
        # errors_obstacle = ((network_output[:, 2] - target[:, 2]) ** 2)[mask_obstacle]

        # beta = 1.5
        return torch.mean(errors)


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
