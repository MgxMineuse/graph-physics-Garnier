import torch
from torch.nn.modules.loss import _Loss
from torch_geometric.data import Batch
from graphphysics.utils.vectorial_operators import *
from graphphysics.utils.nodetype import NodeType

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


class L2Loss_divergence(_Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def __name__(self):
        return "MSE and physics"

    def forward(
        self,
        predicted_outputs: torch.Tensor,
        graph: Batch,
        node_type: torch.Tensor,
        masks: list[NodeType],
    ) -> torch.Tensor:
        """
        Computes L2 loss for nodes of specific types with a physic term.

        Args:
            target (torch.Tensor): The target values.
            predicted_outputs (torch.Tensor): The predicted values from the network.
            node_type (torch.Tensor): Tensor containing the type of each node.
            masks (list[NodeType]): List of NodeTypes to include in the loss calculation.
            selected_indexes (torch.Tensor, optional): Indexes of nodes to exclude from the loss calculation.

        Returns:
            torch.Tensor: The mean squared error for the specified node types.
        """
        mask = _prepare_mask_for_loss(predicted_outputs, node_type, masks)
        predict_divergence = compute_divergence(
            graph, predicted_outputs[:, :2], device=predicted_outputs.device
        )
        physic_loss = torch.mean(torch.abs(predict_divergence[mask]))

        return physic_loss


class L2Loss_convection_diffusion_divergence(_Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def __name__(self):
        return "comparison convection and diffusion terms with groundtruth"

    def forward(
        self,
        predicted_outputs: torch.Tensor,
        graph: Batch,
        node_type: torch.Tensor,
        masks: list[NodeType],
    ) -> torch.Tensor:
        """
        Computes L2 loss for nodes of specific types with a physic term.

        Args:
            target (torch.Tensor): The target values.
            predicted_outputs (torch.Tensor): The predicted values from the network.
            node_type (torch.Tensor): Tensor containing the type of each node.
            masks (list[NodeType]): List of NodeTypes to include in the loss calculation.
            selected_indexes (torch.Tensor, optional): Indexes of nodes to exclude from the loss calculation.

        Returns:
            torch.Tensor: The mean squared error for the specified node types.
        """
        mask = _prepare_mask_for_loss(predicted_outputs, node_type, masks)

        velocity = predicted_outputs[:, :2]
        velocity_gt = graph.y[:, :2]

        grad_velocity = compute_gradient(
            graph,
            field=velocity,
            method="finite_diff",
            device=predicted_outputs.device,
        )
        grad_velocity_gt = compute_gradient(
            graph,
            field=velocity_gt,
            method="finite_diff",
            device=predicted_outputs.device,
        )
        convection = compute_vector_gradient_product(
            graph,
            field=velocity,
            gradient=grad_velocity,
            device=predicted_outputs.device,
        )
        convection_gt = compute_vector_gradient_product(
            graph,
            field=velocity_gt,
            gradient=grad_velocity_gt,
            device=predicted_outputs.device,
        )
        convection_loss = torch.mean((convection_gt[mask] - convection[mask]) ** 2)

        d2U = compute_divergence(
            graph,
            field=grad_velocity[:, 0].squeeze(1),
            method="finite_diff",
            device=predicted_outputs.device,
        ).unsqueeze(1)
        d2V = compute_divergence(
            graph,
            field=grad_velocity[:, 1].squeeze(1),
            method="finite_diff",
            device=predicted_outputs.device,
        ).unsqueeze(1)
        diffusion = torch.concat((d2U, d2V), dim=1)
        d2U_gt = compute_divergence(
            graph,
            field=grad_velocity_gt[:, 0].squeeze(1),
            method="finite_diff",
            device=predicted_outputs.device,
        ).unsqueeze(1)
        d2V_gt = compute_divergence(
            graph,
            field=grad_velocity_gt[:, 1].squeeze(1),
            method="finite_diff",
            device=predicted_outputs.device,
        ).unsqueeze(1)
        diffusion_gt = torch.concat((d2U_gt, d2V_gt), dim=1)
        diffusion_loss = torch.mean((diffusion_gt[mask] - diffusion[mask]) ** 2)

        divergence_velocity = compute_divergence(
            graph,
            field=velocity,
            gradient=grad_velocity,
            method="finite_diff",
            device=predicted_outputs.device,
        )
        divergence_loss = torch.mean(torch.abs(divergence_velocity[mask]))
        return convection_loss / 10 + diffusion_loss / 10000 + divergence_loss / 100


class L2Loss_gradients(_Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def __name__(self):
        return "comparison convection and diffusion terms with groundtruth"

    def forward(
        self,
        predicted_outputs: torch.Tensor,
        graph: Batch,
        node_type: torch.Tensor,
        masks: list[NodeType],
    ) -> torch.Tensor:
        """
        Computes L2 loss for nodes of specific types with a physic term.

        Args:
            target (torch.Tensor): The target values.
            predicted_outputs (torch.Tensor): The predicted values from the network.
            node_type (torch.Tensor): Tensor containing the type of each node.
            masks (list[NodeType]): List of NodeTypes to include in the loss calculation.
            selected_indexes (torch.Tensor, optional): Indexes of nodes to exclude from the loss calculation.

        Returns:
            torch.Tensor: The mean squared error for the specified node types.
        """
        mask = _prepare_mask_for_loss(predicted_outputs, node_type, masks)

        velocity = predicted_outputs[:, :2]
        velocity_gt = graph.y[:, :2]

        grad_velocity = compute_gradient(
            graph,
            field=velocity,
            method="finite_diff",
            device=predicted_outputs.device,
        )
        grad_velocity_gt = compute_gradient(
            graph,
            field=velocity_gt,
            method="finite_diff",
            device=predicted_outputs.device,
        )
        velocity_loss = torch.mean((grad_velocity[mask] - grad_velocity_gt[mask]) ** 2)

        # grad_pressure = compute_gradient(
        #     graph,
        #     field=predicted_outputs[:, 2].unsqueeze(1),
        #     method="finite_diff",
        #     device=predicted_outputs.device,
        # ).squeeze(1)

        # grad_pressure_gt = compute_gradient(
        #     graph,
        #     field=graph.y[:, 2].unsqueeze(1),
        #     method="finite_diff",
        #     device=predicted_outputs.device,
        # )
        # pressure_loss = torch.mean((grad_pressure[mask] - grad_pressure_gt[mask]) ** 2)

        return velocity_loss


class L2Loss_NS(_Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def __name__(self):
        return "full NS physics"

    def forward(
        self,
        predicted_outputs: torch.Tensor,
        graph: Batch,
        node_type: torch.Tensor,
        masks: list[NodeType],
    ) -> torch.Tensor:
        """
        Computes L2 loss for nodes of specific types with a physic term.

        Args:
            target (torch.Tensor): The target values.
            predicted_outputs (torch.Tensor): The predicted values from the network.
            node_type (torch.Tensor): Tensor containing the type of each node.
            masks (list[NodeType]): List of NodeTypes to include in the loss calculation.
            selected_indexes (torch.Tensor, optional): Indexes of nodes to exclude from the loss calculation.

        Returns:
            torch.Tensor: The mean squared error for the specified node types.
        """

        mask = _prepare_mask_for_loss(predicted_outputs, node_type, masks)

        velocity = graph.x[:, :2]
        pressure = graph.x[:, 2]
        grad_velocity = compute_gradient(
            graph,
            field=velocity,
            method="finite_diff",
            device=predicted_outputs.device,
        )
        divergence_velocity = compute_divergence(
            graph,
            field=velocity,
            gradient=grad_velocity,
            method="finite_diff",
            device=predicted_outputs.device,
        )
        convection = compute_vector_gradient_product(
            graph,
            field=velocity,
            gradient=grad_velocity,
            device=predicted_outputs.device,
        )

        d2U = compute_divergence(
            graph,
            field=grad_velocity[:, 0].squeeze(1),
            method="finite_diff",
            device=predicted_outputs.device,
        ).unsqueeze(1)
        d2V = compute_divergence(
            graph,
            field=grad_velocity[:, 1].squeeze(1),
            method="finite_diff",
            device=predicted_outputs.device,
        ).unsqueeze(1)
        diffusion = torch.concat((d2U, d2V), dim=1)

        grad_pressure = compute_gradient(
            graph,
            field=pressure.unsqueeze(1),
            method="finite_diff",
            device=predicted_outputs.device,
        )

        dUdt = (predicted_outputs[:, :2] - velocity) / 0.001

        nu = 0.8e-6
        residuals = dUdt + convection + grad_pressure - nu * diffusion
        NS_loss = torch.mean(torch.abs(residuals[mask]))

        divergence_loss = torch.mean(torch.abs(divergence_velocity[mask]))

        # equation de Poisson
        pressure_poisson = predicted_outputs[:, 2]
        velocity_poisson = predicted_outputs[:, :2]
        grad_velocity_poisson = compute_gradient(
            graph=graph,
            field=velocity_poisson,
            method="finite_diff",
            device=predicted_outputs.device,
        )
        grad_pressure_poisson = compute_gradient(
            graph=graph,
            field=pressure_poisson,
            method="finite_diff",
            device=predicted_outputs.device,
        )
        grad_grad_pressure_poisson = compute_gradient(
            graph=graph,
            field=grad_pressure_poisson,
            method="finite_diff",
            device=predicted_outputs.device,
        )

        p = compute_divergence(
            graph=graph,
            field=pressure_poisson,
            gradient=grad_grad_pressure_poisson,
            device=predicted_outputs.device,
        )
        u = compute_divergence(
            graph=graph,
            field=velocity_poisson,
            gradient=torch.matmul(grad_velocity_poisson, grad_velocity_poisson),
            device=predicted_outputs.device,
        )
        poisson_loss = torch.mean(torch.abs(p[mask] + u[mask]))

        return NS_loss + divergence_loss + poisson_loss


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
        return torch.mean(errors)


class L2Loss_pondération_spatiale(_Loss):
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
        weights = 1 / (kwargs["graph"].x[:, 5] + 1e-6)
        weights_normed = weights / torch.norm(weights)
        errors = ((network_output - target) ** 2) * weights_normed.unsqueeze(1)
        return torch.mean(errors[mask])


class Loss_pressure(_Loss):
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
        mask_obstacle = _prepare_mask_for_loss(
            network_output,
            node_type,
            [NodeType.OBSTACLE],
            selected_indexes,
        )
        errors_pressure_obstacle = ((network_output[:, 2] - target[:, 2]) ** 2)[
            mask_obstacle
        ]
        return torch.max(errors_pressure_obstacle)


class L2Loss_Coefficients(_Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def __name__(self):
        return "Cd et Cl"

    def forward(
        self,
        predicted_outputs: torch.Tensor,
        graph: Batch,
        node_type: torch.Tensor,
        masks: list[NodeType],
    ) -> torch.Tensor:
        """
        Computes L2 loss for nodes of specific types with a physic term.

        Args:
            target (torch.Tensor): The target values.
            predicted_outputs (torch.Tensor): The predicted values from the network.
            node_type (torch.Tensor): Tensor containing the type of each node.
            masks (list[NodeType]): List of NodeTypes to include in the loss calculation.
            selected_indexes (torch.Tensor, optional): Indexes of nodes to exclude from the loss calculation.

        Returns:
            torch.Tensor: The mean squared error for the specified node types.
        """
        total_loss = 0
        num_graphs = graph.num_graphs
        for graph_idx in range(num_graphs):
            graph_mask = graph.batch == graph_idx

            mask = (node_type == NodeType.OBSTACLE) & graph_mask
            pressure = predicted_outputs[mask, 2]
            pressure_gt = graph.y[mask, 2]

            cylinder_points = graph.pos[mask]
            center_cylinder = torch.mean(cylinder_points, axis=0)
            vectors_cylinder = cylinder_points - center_cylinder
            normals = vectors_cylinder / torch.norm(
                vectors_cylinder, dim=1, keepdim=True
            )
            ds_avg = torch.norm(cylinder_points[0] - cylinder_points[1])

            force = -torch.unsqueeze(pressure, 1) * normals * ds_avg
            force = torch.sum(force, dim=0)
            force_true = -torch.unsqueeze(pressure_gt, 1) * normals * ds_avg
            force_true = torch.sum(force_true, dim=0)

            CL, CD = force[0], force[1]
            CL_true, CD_true = force_true[0], force_true[1]
            total_loss += torch.abs(CL - CL_true) + torch.abs(CD - CD_true)
        if num_graphs > 0:
            total_loss /= num_graphs
        return total_loss


class L2Loss_vorticity(_Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def __name__(self):
        return "Vorticity"

    def forward(
        self,
        predicted_outputs: torch.Tensor,
        graph: Batch,
        node_type: torch.Tensor,
        masks: list[NodeType],
    ) -> torch.Tensor:
        """
        Computes L2 loss for nodes of specific types with a physic term.

        Args:
            target (torch.Tensor): The target values.
            predicted_outputs (torch.Tensor): The predicted values from the network.
            node_type (torch.Tensor): Tensor containing the type of each node.
            masks (list[NodeType]): List of NodeTypes to include in the loss calculation.
            selected_indexes (torch.Tensor, optional): Indexes of nodes to exclude from the loss calculation.

        Returns:
            torch.Tensor: The mean squared error for the specified node types.
        """
        mask = _prepare_mask_for_loss(predicted_outputs, node_type, masks)

        velocity = predicted_outputs[:, :2]
        velocity_gt = graph.y[:, :2]

        grad_velocity = compute_gradient(
            graph,
            field=velocity,
            method="finite_diff",
            device=predicted_outputs.device,
        )
        vorticity = grad_velocity[:, 1, 0] - grad_velocity[:, 0, 1]

        grad_velocity_gt = compute_gradient(
            graph,
            field=velocity_gt,
            method="finite_diff",
            device=predicted_outputs.device,
        )
        vorticity_gt = grad_velocity_gt[:, 1, 0] - grad_velocity_gt[:, 0, 1]
        vorticity_loss = torch.mean((vorticity[mask] - vorticity_gt[mask]) ** 2)

        return vorticity_loss


class L2Loss_kineticenergy(_Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def __name__(self):
        return "kinetic energy"

    def forward(
        self,
        predicted_outputs: torch.Tensor,
        graph: Batch,
        node_type: torch.Tensor,
        masks: list[NodeType],
    ) -> torch.Tensor:
        """
        Computes L2 loss for nodes of specific types with a physic term.

        Args:
            target (torch.Tensor): The target values.
            predicted_outputs (torch.Tensor): The predicted values from the network.
            node_type (torch.Tensor): Tensor containing the type of each node.
            masks (list[NodeType]): List of NodeTypes to include in the loss calculation.
            selected_indexes (torch.Tensor, optional): Indexes of nodes to exclude from the loss calculation.

        Returns:
            torch.Tensor: The mean squared error for the specified node types.
        """
        total_loss = 0
        num_graphs = graph.num_graphs
        mask = _prepare_mask_for_loss(predicted_outputs, node_type, masks)
        for graph_idx in range(num_graphs):
            graph_mask = graph.batch == graph_idx
            mask = mask & graph_mask
            energy = torch.sum(predicted_outputs[mask, :2] ** 2)
            energy_gt = torch.sum(graph.y[mask, :2] ** 2)

            total_loss += torch.abs(energy - energy_gt)
        if num_graphs > 0:
            total_loss /= num_graphs
        return total_loss
