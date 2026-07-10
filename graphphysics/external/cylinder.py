import torch
from torch_geometric.data import Data
from graphphysics.utils.nodetype import NodeType

# device = "cuda" if torch.cuda.is_available() else "cpu"


def signed_distance_function(
    pos: torch.Tensor, node_type: torch.Tensor
) -> torch.Tensor:
    # for a cylinder
    points_obstacle_mask = node_type == NodeType.OBSTACLE
    points_cylinder = pos[points_obstacle_mask]
    centre = torch.mean(points_cylinder, axis=0)
    R = torch.linalg.norm(points_cylinder[0] - centre)

    return torch.linalg.norm(pos - centre, dim=1) - R


def build_features(graph: Data) -> Data:
    node_type = graph.x[:, 0]
    timestep = graph.x[:, 4]
    pressure = graph.x[:, 3]
    current_velocity = graph.x[:, 1:3]
    id = torch.ones_like(pressure) * float(graph.id)
    sdf = signed_distance_function(graph.pos, node_type)
    if "previous_data" in graph:
        previous_velocity = torch.tensor(graph.previous_data["velocity"])
        acceleration = current_velocity - previous_velocity
        last_pressure = torch.tensor(graph.previous_data["pressure"])

        graph.x = torch.cat(
            (
                current_velocity,
                # pressure.unsqueeze(1),
                timestep.unsqueeze(1),
                graph.pos,
                sdf.unsqueeze(1),
                acceleration,
                last_pressure,
                id.unsqueeze(1),
                node_type.unsqueeze(1),
            ),
            dim=1,
        )
    else:
        graph.x = torch.cat(
            (
                current_velocity,
                # TAG: if only pressure in output
                # pressure.unsqueeze(1),
                timestep.unsqueeze(1),
                graph.pos,
                sdf.unsqueeze(1),
                id.unsqueeze(1),
                node_type.unsqueeze(1),
            ),
            dim=1,
        )

    return graph
