import torch
from torch_geometric.data import Data

device = "cuda" if torch.cuda.is_available() else "cpu"


def build_features(graph: Data) -> Data:
    node_type = graph.x[:, 0]
    timestep = graph.x[:, 4]
    pressure = graph.x[:, 3]
    current_velocity = graph.x[:, 1:3]
    if "previous_data" in graph:
        previous_velocity = torch.tensor(graph.previous_data["velocity"], device=device)
        acceleration = current_velocity - previous_velocity
        last_pressure = torch.tensor(graph.previous_data["pressure"], device=device)

        graph.x = torch.cat(
            (
                current_velocity,
                pressure.unsqueeze(1),
                timestep.unsqueeze(1),
                graph.pos,
                acceleration,
                last_pressure,
                node_type.to(device).unsqueeze(1),

            ),
            dim=1,
        )
    else:
        graph.x = torch.cat(
            (
                current_velocity,
                pressure.unsqueeze(1),
                timestep.unsqueeze(1),
                graph.pos,
                node_type.to(device).unsqueeze(1),
            ),
            dim=1,
        )

    return graph
