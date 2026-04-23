from typing import Dict, List, Optional, Union

import meshio
import numpy as np
import torch
from meshio import Mesh
from torch_geometric.data import Data


def meshdata_to_graph(
    points: np.ndarray,
    cells: np.ndarray,
    point_data: Optional[Dict[str, np.ndarray]],
    time: Union[int, float] = 1,
    target: Optional[np.ndarray] = None,
    return_only_node_features: bool = False,
    id: Optional[str] = None,
    next_data: Optional[np.ndarray] = None,
) -> Data:
    """Converts mesh data into a PyTorch Geometric Data object.

    Parameters:
        points (np.ndarray): The coordinates of the mesh points.
        cells (np.ndarray): The connectivity of the mesh (how points form cells); either triangles or tetrahedras.
        point_data (Dict[str, np.ndarray]): A dictionary of point-associated data.
        time (int or float): A scalar value representing the time step.
        target (np.ndarray, optional): An optional target tensor.
        return_only_node_features (bool): Whether to return only node features.
        id (str, optional): An optional mesh id to link graph to original dataset mesh.

    Returns:
        Data: A PyTorch Geometric Data object representing the mesh.
    """
    # Combine all point data into a single array
    if point_data is not None:
        if any(data.ndim > 1 for data in point_data.values()):
            # if any(data.shape[1] > 1 for data in point_data.values()):

            node_features = np.hstack(
                [data for data in point_data.values()]
                + [np.full((len(points),), time).reshape((-1, 1))]
            )
            node_features = torch.tensor(node_features, dtype=torch.float32)
        else:
            node_features = np.vstack(
                [data for data in point_data.values()] + [np.full((len(points),), time)]
            ).T
            node_features = torch.tensor(node_features, dtype=torch.float32)
    else:
        node_features = torch.zeros((len(points), 1), dtype=torch.float32)

    if return_only_node_features:
        return node_features

    # Convert target to tensor if provided
    if target is not None:
        if any(data.ndim > 1 for data in target.values()):
            # if any(data.shape[1] > 1 for data in target.values()):
            target_features = np.hstack([data for data in target.values()])
            target_features = torch.tensor(target_features, dtype=torch.float32)
        else:
            target_features = np.vstack([data for data in target.values()]).T
            target_features = torch.tensor(target_features, dtype=torch.float32)
    else:
        target_features = None

    # Get tetrahedras and triangles from cells
    tetra = None
    cells = cells.T
    cells = torch.tensor(cells)
    if cells.shape[0] == 4:
        tetra = cells
        face = torch.cat(
            [
                cells[0:3],
                cells[1:4],
                torch.stack([cells[2], cells[3], cells[0]], dim=0),
                torch.stack([cells[3], cells[0], cells[1]], dim=0),
            ],
            dim=1,
        )
    if cells.shape[0] == 3:
        face = cells

    return Data(
        x=node_features,
        face=face,
        tetra=tetra,
        y=target_features,
        pos=torch.tensor(points, dtype=torch.float32),
        id=id,
        next_data=next_data,
    )


def mesh_to_graph(
    mesh: Mesh,
    time: Union[int, float] = 1,
    target_mesh: Optional[Mesh] = None,
    target_fields: Optional[List[str]] = None,
) -> Data:
    """Converts mesh and optional target mesh data into a PyTorch Geometric Data object.

    Parameters:
        mesh (Mesh): A Mesh object containing the mesh data.
        time (int or float): A scalar value representing the time step.
        target_mesh (Mesh, optional): An optional Mesh object containing target data.
        target_fields (List[str], optional): Fields from the target_mesh to be used as the target data.

    Returns:
        Data: A PyTorch Geometric Data object representing the mesh with optional target data.
    """
    # Prepare target data if a target mesh is provided
    target = None
    if target_mesh is not None and target_fields:
        target_data = [target_mesh.point_data[field] for field in target_fields]
        target = np.hstack(target_data)

    # Extract cells of type 'triangle' and 'quad'
    cells = np.vstack(
        [v for k, v in mesh.cells_dict.items() if k in ["triangle", "quad"]]
    )

    return meshdata_to_graph(
        points=mesh.points,
        cells=cells,
        point_data=mesh.point_data,
        time=time,
        target=target,
    )


def torch_graph_to_mesh(graph: Data, node_features_mapping: dict[str, int]) -> Mesh:
    """Converts a PyTorch Geometric graph to a meshio Mesh object.

    This function takes a graph represented in PyTorch Geometric's `Data` format and
    converts it into a meshio Mesh object. It extracts the positions, faces, and specified
    node features from the graph and constructs a Mesh object.

    Parameters:
        - graph (Data): The graph to convert, represented as a PyTorch Geometric `Data` object.
                      It should contain node positions in `graph.pos` and connectivity
                      (faces) in `graph.face`.
        - node_features_mapping (dict[str, int]): A dictionary mapping feature names to their
                                                corresponding column indices in `graph.x`.
                                                This allows selective inclusion of node features
                                                in the resulting Mesh object's point data.

    Returns:
        - Mesh: A meshio Mesh object containing the graph's geometric and feature data.

    Note:
    The function detaches tensors and moves them to CPU before converting to NumPy arrays,
    ensuring compatibility with meshio and avoiding GPU memory issues.
    """
    point_data = {
        f: graph.x[:, indx].detach().cpu().numpy()
        for f, indx in node_features_mapping.items()
    }

    cells = graph.face.detach().cpu().numpy()
    if graph.pos.shape[1] == 2:
        extra_shape = 3
        cells_type = "triangle"
    elif graph.pos.shape[1] == 3:
        extra_shape = 4
        cells_type = "tetra"
    else:
        raise ValueError(
            f"Graph Pos does not have the right shape. Expected shape[1] to be 2 or 3. Got {graph.pos.shape[1]}"
        )

    if cells.shape[-1] != extra_shape:
        cells = cells.T

    return meshio.Mesh(
        graph.pos.detach().cpu().numpy(),
        [(cells_type, cells)],
        point_data=point_data,
    )
