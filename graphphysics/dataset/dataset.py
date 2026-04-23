import json
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch_geometric.data import Data, Dataset


class BaseDataset(Dataset, ABC):
    def __init__(
        self,
        meta_path: str,
        targets: list[str],
        preprocessing: Optional[Callable[[Data], Data]] = None,
        masking_ratio: Optional[float] = None,
        add_edge_features: bool = True,
        use_previous_data: bool = False,
        world_pos_parameters: Optional[dict] = None,
    ):
        with open(meta_path, "r") as fp:
            meta = json.load(fp)

        # Dataset preprocessing stays on CPU to let DataLoader handle device transfers lazily.
        self.device = torch.device("cpu")

        self.meta: Dict[str, Any] = meta

        # Check targets are properly defined
        if targets is None or len(targets) == 0:
            raise ValueError("At least one target must be specified.")
        for target in targets:
            if target not in self.meta["features"]:
                raise ValueError(f"Target {target} not found in available fields.")
            if self.meta["features"][target]["type"] != "dynamic":
                raise ValueError(f"Target {target} is not a dynamic field.")
        self.targets = targets

        self.trajectory_length: int = self.meta["trajectory_length"]
        self.num_trajectories: Optional[int] = None
        self.num_samples_per_traj = self.trajectory_length - 1 - int(use_previous_data)

        self.preprocessing = preprocessing
        self.masking_ratio = masking_ratio
        self.add_edge_features = add_edge_features
        self.use_previous_data = use_previous_data

        self.world_pos_index_start = None
        self.world_pos_index_end = None
        if world_pos_parameters is not None:
            self.world_pos_index_start = world_pos_parameters.get(
                "world_pos_index_start"
            )
            self.world_pos_index_end = world_pos_parameters.get("world_pos_index_end")

    @property
    @abstractmethod
    def size_dataset(self) -> int:
        """Should return the number of trajectories in the dataset."""

    def get_traj_frame(self, index: int) -> Tuple[int, int]:
        """Calculate the trajectory and frame number based on the given index.

        This method divides the dataset into trajectories and frames. It calculates
        which trajectory and frame the given index corresponds to, considering the
        length of each trajectory.

        Parameters:
            index (int): The index of the item in the dataset.

        Returns:
            Tuple[int, int]: A tuple containing the trajectory number and the frame number within that trajectory.
        """

        traj = index // self.num_samples_per_traj
        frame = index % self.num_samples_per_traj + int(self.use_previous_data)
        return traj, frame

    def __len__(self) -> int:
        return self.size_dataset * (self.num_samples_per_traj)

    @abstractmethod
    def __getitem__(self, index: int) -> Data:
        """Abstract method to retrieve a data sample."""
        raise NotImplementedError

    def _apply_preprocessing(self, graph: Data) -> Data:
        """Applies preprocessing transforms to the graph if provided.

        Parameters:
            graph (Data): The input graph data.

        Returns:
            Data: The preprocessed graph data.
        """
        if self.preprocessing is not None:
            graph = self.preprocessing(graph)
        return graph

    def _may_remove_edges_attr(self, graph: Data) -> Data:
        """Removes edge attributes if they are not needed.

        Parameters:
            graph (Data): The input graph data.

        Returns:
            Data: The graph with edge attributes removed if not needed.
        """
        if not self.add_edge_features:
            graph.edge_attr = None
        return graph
