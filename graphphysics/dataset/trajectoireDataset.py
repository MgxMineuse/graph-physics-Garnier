import torch
import os
from typing import Callable, List, Optional, Sequence, Union, Any
import meshio
import numpy as np

from torch_geometric.data import Dataset, Data
from torch_geometric.data.data import BaseData
from torch_geometric.data.datapipes import DatasetAdapter
from graphphysics.dataset.xdmf_dataset import XDMFDataset
from graphphysics.utils.torch_graph import meshdata_to_graph


class TrajectoryXDMFDataset(XDMFDataset):
    def __init__(
        self,
        xdmf_folder: str,
        meta_path: str,
        targets: list[str] = None,
        preprocessing: Optional[Callable[[Data], Data]] = None,
        masking_ratio: Optional[float] = None,
        add_edge_features: bool = True,
        use_previous_data: bool = False,
        switch_to_val: bool = False,
        seq_length: Optional[int] = None,  # Si None, charge toute la trajectoire
        chunk_size: int = 1,
    ):
        # Désactive le random_prev/next pour avoir des séquences fixes
        super().__init__(
            xdmf_folder=xdmf_folder,
            meta_path=meta_path,
            targets=targets,
            preprocessing=preprocessing,
            masking_ratio=masking_ratio,
            add_edge_features=add_edge_features,
            use_previous_data=use_previous_data,
            switch_to_val=switch_to_val,
            random_prev=1,  # Désactive le random pour avoir des séquences fixes
            random_next=1,
        )
        self.seq_length = seq_length
        self.chunk_size = chunk_size
        self.num_samples_per_traj = (
            self.trajectory_length - 1 - int(self.use_previous_data)
        ) // self.chunk_size

    def get_traj_frame(self, index):
        traj = index // self.num_samples_per_traj
        frame_start = index % self.num_samples_per_traj + int(self.use_previous_data)
        frame_end = min(frame_start + self.chunk_size, self.trajectory_length - 1)
        return traj, frame_start, frame_end

    def __getitem__(self, index: int) -> List[Data]:
        """Retourne TOUTES les frames d'une trajectoire sous forme de liste de graphes.

        Args:
            traj_index: Index de la trajectoire (0 à len(self)-1)

        Returns:
            Liste de Data (1 par timestep), triée par ordre temporel.
        """
        traj_index, frame_start, frame_end = self.get_traj_frame(index)
        xdmf_file = self.file_paths[traj_index]
        mesh_id = os.path.splitext(os.path.basename(xdmf_file))[0].rsplit("_", 1)[-1]

        with meshio.xdmf.TimeSeriesReader(xdmf_file) as reader:
            num_steps = reader.num_steps

            # Limiter à seq_length si spécifié
            if self.seq_length is not None:
                num_steps = min(num_steps, self.seq_length)

            points, cells = reader.read_points_cells()
            mesh = meshio.Mesh(points, cells)

            # Gérer les types de cellules
            if "triangle" in mesh.cells_dict:
                cells = mesh.cells_dict["triangle"]
            elif "tetra" in mesh.cells_dict:
                cells = torch.tensor(mesh.cells_dict["tetra"], dtype=torch.long)
            else:
                raise ValueError(
                    "Unsupported cell type. Only 'triangle' and 'tetra' cells are supported."
                )

            graphs = []
            for frame in range(frame_start, frame_end):
                try:
                    time, point_data, _ = reader.read_data(frame)
                    _, target_point_data, _ = reader.read_data(
                        frame + 1
                    )  # Target = frame+1
                except ValueError:
                    # Fallback si frame+1 dépasse
                    time, point_data, _, _ = reader.read_data(frame)
                    _, target_point_data, _, _ = reader.read_data(frame + 1)

                mesh.point_data = point_data

                # Traiter point_data et target_data
                point_data_dict = {
                    k: np.array(mesh.point_data[k]).astype(
                        self.meta["features"][k]["dtype"]
                    )
                    for k in self.meta["features"]
                    if k in mesh.point_data.keys()
                }

                target_data = {}
                next_data = {}
                for k in self.meta["features"]:
                    if k in self.targets:
                        target_data[k] = np.array(target_point_data[k]).astype(
                            self.meta["features"][k]["dtype"]
                        )
                    else:
                        if (
                            k in target_point_data.keys()
                            and self.meta["features"][k]["type"] == "dynamic"
                        ):
                            next_data[k] = np.array(target_point_data[k]).astype(
                                self.meta["features"][k]["dtype"]
                            )

                # Reshape les arrays 1D
                def _reshape_array(a: dict):
                    for k, v in a.items():
                        if v.ndim == 1:
                            a[k] = v.reshape(-1, 1)

                _reshape_array(point_data_dict)
                _reshape_array(target_data)

                # Créer le graphe pour ce timestep
                graph = meshdata_to_graph(
                    points=points.astype(np.float32),
                    cells=cells,
                    point_data=point_data_dict,
                    time=time,
                    target=target_data,
                    id=mesh_id,
                    next_data=next_data,
                )
                graph.target_dt = self.dt

                if self.use_previous_data:
                    # Pour le premier timestep, pas de previous_data
                    if frame > 0:
                        _, previous_data, _ = reader.read_data(frame - 1)
                        previous = {
                            k: np.array(previous_data[k]).astype(
                                self.meta["features"][k]["dtype"]
                            )
                            for k in self.meta["features"]
                            if k in previous_data.keys()
                            and self.meta["features"][k]["type"] == "dynamic"
                        }
                        _reshape_array(previous)
                        graph.previous_data = previous
                        graph.previous_dt = -self.dt
                    else:
                        # Pour frame=0, previous_data = point_data
                        graph.previous_data = {
                            k: np.zeros_like(point_data_dict[k])
                            for k in point_data_dict
                        }
                        graph.previous_dt = -self.dt

                graph = self._apply_preprocessing(graph)
                graph = self._may_remove_edges_attr(graph)
                graph.edge_index = (
                    graph.edge_index.long() if graph.edge_index is not None else None
                )
                graph.traj_index = traj_index
                graph.frame_index = frame  # Ajout pour débogage

                if hasattr(graph, "next_data"):
                    del graph.next_data
                if hasattr(graph, "previous_data"):
                    del graph.previous_data

                graphs.append(graph)

        return graphs


class TrajectoryCollater:
    def __init__(
        self,
        dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
    ):
        self.dataset = dataset
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch: List[Any]) -> Any:
        if not batch:
            return batch

        elem = batch[0]
        if isinstance(elem, list) and len(elem) > 0 and isinstance(elem[0], BaseData):
            processed_batch = []
            for traj in batch:  # traj = [graph_0, graph_1, ...] (1 trajectoire)
                processed_traj = []
                for graph in traj:
                    new_graph = graph.clone()
                    for key in self.exclude_keys:
                        if hasattr(new_graph, key):
                            delattr(new_graph, key)
                    processed_traj.append(new_graph)
                processed_batch.append(processed_traj)
            return processed_batch

        raise TypeError(f"DataLoader found invalid type: '{type(elem)}'")


class TrajectoryDataLoader(torch.utils.data.DataLoader):
    r"""A data loader which merges data objects from a
    :class:`torch_geometric.data.Dataset` to a mini-batch.
    Data objects can be either of type :class:`~torch_geometric.data.Data` or
    :class:`~torch_geometric.data.HeteroData`.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (List[str], optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`None`)
        exclude_keys (List[str], optional): Will exclude each key in the
            list. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`.
    """

    def __init__(
        self,
        dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        # Remove for PyTorch Lightning:
        kwargs.pop("collate_fn", None)

        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=TrajectoryCollater(dataset, follow_batch, exclude_keys),
            **kwargs,
        )
