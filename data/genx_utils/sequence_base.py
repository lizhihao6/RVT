from pathlib import Path
from typing import Any, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torchdata.datapipes.map import MapDataPipe

from data.genx_utils.labels import ObjectLabelFactory, ObjectLabels
from data.utils.spatial import get_original_hw
from data.utils.types import DatasetType
from utils.timers import TimerDummy as Timer


def get_event_representation_dir(path: Path,
                                 ev_representation_name: str) -> Path:
    ev_repr_dir = path / 'event_representations_v2' / ev_representation_name
    assert ev_repr_dir.is_dir(), f'{ev_repr_dir}'
    return ev_repr_dir


def get_objframe_idx_2_repr_idx(path: Path,
                                ev_representation_name: str) -> np.ndarray:
    ev_repr_dir = get_event_representation_dir(
        path=path, ev_representation_name=ev_representation_name)
    objframe_idx_2_repr_idx = np.load(
        str(ev_repr_dir / 'objframe_idx_2_repr_idx.npy'))
    return objframe_idx_2_repr_idx


class SequenceBase(MapDataPipe):
    """
    Structure example of a sequence:
    .
    ├── event_representations_v2
    │ └── ev_representation_name
    │     ├── event_representations.h5
    │     ├── objframe_idx_2_repr_idx.npy
    │     └── timestamps_us.npy
    └── labels_v2
        ├── labels.npz
        └── timestamps_us.npy
    """

    def __init__(self, path: Path, ev_representation_name: str,
                 sequence_length: int, dataset_type: DatasetType,
                 downsample_by_factor_2: bool, only_load_end_labels: bool):
        assert sequence_length >= 1
        assert path.is_dir()
        assert dataset_type in {DatasetType.GEN1, DatasetType.GEN4
                                }, f'{dataset_type} not implemented'

        self.only_load_end_labels = only_load_end_labels

        ev_repr_dir = get_event_representation_dir(
            path=path, ev_representation_name=ev_representation_name)

        labels_dir = path / 'labels_v2'
        assert labels_dir.is_dir()

        height, width = get_original_hw(dataset_type)
        self.scale = np.array([[width, height]], dtype=np.float32)
        self.seq_len = sequence_length

        ds_factor_str = '_ds2_nearest' if downsample_by_factor_2 else ''
        self.ev_repr_file = ev_repr_dir / f'event_representations{ds_factor_str}.h5'
        assert self.ev_repr_file.exists(), f'{str(self.ev_repr_file)=}'

        with Timer(timer_name='prepare labels'):
            label_data = np.load(str(labels_dir / 'labels.npz'))
            objframe_idx_2_label_idx = label_data['objframe_idx_2_label_idx']
            labels = label_data['labels']
            label_factory = ObjectLabelFactory.from_structured_array(
                object_labels=labels,
                objframe_idx_2_label_idx=objframe_idx_2_label_idx,
                input_size_hw=(height, width),
                downsample_factor=2 if downsample_by_factor_2 else None)
            self.label_factory = label_factory

        with Timer(timer_name='construct repr_idx_2_objframe_idx'):
            self.repr_idx_2_objframe_idx = dict(
                zip(self.objframe_idx_2_repr_idx,
                    range(len(self.objframe_idx_2_repr_idx))))

    def _get_labels_from_repr_idx(self,
                                  repr_idx: int) -> Optional[ObjectLabels]:
        objframe_idx = self.repr_idx_2_objframe_idx.get(repr_idx, None)
        return None if objframe_idx is None else self.label_factory[
            objframe_idx]

    def _get_event_repr_torch(
            self, start_idx: int,
            end_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        assert end_idx > start_idx
        with h5py.File(str(self.ev_repr_file), 'r') as h5f:
            indices = h5f['indices'][start_idx:end_idx][:, 1:]
            ev_start_idx = h5f['indices'][start_idx][0]
            ev_end_idx = h5f['indices'][end_idx][1]
            xy = h5f['pos'][start_idx:end_idx]
            ts = h5f['time'][ev_start_idx:ev_end_idx]
            p = h5f['events'][ev_start_idx:ev_end_idx]

        num_events = ev_end_idx - ev_start_idx
        events = np.zeros([num_events, 4], dtype=np.float32)
        events[:, :2] = xy / self.scale
        events[:, 2] = ts
        events[:, 3] = p

        offsets = indices[:, 1] - indices[0, 0]

        events, offsets = torch.from_numpy(events), torch.from_numpy(offsets)
        return events, offsets

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError
