from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import torch

from data.genx_utils.collate_from_pytorch import (collate,
                                                  default_collate_fn_map)
from data.genx_utils.labels import ObjectLabels, SparselyBatchedObjectLabels
from data.utils.types import DataType


def collate_object_labels(batch,
                          *,
                          collate_fn_map: Optional[Dict[Union[Type,
                                                              Tuple[Type,
                                                                    ...]],
                                                        Callable]] = None):
    return batch


def collate_sparsely_batched_object_labels(
    batch,
    *,
    collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]],
                                  Callable]] = None):
    return SparselyBatchedObjectLabels.transpose_list(batch)


custom_collate_fn_map = deepcopy(default_collate_fn_map)
custom_collate_fn_map[ObjectLabels] = collate_object_labels
custom_collate_fn_map[
    SparselyBatchedObjectLabels] = collate_sparsely_batched_object_labels


def custom_collate(batch: Any):
    # collect ev_repr and offsets from batch
    ev_repr = [sample.pop(DataType.EV_REPR) for sample in batch]
    # todo convert split 
    print(batch[0][DataType.OFFSETS].shape)
    print(len(batch))
    exit()
    offsets = [sample.pop(DataType.OFFSETS) for sample in batch]
    batch = collate(batch, collate_fn_map=custom_collate_fn_map)
    batch[DataType.EV_REPR] = ev_repr
    batch[DataType.OFFSETS] = offsets
    return batch


def custom_collate_rnd(batch: Any):
    samples = batch
    # NOTE: We do not really need the worker id for map style datasets (rnd) but we still provide the id for consistency
    worker_info = torch.utils.data.get_worker_info()
    local_worker_id = 0 if worker_info is None else worker_info.id
    return {
        'data': custom_collate(samples),
        'worker_id': local_worker_id,
    }


def custom_collate_streaming(batch: Any):
    """We assume that we receive a batch collected by a worker of our streaming
    datapipe."""
    samples = batch[0]
    worker_id = batch[1]
    assert isinstance(worker_id, int)
    return {
        'data': custom_collate(samples),
        'worker_id': worker_id,
    }
