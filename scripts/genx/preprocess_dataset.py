import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import argparse
import shutil
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from functools import partial
from multiprocessing import get_context
from pathlib import Path

sys.path.append('../..')
import weakref
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
from numba import jit
from omegaconf import MISSING, DictConfig, OmegaConf
from tqdm import tqdm

from data.utils.representations import EventSurface, RepresentationBase


class DataKeys(Enum):
    InNPY = auto()
    InH5 = auto()
    OutLabelDir = auto()
    OutEvReprDir = auto()
    SplitType = auto()


class SplitType(Enum):
    TRAIN = auto()
    VAL = auto()
    TEST = auto()


split_name_2_type = {
    'train': SplitType.TRAIN,
    'val': SplitType.VAL,
    'test': SplitType.TEST,
}

dataset_2_height = {'gen1': 240, 'gen4': 720}
dataset_2_width = {'gen1': 304, 'gen4': 1280}

# The following sequences would be discarded because all the labels would be removed after filtering:
dirs_to_ignore = {
    'gen1': ('17-04-06_09-57-37_6344500000_6404500000',
             '17-04-13_19-17-27_976500000_1036500000',
             '17-04-06_15-14-36_1159500000_1219500000',
             '17-04-11_15-13-23_122500000_182500000'),
    'gen4': (),
}


class NoLabelsException(Exception):
    # Raised when no labels are present anymore in the sequence after filtering
    ...


class H5Writer:

    def __init__(self, outfile: Path):
        self.h5f = h5py.File(str(outfile), 'w')
        self._finalizer = weakref.finalize(self, self.close_callback, self.h5f)

        # create hdf5 datasets for pos
        maxshape = (None, 2)
        chunkshape = (1, 2)
        self.pos_key = 'pos'
        self.pos_dtype = np.uint16
        self.h5f.create_dataset(self.pos_key,
                                dtype=self.pos_dtype,
                                shape=chunkshape,
                                chunks=chunkshape,
                                maxshape=maxshape)

        # create hdf5 datasets for time
        maxshape = (None, )
        chunkshape = (1, )
        self.time_key = 'time'
        self.time_dtype = np.float32
        self.h5f.create_dataset(self.time_key,
                                dtype=self.time_dtype,
                                shape=chunkshape,
                                chunks=chunkshape,
                                maxshape=maxshape)

        # create hdf5 datasets for polarity
        maxshape = (None, )
        chunkshape = (1, )
        self.events_key = 'events'
        self.events_dtype = np.bool_
        self.h5f.create_dataset(self.events_key,
                                dtype=self.events_dtype,
                                shape=chunkshape,
                                chunks=chunkshape,
                                maxshape=maxshape)

        # create frame_idx, start and stop indices
        maxshape = (None, 3)
        chunkshape = (1, 3)
        self.indices_key = 'indices'
        self.indices_dtype = np.uint32
        self.h5f.create_dataset(self.indices_key,
                                dtype=self.indices_dtype,
                                shape=chunkshape,
                                chunks=chunkshape,
                                maxshape=maxshape)

        self.t_idx = 0
        self.counter = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._finalizer()

    @staticmethod
    def close_callback(h5f: h5py.File):
        h5f.close()

    def close(self):
        self.h5f.close()

    def add_data(self, xytp: np.ndarray, seq_idx: int):
        pos = xytp[:, :2].astype(self.pos_dtype)
        time = xytp[:, 2].astype(self.time_dtype)
        events = xytp[:, 3].astype(self.events_dtype)

        # normalize time
        min_time, max_time = time.min(), time.max()
        time = (time - min_time).astype(
            self.time_dtype) / (max_time - min_time)

        new_size = self.t_idx + xytp.shape[0]

        self.h5f[self.pos_key].resize(new_size, axis=0)
        self.h5f[self.pos_key][self.t_idx:new_size] = pos

        self.h5f[self.time_key].resize(new_size, axis=0)
        self.h5f[self.time_key][self.t_idx:new_size] = time

        self.h5f[self.events_key].resize(new_size, axis=0)
        self.h5f[self.events_key][self.t_idx:new_size] = events

        self.h5f[self.indices_key].resize(self.counter + 1, axis=0)
        self.h5f[self.indices_key][self.counter] = [
            seq_idx, self.t_idx, new_size
        ]

        self.t_idx = new_size
        self.counter += 1


class H5Reader:

    def __init__(self, h5_file: Path, dataset: str = 'gen4'):
        assert h5_file.exists()
        assert h5_file.suffix == '.h5'
        assert dataset in {'gen1', 'gen4'}

        self.h5f = h5py.File(str(h5_file), 'r')
        self._finalizer = weakref.finalize(self, self._close_callback,
                                           self.h5f)
        self.is_open = True

        try:
            self.height = self.h5f['events']['height'][()].item()
            self.width = self.h5f['events']['width'][()].item()
        except KeyError:
            self.height = dataset_2_height[dataset]
            self.width = dataset_2_width[dataset]

        self.all_times = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._finalizer()

    @staticmethod
    def _close_callback(h5f: h5py.File):
        h5f.close()

    def close(self):
        self.h5f.close()
        self.is_open = False

    def get_height_and_width(self) -> Tuple[int, int]:
        return self.height, self.width

    @property
    def time(self) -> np.ndarray:
        # We need to lazy load time because it is typically not sorted everywhere.
        # - Set timestamps of events such they are not decreasing.
        assert self.is_open
        if self.all_times is None:
            self.all_times = np.asarray(self.h5f['events']['t'])
            self._correct_time(self.all_times)
        return self.all_times

    @staticmethod
    @jit(nopython=True)
    def _correct_time(time_array: np.ndarray):
        assert time_array[0] >= 0
        time_last = 0
        for idx, time in enumerate(time_array):
            if time < time_last:
                time_array[idx] = time_last
            else:
                time_last = time

    def get_event_slice(self,
                        idx_start: int,
                        idx_end: int,
                        convert_2_torch: bool = True):
        assert self.is_open
        assert idx_end >= idx_start
        ev_data = self.h5f['events']
        x_array = np.asarray(ev_data['x'][idx_start:idx_end], dtype='int64')
        y_array = np.asarray(ev_data['y'][idx_start:idx_end], dtype='int64')
        p_array = np.asarray(ev_data['p'][idx_start:idx_end], dtype='int64')
        p_array = np.clip(p_array, a_min=0, a_max=None)
        t_array = np.asarray(self.time[idx_start:idx_end], dtype='int64')
        assert np.all(t_array[:-1] <= t_array[1:])
        ev_data = dict(
            x=x_array if not convert_2_torch else torch.from_numpy(x_array),
            y=y_array if not convert_2_torch else torch.from_numpy(y_array),
            p=p_array if not convert_2_torch else torch.from_numpy(p_array),
            t=t_array if not convert_2_torch else torch.from_numpy(t_array),
            height=self.height,
            width=self.width,
        )
        return ev_data


def prophesee_bbox_filter(labels: np.ndarray, dataset_type: str) -> np.ndarray:
    assert dataset_type in {'gen1', 'gen4'}

    # Default values taken from: https://github.com/prophesee-ai/prophesee-automotive-dataset-toolbox/blob/0393adea2bf22d833893c8cb1d986fcbe4e6f82d/src/psee_evaluator.py#L23-L24
    min_box_diag = 60 if dataset_type == 'gen4' else 30
    # Corrected values from supplementary mat from paper for min_box_side!
    min_box_side = 20 if dataset_type == 'gen4' else 10

    w_lbl = labels['w']
    h_lbl = labels['h']

    diag_ok = w_lbl**2 + h_lbl**2 >= min_box_diag**2
    side_ok = (w_lbl >= min_box_side) & (h_lbl >= min_box_side)
    keep = diag_ok & side_ok
    labels = labels[keep]
    return labels


def conservative_bbox_filter(labels: np.ndarray) -> np.ndarray:
    w_lbl = labels['w']
    h_lbl = labels['h']
    min_box_side = 5
    side_ok = (w_lbl >= min_box_side) & (h_lbl >= min_box_side)
    labels = labels[side_ok]
    return labels


def remove_faulty_huge_bbox_filter(labels: np.ndarray,
                                   dataset_type: str) -> np.ndarray:
    """There are some labels which span the frame horizontally without actually
    covering an object."""
    assert dataset_type in {'gen1', 'gen4'}
    w_lbl = labels['w']
    max_width = (9 * dataset_2_width[dataset_type]) // 10
    side_ok = (w_lbl <= max_width)
    labels = labels[side_ok]
    return labels


def crop_to_fov_filter(labels: np.ndarray, dataset_type: str) -> np.ndarray:
    assert dataset_type in {'gen1', 'gen4'}, f'{dataset_type=}'
    # In the gen1 and gen4 datasets the bounding box can be partially or completely outside the frame.
    # We fix this labeling error by cropping to the FOV.
    frame_height = dataset_2_height[dataset_type]
    frame_width = dataset_2_width[dataset_type]
    x_left = labels['x']
    y_top = labels['y']
    x_right = x_left + labels['w']
    y_bottom = y_top + labels['h']
    x_left_cropped = np.clip(x_left, a_min=0, a_max=frame_width - 1)
    y_top_cropped = np.clip(y_top, a_min=0, a_max=frame_height - 1)
    x_right_cropped = np.clip(x_right, a_min=0, a_max=frame_width - 1)
    y_bottom_cropped = np.clip(y_bottom, a_min=0, a_max=frame_height - 1)

    w_cropped = x_right_cropped - x_left_cropped
    assert np.all(w_cropped >= 0)
    h_cropped = y_bottom_cropped - y_top_cropped
    assert np.all(h_cropped >= 0)

    labels['x'] = x_left_cropped
    labels['y'] = y_top_cropped
    labels['w'] = w_cropped
    labels['h'] = h_cropped

    # Remove bboxes that have 0 height or width
    keep = (labels['w'] > 0) & (labels['h'] > 0)
    labels = labels[keep]
    return labels


def prophesee_remove_labels_filter_gen4(labels: np.ndarray) -> np.ndarray:
    # Original gen4 labels: pedestrian, two wheeler, car, truck, bus, traffic sign, traffic light
    # gen4 labels to keep: pedestrian, two wheeler, car
    # gen4 labels to remove: truck, bus, traffic sign, traffic light
    #
    # class_id in {0, 1, 2, 3, 4, 5, 6} in the order mentioned above
    keep = labels['class_id'] <= 2
    labels = labels[keep]
    return labels


def apply_filters(labels: np.ndarray,
                  split_type: SplitType,
                  filter_cfg: DictConfig,
                  dataset_type: str = 'gen1') -> np.ndarray:
    assert isinstance(dataset_type, str)
    if dataset_type == 'gen4':
        labels = prophesee_remove_labels_filter_gen4(labels=labels)
    labels = crop_to_fov_filter(labels=labels, dataset_type=dataset_type)
    if filter_cfg.apply_psee_bbox_filter:
        labels = prophesee_bbox_filter(labels=labels,
                                       dataset_type=dataset_type)
    else:
        labels = conservative_bbox_filter(labels=labels)
    if split_type == SplitType.TRAIN and filter_cfg.apply_faulty_bbox_filter:
        labels = remove_faulty_huge_bbox_filter(labels=labels,
                                                dataset_type=dataset_type)
    return labels


def get_base_delta_ts_for_labels_us(unique_label_ts_us: np.ndarray,
                                    dataset_type: str = 'gen1') -> int:
    if dataset_type == 'gen1':
        delta_t_us_4hz = 250000
        return delta_t_us_4hz
    assert dataset_type == 'gen4'
    diff_us = np.diff(unique_label_ts_us)
    median_diff_us = np.median(diff_us)

    hz = int(np.rint(10**6 / median_diff_us))
    assert hz in {30, 60}, f'{hz=} but should be either 30 or 60'

    delta_t_us_approx_10hz = int(6 * median_diff_us if hz == 60 else 3 *
                                 median_diff_us)
    return delta_t_us_approx_10hz


def save_labels(out_labels_dir: Path,
                labels_per_frame: List[np.ndarray],
                frame_timestamps_us: np.ndarray,
                paired_frame_mask: np.ndarray,
                match_if_exists: bool = True) -> None:
    assert len(labels_per_frame) == len(frame_timestamps_us)
    assert len(labels_per_frame) > 0
    labels_v2 = list()
    objframe_idx_2_label_idx = list()
    start_idx = 0
    _frame_timestamps_us = list()
    for frame_idx, (labels, timestamp) in enumerate(
            zip(labels_per_frame, frame_timestamps_us)):
        if paired_frame_mask[frame_idx] > 0:
            objframe_idx_2_label_idx.append(start_idx)
            labels_v2.append(labels)
            start_idx += len(labels)
            _frame_timestamps_us.append(timestamp)
    frame_timestamps_us = np.asarray(_frame_timestamps_us,
                                     dtype=frame_timestamps_us.dtype)
    assert len(labels_v2) == len(objframe_idx_2_label_idx)
    labels_v2 = np.concatenate(labels_v2)

    outfile_labels = out_labels_dir / 'labels.npz'
    if outfile_labels.exists() and match_if_exists:
        data_existing = np.load(str(outfile_labels))
        labels_existing = data_existing['labels']
        assert np.array_equal(labels_existing, labels_v2)
        oi_2_li_existing = data_existing['objframe_idx_2_label_idx']
        assert np.array_equal(oi_2_li_existing, objframe_idx_2_label_idx)
    else:
        np.savez(str(outfile_labels),
                 labels=labels_v2,
                 objframe_idx_2_label_idx=objframe_idx_2_label_idx)

    out_labels_ts_file = out_labels_dir / 'timestamps_us.npy'
    if out_labels_ts_file.exists() and match_if_exists:
        frame_timestamps_us_existing = np.load(str(out_labels_ts_file))
        assert np.array_equal(frame_timestamps_us_existing,
                              frame_timestamps_us)
    else:
        np.save(str(out_labels_ts_file), frame_timestamps_us)


def labels_and_ev_repr_timestamps(npy_file: Path, split_type: SplitType,
                                  filter_cfg: DictConfig, align_t_ms: int,
                                  ts_step_ev_repr_ms: int, dataset_type: str):
    assert npy_file.exists()
    assert npy_file.suffix == '.npy'
    ts_step_frame_ms = 100
    assert ts_step_frame_ms >= ts_step_ev_repr_ms
    assert ts_step_frame_ms % ts_step_ev_repr_ms == 0 and ts_step_ev_repr_ms > 0

    align_t_us = align_t_ms * 1000
    delta_t_us = ts_step_ev_repr_ms * 1000

    sequence_labels = np.load(str(npy_file))
    assert len(sequence_labels) > 0

    sequence_labels = apply_filters(labels=sequence_labels,
                                    split_type=split_type,
                                    filter_cfg=filter_cfg,
                                    dataset_type=dataset_type)
    if sequence_labels.size == 0:
        raise NoLabelsException

    unique_ts_us = np.unique(np.asarray(sequence_labels['t'], dtype='int64'))

    base_delta_ts_labels_us = get_base_delta_ts_for_labels_us(
        unique_label_ts_us=unique_ts_us, dataset_type=dataset_type)

    # We extract the first label at or after align_t_us to keep it as the reference for the label extraction.
    unique_ts_idx_first = np.searchsorted(unique_ts_us,
                                          align_t_us,
                                          side='left')

    # Extract "frame" timestamps from labels and prepare ev repr ts computation
    num_ev_reprs_between_frame_ts = list()
    frame_timestamps_us = [unique_ts_us[unique_ts_idx_first]]
    for unique_ts_idx in range(unique_ts_idx_first + 1, len(unique_ts_us)):
        reference_time = frame_timestamps_us[-1]
        ts = unique_ts_us[unique_ts_idx]
        diff_to_ref = ts - reference_time
        base_delta_count = round(diff_to_ref / base_delta_ts_labels_us)
        diff_to_ref_rounded = base_delta_count * base_delta_ts_labels_us
        if np.abs(diff_to_ref - diff_to_ref_rounded) <= 2000:
            assert base_delta_count > 0
            # We accept up to 2 millisecond of jitter
            frame_timestamps_us.append(ts)
            num_ev_reprs_between_frame_ts.append(
                base_delta_count * (ts_step_frame_ms // ts_step_ev_repr_ms))
    frame_timestamps_us = np.asarray(frame_timestamps_us, dtype='int64')
    assert len(frame_timestamps_us) > 0, f'{npy_file=}'

    start_indices_per_label = np.searchsorted(sequence_labels['t'],
                                              frame_timestamps_us,
                                              side='left')
    end_indices_per_label = np.searchsorted(sequence_labels['t'],
                                            frame_timestamps_us,
                                            side='right')

    # Create labels per "frame"
    labels_per_frame = list()
    for idx_start, idx_end in zip(start_indices_per_label,
                                  end_indices_per_label):
        labels = sequence_labels[idx_start:idx_end]
        label_time_us = labels['t'][0]
        assert np.all(labels['t'] == label_time_us)
        labels_per_frame.append(labels)

    if len(frame_timestamps_us) > 1:
        assert np.diff(frame_timestamps_us).min(
        ) > 98000, f'{np.diff(frame_timestamps_us).min()=}'

    # Event repr timestamps generation
    ev_repr_timestamps_us_end = list(
        reversed(range(frame_timestamps_us[0], 0, -delta_t_us)))[1:-1]
    assert len(num_ev_reprs_between_frame_ts) == len(
        frame_timestamps_us
    ) - 1, f'{len(num_ev_reprs_between_frame_ts)=}, {len(frame_timestamps_us)=}'
    for idx, (num_ev_repr_between, frame_ts_us_start,
              frame_ts_us_end) in enumerate(
                  zip(num_ev_reprs_between_frame_ts, frame_timestamps_us[:-1],
                      frame_timestamps_us[1:])):
        new_edge_timestamps = np.asarray(np.linspace(frame_ts_us_start,
                                                     frame_ts_us_end,
                                                     num_ev_repr_between + 1),
                                         dtype='int64').tolist()
        is_last_iter = idx == len(num_ev_reprs_between_frame_ts) - 1
        if not is_last_iter:
            new_edge_timestamps = new_edge_timestamps[:-1]
        ev_repr_timestamps_us_end.extend(new_edge_timestamps)
    if len(frame_timestamps_us) == 1:
        # special case not handled in above for loop (no iter in this case)
        # yes, it's hacky ...
        ev_repr_timestamps_us_end.append(frame_timestamps_us[0])
    ev_repr_timestamps_us_end = np.asarray(ev_repr_timestamps_us_end,
                                           dtype='int64')

    frameidx_2_repridx = np.searchsorted(ev_repr_timestamps_us_end,
                                         frame_timestamps_us,
                                         side='left')
    assert len(frameidx_2_repridx) == len(frame_timestamps_us)

    # Some sanity checks:
    assert len(labels_per_frame) == len(frame_timestamps_us)
    assert len(frame_timestamps_us) == len(frameidx_2_repridx)
    for label, frame_ts_us, repr_idx in zip(labels_per_frame,
                                            frame_timestamps_us,
                                            frameidx_2_repridx):
        assert label['t'][0] == frame_ts_us
        assert frame_ts_us == ev_repr_timestamps_us_end[repr_idx]

    return labels_per_frame, frame_timestamps_us, ev_repr_timestamps_us_end, frameidx_2_repridx


def write_event_data(in_h5_file: Path, ev_out_dir: Path, dataset: str,
                     event_representation: RepresentationBase,
                     ev_repr_num_events: Optional[int],
                     ev_repr_delta_ts_ms: Optional[int],
                     ev_repr_timestamps_us: np.ndarray, downsample_by_2: bool,
                     frameidx2repridx: np.ndarray) -> np.ndarray:
    _frameidx2repridx = write_event_representations(
        in_h5_file=in_h5_file,
        ev_out_dir=ev_out_dir,
        dataset=dataset,
        event_representation=event_representation,
        ev_repr_num_events=ev_repr_num_events,
        ev_repr_delta_ts_ms=ev_repr_delta_ts_ms,
        ev_repr_timestamps_us=ev_repr_timestamps_us,
        frameidx2repridx=frameidx2repridx,
        downsample_by_2=downsample_by_2,
        overwrite_if_exists=False)

    # regenerate frameidx2repridx
    frameidx2repridx = _frameidx2repridx[_frameidx2repridx > 0]
    frameidx2repridx = np.cumsum(
        frameidx2repridx) - 1  # -1 because we want including last index

    # save frameidx2repridx
    frameidx2repridx_file = ev_out_dir / 'objframe_idx_2_repr_idx.npy'
    if frameidx2repridx_file.exists():
        frameidx2repridx_loaded = np.load(str(frameidx2repridx_file))
        assert np.array_equal(frameidx2repridx_loaded, frameidx2repridx)
    else:
        np.save(str(frameidx2repridx_file), frameidx2repridx)

    # save timestamps
    timestamps_file = ev_out_dir / 'timestamps_us.npy'
    if timestamps_file.exists():
        timestamps_loaded = np.load(str(timestamps_file))
        assert np.array_equal(timestamps_loaded, ev_repr_timestamps_us)
    else:
        np.save(str(timestamps_file), ev_repr_timestamps_us)

    # generate paired_frame_mask for saving labels
    paired_frame_mask = np.where(_frameidx2repridx > 0, 1, 0)

    return paired_frame_mask


def downsample_ev_repr(x: torch.Tensor, scale_factor: float):
    assert 0 < scale_factor < 1
    orig_dtype = x.dtype
    if orig_dtype == torch.int8:
        x = torch.asarray(x, dtype=torch.int16)
        x = torch.asarray(x + 128, dtype=torch.uint8)
    x = torch.nn.functional.interpolate(x,
                                        scale_factor=scale_factor,
                                        mode='nearest-exact')
    if orig_dtype == torch.int8:
        x = torch.asarray(x, dtype=torch.int16)
        x = torch.asarray(x - 128, dtype=torch.int8)
    return x


def write_event_representations(
        in_h5_file: Path,
        ev_out_dir: Path,
        dataset: str,
        event_representation: RepresentationBase,
        ev_repr_num_events: Optional[int],
        ev_repr_delta_ts_ms: Optional[int],
        ev_repr_timestamps_us: np.ndarray,
        frameidx2repridx: np.ndarray,
        downsample_by_2: bool,
        overwrite_if_exists: bool = False) -> np.ndarray:
    ev_outfile = ev_out_dir / f"event_representations{'_ds2_nearest' if downsample_by_2 else ''}.h5"
    if ev_outfile.exists() and not overwrite_if_exists:
        return
    ev_outfile_in_progress = ev_outfile.parent / (
        ev_outfile.stem + '_in_progress' + ev_outfile.suffix)
    if ev_outfile_in_progress.exists():
        os.remove(ev_outfile_in_progress)
    if downsample_by_2:
        raise NotImplementedError
    with H5Reader(in_h5_file, dataset=dataset) as h5_reader, \
            H5Writer(ev_outfile_in_progress) as h5_writer:
        ev_ts_us = h5_reader.time

        end_indices = np.searchsorted(ev_ts_us,
                                      ev_repr_timestamps_us,
                                      side='right')
        if ev_repr_num_events is not None:
            start_indices = np.maximum(end_indices - ev_repr_num_events, 0)
        else:
            assert ev_repr_delta_ts_ms is not None
            start_indices = np.searchsorted(ev_ts_us,
                                            ev_repr_timestamps_us -
                                            ev_repr_delta_ts_ms * 1000,
                                            side='left')

        _frameidx2repridx = np.zeros_like(frameidx2repridx)
        for seq_idx, (idx_start,
                      idx_end) in enumerate(zip(start_indices, end_indices)):
            ev_window = h5_reader.get_event_slice(idx_start=idx_start,
                                                  idx_end=idx_end)

            # sometimes there are no events in the window
            if ev_window['x'].shape[0] == 0:
                continue

            ev_repr = event_representation.construct(x=ev_window['x'],
                                                     y=ev_window['y'],
                                                     pol=ev_window['p'],
                                                     time=ev_window['t'])

            # sometimes there are no events in the window
            if ev_repr.shape[0] < 2**5:
                continue

            if downsample_by_2:
                raise NotImplementedError
            else:
                ev_repr_numpy = ev_repr.numpy()
            h5_writer.add_data(ev_repr_numpy, seq_idx)
            frame_idx = np.searchsorted(frameidx2repridx, seq_idx, side='left')
            _frameidx2repridx[frame_idx] += 1
    assert seq_idx == len(ev_repr_timestamps_us) - 1
    os.rename(ev_outfile_in_progress, ev_outfile)
    return _frameidx2repridx


def process_sequence(dataset: str, filter_cfg: DictConfig,
                     event_representation: RepresentationBase,
                     ev_repr_num_events: Optional[int],
                     ev_repr_delta_ts_ms: Optional[int],
                     ts_step_ev_repr_ms: int, downsample_by_2: bool,
                     sequence_data: Dict[DataKeys, Union[Path, SplitType]]):
    in_npy_file = sequence_data[DataKeys.InNPY]
    in_h5_file = sequence_data[DataKeys.InH5]
    out_labels_dir = sequence_data[DataKeys.OutLabelDir]
    out_ev_repr_dir = sequence_data[DataKeys.OutEvReprDir]
    split_type = sequence_data[DataKeys.SplitType]
    assert out_labels_dir.is_dir()
    assert ts_step_ev_repr_ms > 0
    assert bool(ev_repr_num_events is not None) ^ bool(ev_repr_delta_ts_ms is not None), \
        f'{ev_repr_num_events=}, {ev_repr_delta_ts_ms=}'

    # 1) extract: labels_per_frame, frame_timestamps_us, ev_repr_timestamps_us, frameidx2repridx
    align_t_ms = 100
    try:
        labels_per_frame, frame_timestamps_us, ev_repr_timestamps_us, frameidx2repridx = \
            labels_and_ev_repr_timestamps(
                npy_file=in_npy_file,
                split_type=split_type,
                filter_cfg=filter_cfg,
                align_t_ms=align_t_ms,
                ts_step_ev_repr_ms=ts_step_ev_repr_ms,
                dataset_type=dataset)
    except NoLabelsException:
        parent_dir = out_labels_dir.parent
        print(f'No labels after filtering. Deleting {str(parent_dir)}')
        shutil.rmtree(parent_dir)
        return

    # 2) retrieve event data, compute event representations and save them
    paired_frame_mask = write_event_data(
        in_h5_file=in_h5_file,
        ev_out_dir=out_ev_repr_dir,
        dataset=dataset,
        event_representation=event_representation,
        ev_repr_num_events=ev_repr_num_events,
        ev_repr_delta_ts_ms=ev_repr_delta_ts_ms,
        ev_repr_timestamps_us=ev_repr_timestamps_us,
        downsample_by_2=downsample_by_2,
        frameidx2repridx=frameidx2repridx)

    # 3) clean and save labels: labels_per_frame, frame_timestamps_us
    save_labels(out_labels_dir=out_labels_dir,
                labels_per_frame=labels_per_frame,
                paired_frame_mask=paired_frame_mask,
                frame_timestamps_us=frame_timestamps_us)


class AggregationType(Enum):
    COUNT = auto()
    DURATION = auto()


aggregation_2_string = {
    AggregationType.DURATION: 'dt',
    AggregationType.COUNT: 'ne',
}


@dataclass
class FilterConf:
    apply_psee_bbox_filter: bool = MISSING
    apply_faulty_bbox_filter: bool = MISSING


@dataclass
class EventWindowExtractionConf:
    method: AggregationType = MISSING
    value: int = MISSING


class EventRepresentationFactory(ABC):

    def __init__(self, config: DictConfig):
        self.config = config

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def create(self, height: int, width: int) -> Any:
        ...


class EventSurfaceFactory(EventRepresentationFactory):

    @property
    def name(self) -> str:
        extraction = self.config.event_window_extraction
        return f'{self.config.name}_{aggregation_2_string[extraction.method]}={extraction.value}'

    def create(self, height: int, width: int) -> EventSurface:
        return EventSurface(height=height, width=width)


name_2_ev_repr_factory = {
    'event_surface': EventSurfaceFactory,
}


def get_configuration(ev_repr_yaml_config: Path,
                      extraction_yaml_config: Path) -> DictConfig:
    config = OmegaConf.load(ev_repr_yaml_config)
    event_window_extraction_config = OmegaConf.load(extraction_yaml_config)
    event_window_extraction_config = OmegaConf.merge(
        OmegaConf.structured(EventWindowExtractionConf),
        event_window_extraction_config)
    config.event_window_extraction = event_window_extraction_config
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir')
    parser.add_argument('target_dir')
    parser.add_argument('ev_repr_yaml_config',
                        help='Path to event representation yaml config file')
    parser.add_argument(
        'extraction_yaml_config',
        help='Path to event window extraction yaml config file')
    parser.add_argument('bbox_filter_yaml_config',
                        help='Path to bbox filter yaml config file')
    parser.add_argument('-ds',
                        '--dataset',
                        default='gen1',
                        help='gen1 or gen4')
    parser.add_argument('-np',
                        '--num_processes',
                        type=int,
                        default=1,
                        help='Num proceesses to run in parallel')
    args = parser.parse_args()

    num_processes = args.num_processes

    dataset = args.dataset
    assert dataset in ('gen1', 'gen4')
    downsample_by_2 = True if dataset == 'gen4' else False

    config = get_configuration(
        ev_repr_yaml_config=Path(args.ev_repr_yaml_config),
        extraction_yaml_config=Path(args.extraction_yaml_config))

    bbox_filter_yaml_config = Path(args.bbox_filter_yaml_config)
    assert bbox_filter_yaml_config.exists()
    filter_cfg = OmegaConf.load(str(bbox_filter_yaml_config))
    filter_cfg = OmegaConf.merge(OmegaConf.structured(FilterConf), filter_cfg)

    print('')
    print(OmegaConf.to_yaml(config))

    ev_repr_factory: EventRepresentationFactory = name_2_ev_repr_factory[
        config.name](config)
    height = dataset_2_height[args.dataset]
    width = dataset_2_width[args.dataset]
    ev_repr = ev_repr_factory.create(height=height, width=width)
    ev_repr_string = ev_repr_factory.name

    dataset_input_path = Path(args.input_dir)
    train_path = dataset_input_path / 'train'
    val_path = dataset_input_path / 'val'
    test_path = dataset_input_path / 'test'
    target_dir = Path(args.target_dir)
    os.makedirs(target_dir, exist_ok=True)

    assert train_path.exists(), f'{train_path=}'
    assert val_path.exists(), f'{val_path=}'
    assert test_path.exists(), f'{test_path=}'

    seq_data_list = list()
    for split in [train_path, val_path, test_path]:
        split_out_dir = target_dir / split.name
        os.makedirs(split_out_dir, exist_ok=True)
        for npy_file in split.iterdir():
            if npy_file.suffix != '.npy':
                continue
            h5f_path = npy_file.parent / (
                npy_file.stem.split('bbox')[0] +
                f"td{'.dat' if dataset == 'gen1' else ''}.h5")
            assert h5f_path.exists(), f'{h5f_path=}'

            dir_name = npy_file.stem.split('_bbox')[0]
            if dir_name in dirs_to_ignore[dataset]:
                continue
            out_seq_path = split_out_dir / dir_name

            out_labels_path = out_seq_path / 'labels_v2'
            os.makedirs(out_labels_path, exist_ok=True)

            out_ev_repr_parent_path = out_seq_path / 'event_representations_v2'
            out_ev_repr_path = out_ev_repr_parent_path / ev_repr_string
            os.makedirs(out_ev_repr_path, exist_ok=True)

            sequence_data = {
                DataKeys.InNPY: npy_file,
                DataKeys.InH5: h5f_path,
                DataKeys.OutLabelDir: out_labels_path,
                DataKeys.OutEvReprDir: out_ev_repr_path,
                DataKeys.SplitType: split_name_2_type[split.name],
            }
            seq_data_list.append(sequence_data)

    ev_repr_num_events = None
    ev_repr_delta_ts_ms = None
    if config.event_window_extraction.method == AggregationType.COUNT:
        ev_repr_num_events = config.event_window_extraction.value
    else:
        assert config.event_window_extraction.method == AggregationType.DURATION
        ev_repr_delta_ts_ms = config.event_window_extraction.value
    ts_step_ev_repr_ms = 50  # Could be an argument of the script.

    if num_processes > 1:
        chunksize = 1
        func = partial(process_sequence, dataset, filter_cfg, ev_repr,
                       ev_repr_num_events, ev_repr_delta_ts_ms,
                       ts_step_ev_repr_ms, downsample_by_2)
        with get_context('spawn').Pool(num_processes) as pool:
            with tqdm(total=len(seq_data_list), desc='sequences') as pbar:
                for _ in pool.imap_unordered(func,
                                             iterable=seq_data_list,
                                             chunksize=chunksize):
                    pbar.update()
    else:
        for entry in tqdm(seq_data_list, desc='sequences'):
            process_sequence(dataset=dataset,
                             filter_cfg=filter_cfg,
                             event_representation=ev_repr,
                             ev_repr_num_events=ev_repr_num_events,
                             ev_repr_delta_ts_ms=ev_repr_delta_ts_ms,
                             ts_step_ev_repr_ms=ts_step_ev_repr_ms,
                             downsample_by_2=downsample_by_2,
                             sequence_data=entry)
