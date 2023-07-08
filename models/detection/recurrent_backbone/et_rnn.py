from typing import Dict, Optional, Tuple

import torch as th
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

try:
    from torch import compile as th_compile
except ImportError:
    th_compile = None

from data.utils.types import (BackboneFeatures, FeatureMap, LstmState,
                              LstmStates)
from models.layers.et import EventTransformer
from models.layers.rnn import DWSConvLSTM2d

from .base import BaseDetector


class RNNDetector(BaseDetector):

    def __init__(self, mdl_config: DictConfig):
        super().__init__()

        ###### Compile if requested ######
        compile_cfg = mdl_config.get('compile', None)
        if compile_cfg is not None:
            compile_mdl = compile_cfg.enable
            if compile_mdl and th_compile is not None:
                compile_args = OmegaConf.to_container(compile_cfg.args,
                                                      resolve=True,
                                                      throw_on_missing=True)
                self.forward = th_compile(self.forward, **compile_args)
            elif compile_mdl:
                print(
                    'Could not compile backbone because torch.compile is not available'
                )
        ##################################

        # build backbone
        h, w = mdl_config.in_res_hw
        mdl_config.feature_extractor.height = h
        mdl_config.feature_extractor.width = w
        self.feature_extractor = EventTransformer(
            **mdl_config.feature_extractor)
        self.strides = self.feature_extractor.strides
        self.stage_dims = self.feature_extractor.stage_dims
        self.num_stages = self.feature_extractor.num_stages

        # build rnns
        self.rnn_stages = nn.ModuleList()
        for stage_dim in self.stage_dims:
            if stage_dim > 0:
                stage = RNNDetectorStage(stage_dim=stage_dim,
                                         stage_cfg=mdl_config.stage)
                self.rnn_stages.append(stage)
            else:
                self.rnn_stages.append(None)

    def get_stage_dims(self, stages: Tuple[int, ...]) -> Tuple[int, ...]:
        stage_indices = [x - 1 for x in stages]
        assert min(stage_indices) >= 0, stage_indices
        assert max(stage_indices) < self.num_stages, stage_indices
        return tuple(self.stage_dims[stage_idx] for stage_idx in stage_indices)

    def get_strides(self, stages: Tuple[int, ...]) -> Tuple[int, ...]:
        stage_indices = [x - 1 for x in stages]
        assert min(stage_indices) >= 0, stage_indices
        assert max(stage_indices) < self.num_stages, stage_indices
        return tuple(self.strides[stage_idx] for stage_idx in stage_indices)

    def forward(self, events: th.Tensor, offsets: th.Tensor,
                prev_states: Optional[LstmStates] = None,
                token_mask: Optional[th.Tensor] = None) \
            -> Tuple[BackboneFeatures, LstmStates]:
        if prev_states is None:
            prev_states = [None] * self.num_stages
        assert len(prev_states) == self.num_stages

        features = self.feature_extractor(events, offsets)
        states = []
        output: Dict[int, FeatureMap] = {}
        for stage_idx, rnn in enumerate(self.rnn_stages):
            if rnn is not None:
                x = features.pop(0)
                x, state = rnn(x, prev_states[stage_idx],
                               token_mask if stage_idx == 0 else None)
            states.append(state)
            stage_number = stage_idx + 1
            output[stage_number] = x
        return output, states


class RNNDetectorStage(nn.Module):
    """Operates with NCHW [channel-first] format as input and output."""

    def __init__(self, stage_dim: int, stage_cfg: DictConfig):
        super().__init__()
        lstm_cfg = stage_cfg.lstm

        self.lstm = DWSConvLSTM2d(
            dim=stage_dim,
            dws_conv=lstm_cfg.dws_conv,
            dws_conv_only_hidden=lstm_cfg.dws_conv_only_hidden,
            dws_conv_kernel_size=lstm_cfg.dws_conv_kernel_size,
            cell_update_dropout=lstm_cfg.get('drop_cell_update', 0))

    def forward(self, x: th.Tensor,
                h_and_c_previous: Optional[LstmState] = None) \
            -> Tuple[FeatureMap, LstmState]:
        h_c_tuple = self.lstm(x, h_and_c_previous)
        x = h_c_tuple[0]
        return x, h_c_tuple
