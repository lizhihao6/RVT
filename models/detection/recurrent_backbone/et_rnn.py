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
from models.layers.et.et import (PartitionAttentionCl, PartitionType,
                                 get_downsample_layer_Cf2Cl, nhwC_2_nChw)
from models.layers.rnn import DWSConvLSTM2d

from .base import BaseDetector


class RNNDetector(BaseDetector):

    def __init__(self, mdl_config: DictConfig):
        super().__init__()

        ###### Config ######
        embed_dim = mdl_config.embed_dim
        dim_multiplier_per_stage = tuple(mdl_config.dim_multiplier)
        num_blocks_per_stage = tuple(mdl_config.num_blocks)
        T_max_chrono_init_per_stage = tuple(mdl_config.T_max_chrono_init)

        num_stages = len(num_blocks_per_stage)
        assert num_stages == 4

        assert isinstance(embed_dim, int)
        assert num_stages == len(dim_multiplier_per_stage)
        assert num_stages == len(num_blocks_per_stage)
        assert num_stages == len(T_max_chrono_init_per_stage)

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

        patch_size = mdl_config.stem.patch_size
        stride = 1
        self.stage_dims = [embed_dim * x for x in dim_multiplier_per_stage]

        self.rnn_stages = nn.ModuleList()
        self.strides = []
        for stage_idx, (num_blocks, T_max_chrono_init_stage) in \
                enumerate(zip(num_blocks_per_stage, T_max_chrono_init_per_stage)):
            spatial_downsample_factor = patch_size if stage_idx == 0 else 2
            stage_dim = self.stage_dims[stage_idx]
            stage = RNNDetectorStage(stage_dim=stage_dim,
                                     stage_cfg=mdl_config.stage)
            stride = stride * spatial_downsample_factor
            self.strides.append(stride)

            self.rnn_stages.append(stage)

        self.num_stages = num_stages

    def get_stage_dims(self, stages: Tuple[int, ...]) -> Tuple[int, ...]:
        stage_indices = [x - 1 for x in stages]
        assert min(stage_indices) >= 0, stage_indices
        assert max(stage_indices) < len(self.stages), stage_indices
        return tuple(self.stage_dims[stage_idx] for stage_idx in stage_indices)

    def get_strides(self, stages: Tuple[int, ...]) -> Tuple[int, ...]:
        stage_indices = [x - 1 for x in stages]
        assert min(stage_indices) >= 0, stage_indices
        assert max(stage_indices) < len(self.stages), stage_indices
        return tuple(self.strides[stage_idx] for stage_idx in stage_indices)

    def forward(self, x: th.Tensor, prev_states: Optional[LstmStates] = None, token_mask: Optional[th.Tensor] = None) \
            -> Tuple[BackboneFeatures, LstmStates]:
        if prev_states is None:
            prev_states = [None] * self.num_stages
        assert len(prev_states) == self.num_stages
        states: LstmStates = list()
        output: Dict[int, FeatureMap] = {}
        for stage_idx, stage in enumerate(self.stages):
            x, state = stage(x, prev_states[stage_idx],
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
