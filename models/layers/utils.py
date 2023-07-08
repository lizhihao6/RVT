# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------

import pointops
import sparseconvnet as scn
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from torch.nn import init as init


def default_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm) or isinstance(
            m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class MLP(nn.Module):

    def __init__(self, in_ch, hidden_ch=None, out_ch=None, act_layer=nn.GELU):
        super().__init__()
        out_ch = out_ch or in_ch
        hidden_ch = hidden_ch or in_ch
        self.net = nn.Sequential(nn.Linear(in_ch, hidden_ch), act_layer(),
                                 nn.Linear(hidden_ch, out_ch))

    def forward(self, x):
        return self.net(x)


class TransitionDown(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, k_nearest: int, stride: int):
        super().__init__()
        self.stride = stride
        self.k_nearest = k_nearest
        self.mlp = nn.Sequential(nn.Linear(4 + in_ch, out_ch), SwapAxes(1, 2),
                                 nn.BatchNorm1d(out_ch), SwapAxes(1, 2),
                                 nn.ReLU())

    def forward(self, data_dict):
        events = data_dict['events']
        features = data_dict['features']
        offsets = data_dict['offsets']
        down_offsets = offsets // self.stride

        xyz = events[:, :3].contiguous()
        down_idx = pointops.farthest_point_sampling(xyz, offsets, down_offsets)
        down_xyz = pointops.grouping(down_idx[:, None], xyz, xyz)

        idx, _ = pointops.knn_query(self.k_nearest, xyz, offsets, down_xyz,
                                    down_offsets)  # [M, k_nearest]
        down_events = pointops.grouping(down_idx[:, None], events,
                                        xyz)[:, 0, :]  # [M, 4]
        grouped_events = pointops.grouping(idx, events,
                                           xyz)  # [M, k_nearest, 4]
        grouped_events_norm = grouped_events - \
            down_events[:, None]  # [M, k_nearest, 4]
        grouped_features = pointops.grouping(idx, features,
                                             xyz)  # [M, k_nearest, F]
        grouped_features = torch.cat([grouped_events_norm, grouped_features],
                                     dim=-1)  # [M, k_nearest, 4+F]
        down_features = self.mlp(grouped_features)  # [M, k_nearest, out_ch]
        down_features = torch.max(down_features, 1)[0]  # [M, out_ch]

        data_dict = dict(events=down_events,
                         features=down_features,
                         offsets=down_offsets)
        return data_dict


class SwapAxes(nn.Module):

    def __init__(self, dim1: int = 1, dim2: int = 2):
        self.dim1 = dim1
        self.dim2 = dim2
        super().__init__()

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)


class PositionEncoder(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_ch, in_ch), nn.LayerNorm(in_ch),
                                 nn.GELU(), nn.Linear(in_ch, out_ch))

    def forward(self, x):
        return self.mlp(x)


class SparseConv(nn.Module):

    def __init__(self, h, w, in_ch, out_ch, filter_size=3):
        super().__init__()
        self.h = h
        self.w = w
        self.conv = scn.Sequential(
            scn.InputLayer(dimension=2,
                           spatial_size=torch.LongTensor([h, w]),
                           mode=4),  # mean
            scn.SubmanifoldConvolution(dimension=2,
                                       nIn=in_ch + 2,
                                       nOut=out_ch,
                                       filter_size=filter_size,
                                       bias=True),
            scn.OutputLayer(out_ch))

    def forward(self, data_dict):
        events = data_dict['events']
        features = data_dict['features']
        offsets = data_dict['offsets']
        batch = offset2batch(offsets)
        pos = events[:, 3:]
        neg = 1 - events[:, 3:]
        sparse = torch.cat([pos, neg, features], dim=1)  # [N, 2 + in_ch]
        yx = events[:, [1, 0]]
        yx[:, 0] = torch.round(yx[:, 0] * self.h)
        yx[:, 1] = torch.round(yx[:, 1] * self.w)
        yxb = torch.cat([yx.long(), batch[:, None]], dim=1)  # [N, 3]
        f = self.conv([yxb, sparse]).reshape([yxb.shape[0], -1])
        return f


class EventEmbed(nn.Module):

    def __init__(self, dim, norm_layer=None):
        super().__init__()
        self.embed = nn.Linear(4, dim)
        self.norm = norm_layer(dim) if norm_layer else None

    def forward(self, data_dict):
        features = self.embed(data_dict['events'])
        if self.norm:
            features = self.norm(features)
        data_dict['features'] = features
        return data_dict


class SparseToDense(nn.Module):

    def __init__(self, h, w, dim):
        super().__init__()
        self.h = h
        self.w = w
        self.sparse_to_dense = scn.Sequential(
            scn.InputLayer(2, (h, w), mode=4),  # mean
            scn.SparseToDense(2, dim))

    def forward(self, data_dict):
        events = data_dict['events']
        features = data_dict['features']
        offsets = data_dict['offsets']
        batch = offset2batch(offsets)
        yx = events[:, [1, 0]]
        yx[:, 0] = torch.round(yx[:, 0] * self.h)
        yx[:, 1] = torch.round(yx[:, 1] * self.w)
        yxb = torch.cat([yx.long(), batch[:, None]], dim=1)  # [N, 3]
        features = self.sparse_to_dense([yxb, features])
        return features


def offset2batch(offset):
    return torch.cat([
        torch.tensor([i] *
                     (o - offset[i - 1])) if i > 0 else torch.tensor([i] * o)
        for i, o in enumerate(offset)
    ],
                     dim=0).long().to(offset.device)


def batch2offset(batch):
    return torch.cumsum(batch.bincount(), dim=0).long()
