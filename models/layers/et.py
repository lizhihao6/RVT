from math import sqrt

import pointops
import pytorch3d
import torch.nn as nn
import torch.nn.functional
from timm.models.layers import DropPath
from torch_scatter import scatter

from .utils import (MLP, EventEmbed, PositionEncoder, SparseConv,
                    SparseToDense, TransitionDown, offset2batch)


class EventAttention(nn.Module):

    def __init__(self,
                 dim,
                 attn_dim,
                 k_nearest=16,
                 h=128,
                 w=128,
                 conv_kernel_size=3,
                 global_step=8,
                 drop_out=0.1):
        super().__init__()
        self.k_nearest = k_nearest
        self.attn_scale = sqrt(attn_dim)
        self.h, self.w = h, w
        self.conv_kernel_size = conv_kernel_size
        self.global_step = global_step

        self.local_qkv = nn.Linear(dim, attn_dim * 3)
        self.local_pe = PositionEncoder(4, attn_dim)
        self.local_fc = nn.LayerNorm(attn_dim)
        proj_dim = attn_dim

        if self.conv_kernel_size > 0:
            self.conv_qkv = SparseConv(in_ch=dim,
                                       out_ch=attn_dim * 3,
                                       h=h,
                                       w=w,
                                       filter_size=conv_kernel_size)
            self.conv_pe = PositionEncoder(2, attn_dim)
            self.conv_fc = nn.LayerNorm(attn_dim)
            proj_dim += attn_dim

        if self.global_step > 0:
            self.global_qkv = nn.Linear(dim, attn_dim * 3)
            self.global_pe = PositionEncoder(4, attn_dim)
            self.global_fc = nn.LayerNorm(attn_dim)
            proj_dim += attn_dim

        self.proj = MLP(in_ch=proj_dim, hidden_ch=dim, out_ch=dim)
        self.softmax = nn.Softmax(dim=1)
        self.drop_out = nn.Dropout(drop_out)

    def forward(self, data_dict):
        attn = self.local_attn(data_dict)
        if self.conv_kernel_size > 0:
            conv_attn = self.conv_attn(data_dict)
            attn = torch.cat([attn, conv_attn], dim=-1)
        if self.global_step > 0:
            global_attn = self.global_attn(data_dict)
            attn = torch.cat([attn, global_attn], dim=-1)
        attn = self.proj(attn)
        data_dict['features'] = attn
        return data_dict

    def local_attn(self, data_dict):
        events = data_dict['events']
        offsets = data_dict['offsets']
        features = data_dict['features']

        # local position encoding
        xyz = events[:, :3].clone().detach()
        idx, _ = pointops.knn_query(self.k_nearest, xyz, offsets)

        # [N, k_nearest, attn_dim]
        pos_enc = self.local_pe(events[:, None, :] \
                                - pointops.grouping(idx, events, xyz))
        pos_enc = self.drop_out(pos_enc)

        # local attention
        qkv = self.local_qkv(features)
        # [N, attn_dim]
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        k = pointops.grouping(idx, k, xyz)  # [N, k_nearest, attn_dim]
        v = pointops.grouping(idx, v, xyz)  # [N, k_nearest, attn_dim]
        local_attn = self.local_fc(q[:, None, :] - k +
                                   pos_enc) / self.attn_scale  # noqa E127
        local_attn = self.softmax(local_attn)
        local_attn = self.drop_out(local_attn)
        local_attn = local_attn * (v + pos_enc)
        local_attn = torch.sum(local_attn, dim=1)  # [N, attn_dim]
        return local_attn

    def conv_attn(self, data_dict):
        events = data_dict['events']
        offsets = data_dict['offsets']

        # conv position encoding
        xyz = events[:, :3].clone().detach()
        xyz[..., 2] = 0
        xy = events[:, :2].contiguous()
        batch = offset2batch(offsets)
        idx = pointops.conv_sampling(xy, batch, self.h, self.w, 11,
                                     self.k_nearest)
        mask = torch.where(idx == -1, torch.zeros_like(idx),
                           torch.ones_like(idx))
        idx = torch.where(idx == -1, self._get_self_idx(idx),
                          idx)  # replace -1 idx to self idx

        # [N, k_nearest, attn_dim]
        pos_enc = self.conv_pe(xy[:, None, :] \
                               - pointops.grouping(idx, xy, xyz))
        pos_enc = self.drop_out(pos_enc)
        # conv attention
        qkv = self.conv_qkv(data_dict)
        # [N, attn_dim]
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        k = pointops.grouping(idx, k, xyz)  # [N, k_nearest, attn_dim]
        v = pointops.grouping(idx, v, xyz)  # [N, k_nearest, attn_dim]
        conv_attn = self.conv_fc(q[:, None, :] - k + pos_enc) \
            / self.attn_scale
        conv_attn = conv_attn * mask[:, :,
                                     None] + (1 - mask[:, :, None]) * -100
        conv_attn = self.softmax(conv_attn)
        conv_attn = self.drop_out(conv_attn)
        conv_attn = conv_attn * (v + pos_enc)
        conv_attn = torch.sum(conv_attn, dim=1)  # [N, attn_dim]
        return conv_attn

    def global_attn(self, data_dict):
        events = data_dict['events']
        offsets = data_dict['offsets']
        features = data_dict['features']

        # global position encoding
        xyz = events[:, :3].clone().detach()
        down_offsets = offsets // self.global_step

        down_idx = pointops.farthest_point_sampling(xyz, offsets, down_offsets)
        down_xyz = pointops.grouping(down_idx[:, None], xyz, xyz)[:, 0, :]
        down_events = pointops.grouping(down_idx[:, None], events, xyz)[:,
                                                                        0, :]
        # [M, k_nearest]
        pair_idx, _ = pointops.knn_query(self.k_nearest, xyz, offsets,
                                         down_xyz, down_offsets)
        # [N, k_nearest]
        inv_pair_idx, _ = pointops.knn_query(self.k_nearest, down_xyz,
                                             down_offsets, xyz, offsets)
        # [N, k_nearest, attn_dim]
        pos_enc = self.global_pe(events[:, None, :] - \
                                 pointops.grouping(inv_pair_idx, down_events, xyz))
        pos_enc = self.drop_out(pos_enc)

        # global attention
        qkv = self.global_qkv(features)
        # [N, attn_dim]
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        k = pointops.grouping(pair_idx, k,
                              down_xyz)  # [M, k_nearest, attn_dim]
        v = pointops.grouping(pair_idx, v,
                              down_xyz)  # [M, k_nearest, attn_dim]
        k = torch.max(k, dim=1)[0]  # [M, attn_dim]
        v = torch.max(v, dim=1)[0]  # [M, attn_dim]
        k = pointops.grouping(inv_pair_idx, k, xyz)  # [N, k_nearest, attn_dim]
        v = pointops.grouping(inv_pair_idx, v, xyz)  # [N, k_nearest, attn_dim]
        global_attn = self.global_fc(q[:, None, :] - k + pos_enc) \
            / self.attn_scale
        global_attn = self.softmax(global_attn)
        global_attn = self.drop_out(global_attn)
        global_attn = global_attn * (v + pos_enc)
        global_attn = torch.sum(global_attn, dim=1)  # [N, attn_dim]
        return global_attn

    def _get_self_idx(self, k_idx):
        n, k = k_idx.shape
        if not hasattr(
                self, 'idx'
        ) or self.idx.shape != k_idx.shape or self.idx.device != k_idx.device:
            self.idx = torch.arange(n, device=k_idx.device)[:,
                                                            None].repeat(1, k)
        return self.idx


class EventTransformerBlock(nn.Module):

    def __init__(self, dim, attn_dim, mlp_ratio, k_nearest, h, w,
                 conv_kernel_size, global_step, drop_out, drop_path):
        super().__init__()
        self.dim = dim
        mlp_dim = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EventAttention(dim=dim,
                                   attn_dim=attn_dim,
                                   k_nearest=k_nearest,
                                   h=h,
                                   w=w,
                                   conv_kernel_size=conv_kernel_size,
                                   global_step=global_step,
                                   drop_out=drop_out)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(in_ch=dim, hidden_ch=mlp_dim, out_ch=dim)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, data_dict):
        shortcut = data_dict['features'].clone()
        data_dict['features'] = self.norm1(data_dict['features'])
        data_dict = self.attn(data_dict)
        features = data_dict['features']
        features = features + shortcut

        shortcut = features
        features = self.mlp(self.norm2(features))
        features = self.drop_path(features) + shortcut
        data_dict['features'] = features
        return data_dict


class BasicLayer(nn.Module):

    def __init__(self, depth, dim, attn_dim, mlp_ratio, k_nearest, h, w,
                 conv_kernel_size, global_step, drop_out, drop_path):
        super().__init__()
        self.blocks = nn.ModuleList(
            EventTransformerBlock(dim=dim,
                                  attn_dim=attn_dim,
                                  mlp_ratio=mlp_ratio,
                                  k_nearest=k_nearest,
                                  h=h,
                                  w=w,
                                  conv_kernel_size=conv_kernel_size,
                                  global_step=global_step,
                                  drop_out=drop_out,
                                  drop_path=drop_path[i] if isinstance(
                                      drop_path, list) else drop_path)
            for i in range(depth))

    def forward(self, data_dict):
        for blk in self.blocks:
            data_dict = blk(data_dict)
        return data_dict


class EventTransformer(nn.Module):

    def __init__(self,
                 embed_dim=32,
                 embed_norm=True,
                 drop_out=0.,
                 drop_path_rate=0.,
                 height=240,
                 width=304,
                 conv_ks_list=[5, 3, 3, 3],
                 depth_list=[1, 1, 1, 1],
                 k_nearest_list=[8, 16, 16, 16],
                 mlp_ratio_list=[4, 4, 4, 4],
                 global_step_list=[16, 16, 8, 4],
                 down_stride_list=[4, 4, 4, -1],
                 out_strides=[-1, 8, 16, 32]):
        super().__init__()
        # build embed layer
        self.embed = EventEmbed(
            dim=embed_dim, norm_layer=nn.LayerNorm if embed_norm else None)

        # stochastic depth
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(depth_list))
        ]  # stochastic depth decay rule

        # build layers
        self.attn_layers = nn.ModuleList()
        self.sparse_to_denses = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        for i in range(len(depth_list)):
            dim = int(embed_dim * 2**i)

            # build attention layer
            attn_layer = BasicLayer(
                depth=depth_list[i],
                dim=dim,
                attn_dim=min(dim, 128),
                mlp_ratio=mlp_ratio_list[i],
                k_nearest=k_nearest_list[i],
                h=height // (2**i),
                w=width // (2**i),
                conv_kernel_size=conv_ks_list[i],
                global_step=global_step_list[i],
                drop_out=drop_out,
                drop_path=dpr[sum(depth_list[:i]):sum(depth_list[:i + 1])])
            self.attn_layers.append(attn_layer)

            # build sparse to dense layer
            out_stride = out_strides[i]
            if out_stride > 0:
                sparse_to_dense = SparseToDense(h=height // out_stride,
                                                w=width // out_stride,
                                                dim=dim)
                self.sparse_to_denses.append(sparse_to_dense)
            else:
                self.sparse_to_denses.append(None)

            # build downsample layer
            down_stride = down_stride_list[i]
            if down_stride > 0:
                down_sample = TransitionDown(
                    in_ch=dim,
                    out_ch=dim * 2,
                    k_nearest=k_nearest_list[i],
                    stride=down_stride) if down_stride > 0 else None
                self.down_samples.append(down_sample)
            else:
                self.down_samples.append(None)

    def forward_features(self, data_dict):
        out_features = []
        data_dict = self.embed(data_dict)
        for attn_layer, sparse_to_dense, down_sample in zip(
                self.attn_layers, self.sparse_to_denses, self.down_samples):
            data_dict = attn_layer(data_dict)
            if sparse_to_dense is not None:
                out = sparse_to_dense(data_dict)
                out_features.append(out)
            if down_sample is not None:
                data_dict = down_sample(data_dict)
        return out_features

    def forward(self, events, offsets):
        data_dict = dict(events=events, offsets=offsets)
        return self.forward_features(data_dict)
