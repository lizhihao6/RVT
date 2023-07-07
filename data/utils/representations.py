from abc import ABC, abstractmethod

import sparseconvnet as scn
import torch as th


class RepresentationBase(ABC):

    @abstractmethod
    def construct(self, x: th.Tensor, y: th.Tensor, pol: th.Tensor,
                  time: th.Tensor) -> th.Tensor:
        ...

    @staticmethod
    def _is_int_tensor(tensor: th.Tensor) -> bool:
        return not th.is_floating_point(tensor) and not th.is_complex(tensor)


class EventSurface(RepresentationBase):

    def __init__(self, height: int, width: int):
        self.sparse_to_dense = scn.Sequential(
            scn.InputLayer(2, (height, width), mode=1),
            scn.SparseToDense(2, 4))

    def construct(self, x: th.Tensor, y: th.Tensor, pol: th.Tensor,
                  time: th.Tensor) -> th.Tensor:
        device = x.device
        assert y.device == pol.device == time.device == device
        assert self._is_int_tensor(x)
        assert self._is_int_tensor(y)
        assert self._is_int_tensor(pol)
        assert self._is_int_tensor(time)

        assert pol.min() >= 0
        assert pol.max() <= 1

        # NOTE: assume sorted time
        t0_int = time[0]
        t1_int = time[-1]
        assert t1_int >= t0_int
        t_norm = time - t0_int

        yxb = th.stack((y, x, th.zeros_like(x)), dim=1)
        xytp = th.stack((x, y, t_norm, pol), dim=1)

        dense = self.sparse_to_dense((yxb, xytp.float()))
        assert len(dense.shape) == 4
        dense = dense[0].permute([1, 2, 0])  # [H, W, C]
        sparse = dense.to_sparse(2)

        xytp = sparse.values().round().long()
        return xytp
