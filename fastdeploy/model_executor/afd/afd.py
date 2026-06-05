from __future__ import annotations

import paddle
from paddleformers.utils.log import logger

from fastdeploy import envs


class AFDA2ABackendBase:
    """Communication backend used by AFD decode workers."""

    name = "base"

    def dispatch_physical(self, x, physical_topk_idx, topk_weights, **kwargs):
        raise NotImplementedError

    def combine(self, ffn_out, physical_topk_idx, topk_weights, handle, **kwargs):
        raise NotImplementedError


class AFDDecodeRunner:
    """Decode-phase runner that drives dispatch / combine for AFD."""

    def __init__(self, fd_config):
        self.fd_config = fd_config
        self.static_log2phy_tensor = None
        if not fd_config.eplb_config.enable_eplb:
            self.static_log2phy_tensor = paddle.to_tensor(
                self.fd_config.afd_config.afd_static_log2phy,
                dtype=paddle.int64,
            )

        if envs.FD_MOE_A2A_BACKEND == "deepep":
            from fastdeploy.model_executor.afd.deepep_backend import AFDDeepEPA2ABackend

            self.a2a_backend = AFDDeepEPA2ABackend(fd_config)
        elif envs.FD_MOE_A2A_BACKEND == "mooncake":
            from fastdeploy.model_executor.afd.mooncake_backend import AFDMooncakeM2NA2ABackend

            self.a2a_backend = AFDMooncakeM2NA2ABackend(fd_config)
        else:
            raise ValueError(
                f"Unknown FD_MOE_A2A_BACKEND={envs.FD_MOE_A2A_BACKEND!r}. "
                "Valid options for AFD are ['deepep', 'mooncake']."
            )

        logger.info(
            f"AFDDecodeRunner created: physical_experts={self.fd_config.afd_config.afd_num_physical_experts}, "
            f"local_physical_experts={self.fd_config.afd_config.afd_num_local_physical_experts}, "
            f"attn_ranks={self.fd_config.afd_config.afd_attn_ranks}, ffn_ranks={self.fd_config.afd_config.afd_ffn_ranks}, "
            f"a2a_backend={self.a2a_backend.name}, "
            f"ep_rank={fd_config.parallel_config.expert_parallel_rank}, "
            f"current_device={paddle.device.get_device()}"
        )

    def routing_logical_to_physical(self, topk_idx: paddle.Tensor) -> paddle.Tensor:
        if self.fd_config.eplb_config.enable_eplb:
            raise RuntimeError("AFD routing_logical_to_physical is only valid when EPLB is disabled.")
        if topk_idx.shape[0] == 0:
            return topk_idx
        orig_shape = topk_idx.shape
        return paddle.index_select(self.static_log2phy_tensor, topk_idx.reshape([-1]), axis=0).reshape(orig_shape)

    def dispatch_physical(self, x, physical_topk_idx, topk_weights, **kwargs):
        return self.a2a_backend.dispatch_physical(x, physical_topk_idx, topk_weights, **kwargs)

    def combine(self, ffn_out, physical_topk_idx, topk_weights, handle, **kwargs):
        return self.a2a_backend.combine(ffn_out, physical_topk_idx, topk_weights, handle, **kwargs)
