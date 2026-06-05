# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle

from .afd import AFDA2ABackendBase


class AFDMooncakeM2NA2ABackend(AFDA2ABackendBase):
    """Mooncake M2N implementation for AFD all-to-all communication."""

    name = "mooncake"

    def __init__(self, fd_config):
        if fd_config.model_config.num_max_dispatch_tokens_per_rank > 1024:
            raise ValueError(
                "Mooncake AFD requires num_max_dispatch_tokens_per_rank <= 1024, "
                f"got {fd_config.model_config.num_max_dispatch_tokens_per_rank}."
            )

        try:
            paddle.enable_compat(scope={"mooncake"})
            from mooncake.mooncake_ep_m2n_buffer import M2NBuffer
        except ImportError as exc:
            raise ImportError(
                "FD_MOE_A2A_BACKEND=mooncake in AFD requires mooncake.mooncake_ep_m2n_buffer. "
                "Please install a Mooncake wheel that includes the M2N EP wrapper."
            ) from exc

        role = fd_config.afd_config.afd_role
        if role not in ("attn", "ffn"):
            raise ValueError(f"Mooncake AFD requires afd_role to be 'attn' or 'ffn', got {role!r}.")

        ep_group = fd_config.parallel_config.ep_group
        try:
            mooncake_group = ep_group.process_group._mc
        except AttributeError as exc:
            raise TypeError(
                "Mooncake AFD requires fd_config.parallel_config.ep_group to be a Mooncake process group. "
                "Set FD_MOE_A2A_BACKEND=mooncake so FastDeploy initializes Mooncake PG."
            ) from exc

        self.role = role
        self.top_k = fd_config.model_config.num_experts_per_tok
        self.active_ranks = paddle.ones((fd_config.afd_config.afd_world_size,), dtype=paddle.int32)
        self._first_execution = True
        self._timeout_us = 10_000_000
        self.m2n_buffer = M2NBuffer(
            mooncake_group,
            attention_ranks=fd_config.afd_config.afd_attn_ranks,
            ffn_ranks=fd_config.afd_config.afd_ffn_ranks,
            num_experts_per_rank=fd_config.afd_config.afd_num_local_physical_experts,
            num_max_dispatch_tokens_per_rank=fd_config.model_config.num_max_dispatch_tokens_per_rank,
            hidden=fd_config.model_config.hidden_size,
        )

    def _get_timeout(self) -> int:
        return -1 if self._first_execution else self._timeout_us

    @staticmethod
    def _finish(event, hook) -> None:
        if hook is not None:
            hook()
        elif event is not None:
            event.current_stream_wait()

    def dispatch_physical(self, x, physical_topk_idx, topk_weights, **kwargs):
        timeout = self._get_timeout()
        use_fp8 = kwargs.get("use_fp8", False)
        if self.role == "attn":
            recv_hidden, recv_count, handle, event, hook = self.m2n_buffer.a2e_isend(
                x,
                physical_topk_idx,
                self.active_ranks,
                timeout_us=timeout,
                use_fp8=use_fp8,
                async_finish=False,
                return_recv_hook=True,
            )
        else:
            recv_hidden, recv_count, handle, event, hook = self.m2n_buffer.a2e_irecv(
                self.active_ranks,
                self.top_k,
                timeout_us=timeout,
                use_fp8=use_fp8,
                async_finish=False,
                return_recv_hook=True,
            )
        self._finish(event, hook)
        return recv_hidden, recv_count, handle

    def combine(self, ffn_out, physical_topk_idx, topk_weights, handle, **kwargs):
        timeout = self._get_timeout()
        if self.role == "attn":
            combined, event, hook = self.m2n_buffer.e2a_irecv(
                physical_topk_idx,
                topk_weights,
                self.active_ranks,
                handle,
                timeout_us=timeout,
                async_finish=False,
                return_recv_hook=True,
            )
        else:
            combined, event, hook = self.m2n_buffer.e2a_isend(
                ffn_out,
                self.active_ranks,
                handle,
                timeout_us=timeout,
                zero_copy=False,
                async_finish=False,
                return_recv_hook=True,
            )
        self._finish(event, hook)
        self._first_execution = False
        return combined
