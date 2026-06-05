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

from __future__ import annotations

from .afd import AFDA2ABackendBase


class AFDDeepEPA2ABackend(AFDA2ABackendBase):
    """DeepEP implementation for AFD all-to-all communication."""

    name = "deepep"

    def __init__(self, fd_config):
        from fastdeploy.config import MoEPhase
        from fastdeploy.model_executor.layers.moe.ep_deepep_backend import DeepEPEngine

        self.ep_engine = DeepEPEngine(
            num_max_dispatch_tokens_per_rank=fd_config.model_config.num_max_dispatch_tokens_per_rank,
            hidden_size=fd_config.model_config.hidden_size,
            num_experts=fd_config.afd_config.afd_num_physical_experts,
            ep_size=fd_config.afd_config.afd_world_size,
            ep_rank=fd_config.parallel_config.expert_parallel_rank,
            splitwise_role=fd_config.scheduler_config.splitwise_role,
            moe_phase=MoEPhase("decode"),
            group=fd_config.parallel_config.ep_group,
            use_internode_ll_two_stage=False,
            top_k=fd_config.model_config.num_experts_per_tok,
        )

    def dispatch_physical(self, x, physical_topk_idx, topk_weights, **kwargs):
        recv_hidden, recv_count, handle, dispatch_hook = self.ep_engine.low_latency_dispatch(
            x,
            physical_topk_idx,
            kwargs.get("expertwise_scale", None),
            kwargs.get("use_fp8", False),
            kwargs.get("quant_group_size", 128),
            kwargs.get("use_ue8m0", False),
        )
        if dispatch_hook is not None:
            dispatch_hook()
        return recv_hidden, recv_count, handle

    def combine(self, ffn_out, physical_topk_idx, topk_weights, handle, **kwargs):
        combined, combine_hook = self.ep_engine.low_latency_combine(
            ffn_out,
            physical_topk_idx,
            topk_weights,
            handle,
        )
        if combine_hook is not None:
            combine_hook()
        return combined
