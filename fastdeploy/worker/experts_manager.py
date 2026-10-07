"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
"""

"""redundant expert manager."""
import os
import time
from typing import Tuple

import numpy as np
import paddle
from paddleformers.utils.log import logger

from fastdeploy import envs
from fastdeploy.eplb.eplb import rebalance_experts
from fastdeploy.eplb.utils import (
    derive_expert_tables,
    dump_redundant_expert_table_snapshot,
    expand_expert_rank_table,
    read_shared_expert_rank_table,
)


class RedundantExpertManger:
    """
    RedundantExpertManger
    """

    def __init__(
        self,
        n_routed_experts: int,
        num_hidden_layers: int,
        redundant_experts_num: int,
        ep_size: int,
        fd_config=None,
    ) -> None:
        """Initialize a redundant expert manager"""
        self.num_expert = n_routed_experts if isinstance(n_routed_experts, int) else n_routed_experts[0]
        self.redundant_experts_num = redundant_experts_num
        self.num_hidden_layers = num_hidden_layers
        self.ep_size = ep_size

        self.fd_config = fd_config

        if fd_config is None or not fd_config.afd_config.enable_afd:
            self.num_replicas = self.num_expert + self.redundant_experts_num
            self.num_nodes = max(ep_size // 8, 8)
            self.num_gpus = ep_size
        else:
            self.num_replicas = fd_config.afd_config.num_physical_experts
            self.num_nodes = max(fd_config.afd_config.num_ffn_ranks // 8, 1)
            self.num_gpus = fd_config.afd_config.num_ffn_ranks
        self.num_groups = 1

        self.export_per_rank = self.num_replicas // ep_size
        assert (
            self.num_replicas % ep_size == 0
        ), f"num_replicas must be divisible by ep_size, \
                but got num_replicas = {self.num_replicas}, ep_size = {ep_size}"

        self.model_ep_rank_to_expert_id_list = paddle.full(
            shape=[
                self.num_hidden_layers,
                self.num_replicas,
            ],
            fill_value=-1,
            dtype="int32",
        )
        self.model_expert_id_to_ep_rank_array = paddle.full(
            shape=[
                self.num_hidden_layers,
                self.num_expert,
                self.redundant_experts_num + 1,
            ],
            fill_value=-1,
            dtype="int32",
        )
        self.model_expert_in_rank_num_list = paddle.full(
            shape=[self.num_hidden_layers, self.num_expert],
            fill_value=0,
            dtype="int32",
        )
        self.model_active_expert_id_to_ep_rank_array = self.model_expert_id_to_ep_rank_array.clone()
        self.model_active_expert_in_rank_num_list = self.model_expert_in_rank_num_list.clone()

        self.model_tokens_per_expert_stats_list = paddle.ones(
            shape=[self.num_hidden_layers, self.num_expert], dtype="int32"
        )
        self.model_tokens_per_expert_stats_list_cpu = paddle.zeros_like(
            self.model_tokens_per_expert_stats_list
        ).pin_memory()
        self.model_tokens_per_expert_stats_ready = paddle.device.Event()
        self.last_stats_copy_ts = 0.0

        shm_expert_rank_table = read_shared_expert_rank_table(fd_config) if fd_config is not None else None
        if shm_expert_rank_table is not None:
            logger.info("read the expert rank table from shared memory")
            rank_expert_list = expand_expert_rank_table(shm_expert_rank_table, fd_config.afd_config)
            logical_to_physical_map, expert_count = derive_expert_tables(
                rank_expert_list, self.num_expert, self.redundant_experts_num + 1
            )
        else:
            rank_expert_list, logical_to_physical_map, expert_count = self._rebalance_experts(
                self.model_tokens_per_expert_stats_list.cpu().numpy()
            )

        self.update_expert_rank_table(rank_expert_list, logical_to_physical_map, expert_count, False)

        logger.info(
            f"moe experts table manager init successfully, ep_size {ep_size} \
            num_replicas {self.num_replicas} export_per_rank {self.export_per_rank}"
        )

    def _rebalance_experts(self, weight: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return rebalance_experts(
            weight=weight,
            num_replicas=self.num_replicas,
            num_groups=self.num_groups,
            num_nodes=self.num_nodes,
            num_gpus=self.num_gpus,
            fd_config=self.fd_config,
        )

    def get_ep_rank_to_expert_id_list_by_layer(
        self, layer_id: int
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """
        get_ep_rank_to_expert_id_list_by_layer
        """
        return (
            self.model_ep_rank_to_expert_id_list[layer_id],
            self.model_active_expert_id_to_ep_rank_array[layer_id],
            self.model_active_expert_in_rank_num_list[layer_id],
            self.model_tokens_per_expert_stats_list[layer_id],
        )

    def get_ep_rank_to_expert_id_list(self) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """
        get_ep_rank_to_expert_id_list
        """
        return (
            self.model_ep_rank_to_expert_id_list,
            self.model_expert_id_to_ep_rank_array,
            self.model_expert_in_rank_num_list,
        )

    def maybe_copy_tokens_stats(self, interval: float) -> None:
        now = time.time()
        if now - self.last_stats_copy_ts <= interval:
            return
        self.last_stats_copy_ts = now
        self.model_tokens_per_expert_stats_list_cpu.copy_(self.model_tokens_per_expert_stats_list, False)
        self.model_tokens_per_expert_stats_ready.record()
        self.model_tokens_per_expert_stats_list.zero_()

    def get_tokens_stats_snapshot(self) -> paddle.Tensor:
        self.model_tokens_per_expert_stats_ready.synchronize()
        return self.model_tokens_per_expert_stats_list_cpu.cpu()

    def get_expert_id_to_ep_rank_array(self) -> np.ndarray:
        """
        get_expert_id_to_ep_rank_array
        """
        return self.model_expert_id_to_ep_rank_array.cpu().numpy()

    def update_expert_rank_table(
        self,
        rank_expert_list: np.ndarray,
        logical_to_physical_map: np.ndarray,
        expert_count: np.ndarray,
        clear_stat: bool = True,
    ) -> None:
        """
        update_expert_rank_table
        """
        if self.fd_config is not None and clear_stat:
            try:
                dump_path, snapshot = dump_redundant_expert_table_snapshot(
                    fd_config=self.fd_config,
                    rank_expert_list=self.model_ep_rank_to_expert_id_list,
                    logical_to_physical_map=self.model_expert_id_to_ep_rank_array,
                    expert_count=self.model_expert_in_rank_num_list,
                    source="model_table_before",
                    local_rank=self.fd_config.parallel_config.expert_parallel_rank,
                    clear_stat=clear_stat,
                )
                logger.info(
                    "redundant_expert: dump model routing table before update, "
                    f"path={dump_path}, hash={snapshot['table_hash']}, role={snapshot['role']}, "
                    f"shape={snapshot['shape']}"
                )
            except Exception as e:
                logger.warning(f"redundant_expert: dump model routing table before update failed, {e}")

        # update model info
        self.model_ep_rank_to_expert_id_list.copy_(paddle.to_tensor(rank_expert_list), True)
        self.model_expert_id_to_ep_rank_array.fill_(-1)
        self.model_expert_id_to_ep_rank_array[:, :, : logical_to_physical_map.shape[-1]] = paddle.to_tensor(
            logical_to_physical_map
        )
        self.model_expert_in_rank_num_list.copy_(paddle.to_tensor(expert_count), True)

        self.model_active_expert_id_to_ep_rank_array.copy_(self.model_expert_id_to_ep_rank_array, True)
        self.model_active_expert_in_rank_num_list.copy_(self.model_expert_in_rank_num_list, True)

        if self.fd_config is not None and clear_stat:
            try:
                dump_path, snapshot = dump_redundant_expert_table_snapshot(
                    fd_config=self.fd_config,
                    rank_expert_list=rank_expert_list,
                    logical_to_physical_map=logical_to_physical_map,
                    expert_count=expert_count,
                    source="model_table_after",
                    local_rank=self.fd_config.parallel_config.expert_parallel_rank,
                    clear_stat=clear_stat,
                )
                logger.info(
                    "redundant_expert: dump model routing table after update, "
                    f"path={dump_path}, hash={snapshot['table_hash']}, role={snapshot['role']}, "
                    f"shape={snapshot['shape']}"
                )
            except Exception as e:
                logger.warning(f"redundant_expert: dump model routing table after update failed, {e}")

        # reset
        if clear_stat:
            self.model_tokens_per_expert_stats_list.zero_()

    def refresh_active_expert_rank_table(self):
        """
        Rebuild the active expert routing table after rank liveness changes.
        """
        if not self.fd_config.afd_config.enable_afd or envs.FD_MOE_A2A_BACKEND != "mooncake":
            return

        from fastdeploy.model_executor.layers.moe.ep import EPBackend

        active_ranks = EPBackend().active_ranks

        local_physical_experts = self.fd_config.afd_config.num_local_physical_experts

        # Candidate physical-expert ids per (layer, logical_expert), -1 padded.
        # In AFD, valid physical experts are only placed on FFN ranks. ATTN rank slots stay -1.
        base = self.model_expert_id_to_ep_rank_array  # [L, E, C] int32
        num_layers, num_experts, num_candidates = base.shape

        owner_rank = (base.clip(min=0) // local_physical_experts).reshape([-1]).astype("int64")
        keep = paddle.logical_and(
            base >= 0,
            paddle.index_select(active_ranks, owner_rank).reshape(base.shape) > 0,
        )
        keep_int = keep.astype("int32")

        # Stable compaction without a sort: the k-th survivor of a row lands on column k,
        # dropped candidates are parked in a scratch column that is sliced off, and columns
        # nobody writes keep the -1 fill.
        target_col = paddle.where(
            keep,
            keep_int.cumsum(axis=-1).astype("int32") - 1,
            paddle.full_like(keep_int, num_candidates),
        )
        active_table = paddle.put_along_axis(
            paddle.full([num_layers, num_experts, num_candidates + 1], -1, dtype="int32"),
            target_col.astype("int64"),
            base,
            axis=-1,
        )[:, :, :num_candidates]
        active_count = keep_int.sum(axis=-1).astype("int32")

        # Publish refreshed active expert table.
        self.model_active_expert_id_to_ep_rank_array.copy_(active_table, False)
        self.model_active_expert_in_rank_num_list.copy_(active_count, False)
        lost_per_layer = (active_count <= 0).astype("int32").sum(axis=1).tolist()
        logger.info(
            "redundant_expert: refresh active expert table, "
            f"mode={os.getenv('FD_LOST_EXPERT_MODE', 'drop')}, "
            f"logical experts with no live replica per layer={lost_per_layer}"
        )

if __name__ == "__main__":
    print(RedundantExpertManger(64, 2, 8, 8).model_expert_id_to_ep_rank_array)
