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

import argparse
import asyncio
import json
import os
import time
import traceback
from typing import List, Tuple

import numpy as np

from fastdeploy.logger.logger import intercept_paddle_loggers

with intercept_paddle_loggers():
    import paddle
    import paddle.distributed as dist
    from paddle.distributed import fleet

from fastdeploy import envs
from fastdeploy.config import (
    AFDConfig,
    CacheConfig,
    DeployModality,
    DeviceConfig,
    EarlyStopConfig,
    EPLBConfig,
    ErnieArchitectures,
    FDConfig,
    GraphOptimizationConfig,
    LoadConfig,
    ModelConfig,
    ParallelConfig,
    PlasAttentionConfig,
    RoutingReplayConfig,
    SpeculativeConfig,
    StructuredOutputsConfig,
)
from fastdeploy.engine.request import ControlRequest, ControlResponse, RequestType
from fastdeploy.eplb.async_expert_loader import (
    MODEL_MAIN_NAME,
    REARRANGE_EXPERT_MAGIC_NUM,
    create_mmap,
    load_tensor_from_shm_mem,
)
from fastdeploy.eplb.experts_manager import RedundantExpertManager
from fastdeploy.eplb.utils import dump_redundant_expert_table_snapshot
from fastdeploy.inter_communicator import EngineWorkerQueue as TaskQueue
from fastdeploy.inter_communicator import (
    ExistTaskStatus,
    IPCSignal,
    ModelWeightsStatus,
    RearrangeExpertStatus,
)
from fastdeploy.inter_communicator.fmq import FMQ
from fastdeploy.model_executor.layers.quantization import parse_quant_config
from fastdeploy.model_executor.utils import v1_loader_support
from fastdeploy.platforms import current_platform
from fastdeploy.scheduler import SchedulerConfig
from fastdeploy.utils import all_gather_values, get_logger, optional_type, get_host_ip
from fastdeploy.cache_manager.transfer_factory.utils import get_rdma_nics
from fastdeploy.worker.worker_base import WorkerBase

logger = get_logger("worker_process", "worker_process.log")

def dump_mooncake_debug_state(worker, fd_config, prefix: str = "") -> None:

    mc_model = worker.model_runner.get_model()
    mooncake_a2a = mc_model._afd_runner.a2a_backend

    if mooncake_a2a is None:
        logger.warning("AFD Mooncake backend is not available on worker model")
        return

    mc_ep_group = fd_config.parallel_config.ep_group.process_group._mc._get_backend(paddle.device("cuda"))
    mc_tp_group = fd_config.parallel_config.tp_group.process_group._mc._get_backend(paddle.device("cuda"))


    mc_ep_active_ranks = mooncake_a2a.active_ranks

    from mooncake.ep import get_active_ranks

    logger.info(
        f"{prefix} mooncake debug state: "
        f"worker_rank={getattr(worker, 'rank', None)}, "
        f"local_rank={getattr(worker, 'local_rank', None)}, "
        f"mc_ep_active_ranks={mc_ep_active_ranks},"
        f"ep_active={get_active_ranks(mc_ep_group)}, "
        f"tp_active={get_active_ranks(mc_tp_group)}"
    )



def get_worker(fd_config: FDConfig, local_rank: int, rank: int) -> WorkerBase:
    """
    get worker of different device
    """
    if fd_config.model_config.enable_logprob and not current_platform.is_cuda() and not current_platform.is_xpu():
        raise NotImplementedError("Only CUDA and XPU platforms support logprob.")
    if current_platform.is_dcu():
        from fastdeploy.worker.dcu_worker import DcuWorker

        return DcuWorker(fd_config=fd_config, local_rank=local_rank, rank=rank)
    if current_platform.is_cuda():
        from fastdeploy.worker.gpu_worker import GpuWorker

        return GpuWorker(fd_config=fd_config, local_rank=local_rank, rank=rank)
    if current_platform.is_xpu():
        from fastdeploy.worker.xpu_worker import XpuWorker

        return XpuWorker(fd_config=fd_config, local_rank=local_rank, rank=rank)
    if current_platform.is_iluvatar():
        from fastdeploy.worker.iluvatar_worker import IluvatarWorker

        return IluvatarWorker(fd_config=fd_config, local_rank=local_rank, rank=rank)
    if current_platform.is_gcu():
        from fastdeploy.worker.gcu_worker import GcuWorker

        return GcuWorker(fd_config=fd_config, local_rank=local_rank, rank=rank)
    if current_platform.is_maca():
        from fastdeploy.worker.metax_worker import MetaxWorker

        return MetaxWorker(fd_config=fd_config, local_rank=local_rank, rank=rank)
    if current_platform.is_intel_hpu():
        from fastdeploy.worker.hpu_worker import HpuWorker

        return HpuWorker(fd_config=fd_config, local_rank=local_rank, rank=rank)


def _get_eplb_shmem_parallel_size(fd_config: FDConfig, parallel_config: ParallelConfig) -> int:
    """Return the rank count used to size EPLB async-load shared memory."""
    if fd_config.afd_config.enable_afd:
        if fd_config.afd_config.afd_num_ffn_ranks <= 0:
            raise ValueError("AFD EPLB requires at least one FFN rank for async-load shared memory sizing.")
        return fd_config.afd_config.afd_num_ffn_ranks
    return parallel_config.expert_parallel_size


def init_distributed_environment(seed: int = 20) -> Tuple[int, int]:
    """Initialize Paddle Fleet and return world size and global rank."""

    world_size = dist.get_world_size()
    dist_strategy = fleet.DistributedStrategy()
    if world_size > 0:
        dist_strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": world_size,
            "pp_degree": 1,
            "sharding_degree": 1,
        }

        # Set control in tensor parallel
        dist_strategy.tensor_parallel_configs = {"tensor_init_seed": seed}
        fleet.init(is_collective=True, strategy=dist_strategy)

        global_rank = fleet.worker_index()
    else:
        global_rank = 0

    if envs.FD_USE_MOONCAKE_PG:
        from mooncake.paddle_integration import init_mooncake_pg
        init_mooncake_pg(
            world_size, global_rank,
            host_ip=get_host_ip(),
            ib_device_filter=get_rdma_nics().split(","),
            clean_existed_groups=True, logger=logger)

    return world_size, global_rank


def collect_afd_rank_info(args, world_size: int, global_rank: int) -> Tuple[int, int, List[int], List[int], int, int]:
    """Collect per-rank AFD topology and return node start rank, node-local rank and ATTN TP size."""
    info = {
        "afd_role": args.afd_role,
        "afd_nnode_rank": args.afd_nnode_rank,
        "global_rank": global_rank,
        "tp_size": args.tensor_parallel_size,
    }
    infos = []
    dist.all_gather_object(infos, info)
    infos = sorted(infos, key=lambda item: item["global_rank"])
    if len(infos) != world_size:
        raise ValueError(
            "AFD topology mismatch: gathered rank count does not match world_size. "
            f"world_size={world_size}, gathered_ranks={len(infos)}"
        )

    ranks_map = {}
    tp_map = {}
    role_map = {}
    attn_ranks = []
    ffn_ranks = []
    afd_attn_tp_size = None
    for item in infos:
        if item["afd_role"] == "attn":
            attn_ranks.append(item["global_rank"])
            if afd_attn_tp_size is None:
                afd_attn_tp_size = item["tp_size"]
            elif afd_attn_tp_size != item["tp_size"]:
                raise ValueError(
                    "AFD requires one ATTN tp_size, "
                    f"got {afd_attn_tp_size} and {item['tp_size']} from ranks={infos}"
                )
        elif item["afd_role"] == "ffn":
            ffn_ranks.append(item["global_rank"])
        else:
            raise ValueError(f"Unknown AFD role: {item['afd_role']}")

        node_ranks = ranks_map.setdefault(item["afd_nnode_rank"], [])
        node_ranks.append(item["global_rank"])

        node_role = role_map.setdefault(item["afd_nnode_rank"], item["afd_role"])
        if node_role != item["afd_role"]:
            raise ValueError(
                "AFD topology mismatch: ranks from the same node report different afd_role. "
                f"afd_nnode_rank={item['afd_nnode_rank']}, ranks={infos}"
            )

        tp_size = tp_map.setdefault(item["afd_nnode_rank"], item["tp_size"])
        if tp_size != item["tp_size"]:
            raise ValueError(
                "AFD topology mismatch: ranks from the same node report different tp_size. "
                f"afd_nnode_rank={item['afd_nnode_rank']}, ranks={infos}"
            )

    if len(ranks_map) != args.afd_nnodes:
        raise ValueError(
            "AFD topology mismatch: gathered node count does not match afd_nnodes. "
            f"afd_nnodes={args.afd_nnodes}, gathered_nodes={len(ranks_map)}, "
            f"afd_nnode_ranks={sorted(ranks_map)}"
        )

    for afd_nnode_rank, node_ranks in ranks_map.items():
        tp_size = tp_map[afd_nnode_rank]
        if len(node_ranks) != tp_size:
            raise ValueError(
                "AFD topology mismatch: gathered global ranks do not match tensor_parallel_size. "
                f"afd_nnode_rank={afd_nnode_rank}, tp={tp_size}, ranks={node_ranks}"
            )
        expect_ranks = list(range(node_ranks[0], node_ranks[0] + tp_size))
        if node_ranks != expect_ranks:
            raise ValueError(
                "AFD topology mismatch: ranks inside one node must be contiguous. "
                f"afd_nnode_rank={afd_nnode_rank}, expected={expect_ranks}, got={node_ranks}"
            )

    if not attn_ranks or not ffn_ranks:
        raise ValueError(f"AFD requires both ATTN and FFN ranks, got attn={attn_ranks}, ffn={ffn_ranks}")

    curr_node_ranks = ranks_map.get(args.afd_nnode_rank)
    if curr_node_ranks is None:
        raise ValueError(f"Can not resolve AFD instance ranks for afd_nnode_rank={args.afd_nnode_rank}")

    afd_node_srank = curr_node_ranks[0]
    local_rank = global_rank - afd_node_srank
    ranks = len(curr_node_ranks)
    return ranks, local_rank, attn_ranks, ffn_ranks, afd_node_srank, afd_attn_tp_size


def update_fd_config_for_mm(fd_config: FDConfig) -> None:
    architectures = fd_config.model_config.architectures
    if fd_config.enable_mm_runtime and ErnieArchitectures.contains_ernie_arch(architectures):
        fd_config.model_config.tensor_model_parallel_size = fd_config.parallel_config.tensor_parallel_size
        fd_config.model_config.tensor_parallel_rank = fd_config.parallel_config.tensor_parallel_rank
        fd_config.model_config.vision_config.dtype = fd_config.model_config.dtype


class PaddleDisWorkerProc:
    """
    Paddle Distributed wrapper for fastdeploy.worker.Worker,
        for handling single-node multi-GPU tensor parallel.
    The wrapper internally executes an event loop that continuously executes requests
        in the task queue. Control flow is transmitted by IPC.
    """

    def __init__(self, fd_config: FDConfig, world_size: int = 1, ranks: int = 1, global_rank: int = 0, local_rank: int = 0) -> None:
        """
        Initialize a distributed worker and task queue for single-node multi-GPU setup.
        Args:
            fd_config (FDConfig): Arguments related to inference, containing
                attributes such as weight_dtype, act_dtype, mp_size, hidden_size, head_dim,
                num_attention_heads, and ffn_hidden_size.
        """
        self.world_size = world_size
        self.ranks = ranks
        self.global_rank = global_rank
        self.local_rank = local_rank
        self.fd_config = fd_config
        self.parallel_config = fd_config.parallel_config
        self.cache_config = fd_config.cache_config
        self.scheduler_config = fd_config.scheduler_config
        self.eplb_config = fd_config.eplb_config

        # TODO(gongshaotian): Use worker factory to get worker
        self.worker = get_worker(fd_config=fd_config, local_rank=self.local_rank, rank=self.ranks)

        self.max_chips_per_node = 16 if current_platform.is_iluvatar() else 8
        self.enable_overlap_schedule = self.scheduler_config.enable_overlap_schedule
        self.cached_control_reqs = []

    def init_control(self):
        engine_worker_queue_port = self.parallel_config.local_engine_worker_queue_port
        queue_name = f"ctrl_w2e_rank{self.local_rank}_{engine_worker_queue_port}"
        logger.info(f"Init Control Output Queue: {queue_name}(producer)")
        self._ctrl_output = FMQ().queue(queue_name, "producer")

    def init_health_status(self) -> None:
        """
        Initialize the health status of the worker.
        Worker Status:
            worker_ready_signal:
            worker_healthy_live_signal:
            exist_task_signal:
            exist_swapped_task_signal:
            model_weights_status:
        """
        if self.parallel_config.data_parallel_size > 1 and not envs.FD_ENABLE_MULTI_API_SERVER:
            launched_expert_service_signal_data = np.zeros(
                shape=[self.parallel_config.data_parallel_size // self.fd_config.nnode], dtype=np.int32
            )
            self.launched_expert_service_signal = IPCSignal(
                name="launched_expert_service_signal",
                array=launched_expert_service_signal_data,
                dtype=np.int32,
                suffix=self.parallel_config.engine_pid,
                create=False,
            )
            while (
                self.launched_expert_service_signal.value[
                    self.parallel_config.local_data_parallel_id % self.max_chips_per_node
                ]
                == 0
            ):
                pass

        # init worker_ready_signal
        array_size = min(
            self.max_chips_per_node,
            self.parallel_config.tensor_parallel_size * self.parallel_config.data_parallel_size,
        )

        workers_ready = np.zeros(shape=[array_size], dtype=np.int32)
        self.worker_ready_signal = IPCSignal(
            name="worker_ready_signal",
            array=workers_ready,
            dtype=np.int32,
            suffix=self.parallel_config.engine_pid,
            create=False,
        )
        local_device_rank = self.local_rank % array_size
        self.worker_ready_signal.value[local_device_rank] = 1
        # init worker_healthy_live_signal
        workers_alive = np.zeros(shape=[min(array_size, self.parallel_config.tensor_parallel_size)], dtype=np.int32)
        self.worker_healthy_live_signal = IPCSignal(
            name="worker_healthy_live_signal",
            array=workers_alive,
            dtype=np.int32,
            suffix=self.parallel_config.local_engine_worker_queue_port,
            create=False,
        )
        tp_rank = self.parallel_config.tensor_parallel_rank
        self.worker_healthy_live_signal.value[tp_rank] = int(time.time())

        # init model_weights_status
        workers_model_weights = np.zeros(shape=[1], dtype=np.int32)
        self.model_weights_status = IPCSignal(
            name="model_weights_status",
            array=workers_model_weights,
            dtype=np.int32,
            suffix=self.parallel_config.local_engine_worker_queue_port,
            create=False,
        )

        # init kv_cache_status
        kv_cache_status_data = np.zeros(shape=[1], dtype=np.int32)
        self.kv_cache_status = IPCSignal(
            name="kv_cache_status",
            array=kv_cache_status_data,
            dtype=np.int32,
            suffix=self.parallel_config.local_engine_worker_queue_port,
            create=False,
        )

        # init exist_task_signal
        workers_exist_task = np.zeros([1], dtype=np.int32)
        self.exist_task_signal = IPCSignal(
            name="exist_task_signal",
            array=workers_exist_task,
            dtype=np.int32,
            suffix=self.parallel_config.local_engine_worker_queue_port,
            create=False,
        )

        # init exist_swapped_task_signal
        workers_swapped_task = np.zeros(shape=[1], dtype=np.int32)
        self.exist_swapped_task_signal = IPCSignal(
            name="exist_swapped_task_signal",
            array=workers_swapped_task,
            dtype=np.int32,
            suffix=self.parallel_config.local_engine_worker_queue_port,
            create=False,
        )

        # init exist_prefill_task_signal
        exist_prefill_task_signal_data = np.zeros([1], dtype=np.int32)
        self.exist_prefill_task_signal = IPCSignal(
            name="exist_prefill_task_signal",
            array=exist_prefill_task_signal_data,
            dtype=np.int32,
            suffix=self.parallel_config.local_engine_worker_queue_port,
            create=False,
        )

        # init engine forward signal
        # If engine is being forward, engine_forward_signal_data should be 1.
        # If engine is out of forward, engine_forward_signal_data should be 0.
        # In pd disaggregation + EP parallel, only when engine is out of forward, scheduler send next batch to worker.
        # When engine is out of forward, engine_forward_signal_data must be 0, otherwise scheduler will not schedule next batch.
        engine_forward_signal_data = np.zeros([1], dtype=np.int32)
        self.engine_forward_signal = IPCSignal(
            name="engine_forward_signal",
            array=engine_forward_signal_data,
            dtype=np.int32,
            suffix=self.parallel_config.local_engine_worker_queue_port,
            create=False,
        )

    def update_weights_from_tensor(self, mmap_infos):
        """
        update_weights_from_tensor
        """
        import time

        if self.fd_config.afd_config.afd_role == "attn":
            redundant_table_manger = self.worker.get_model().redundant_table_manger
            if redundant_table_manger is None:
                raise RuntimeError("AFD ATTN EPLB update requires a redundant table manager.")
            rank_expert_list, logical_to_physical_map, expert_count = (
                self.experts_manager.get_ep_rank_to_expert_id_list()
            )
            redundant_table_manger.update_expert_rank_table(rank_expert_list, logical_to_physical_map, expert_count)
            valid_phy = rank_expert_list[rank_expert_list >= 0]
            valid_log2phy = logical_to_physical_map[logical_to_physical_map >= 0]
            logger.info(
                "redundant_expert: AFD ATTN routing table updated, "
                f"phy_shape={rank_expert_list.shape}, log2phy_shape={logical_to_physical_map.shape}, "
                f"valid_phy={valid_phy.size}, valid_log2phy={valid_log2phy.size}, "
                f"phy_sum={int(np.sum(valid_phy)) if valid_phy.size > 0 else 0}, "
                f"log2phy_sum={int(np.sum(valid_log2phy)) if valid_log2phy.size > 0 else 0}"
            )
            return

        while True:
            if self.experts_manager.tensor_infos is None:
                time.sleep(0.1)
            else:
                break
        state_dicts = load_tensor_from_shm_mem(self.experts_manager.tensor_infos, mmap_infos[MODEL_MAIN_NAME], logger)
        rank_expert_list, logical_to_physical_map, expert_count = self.experts_manager.get_ep_rank_to_expert_id_list()
        self.worker.get_model().redundant_table_manger.update_expert_rank_table(
            rank_expert_list, logical_to_physical_map, expert_count
        )
        # TO BE FIXED
        self.worker.get_model().update_state_dict(state_dicts)
        self.experts_manager.tensor_infos = None

    def _get_eplb_update_group_and_src(self):
        """Get the TP-local group/source for broadcasting EPLB update signals."""
        if self.fd_config.afd_config.enable_afd:
            return self.parallel_config.tp_group, self.fd_config.afd_config.afd_node_srank
        return None, 0

    def _broadcast_model_weights_signal(self, src: int, group) -> int:
        model_weights_signal_tensor = paddle.full(shape=[1], fill_value=self.model_weights_signal[0], dtype="int32")
        paddle.distributed.broadcast(model_weights_signal_tensor, src=src, group=group)
        value = model_weights_signal_tensor.numpy()[0]
        return int(value)

    def _tp_barrier_wait(self):
        if current_platform.is_xpu() or self.enable_overlap_schedule:
            self.task_queue.worker_process_tp_barrier.wait()
        else:
            paddle.distributed.barrier(self.parallel_config.tp_group)

    def _init_eplb_signal(self):
        if not self.eplb_config.enable_eplb:
            return

        self.last_dump_expert_workload_ts = 0
        self.experts_manager = RedundantExpertManager(
            rank=self.parallel_config.expert_parallel_rank,
            ep_size=self.parallel_config.expert_parallel_size,
            fd_config=self.fd_config,
            ipc_signal_suffix=self.parallel_config.local_engine_worker_queue_port,
        )

        dp_ipc_signal_suffix = (
            f"{self.parallel_config.local_engine_worker_queue_port}_dp{self.parallel_config.local_data_parallel_id}"
        )
        tp_rank = self.parallel_config.tensor_parallel_rank
        if tp_rank == 0:  # master rank0
            signal_update_weight_from_tensor = np.zeros([1], dtype=np.int32)
            self.signal_update_weight_from_tensor_array = IPCSignal(
                name="signal_update_weight_from_tensor",
                array=signal_update_weight_from_tensor,
                dtype=np.int32,
                suffix=dp_ipc_signal_suffix,
                create=False,
            )

            rearrange_experts_status = np.zeros([1], dtype=np.int32)
            self.rearrange_experts_signal = IPCSignal(
                name="rearrange_experts_status",
                array=rearrange_experts_status,
                dtype=np.int32,
                suffix=dp_ipc_signal_suffix,
                create=False,
            )

        tp_ipc_signal_suffix = f"{dp_ipc_signal_suffix}_tp{tp_rank}"
        experts_token_stats = np.zeros(
            (self.fd_config.model_config.num_hidden_layers, self.fd_config.model_config.moe_num_experts),
            dtype=np.int32,
        )
        self.local_experts_token_stats_array = IPCSignal(
            name="local_experts_token_stats",
            array=experts_token_stats,
            dtype=np.int32,
            suffix=tp_ipc_signal_suffix,
            create=False,
        )

        clear_experts_token_stats = np.zeros([1], dtype=np.int32)
        self.signal_clear_experts_token_stats = IPCSignal(
            name="signal_clear_experts_token_stats",
            array=clear_experts_token_stats,
            dtype=np.int32,
            suffix=tp_ipc_signal_suffix,
            create=False,
        )

        if self.fd_config.afd_config.afd_role == "attn":
            self.mmap_infos = None
        else:
            self.mmap_infos = create_mmap(
                [MODEL_MAIN_NAME],
                self.parallel_config.expert_parallel_rank,
                _get_eplb_shmem_parallel_size(self.fd_config, self.parallel_config),
                shm_uuid=self.parallel_config.local_engine_worker_queue_port,
                eplb_config=self.eplb_config,
                logger=logger,
            )

    def _maybe_refresh_eplb_expert_rank_table(self):

        mc_backend = self.worker.get_model()._afd_runner.a2a_backend

        if mc_backend.name != "mooncake":
            return

        changed_ranks = paddle.nonzero(mc_backend.last_active_ranks != mc_backend.active_ranks).reshape([-1])
        if changed_ranks.shape[0] == 0:
            return

        ffn_ranks = paddle.to_tensor(
        self.fd_config.afd_config.afd_ffn_ranks,
            dtype=changed_ranks.dtype,
            place=changed_ranks.place,
        )
        is_ffn_changed = bool(paddle.any(changed_ranks.unsqueeze(-1) == ffn_ranks).item())
        if not is_ffn_changed:
            logger.info(f"active_ranks changed only on non-FFN ranks, skip expert rank table refresh: {changed_ranks.tolist()}")
            mc_backend.last_active_ranks.copy_(mc_backend.active_ranks, True)
            return

        logger.info("active_ranks change, need to refresh expert rank table")
        redundant_table_manger = self.worker.get_model().redundant_table_manger

        dump_redundant_expert_table_snapshot(
            fd_config=self.fd_config,
            rank_expert_list=redundant_table_manger.model_ep_rank_to_expert_id_list,
            logical_to_physical_map=redundant_table_manger.model_active_expert_id_to_ep_rank_array,
            expert_count=redundant_table_manger.model_active_expert_in_rank_num_list,
            source="active_table_before",
            local_rank=self.fd_config.parallel_config.expert_parallel_rank,
            clear_stat=False,
        )

        redundant_table_manger.refresh_active_expert_rank_table_by_ranks(
            mc_backend.last_active_ranks,
            mc_backend.active_ranks,
        )

        dump_redundant_expert_table_snapshot(
            fd_config=self.fd_config,
            rank_expert_list=redundant_table_manger.model_ep_rank_to_expert_id_list,
            logical_to_physical_map=redundant_table_manger.model_active_expert_id_to_ep_rank_array,
            expert_count=redundant_table_manger.model_active_expert_in_rank_num_list,
            source="active_table_after",
            local_rank=self.fd_config.parallel_config.expert_parallel_rank,
            clear_stat=False,
        )
        mc_backend.last_active_ranks.copy_(mc_backend.active_ranks, True)

    def _run_eplb(self, tp_rank):
        """internal call to run eplb"""
        if not self.eplb_config.enable_eplb:
            return

        # if self.fd_config.afd_config.enable_afd and self.fd_config.afd_config.afd_role != "attn":
        if self.fd_config.afd_config.enable_afd:
            self._maybe_refresh_eplb_expert_rank_table()

        rearrange_time = time.time()
        # Get expert load
        if self.local_experts_token_stats_array.value is not None and (
            int(rearrange_time) - self.last_dump_expert_workload_ts
            > self.eplb_config.redundant_expert_dump_workload_interval
        ):
            self.last_dump_expert_workload_ts = int(rearrange_time)
            clear_stat = False
            if self.signal_clear_experts_token_stats.value[0] == 1:
                clear_stat = True
                self.signal_clear_experts_token_stats.value[0] = 0
            (
                new_stats_array,
                _,
                _,
                _,
            ) = self.worker.get_model().redundant_table_manger.get_expert_tokens_stats(clear_stat=clear_stat)
            self.local_experts_token_stats_array.value[:] = new_stats_array[:]
        elif self.local_experts_token_stats_array.value is None:
            logger.warning("redundant_expert: local_experts_token_stats not init")

        # All DP synchronously update weights
        broadcast_value = 0
        if tp_rank == 0 and self.signal_update_weight_from_tensor_array.value[0] == 1:
            logger.info("redundant_expert: update_weight_from_tensor broadcast signal")
            self.signal_update_weight_from_tensor_array.value[0] = 0
            broadcast_value = REARRANGE_EXPERT_MAGIC_NUM
        data = paddle.to_tensor([broadcast_value])
        paddle.distributed.broadcast(data, 0, group=self.parallel_config.ep_group)
        if data[0] == REARRANGE_EXPERT_MAGIC_NUM:
            self.update_weights_from_tensor(self.mmap_infos)
            logger.info(
                f"redundant_expert: update_weight_from_tensor success, cost {(time.time() - rearrange_time)*1000}ms"
            )
            paddle.distributed.barrier(self.parallel_config.ep_group)
            if tp_rank == 0:
                self.rearrange_experts_signal.value[0] = RearrangeExpertStatus.DONE.value
            logger.info("redundant_expert: done")

    def event_loop_normal(self) -> None:
        """Main event loop for Paddle Distributed Workers.
        TODO(gongshaotian): support remote calling of functions that control worker.
        """
        # init eplb signal
        self._init_eplb_signal()
        tp_size = self.parallel_config.tensor_parallel_size
        # Currently, only support single node
        self.nnode = (tp_size + self.max_chips_per_node) // self.max_chips_per_node
        max_occupied_batch_index = 0
        tp_rank = self.parallel_config.tensor_parallel_rank

        paddle.distributed.barrier(self.parallel_config.ep_group)
        # TODO: Unify status variables model_weights_status (shared memory) and model_weights_signal (numpy array) to one
        self.model_weights_signal = np.zeros([1], dtype=np.int32)
        if self.fd_config.afd_config.afd_role == "ffn":
            self._event_loop_afd_ffn(tp_rank)
            return

        attn_iter = 0
        while True:
            if self.fd_config.load_config.dynamic_load_weight and not envs.FD_ENABLE_V1_UPDATE_WEIGHTS:
                self.model_weights_signal[0] = int(self.model_weights_status.value[0])
                if self.ranks > 1:
                    self.model_weights_signal[0] = self._broadcast_model_weights_signal(src=0, group=None)

            req_dicts = None
            self.worker_healthy_live_signal.value[tp_rank] = int(time.time())

            # The first worker detects whether there are tasks in the task queue
            if tp_rank == 0:
                if self.task_queue.exist_tasks():
                    if envs.ENABLE_V1_KVCACHE_SCHEDULER or not (
                        self.fd_config.enable_mm_runtime and self.worker.exist_prefill()
                    ):
                        if self.nnode > 1:
                            self.task_queue.read_finish_flag.set(1)
                        else:
                            self.exist_task_signal.value[0] = ExistTaskStatus.EXIST

            # Synchronize the signal set by tp_rank0 visiable to other workers
            self._tp_barrier_wait() if tp_size > 1 else None

            if self.fd_config.load_config.dynamic_load_weight and not envs.FD_ENABLE_V1_UPDATE_WEIGHTS:
                if self.ranks > 1:
                    paddle.distributed.barrier()
                if self.model_weights_signal[0] != ModelWeightsStatus.NORMAL:
                    logger.info(
                        f"Global rank: {self.global_rank}, Local rank: {self.local_rank} to update or clear parameters, signal is {self.model_weights_signal[0]}, [-1:clear, 1:update]"
                    )
                    from fastdeploy.rl.dynamic_weight_manager import (
                        DynamicWeightManager,
                    )

                    self.model_weights_status.value[0] = self.model_weights_signal[0]
                    self.kv_cache_status.value[0] = self.model_weights_signal[0]
                    cache_flag = (
                        self.fd_config.cache_config.num_cpu_blocks > 0
                        or self.fd_config.cache_config.kvcache_storage_backend is not None
                    )
                    DynamicWeightManager.check_model_weights_status(
                        self.model_weights_status,
                        self.kv_cache_status if cache_flag else None,
                        # model_weights_signal
                        self.worker.model_runner,
                        self.parallel_config.local_engine_worker_queue_port,
                        self.parallel_config.shutdown_comm_group_if_worker_idle,
                    )
                    logger.info(f"current task queue data: {self.task_queue.num_tasks()}")
                    self.task_queue.clear_data()

                    if self.model_weights_signal[0] == ModelWeightsStatus.UPDATING:
                        logger.info(
                            f"Global rank: {self.global_rank}, Local rank: {self.local_rank} has updated parameters. {self.model_weights_status.value[0]}"
                        )
                        self.model_weights_signal[0] = ModelWeightsStatus.NORMAL
                    elif self.model_weights_signal[0] == ModelWeightsStatus.CLEARING:
                        logger.info(
                            f"Global rank: {self.global_rank}, Local rank: {self.local_rank} has cleared parameters. {self.model_weights_status.value[0]}"
                        )
                        # 如果清理权重后不关闭通信组，那么将推理进程统一阻塞在下面的循环中，否则信号量可能同步混乱；直到下次权重更新时唤醒
                        if not self.fd_config.parallel_config.shutdown_comm_group_if_worker_idle:
                            if self.ranks > 1:  # 所有 Rank 同时入睡，监听下次的更新信号
                                paddle.distributed.barrier()
                            while self.model_weights_signal[0] != ModelWeightsStatus.UPDATING:
                                self.model_weights_signal[0] = self.model_weights_status.value[0]
                                if self.ranks > 1:
                                    self.model_weights_signal[0] = self._broadcast_model_weights_signal(
                                        src=0, group=None
                                    )
                                time.sleep(1)
                            self.model_weights_status.value[0] = (
                                ModelWeightsStatus.UPDATING
                            )  # 所有 Rank 已同步唤醒，启动权重更新流程
                    continue

            if self.exist_task_signal.value[0] == ExistTaskStatus.EXIST or self.task_queue.read_finish_flag.get() == 1:
                logger.debug(f"Global rank: {self.global_rank}, Local rank: {self.local_rank} detected new requests.")
                self.engine_forward_signal.value[0] = 1
                tasks, read_finish = self.task_queue.get_tasks()
                # Only one of all tp_size client will get read_finish == True.
                if read_finish:
                    # Reset the two signal.
                    if self.nnode > 1:
                        self.task_queue.read_finish_flag.set(0)
                    else:
                        self.exist_task_signal.value[0] = ExistTaskStatus.EMPTY
                # In EP parallel(corresponing to dp attention), we need to barrier for prefill to prevent data imbalance due to inconsistent data arrival.
                # Only EP + DP prefill should barrier for data arrival.
                # In mixed mode and decoder in D, we should not barrier to influence decoding.
                if self.parallel_config.use_ep and self.scheduler_config.splitwise_role == "prefill":
                    paddle.distributed.barrier(self.parallel_config.ep_group)

                req_dicts, control_reqs = [], []
                assert (
                    len(tasks) > 0
                ), f"task_queue.get_tasks() should contain at least one tuple, [([req1, ...] ,real_bsz)], but got len(tasks)={len(tasks)}"
                # In EP + DP prefill, empty task ([]) is delived in worker to barrier. For empty task, just skip and continue.
                # tasks[0] contains two part, ([req1, ...] ,real_bsz)
                # tasks[0][0] is [req1, ...]
                # if empty batch is delived, eval(tasks[0][0]) should be False ([]),
                # if batch with requests is delived, eval(tasks[0][0]) should be True, then to be processed as below.
                if tasks[0][0]:
                    for req_dict, bsz in tasks:
                        if len(req_dict) > 0 and isinstance(req_dict[0], ControlRequest):
                            control_reqs.append(req_dict[0])
                        else:
                            max_occupied_batch_index = int(bsz)
                            req_dicts.extend(req_dict)

                    # todo: run control request async
                    if len(control_reqs) > 0:
                        logger.info(f"Global rank: {self.global_rank}, Local rank: {self.local_rank} received {len(control_reqs)} control request.")
                        for control_req in control_reqs:
                            if self.parallel_config.use_ep:
                                self.cached_control_reqs.append(control_req)
                                logger.info(
                                    f"Global rank: {self.global_rank}, Local rank: {self.local_rank} cached ep control request: {control_req}"
                                )
                            else:
                                self.run_control_method(control_req)
                                self._tp_barrier_wait() if tp_size > 1 else None

                if len(req_dicts) > 0:
                    # Count prefill requests in current batch
                    num_prefill_requests = sum(1 for req in req_dicts if req.task_type == RequestType.PREFILL)
                    num_scheduled_requests = len(req_dicts)
                    scheduled_request_ids = [req.request_id for req in req_dicts]
                    logger.info(
                        f"Global rank: {self.global_rank}, Local rank: {self.local_rank}, num_prefill_requests: {num_prefill_requests}, "
                        f"max_occupied_batch_index: {max_occupied_batch_index}, "
                        f"num_scheduled_requests: {num_scheduled_requests}, "
                        f"scheduled_request_ids: {scheduled_request_ids}"
                    )

                    # Process prefill inputs
                    self.worker.preprocess_new_task(req_dicts, max_occupied_batch_index)
            else:
                if self.scheduler_config.splitwise_role == "prefill":
                    if tp_size > 1:
                        # Synchronize the signal for other workers
                        self._tp_barrier_wait()
                    continue

            # Let the ep group run control method synchronically
            if envs.FD_ENABLE_V1_UPDATE_WEIGHTS and self.parallel_config.use_ep:
                pendings = all_gather_values(len(self.cached_control_reqs), self.parallel_config.ep_group)
                if all([p > 0 for p in pendings]):
                    logger.info(f"Global rank: {self.global_rank}, Local rank: {self.local_rank} detected all ep ranks have pending control tasks.")
                    self.run_control_method(self.cached_control_reqs.pop(0))

            if (
                not self.parallel_config.use_ep
                and hasattr(self.worker.model_runner, "not_need_stop")
                and not self.worker.model_runner.not_need_stop()
            ):
                self._tp_barrier_wait() if tp_size > 1 else None
                self.engine_forward_signal.value[0] = 0
                time.sleep(0.001)
                continue

            # Check if worker is paused (V1 update weights flow)
            if (
                self.fd_config.load_config.dynamic_load_weight
                and hasattr(self.worker.model_runner, "is_sleeping")
                and self.worker.model_runner.is_sleeping
            ):
                self._tp_barrier_wait() if tp_size > 1 else None
                continue

            # Execute model to generate token. The generated token will be written to the buffer.
            # These generated tokens can be obtained through get_output op.
            start_execute_time = time.time()

            self.worker.execute_model(req_dicts, max_occupied_batch_index)

            # Only v0 use this signal
            if not envs.ENABLE_V1_KVCACHE_SCHEDULER:
                self.exist_prefill_task_signal.value[0] = self.worker.exist_prefill()
            logger.debug(f"execute model cost: {time.time()-start_execute_time:.5f} s")
            # run eplb
            self._run_eplb(tp_rank)
            self.engine_forward_signal.value[0] = 0

            if (
                not self.parallel_config.use_ep
                and hasattr(self.worker.model_runner, "current_launch_token_num")
                and self.worker.model_runner.current_launch_token_num == 0
            ):
                self._tp_barrier_wait() if tp_size > 1 else None
                time.sleep(0.001)

    def _event_loop_afd_ffn(self, tp_rank: int) -> None:
        """Run the AFD FFN participant loop on the normal worker event-loop thread."""
        if hasattr(self.worker, "device"):
            paddle.device.set_device(self.worker.device)

        logger.info(
            "Start AFD FFN participant loop in event_loop_normal. "
            f"global_rank={self.global_rank}, local_rank={self.local_rank}"
        )
        while True:
            try:
                self.worker_healthy_live_signal.value[tp_rank] = int(time.time())
                with paddle.no_grad():
                    self.worker.execute_model(None, 0)
                self._run_eplb(tp_rank)
            except Exception:
                logger.error(
                    "AFD FFN participant loop failed. "
                    f"global_rank={self.global_rank}, local_rank={self.local_rank}\n{traceback.format_exc()}"
                )
                os._exit(1)

    def initialize_kv_cache(self) -> None:
        """Profiles the peak memory usage of the model to determine how many
        KV blocks may be allocated without OOMs.

        The engine will first conduct a profiling of the existing memory usage.
        Then, it calculate the maximum possible number of GPU and CPU blocks
        that can be allocated with the remaining free memory.

        .. tip::
            You may limit the usage of GPU memory
            by adjusting the `gpu_memory_utilization` parameter.
        """
        if self.fd_config.parallel_config.do_profile:
            # 1. Get available memory(bytes)
            available_kv_cache_memory = self.worker.determine_available_memory()
            logger.info(f"------- available_kv_cache_memory:{available_kv_cache_memory / 1024**3} GB --------")

            # 2. Calculate the appropriate number of blocks
            model_block_memory_used = self.worker.cal_theortical_kvcache()
            num_blocks_local = int(available_kv_cache_memory // model_block_memory_used)
            logger.info(f"------- model_block_memory_used:{model_block_memory_used / 1024**3} GB --------")
            logger.info(f"------- num_blocks_local:{num_blocks_local} --------")

            if num_blocks_local <= 0:
                raise ValueError(
                    f"The total number of blocks cannot be less than zero bug got {num_blocks_local}. "
                    "Please increase gpu_memory_utilization "
                    "Or decrease max_num_batched_tokens(max model length)."
                )

            if self.ranks > 1:
                num_blocks_local = paddle.full(shape=[1], fill_value=num_blocks_local, dtype="int32")
                group = self.parallel_config.tp_group if self.fd_config.afd_config.enable_afd else None
                dist.all_reduce(num_blocks_local, op=dist.ReduceOp.MIN, group=group)
                num_blocks_local = num_blocks_local.item()

            local_device_rank = self.local_rank % len(self.parallel_config.device_ids.split(","))
            if local_device_rank == 0:
                # 3. Send IPCSignal
                get_profile_block_num = np.zeros(shape=[1], dtype=np.int32)
                self.get_profile_block_num_signal = IPCSignal(
                    name="get_profile_block_num",
                    array=get_profile_block_num,
                    dtype=np.int32,
                    suffix=self.parallel_config.engine_pid,
                    create=False,
                )
                self.get_profile_block_num_signal.value[0] = num_blocks_local
        else:
            num_blocks_local = self.fd_config.cache_config.total_block_num

        if not self.fd_config.afd_config.afd_role == "ffn":
            # 4. init kv_cache with accurate num_blocks
            logger.info(f"------- num_blocks_global: {num_blocks_local} --------")
            self.worker.initialize_cache(num_gpu_blocks=num_blocks_local)
        else:
            logger.info("Collaborate with attention participants to initialize kv cache.")
            # TODO: directly call worker.model_runner.init_cache_ffn_only()?
            self.worker.model_runner.init_cache_ffn_only()

        if self.fd_config.afd_config.enable_afd:
            dist.barrier(self.parallel_config.ep_group)

    def graph_optimize_and_warm_up_model(self) -> None:
        self.worker.graph_optimize_and_warm_up_model()
        # reset cache_messager prefilled_step signal
        if not envs.ENABLE_V1_KVCACHE_SCHEDULER and self.scheduler_config.splitwise_role == "prefill":
            gpu_id = self.worker.model_runner.device_id
            prefilled_step_name = f"splitwise_complete_prefilled_step_{self.local_rank}"
            prefilled_step_idx_data = np.zeros(shape=[1], dtype=np.int32)
            step_shm_value = IPCSignal(
                name=prefilled_step_name, array=prefilled_step_idx_data, dtype=np.int32, suffix=gpu_id, create=False
            )
            step_shm_value.value[0] = -1

    def init_device(self) -> None:
        """Initialize device and Construct model runner"""
        self.worker.init_device()

    def start_task_queue_service(self):
        # Initialize task queue
        if not envs.FD_ENGINE_TASK_QUEUE_WITH_SHM:
            task_address = (
                self.parallel_config.pod_ip,
                self.parallel_config.local_engine_worker_queue_port,
            )
        else:
            task_address = f"/dev/shm/fd_task_queue_{self.parallel_config.local_engine_worker_queue_port}.sock"
        logger.info(f"connect task queue address {task_address}")
        self.task_queue = TaskQueue(
            address=task_address,
            is_server=False,
            num_client=self.parallel_config.tensor_parallel_size,
            client_id=self.parallel_config.tensor_parallel_rank,
            local_data_parallel_id=self.parallel_config.local_data_parallel_id,
        )

    def load_model(self) -> None:
        """Load weights and create model"""

        self.worker.load_model()
        loaded_model_signal_data = np.zeros(shape=[1], dtype=np.int32)
        self.loaded_model_signal = IPCSignal(
            name="loaded_model_signal",
            array=loaded_model_signal_data,
            dtype=np.int32,
            suffix=self.parallel_config.engine_pid,
            create=False,
        )
        if self.ranks > 1:
            if self.fd_config.afd_config.enable_afd:
                paddle.distributed.barrier(self.parallel_config.tp_group)
            else:
                paddle.distributed.barrier()
        self.loaded_model_signal.value[0] = 1

    def run_control_method(self, control_request: ControlRequest) -> None:
        logger.info(f"Global rank: {self.global_rank}, Local rank: {self.local_rank} start to run control request: {control_request}")
        request_id = control_request.request_id
        method = control_request.method
        kwargs = control_request.args

        handler = getattr(self.worker, method, None)
        if handler is None or not callable(handler):
            error_msg = f"Global rank: {self.global_rank}, Local rank: {self.local_rank} unknown control method {method}"
            error_result = ControlResponse(request_id, 400, error_msg)
            asyncio.run(self._ctrl_output.put(error_result))
            return

        try:
            result = handler(**kwargs)
            succ_result = ControlResponse(request_id, 200, "Success", result)
            logger.info(
                f"Global rank: {self.global_rank}, Local rank: {self.local_rank} successfully ran control request: {control_request}, response: {succ_result}"
            )
            asyncio.run(self._ctrl_output.put(succ_result, shm_threshold=100 * 1024 * 1024))
        except Exception as e:
            error_msg = f"Global rank: {self.global_rank}, Local rank: {self.local_rank} failed to run control method {method}: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            error_result = ControlResponse(request_id, 500, error_msg)
            asyncio.run(self._ctrl_output.put(error_result))


def parse_args():
    """
    Parse args from command line
    """
    parser = argparse.ArgumentParser("FastDeploy LLM Inference")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="./output",
        help="model dir",
    )
    parser.add_argument("-mbs", "--max_num_seqs", type=int, default=34, help="max batch size")
    parser.add_argument("--num_gpu_blocks_override", type=int, default=None)
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--pod_ip", type=str, default="127.0.0.1")
    parser.add_argument("--engine_worker_queue_port", type=str, default="9923")
    parser.add_argument("--max_model_len", type=int, default=3072, help="max model len")
    parser.add_argument("--device_ids", type=str, default="0", help="cuda visible devices")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="input dtype")
    parser.add_argument("--enc_dec_block_num", type=int, default=1, help="encoder's decoder num")
    parser.add_argument(
        "--kv_cache_ratio",
        type=float,
        default=0.7,
        help="kv cache ratio for input",
    )
    parser.add_argument("--first_token_id", type=int, default=1, help="first token id")
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="gpu memory utilization",
    )
    parser.add_argument("--engine_pid", type=int, default=None, help="Process ID of engine")
    parser.add_argument("--do_profile", action="store_true", help="do profile or not")
    parser.add_argument("--pad_token_id", type=int, default=-1, help="pad token id")
    parser.add_argument("--eos_tokens_lens", type=int, default=2, help="eos token lens")
    parser.add_argument(
        "--enable_chunked_prefill",
        action="store_true",
        help="enable chunked prefill",
    )
    parser.add_argument(
        "--use_internode_ll_two_stage",
        action="store_true",
        help="enable internode_ll_two_stage",
    )
    parser.add_argument(
        "--speculative_config",
        type=json.loads,
        default=None,
        help="Configuration of SpeculativeConfig.",
    )
    parser.add_argument(
        "--max_num_batched_tokens",
        type=int,
        default=2048,
        help="max num batched tokens",
    )

    parser.add_argument(
        "--enable_prefix_caching",
        action="store_true",
        help="enable prefix cache",
    )
    parser.add_argument(
        "--disable_custom_all_reduce",
        action="store_true",
        help="enable custom all-reduce",
    )
    parser.add_argument(
        "--disable_sequence_parallel_moe",
        action="store_true",
        help="disable sequence parallel moe",
    )
    parser.add_argument("--splitwise_role", type=str, default="mixed", help="splitwise role")
    parser.add_argument("--afd_role", type=str, default=None, help="AFD role: attn, ffn, or None (disabled)")
    parser.add_argument("--afd_master", type=str, default=None, help="AFD master endpoint.")
    parser.add_argument("--afd_nnodes", type=int, default=1, help="AFD instance count.")
    parser.add_argument("--afd_nnode_rank", type=int, default=0, help="AFD instance rank.")
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="tensor parallel size",
    )
    parser.add_argument(
        "--expert_parallel_size",
        type=int,
        default=1,
        help="expert parallel size",
    )
    parser.add_argument(
        "--data_parallel_size",
        type=int,
        default=1,
        help="data parallel size",
    )
    parser.add_argument(
        "--enable_expert_parallel",
        action="store_true",
        help="enable expert parallel",
    )
    parser.add_argument(
        "--enable_chunked_moe",
        action="store_true",
        help="enable chunked moe",
    )
    parser.add_argument(
        "--chunked_moe_size",
        type=int,
        default=256,
        help="chunk size of moe input",
    )
    parser.add_argument("--ori_vocab_size", type=int, default=None)
    parser.add_argument("--think_start_id", type=int, default=-1)
    parser.add_argument("--think_end_id", type=int, default=-1)
    parser.add_argument("--image_patch_id", type=int, default=-1)
    parser.add_argument("--line_break_id", type=int, default=-1)
    parser.add_argument("--think_truncate_prompt_ids", type=json.loads, default=[])

    parser.add_argument(
        "--quantization",
        type=json.loads,
        default=None,
        help="Quantization name for the model, currently support "
        "'wint4', 'wint8',"
        "default is None. The priority of this configuration "
        "is lower than that of the config file. "
        "More complex quantization methods need to be configured via the config file.",
    )
    parser.add_argument(
        "--graph_optimization_config",
        type=json.loads,
        default=None,
        help="Configuration of Graph optimization backend.",
    )
    parser.add_argument(
        "--plas_attention_config",
        type=json.loads,
        default=None,
        help="Configation of plas attention.",
    )
    parser.add_argument(
        "--guided_decoding_backend",
        type=str,
        default="off",
        help="guided decoding backend",
    )
    parser.add_argument(
        "--disable_any_whitespace",
        action="store_true",
        help="Disable any whitespace for guided decoding.",
    )
    parser.add_argument(
        "--dynamic_load_weight",
        action="store_true",
        help="Enable dynamic weight loading strategy",
    )
    parser.add_argument(
        "--load_strategy",
        type=str,
        choices=["ipc", "ipc_snapshot", "meta", "normal", "rsync"],
        default="ipc_snapshot",
        help="Weight loading method when dynamic loading is enabled: "
        "'ipc': real-time IPC streaming with automatic resharding, "
        "'ipc_snapshot': load from disk snapshot of IPC weights.",
    )
    parser.add_argument(
        "--rsync_config",
        type=json.loads,
        default=None,
        help="Rsync weights config",
    )
    parser.add_argument(
        "--enable_logprob",
        action="store_true",
        help="Enable output of token-level log probabilities.",
    )
    parser.add_argument(
        "--max_logprobs",
        type=int,
        default=20,
        help="Maximum number of log probabilities.",
    )
    parser.add_argument(
        "--logprobs_mode",
        type=str,
        default="raw_logprobs",
        help="Indicates the content returned in the logprobs.",
    )
    parser.add_argument(
        "--reasoning_parser",
        type=str,
        default=None,
        help="Flag specifies the reasoning parser to use for extracting reasoning content from the model output",
    )
    parser.add_argument(
        "--early_stop_config",
        type=json.loads,
        default=None,
        help="Configuration of early stop.",
    )

    parser.add_argument(
        "--load_choices",
        type=str,
        default="default_v1",
        help="The format of the model weights to load. default/default_v1/dummy.",
    )

    parser.add_argument(
        "--model_loader_extra_config",
        type=json.loads,
        default=None,
        help="Additional configuration for model loader (JSON format). "
        'e.g., \'{"enable_multithread_load": true, "num_threads": 8}\'',
    )

    parser.add_argument(
        "--ips",
        type=str,
        default=None,
        help="The ips of multinode deployment.",
    )

    parser.add_argument(
        "--lm_head_fp32",
        action="store_true",
        help="Flag to specify dtype of lm_head as FP32",
    )

    parser.add_argument(
        "--moe_gate_fp32",
        action="store_true",
        help="Flag to specify dtype of gate as FP32",
    )

    parser.add_argument(
        "--max_encoder_cache",
        type=int,
        help="Maximum encoder cache tokens(use 0 to disable).",
    )

    parser.add_argument(
        "--model-impl",
        type=str,
        choices=["auto", "fastdeploy", "paddleformers"],
        default="auto",
        help="Model implementation backend (auto, fastdeploy, paddleformers)",
    )

    parser.add_argument(
        "--cache-transfer-protocol",
        type=str,
        default="ipc",
        help="support protocol list, comma separated, default is ipc",
    )
    parser.add_argument(
        "--runner",
        type=str,
        default="auto",
        help="The type of model runner to use.Each FD instance only supports one model runner.even if the same model can be used for multiple types.",
    )

    parser.add_argument(
        "--convert",
        type=str,
        default="auto",
        help="Convert the model using adapters. The most common use case is to adapt a text generation model to be used for pooling tasks.",
    )

    parser.add_argument(
        "--override-pooler-config",
        type=optional_type(json.loads),
        default=None,
        help="Override configuration for the pooler.",
    )

    parser.add_argument(
        "--logits-processors",
        type=str,
        nargs="+",
        default=[],
        help="FQCNs (Fully Qualified Class Names) of logits processors supported by the service.",
    )

    parser.add_argument(
        "--eplb_config",
        type=json.loads,
        default=None,
        help="EPLB Configuration.",
    )

    parser.add_argument(
        "--routing_replay_config",
        type=json.loads,
        default=None,
        help="Configation of Rollout Routing Replay.",
    )

    parser.add_argument(
        "--shutdown_comm_group_if_worker_idle",
        action="store_true",
        help="Shutdown comm group if worker idle.",
    )

    parser.add_argument(
        "--enable_entropy",
        action="store_true",
        help="Enable output of token-level entropy.",
    )

    parser.add_argument(
        "--mm_max_tokens_per_item",
        type=json.loads,
        default=None,
        help="Maximum tokens per item in mm input.",
    )

    parser.add_argument(
        "--num_cpu_blocks",
        type=int,
        default=0,
        help="Number of cpu blocks.",
    )
    parser.add_argument(
        "--kvcache_storage_backend",
        type=str,
        help="KVCache storage backend.",
    )

    parser.add_argument(
        "--enable_overlap_schedule",
        action="store_true",
        help="Enable overlap schedule",
    )

    parser.add_argument(
        "--ep_prefill_use_worst_num_tokens",
        action="store_true",
        help="enable to avoid cpu sync",
    )

    parser.add_argument(
        "--deploy_modality",
        type=str,
        default="mixed",
        choices=["mixed", "text"],
        help="Deploy modality: 'mixed' for multimodal, 'text' for text-only.",
    )

    args = parser.parse_args()
    return args


def initialize_fd_config(
    args,
    world_size: int = 1,
    ranks: int = 1,
    global_rank: int = 0,
    local_rank: int = 0,
    afd_node_srank: int = 0,
    afd_attn_tp_size: int = 1,
    afd_attn_ranks: List[int] | None = None,
    afd_ffn_ranks: List[int] | None = None,
) -> FDConfig:
    """Initialize FDConfig from either RolloutModelConfig or argparse.Namespace

    Args:
        config: Configuration object containing all parameters (either RolloutModelConfig or argparse.Namespace)

    Returns:
        FDConfig: Initialized FastDeploy configuration object
    """
    # RL rollout
    paddle.set_default_dtype(args.dtype)
    model_config = ModelConfig(vars(args))
    device_config = DeviceConfig(vars(args))
    speculative_config = SpeculativeConfig(args.speculative_config)
    parallel_config = ParallelConfig(vars(args), world_size)
    cache_config = CacheConfig(vars(args))
    scheduler_config = SchedulerConfig(vars(args))
    eplb_config = EPLBConfig(args.eplb_config)
    afd_config = AFDConfig(vars(args), afd_node_srank, afd_attn_tp_size)
    if afd_config.enable_afd:
        afd_config.set_rank_topology(world_size, afd_attn_ranks or [], afd_ffn_ranks or [])
        num_logical_experts = (
            model_config.moe_num_experts[0]
            if isinstance(model_config.moe_num_experts, list)
            else model_config.moe_num_experts
        )
        num_redundant_experts = eplb_config.redundant_experts_num if eplb_config.enable_eplb else 0
        afd_config.set_expert_layout(num_logical_experts, num_redundant_experts)

    parallel_config.tensor_parallel_rank = local_rank % parallel_config.tensor_parallel_size
    parallel_config.data_parallel_rank = local_rank // parallel_config.tensor_parallel_size
    # config for DP
    if parallel_config.data_parallel_size > 1:
        max_chips_per_node = 16 if current_platform.is_iluvatar() else 8
        parallel_config.local_data_parallel_id = parallel_config.data_parallel_rank % (
            max_chips_per_node // parallel_config.tensor_parallel_size
        )
    # config for EP, may be need change
    if parallel_config.expert_parallel_size > 1:
        expert_parallel_rank = global_rank if afd_config.enable_afd else local_rank % parallel_config.expert_parallel_size
        if afd_config.enable_afd:
            num_experts_per_rank = afd_config.afd_num_local_physical_experts
        else:
            if isinstance(model_config.moe_num_experts, list):
                num_experts = model_config.moe_num_experts[0] + eplb_config.redundant_experts_num
            elif hasattr(model_config, "num_local_experts") and model_config.num_local_experts is not None:
                num_experts = model_config.num_local_experts + eplb_config.redundant_experts_num
            else:
                num_experts = model_config.moe_num_experts + eplb_config.redundant_experts_num
            num_experts_per_rank = num_experts // parallel_config.expert_parallel_size
        num_experts_start_offset = expert_parallel_rank * num_experts_per_rank
        parallel_config.expert_parallel_rank = expert_parallel_rank
        parallel_config.num_experts_per_rank = num_experts_per_rank
        parallel_config.num_experts_start_offset = num_experts_start_offset

    parallel_config.set_communicate_group(afd_config)

    load_config = LoadConfig(vars(args))

    graph_opt_config = GraphOptimizationConfig(args.graph_optimization_config)

    plas_attention_config = PlasAttentionConfig(args.plas_attention_config)

    early_stop_config = EarlyStopConfig(args.early_stop_config)

    structured_outputs_config: StructuredOutputsConfig = StructuredOutputsConfig(args=vars(args))
    routing_replay_config = RoutingReplayConfig(args.routing_replay_config)

    # Note(tangbinhan): used for load_checkpoint
    model_config.pretrained_config.tensor_parallel_rank = parallel_config.tensor_parallel_rank
    model_config.pretrained_config.tensor_model_parallel_size = parallel_config.tensor_parallel_size
    model_config.pretrained_config.is_mtp = False
    model_config.pretrained_config.head_dim = model_config.head_dim

    logger.info(f"parallel_config.use_ep {parallel_config.use_ep}")
    logger.info(f"parallel_config.tensor_parallel_size {parallel_config.tensor_parallel_size}")
    logger.info(f"parallel_config.tensor_parallel_rank {parallel_config.tensor_parallel_rank}")
    logger.info(f"parallel_config.engine_worker_queue_port {parallel_config.engine_worker_queue_port}")

    if getattr(model_config, "num_hidden_layers", None) is None:
        raise ValueError("num_hidden_layers is None")

    quant_config = parse_quant_config(
        args,
        model_config,
        is_ernie=(
            ErnieArchitectures.contains_ernie_arch(model_config.architectures)
            or ErnieArchitectures.is_ernie5_arch(model_config.architectures)
        ),
        is_v1_loader=load_config.load_choices == "default_v1",
    )

    # Log quantization info
    logger.info("===========quantization_config==============")
    if quant_config is not None:
        if model_config.is_quantized:
            logger.info("Model Status: Offline Quantized (pre-quantized weights loaded)")
        else:
            logger.info("Model Status: Original (will apply online quantization)")

        logger.info(f"{model_config.quantization_config}")
    else:
        logger.info("No quantization config found and use original weight and act dtype.")

    logger.info(f"- Dynamic load weight: {load_config.dynamic_load_weight}")
    logger.info(f"- Load strategy: {load_config.load_strategy}")
    logger.info(f"- Rsync config: {load_config.rsync_config}, {type(load_config.rsync_config)}")

    if not (
        current_platform.is_cuda()
        or current_platform.is_xpu()
        or current_platform.is_maca()
        or current_platform.is_iluvatar()
        or current_platform.is_intel_hpu()
    ):
        logger.info("Set ENABLE_V1_KVCACHE_SCHEDULER to 0 due to not supported.")
        envs.ENABLE_V1_KVCACHE_SCHEDULER = 0

    if envs.ENABLE_V1_KVCACHE_SCHEDULER and args.splitwise_role == "prefill":
        os.environ["PREFILL_NODE_ONE_STEP_STOP_V1"] = "1"
    elif envs.ENABLE_V1_KVCACHE_SCHEDULER and args.splitwise_role == "decode":
        os.environ["PREFILL_NODE_ONE_STEP_STOP_V1"] = "0"

    fd_config = FDConfig(
        model_config=model_config,
        parallel_config=parallel_config,
        speculative_config=speculative_config,
        device_config=device_config,
        load_config=load_config,
        quant_config=quant_config,
        graph_opt_config=graph_opt_config,
        early_stop_config=early_stop_config,
        cache_config=cache_config,
        scheduler_config=scheduler_config,
        afd_config=afd_config,
        ips=args.ips,
        plas_attention_config=plas_attention_config,
        structured_outputs_config=structured_outputs_config,
        eplb_config=eplb_config,
        routing_replay_config=routing_replay_config,
        deploy_modality=DeployModality.from_str(getattr(args, "deploy_modality", "mixed")),
    )
    logger.info(f"parallel_config.local_engine_worker_queue_port {parallel_config.local_engine_worker_queue_port}")

    update_fd_config_for_mm(fd_config)
    if fd_config.load_config.load_choices == "default_v1" and not v1_loader_support(fd_config):
        fd_config.load_config.load_choices = "default"

    architecture = fd_config.model_config.architectures[0]
    if "PaddleOCR" in architecture:
        envs.FD_ENABLE_MAX_PREFILL = 1
        fd_config.cache_config.enable_prefix_caching = False
        fd_config.cache_config.max_encoder_cache = 0

    return fd_config


@paddle.no_grad()
def run_worker_proc() -> None:
    """
    start worker process
    """
    # Get args form Engine
    args = parse_args()

    # Note:
    # world_size: 全部 AFD world 的 rank 数
    # global_rank: 当前 worker 在整个 world 中的 rank
    # local_rank: 当前 worker 在当前 api_server 实例内的 rank
    # ranks: 当前 api_server 实例内的 worker 数
    # afd_node_srank: 当前 api_server 实例在整个 world 中的起始 rank

    world_size, global_rank = init_distributed_environment()

    local_rank = global_rank
    ranks = world_size
    afd_node_srank = 0
    afd_attn_tp_size = 1
    attn_ranks = []
    ffn_ranks = []

    if args.afd_role is not None:
        ranks, local_rank, attn_ranks, ffn_ranks, afd_node_srank, afd_attn_tp_size = collect_afd_rank_info(
            args, world_size, global_rank
        )

    # Get fd_config
    fd_config = initialize_fd_config(
        args,
        world_size=world_size,
        ranks=ranks,
        global_rank=global_rank,
        local_rank=local_rank,
        afd_node_srank=afd_node_srank,
        afd_attn_tp_size=afd_attn_tp_size,
        afd_attn_ranks=attn_ranks,
        afd_ffn_ranks=ffn_ranks,
    )

    # Create worker process
    if current_platform.is_iluvatar():
        from fastdeploy.worker.iluvatar_worker import IluvatarPaddleDisWorkerProc

        worker_proc = IluvatarPaddleDisWorkerProc(fd_config, ranks, local_rank)
    else:
        worker_proc = PaddleDisWorkerProc(fd_config, world_size, ranks, global_rank, local_rank)
        worker_proc.init_control()

    # Enable batch-invariant mode for deterministic inference.
    # This must happen AFTER worker creation but BEFORE model loading,
    # because enable_batch_invariant_mode() calls paddle.enable_compat()
    # which makes torch appear available via proxy. If called before worker creation,
    # the gpu_model_runner import chain (ernie4_5_vl_processor → paddleformers →
    # transformers) will fail when transformers tries to query torch metadata.
    if envs.FD_DETERMINISTIC_MODE:
        from fastdeploy.model_executor.layers.batch_invariant_ops import (
            init_deterministic_mode,
        )

        init_deterministic_mode()

    # Initialize device and create model runner
    worker_proc.init_device()

    # Load model
    worker_proc.load_model()
    # Initialize KV Cache
    worker_proc.initialize_kv_cache()

    # Trigger CUDAGraph capture
    worker_proc.graph_optimize_and_warm_up_model()

    # Initialize health status
    worker_proc.init_health_status()

    worker_proc.start_task_queue_service()

    # Start event loop
    worker_proc.event_loop_normal()


if __name__ == "__main__":
    run_worker_proc()
