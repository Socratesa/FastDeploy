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
    LaunchConfig,
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
    create_mmap,
    load_tensor_from_shm_mem,
)
from fastdeploy.eplb.experts_manager import RedundantExpertManager
from fastdeploy.fault_tolerant.strategy import coverage_repair, reserve_experts, keep_from_order
from fastdeploy.fault_tolerant.stats_manager import FaultTolStatsManager
from fastdeploy.eplb.utils import (
    derive_expert_tables,
    dump_redundant_expert_table_snapshot,
)
from fastdeploy.inter_communicator import EngineWorkerQueue as TaskQueue
from fastdeploy.inter_communicator import (
    EplbCollectiveOp,
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
from fastdeploy.utils import all_gather_values, get_logger, optional_type
from fastdeploy.weight_cache.expert_weight_daemon import ExpertBlockSpec, READY_TIMEOUT
from fastdeploy.weight_cache.expert_weight_shm import ExpertWeightShm, block_paths
from fastdeploy.worker.worker_base import WorkerBase

if envs.FD_USE_MOONCAKE_PG:
    assert paddle.in_dynamic_mode(), "Mooncake process group only supports paddle dynamic mode."
    from fastdeploy.utils import get_host_ip
    from fastdeploy.cache_manager.transfer_factory.utils import get_rdma_nics
    from mooncake.integration.paddle import ProcessGroupMooncake as pgmc
    pgmc.set_host_ip(get_host_ip())
    pgmc.set_device_filter(get_rdma_nics().split(","))
    dist.register_process_group_backend("mooncake", pgmc.create)
    # select backend for ``paddle.distributed.init_parallel_env()``
    os.environ["PADDLE_DISTRI_BACKEND"] = "mooncake"

logger = get_logger("worker_process", "worker_process.log")

def dump_mooncake_debug_state(worker, fd_config, prefix: str = "") -> None:
    ep_group = fd_config.parallel_config.ep_group.process_group
    tp_group = fd_config.parallel_config.tp_group.process_group

    from fastdeploy.model_executor.layers.moe.ep import EPBackend
    backend = EPBackend()
    ep_buffer_active_ranks = backend.active_ranks.clone()

    log_str = (
        f"{prefix} mooncake debug state: "
        f"worker_rank={getattr(worker, 'rank', None)}, "
        f"local_rank={getattr(worker, 'local_rank', None)}, "
        f"ep_buffer_active_ranks={ep_buffer_active_ranks},"
        f"ep_group={ep_group.get_active_ranks()}"
    )
    log_str += f", tp_group={tp_group.get_active_ranks()}" if tp_group else ""

    logger.info(log_str)


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
        if fd_config.afd_config.num_ffn_ranks <= 0:
            raise ValueError("AFD EPLB requires at least one FFN rank for async-load shared memory sizing.")
        return fd_config.afd_config.num_ffn_ranks
    return parallel_config.expert_parallel_size


def init_distributed_environment(seed: int = 20, is_extension: bool = False) -> Tuple[int, int]:
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

        comm_group_mode = "full"
        world_pg_opts = None
        if envs.FD_USE_MOONCAKE_PG:
            comm_group_mode = "minimal"
            world_pg_opts = {
                "active_value": 0,
                "is_extension": is_extension,
                "max_world_size": None,
                "global_ranks_in_group": list(range(world_size)),
            }

        fleet.init(
            is_collective=True,
            strategy=dist_strategy,
            comm_group_mode=comm_group_mode,
            pg_options=world_pg_opts,
        )

        global_rank = fleet.worker_index()
    else:
        global_rank = 0

    return world_size, global_rank


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

    def __init__(self, fd_config: FDConfig, world_size: int = 1, global_rank: int = 0) -> None:
        """
        Initialize a distributed worker and task queue for single-node multi-GPU setup.
        Args:
            fd_config (FDConfig): Arguments related to inference, containing
                attributes such as weight_dtype, act_dtype, mp_size, hidden_size, head_dim,
                num_attention_heads, and ffn_hidden_size.
        """
        self.world_size = world_size
        self.ranks = world_size
        self.global_rank = global_rank
        self.local_rank = global_rank
        self.fd_config = fd_config
        self.parallel_config = fd_config.parallel_config
        self.cache_config = fd_config.cache_config
        self.scheduler_config = fd_config.scheduler_config
        self.eplb_config = fd_config.eplb_config

        if fd_config.afd_config.enable_afd:
            self.ranks = len(fd_config.afd_config.inst_ranks)
            self.local_rank = global_rank - fd_config.afd_config.inst_first_rank

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

        redundant_table_manger = self.worker.get_model().redundant_table_manger
        if self.fd_config.afd_config.is_attn:
            if redundant_table_manger is None:
                raise RuntimeError("AFD ATTN EPLB update requires a redundant table manager.")
            rank_expert_list, logical_to_physical_map, expert_count = (
                self.experts_manager.get_ep_rank_to_expert_id_list()
            )
            redundant_table_manger.update_expert_rank_table(rank_expert_list, logical_to_physical_map, expert_count)
            redundant_table_manger.refresh_active_expert_rank_table()
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
        redundant_table_manger.update_expert_rank_table(rank_expert_list, logical_to_physical_map, expert_count)
        redundant_table_manger.refresh_active_expert_rank_table()
        # TO BE FIXED
        self.worker.get_model().update_state_dict(state_dicts)
        if self.expert_weight_shm is None:
            self.experts_manager.tensor_infos = None

    def _broadcast_model_weights_signal(self, src: int, group) -> int:
        model_weights_signal_tensor = paddle.full(shape=[1], fill_value=self.model_weights_signal[0], dtype="int32")
        paddle.distributed.broadcast(model_weights_signal_tensor, src=src, group=group)
        value = model_weights_signal_tensor.numpy()[0]
        return int(value)

    def _get_exist_task_flag(self) -> bool:
        if self.nnode > 1:
            return self.task_queue.read_finish_flag.get() == 1
        else:
            return self.exist_task_signal.value[0] == ExistTaskStatus.EXIST

    def _update_exist_task_flag(self, flag: bool) -> None:
        if self.nnode > 1:
            self.task_queue.read_finish_flag.set(int(flag))
        else:
            self.exist_task_signal.value[0] = ExistTaskStatus.EXIST if flag else ExistTaskStatus.EMPTY

    def _tp_barrier_wait(self):
        if (
            current_platform.is_xpu()
            or self.enable_overlap_schedule
            and not self.fd_config.launch_config.enable_fault_tolerant
        ):
            self.task_queue.worker_process_tp_barrier.wait()
        else:
            # A host-side barrier does not enter the compute stream,
            # so it does not drain the pending forward.
            if envs.FD_ENABLE_CPU_GROUP:
                paddle.distributed.barrier(self.parallel_config.tp_group_cpu)
            else:
                paddle.distributed.barrier(self.parallel_config.tp_group)

    def init_eplb(self):
        if not self.eplb_config.enable_eplb:
            logger.info(f"Skip eplb init. enable_eplb: {self.eplb_config.enable_eplb}")
            return

        assert (
            envs.FD_ENABLE_CPU_GROUP and getattr(self.parallel_config, "ep_group_cpu", None) is not None
        ), "eplb requires FD_ENABLE_CPU_GROUP=1"

        start_time = time.perf_counter()
        # Run before RedundantExpertManager forks the async loader: the loader
        # inherits shm_fd, which is the only handle left once create_mmap unlinks the file.
        if self.fd_config.afd_config.is_attn:
            self.mmap_infos = None
            shm_fd = -1
        elif self.expert_weight_shm is not None:
            # The daemon's block already holds every expert, so there is nothing to load
            # from disk and no loader process to fork.
            self.mmap_infos = {MODEL_MAIN_NAME: self.expert_weight_shm.ptr}
            shm_fd = -1
        else:
            logger.info("Creating shared memory for eplb.")
            self.mmap_infos, shm_fds = create_mmap(
                [MODEL_MAIN_NAME],
                self.parallel_config.expert_parallel_rank,
                _get_eplb_shmem_parallel_size(self.fd_config, self.parallel_config),
                shm_uuid=self.parallel_config.local_engine_worker_queue_port,
                eplb_config=self.eplb_config,
                logger=logger,
            )
            shm_fd = shm_fds[MODEL_MAIN_NAME]

        self.experts_manager = RedundantExpertManager(
            rank=self.parallel_config.expert_parallel_rank,
            ep_size=self.parallel_config.expert_parallel_size,
            fd_config=self.fd_config,
            ipc_signal_suffix=self.parallel_config.local_engine_worker_queue_port,
            shm_fd=shm_fd,
            ft_stats=self.ft_stats,
        )
        if self.expert_weight_shm is not None:
            # The block is immutable, so its infos stay valid across rearranges.
            self.experts_manager.tensor_infos = self.expert_weight_shm.tensor_infos()

        end_time = time.perf_counter()
        logger.info(f"Initialize eplb took {end_time - start_time} seconds.")
        paddle.distributed.barrier(self.parallel_config.ep_group)

    def refresh_expert_rank_table(self, dump_table_snapshot: bool = False):
        redundant_table_manger = self.worker.get_model().redundant_table_manger

        if dump_table_snapshot:
            dump_redundant_expert_table_snapshot(
                fd_config=self.fd_config,
                rank_expert_list=redundant_table_manger.model_ep_rank_to_expert_id_list,
                logical_to_physical_map=redundant_table_manger.model_active_expert_id_to_ep_rank_array,
                expert_count=redundant_table_manger.model_active_expert_in_rank_num_list,
                source="active_table_before",
                local_rank=self.fd_config.parallel_config.expert_parallel_rank,
                clear_stat=False,
            )

        redundant_table_manger.refresh_active_expert_rank_table()

        if dump_table_snapshot:
            dump_redundant_expert_table_snapshot(
                fd_config=self.fd_config,
                rank_expert_list=redundant_table_manger.model_ep_rank_to_expert_id_list,
                logical_to_physical_map=redundant_table_manger.model_active_expert_id_to_ep_rank_array,
                expert_count=redundant_table_manger.model_active_expert_in_rank_num_list,
                source="active_table_after",
                local_rank=self.fd_config.parallel_config.expert_parallel_rank,
                clear_stat=False,
            )

    def _handle_rank_failures(self):
        from fastdeploy.model_executor.layers.moe.ep import EPBackend

        afd_config = self.fd_config.afd_config
        local = afd_config.num_local_physical_experts
        table_manager = self.worker.get_model().redundant_table_manger

        active = EPBackend().active_ranks.numpy()
        survivors = [r for r in afd_config.ffn_ranks if active[r] > 0]
        casualties = [r for r in afd_config.ffn_ranks if active[r] <= 0]
        if not survivors or not casualties:
            logger.info(f"no ffn rank failed (live ffn ranks={survivors})")
            return

        # Every survivor adds its own counts, the same way the eplb rebalance does.
        load = table_manager.get_tokens_stats_snapshot()
        paddle.distributed.all_reduce(
            load, op=paddle.distributed.ReduceOp.SUM, group=self.parallel_config.ep_group_cpu
        )

        # Plan over surviving slots only, compacted to survivor order.
        phy2log = table_manager.model_ep_rank_to_expert_id_list.numpy()
        slots = np.concatenate([phy2log[:, g * local : (g + 1) * local] for g in survivors], axis=1)

        # drop / retopk: no importance stats are supplied, so we do not try to pull
        # the lost experts back onto the survivors. Rebuild the table with the dead
        # ranks' slots emptied (-1) and keep every survivor's own experts in place;
        # get_moe_scores then drops (topk id -1) or re-topks the logical experts that
        # no longer have a live replica. When stats are provided we still plan a
        # coverage repair that brings the most important lost experts back.
        stats = self.ft_stats
        if stats is None:
            moves, new_slots = [], slots
        else:
            keep = keep_from_order(stats.order, slots.shape[1])
            # coverage_repair orders which lost expert to bring back first, so collapse the
            # domains to the most demanding one.
            moves, new_slots = coverage_repair(
                keep, slots, stats.importance.max(axis=0), load.numpy(), len(survivors)
            )
            if not moves:
                logger.info("rank failure needs no expert moves")
                return

        # Survivors take the new placement, while casualties reserve experts nobody kept.
        phy2log = np.full_like(phy2log, -1)
        for index, g in enumerate(survivors):
            phy2log[:, g * local : (g + 1) * local] = new_slots[:, index * local : (index + 1) * local]
        max_replicas = table_manager.model_expert_id_to_ep_rank_array.shape[-1]
        dropped = stats.order[:, : max(0, afd_config.num_logical_experts - slots.shape[1])]
        reserve_experts(phy2log, dropped, load.numpy(), casualties, local, max_replicas)
        log2phy, expert_count = derive_expert_tables(phy2log, afd_config.num_logical_experts, max_replicas)

        # Table first: update_state_dict resolves logical experts to slots through it.
        table_manager.update_expert_rank_table(phy2log, log2phy, expert_count, clear_stat=False)
        logger.info(
            f"rebalanced {len(moves)} experts after "
            f"losing {len(afd_config.ffn_ranks) - len(survivors)} rank(s)"
        )

        if moves and self.expert_weight_shm is not None:
            wanted = {(layer, expert) for layer, _, expert in moves}
            names = {
                name
                for name, entry in self.expert_weight_shm.entries.items()
                if (entry["layer"], entry["expert"]) in wanted
            }
            infos = self.expert_weight_shm.tensor_infos(names)
            state_dicts = load_tensor_from_shm_mem(infos, self.expert_weight_shm.ptr)
            self.worker.get_model().update_state_dict(state_dicts)
        # ATTN waits here until every FFN rank has loaded the moved expert weights.
        paddle.distributed.barrier(self.parallel_config.ep_group)
        return phy2log

    def _run_ft(self):
        if (
            not self.fd_config.afd_config.enable_afd
            or not hasattr(self.worker.model_runner, "rank_liveness_changed")
            or not self.worker.model_runner.rank_liveness_changed
        ):
            return

        logger.info(f">>> start handling rank failures")
        begin_ts = time.perf_counter()
        phy2log = self._handle_rank_failures()
        logger.info(f"<<< rank failures handled in {(time.perf_counter() - begin_ts):.3f}s")

        self.refresh_expert_rank_table(dump_table_snapshot=False)
        if phy2log is not None and self.eplb_config.enable_eplb:
            self.experts_manager.set_expert_rank_table(phy2log)

    def _sync_eplb_op(self, tp_rank):
        local_op, local_signal = EplbCollectiveOp.NONE, None
        if tp_rank == 0:
            for signal, candidate in (
                (
                    self.experts_manager.signal_allreduce_expert_tokens_stats_array,
                    EplbCollectiveOp.ALLREDUCE_TOKENS_STATS,
                ),
                (
                    self.experts_manager.signal_allreduce_load_weight_result_array,
                    EplbCollectiveOp.ALLREDUCE_LOAD_WEIGHT_RESULT,
                ),
                (
                    self.experts_manager.signal_update_weight_from_tensor_array,
                    EplbCollectiveOp.UPDATE_WEIGHT_FROM_TENSOR,
                ),
            ):
                if signal.value[0] == 1:
                    local_op, local_signal = candidate, signal
                    break

        data = paddle.to_tensor([local_op.value], dtype="int32", place="cpu")
        paddle.distributed.all_reduce(
            data, op=paddle.distributed.ReduceOp.MAX, group=self.parallel_config.ep_group_cpu
        )
        op = EplbCollectiveOp(int(data.numpy()[0]))

        if local_signal is not None and local_op is op:
            local_signal.value[0] = 0
        if op is not EplbCollectiveOp.NONE:
            logger.info(f"redundant_expert: ep group runs {op.name}")
        return op

    def _apply_expert_weights(self, tp_rank):
        """Swap the preloaded expert weights and the new routing table into the model."""
        begin_ts = time.time()
        # update_weights_from_tensor overwrites model weights in place, so the pending
        # forward has to be drained first: the op negotiation runs on the cpu group and
        # therefore does not order device work.
        paddle.device.synchronize()
        self.update_weights_from_tensor(self.mmap_infos)
        logger.info(f"redundant_expert: update_weight_from_tensor success, cost {(time.time() - begin_ts)*1000}ms")
        paddle.distributed.barrier(self.parallel_config.ep_group)
        if tp_rank == 0:
            self.experts_manager.rearrange_experts_signal.value[0] = RearrangeExpertStatus.DONE.value
        logger.info("redundant_expert: done")

    def _run_eplb(self, tp_rank):
        """internal call to run eplb"""
        if not self.eplb_config.enable_eplb:
            return

        cpu_group = self.parallel_config.ep_group_cpu
        op = self._sync_eplb_op(tp_rank)
        if op is EplbCollectiveOp.ALLREDUCE_TOKENS_STATS:
            tokens_stats = self.worker.get_model().redundant_table_manger.get_tokens_stats_snapshot()
            self.experts_manager.allreduce_expert_tokens_stats(tokens_stats, cpu_group)
        elif op is EplbCollectiveOp.ALLREDUCE_LOAD_WEIGHT_RESULT:
            result = paddle.to_tensor(self.experts_manager.update_weight_from_disk_result.value, place="cpu")
            paddle.distributed.all_reduce(result, op=paddle.distributed.ReduceOp.MIN, group=cpu_group)
            self.experts_manager.on_load_weight_result_allreduced(int(result.numpy()[0]))
        elif op is EplbCollectiveOp.UPDATE_WEIGHT_FROM_TENSOR:
            self._apply_expert_weights(tp_rank)

    def event_loop_normal(self) -> None:
        """Main event loop for Paddle Distributed Workers.
        TODO(gongshaotian): support remote calling of functions that control worker.
        """
        tp_size = self.parallel_config.tensor_parallel_size
        # Currently, only support single node
        self.nnode = (tp_size + self.max_chips_per_node) // self.max_chips_per_node
        max_occupied_batch_index = 0
        tp_rank = self.parallel_config.tensor_parallel_rank

        # TODO: Unify status variables model_weights_status (shared memory) and model_weights_signal (numpy array) to one
        self.model_weights_signal = np.zeros([1], dtype=np.int32)

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
                        self._update_exist_task_flag(True)

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

            if self._get_exist_task_flag():
                logger.debug(f"Global rank: {self.global_rank}, Local rank: {self.local_rank} detected new requests.")
                if not self.fd_config.launch_config.is_extension:
                    self.engine_forward_signal.value[0] = 1
                tasks, read_finish, dropped = self.task_queue.get_tasks()
                # Only one of all tp_size client will get read_finish == True.
                if read_finish:
                    # Reset the two signal.
                    self._update_exist_task_flag(False)
                # Some ranks are deactivated, skip this round.
                if dropped:
                    continue
                # In EP parallel(corresponing to dp attention), we need to barrier for prefill to prevent data imbalance due to inconsistent data arrival.
                # Only EP + DP prefill should barrier for data arrival.
                # In mixed mode and decoder in D, we should not barrier to influence decoding.
                if self.parallel_config.use_ep and self.scheduler_config.splitwise_role == "prefill":
                    paddle.distributed.barrier(self.parallel_config.ep_group)

                req_dicts, control_reqs = [], []
                assert (
                    len(tasks) > 0 or self.fd_config.launch_config.is_extension
                ), f"task_queue.get_tasks() should contain at least one tuple, [([req1, ...] ,real_bsz)], but got len(tasks)={len(tasks)}"
                # In EP + DP prefill, empty task ([]) is delived in worker to barrier. For empty task, just skip and continue.
                # tasks[0] contains two part, ([req1, ...] ,real_bsz)
                # tasks[0][0] is [req1, ...]
                # if empty batch is delived, eval(tasks[0][0]) should be False ([]),
                # if batch with requests is delived, eval(tasks[0][0]) should be True, then to be processed as below.
                if tasks and tasks[0][0]:
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
                # Synchronize the signal for other workers
                self._tp_barrier_wait() if tp_size > 1 else None
                if self.scheduler_config.splitwise_role == "prefill":
                    continue

            # Let the ep group run control method synchronically
            if envs.FD_ENABLE_EP_CONTROL_REQ_POLL and self.parallel_config.use_ep:
                pendings = all_gather_values(
                    len(self.cached_control_reqs),
                    self.parallel_config.ep_group_cpu if envs.FD_ENABLE_CPU_GROUP else self.parallel_config.ep_group,
                    on_cpu=envs.FD_ENABLE_CPU_GROUP,
                )
                if hasattr(self.parallel_config, "ep_active_ranks"):
                    pendings = [p for p, active in zip(pendings, self.parallel_config.ep_active_ranks.tolist()) if active]
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

            if self.fd_config.launch_config.is_extension:
                time.sleep(0.001)
                continue

            # Execute model to generate token. The generated token will be written to the buffer.
            # These generated tokens can be obtained through get_output op.
            start_execute_time = time.time()

            self.worker.execute_model(req_dicts, max_occupied_batch_index)

            # Only v0 use this signal
            if not envs.ENABLE_V1_KVCACHE_SCHEDULER:
                self.exist_prefill_task_signal.value[0] = self.worker.exist_prefill()
            logger.debug(f"execute model cost: {time.time()-start_execute_time:.5f} s")
            # handle rank failures
            self._run_ft()
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

        if not self.fd_config.afd_config.is_ffn:
            # 4. init kv_cache with accurate num_blocks
            logger.info(f"------- num_blocks_global: {num_blocks_local} --------")
            self.worker.initialize_cache(num_gpu_blocks=num_blocks_local)
        elif not self.fd_config.launch_config.is_extension:
            logger.info("Collaborate with attention participants to initialize kv cache.")
            # TODO: directly call worker.model_runner.init_cache_ffn_only()?
            self.worker.model_runner.init_cache_ffn_only()

        if self.fd_config.afd_config.enable_afd and not self.fd_config.launch_config.is_extension:
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

    def maybe_attach_expert_weight_shm(self) -> None:
        """Attach the daemon's expert weight block, waiting for it if it is still building."""
        self.expert_weight_shm = None
        if not self.fd_config.launch_config.enable_expert_weight_daemon or self.fd_config.afd_config.is_attn:
            return
        inst_id = self.parallel_config.engine_worker_queue_port[0]
        _, _, ready_path = block_paths(inst_id)
        deadline = time.time() + READY_TIMEOUT
        while not os.path.exists(ready_path) and time.time() < deadline:
            time.sleep(1)
        try:
            shm = ExpertWeightShm(inst_id)
        except FileNotFoundError:
            logger.warning(
                f"expert weight block not ready after {READY_TIMEOUT}s, falling back to disk: {inst_id}"
            )
            return
        fingerprint = ExpertBlockSpec.from_fd_config(self.fd_config).fingerprint()
        if shm.fingerprint != fingerprint:
            logger.warning(f"expert weight block fingerprint mismatch, ignoring {inst_id}")
            shm.close()
            return
        self.expert_weight_shm = shm
        self.fd_config.load_config.weight_cache = shm
        logger.info(f"attached expert weight block {inst_id} from pid {shm.daemon_pid}, {len(shm.entries)} entries")

    def maybe_load_ft_stats(self) -> None:
        """Read the externally produced importance/similarity tables."""
        self.ft_stats = None
        path = self.fd_config.launch_config.fault_tolerance_stats_path
        if path is None:
            return
        model_config = self.fd_config.model_config
        self.ft_stats = FaultTolStatsManager(
            path,
            num_layers=model_config.num_hidden_layers,
            num_experts=model_config.moe_num_experts,
        )
        logger.info(f"loaded fault tolerance stats from {path}: {self.ft_stats.importance.shape}")

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
    parser.add_argument("--ninsts", type=int, default=1, help="AFD instance count.")
    parser.add_argument("--inst_rank", type=int, default=0, help="AFD instance rank.")
    parser.add_argument(
        "--enable_dbo",
        action="store_true",
        help="Enable AFD dual-batch overlap.",
    )
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

    parser.add_argument(
        "--enable_fault_tolerant",
        action="store_true",
        default=False,
        help="Enable fault tolerant mechanism.",
    )

    parser.add_argument(
        "--enable_expert_weight_daemon",
        action="store_true",
        default=False,
        help="Serve expert weights from a pinned shared-memory block.",
    )

    parser.add_argument(
        "--fault_tolerance_stats_path",
        type=str,
        default=None,
        help="File holding the per-expert importance and similarity tables.",
    )

    parser.add_argument(
        "--is_extension",
        action="store_true",
        default=False,
        help="Whether this worker is an extension worker.",
    )

    args = parser.parse_args()
    return args


def initialize_fd_config(
    args,
    world_size: int = 1,
    global_rank: int = 0,
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
    parallel_config = ParallelConfig(vars(args))
    cache_config = CacheConfig(vars(args))
    scheduler_config = SchedulerConfig(vars(args))
    eplb_config = EPLBConfig(args.eplb_config)
    afd_config = AFDConfig(vars(args))
    launch_config = LaunchConfig(vars(args))

    if afd_config.enable_afd:
        parallel_config.expert_parallel_size = afd_config.afd_world_size
        num_logical_experts = (
            model_config.moe_num_experts[0]
            if isinstance(model_config.moe_num_experts, list)
            else model_config.moe_num_experts
        )
        num_redundant_experts = eplb_config.redundant_experts_num if eplb_config.enable_eplb else 0
        afd_config.set_expert_layout(num_logical_experts, num_redundant_experts)
        # The token dimension is sharded by the ATTN instance, so both roles must
        # decide this from the ATTN instance size.
        parallel_config.use_sequence_parallel_moe = (
            parallel_config.use_sequence_parallel_moe and afd_config.attn_inst_size > 1
        )

    local_rank = global_rank
    if afd_config.enable_afd:
        # reset local_rank
        local_rank = global_rank - afd_config.inst_first_rank

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
            num_experts_per_rank = afd_config.num_local_physical_experts
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

    parallel_config.set_communicate_group(afd_config, launch_config)

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
        launch_config=launch_config,
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
    # inst_first_rank: 当前 api_server 实例在整个 world 中的起始 rank

    world_size, global_rank = init_distributed_environment(is_extension=args.is_extension)

    # Get fd_config
    fd_config = initialize_fd_config(args, world_size, global_rank)

    # Create worker process
    if current_platform.is_iluvatar():
        from fastdeploy.worker.iluvatar_worker import IluvatarPaddleDisWorkerProc

        worker_proc = IluvatarPaddleDisWorkerProc(fd_config, world_size, global_rank)
    else:
        worker_proc = PaddleDisWorkerProc(fd_config, world_size, global_rank)
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

    # Attach the expert weight block.
    worker_proc.maybe_attach_expert_weight_shm()

    # Read the fault tolerance stats.
    worker_proc.maybe_load_ft_stats()

    # Load model
    worker_proc.load_model()
    # Initialize KV Cache
    worker_proc.initialize_kv_cache()

    # Trigger CUDAGraph capture
    worker_proc.graph_optimize_and_warm_up_model()

    # Initialize EPLB
    worker_proc.init_eplb()

    # Initialize health status
    worker_proc.init_health_status()

    worker_proc.start_task_queue_service()

    # Start event loop
    worker_proc.event_loop_normal()


if __name__ == "__main__":
    run_worker_proc()
