from __future__ import annotations

import os
import threading
from typing import Dict, List

import paddle
from paddleformers.utils.log import logger

from fastdeploy.model_executor.afd.utils import resolve_afd_worker_device
from fastdeploy.utils import singleton


@singleton
class AFDWorldTopology:
    """Runtime world topology for one AFD worker."""

    def __init__(self, world_size, attn_ranks, ffn_ranks) -> None:
        if world_size <= 0:
            raise ValueError(f"AFD world_size must be positive, got {world_size}")
        if not attn_ranks:
            raise ValueError("AFD requires at least one ATTN rank.")
        if not ffn_ranks:
            raise ValueError("AFD requires at least one FFN rank.")
        combined_ranks = list(attn_ranks) + list(ffn_ranks)
        duplicated_ranks = sorted({rank for rank in combined_ranks if combined_ranks.count(rank) > 1})
        if duplicated_ranks:
            raise ValueError(f"AFD ranks must be unique, duplicated={duplicated_ranks}")
        invalid_ranks = sorted(rank for rank in combined_ranks if rank < 0 or rank >= world_size)
        if invalid_ranks:
            raise ValueError(f"AFD ranks must be in [0, {world_size}), invalid={invalid_ranks}")
        if len(set(attn_ranks).intersection(ffn_ranks)) > 0:
            raise ValueError(f"AFD ATTN/FFN ranks overlap: attn={attn_ranks}, ffn={ffn_ranks}")
        if len(attn_ranks) + len(ffn_ranks) != world_size:
            raise ValueError(
                "AFD rank count mismatch: "
                f"world_size={world_size}, attn={attn_ranks}, ffn={ffn_ranks}"
            )
        missing_ranks = sorted(set(range(world_size)).difference(combined_ranks))
        if missing_ranks:
            raise ValueError(f"AFD topology misses world ranks: {missing_ranks}")
        self.world_size = world_size
        self.attn_ranks: List[int] = sorted(attn_ranks)
        self.ffn_ranks: List[int] = sorted(ffn_ranks)


@singleton
class AFDExpertLayout:
    """Runtime logical-to-physical expert layout for AFD.

    Physical expert space is inflated so that every rank (including ATTN ranks)
    has ``num_local_physical_experts`` DeepEP expert positions.  ATTN-rank
    positions are phantom (never routed to); only FFN-rank positions carry real
    experts.

    Mapping formula (logical -> physical):
        ffn_rank_index  = logical_id // num_local_physical_experts
        ffn_global_rank = ffn_ranks[ffn_rank_index]
        physical_id     = ffn_global_rank * num_local_physical_experts
                        + (logical_id % num_local_physical_experts)
    """

    def __init__(self, n_routed_experts: int) -> None:
        self.afd_world_topology = AFDWorldTopology()
        self.num_attn_ranks = len(self.afd_world_topology.attn_ranks)
        self.num_ffn_ranks = len(self.afd_world_topology.ffn_ranks)
        self.world_size = self.afd_world_topology.world_size

        self.num_logical_experts = n_routed_experts
        if n_routed_experts <= 0:
            raise ValueError(f"n_routed_experts must be positive, got {n_routed_experts}")
        if n_routed_experts % self.num_ffn_ranks != 0:
            raise ValueError(
                "AFD requires logical experts to be evenly sharded over FFN ranks: "
                f"n_routed_experts={n_routed_experts}, num_ffn_ranks={self.num_ffn_ranks}"
            )
        self.num_local_physical_experts = n_routed_experts // self.num_ffn_ranks
        self.num_physical_experts = self.num_local_physical_experts * self.world_size

        # log2phy: logical expert -> list of physical expert IDs
        # (one logical expert may map to multiple physical replicas in the future)
        self.log2phy: Dict[int, List[int]] = {}
        self.phy2log: List[int] = [-1] * self.num_physical_experts
        self.log2phy_by_layer: Dict[int, Dict[int, List[int]]] = {}
        self.phy2log_by_layer: Dict[int, List[int]] = {}
        self._layout_lock = threading.RLock()

        for logical_id in range(n_routed_experts):
            ffn_rank_index = logical_id // self.num_local_physical_experts
            ffn_global_rank = self.afd_world_topology.ffn_ranks[ffn_rank_index]
            local_offset = logical_id % self.num_local_physical_experts
            physical_id = ffn_global_rank * self.num_local_physical_experts + local_offset

            if logical_id not in self.log2phy:
                self.log2phy[logical_id] = []
            self.log2phy[logical_id].append(physical_id)
            self.phy2log[physical_id] = logical_id

        self._log2phy_flat = [self.log2phy[i][0] for i in range(self.num_logical_experts)]
        self._log2phy_flat_by_layer: Dict[int, List[int]] = {}
        self._replica_capacity = self.num_physical_experts
        self._log2phy_replica_matrix, self._log2phy_replica_count = self._build_replica_table(self.log2phy)
        self._log2phy_replica_matrix_by_layer: Dict[int, List[List[int]]] = {}
        self._log2phy_replica_count_by_layer: Dict[int, List[int]] = {}
        self._log2phy_tensor_cache: Dict[str, paddle.Tensor] = {}
        self._log2phy_replica_tensor_cache: Dict[str, paddle.Tensor] = {}
        self._log2phy_replica_count_tensor_cache: Dict[str, paddle.Tensor] = {}

        logger.info(
            f"AFDExpertLayout: logical={n_routed_experts}, "
            f"physical={self.num_physical_experts}, "
            f"local_per_rank={self.num_local_physical_experts}, "
            f"attn_ranks={self.afd_world_topology.attn_ranks}, "
            f"ffn_ranks={self.afd_world_topology.ffn_ranks}"
        )

    # ------------------------------------------------------------------
    # scalar helpers
    # ------------------------------------------------------------------
    def router_log2phy(self, logical_expert_id: int, layer_id: int | None = None) -> int:
        """Convert a single logical expert ID to physical expert ID.

        Currently returns the first physical replica; will support
        load-balanced selection among replicas in the future.
        """
        with self._layout_lock:
            if layer_id is not None and layer_id in self.log2phy_by_layer:
                return self.log2phy_by_layer[layer_id][logical_expert_id][0]
            return self.log2phy[logical_expert_id][0]

    def router_phy2log(self, physical_expert_id: int, layer_id: int | None = None) -> int:
        """Convert a single physical expert ID to logical expert ID."""
        with self._layout_lock:
            if layer_id is not None and layer_id in self.phy2log_by_layer:
                return self.phy2log_by_layer[layer_id][physical_expert_id]
            return self.phy2log[physical_expert_id]

    # ------------------------------------------------------------------
    # batched GPU conversion
    # ------------------------------------------------------------------
    @property
    def log2phy_tensor(self) -> paddle.Tensor:
        """Flat tensor: log2phy_tensor[logical_id] = first physical id.

        Uses the first replica for each logical expert (same as
        ``router_log2phy``).  Will be extended for multi-replica
        selection in the future.
        """
        return self._log2phy_tensor_for_device(paddle.device.get_device(), None)

    def _cache_key(self, place, layer_id: int | None) -> str:
        layer_key = "default" if layer_id is None else str(layer_id)
        return f"{layer_key}:{place}"

    def _flat_for_layer(self, layer_id: int | None) -> List[int]:
        if layer_id is not None and layer_id in self._log2phy_flat_by_layer:
            return self._log2phy_flat_by_layer[layer_id]
        return self._log2phy_flat

    def _build_replica_table(self, log2phy: Dict[int, List[int]]) -> tuple[List[List[int]], List[int]]:
        matrix: List[List[int]] = []
        counts: List[int] = []
        for logical_id in range(self.num_logical_experts):
            physical_ids = log2phy[logical_id]
            first_physical = physical_ids[0]
            capped = physical_ids[: self._replica_capacity]
            row = [first_physical] * self._replica_capacity
            for replica_idx, physical_id in enumerate(capped):
                row[replica_idx] = physical_id
            matrix.append(row)
            counts.append(max(1, len(capped)))
        return matrix, counts

    def _replica_table_for_layer(self, layer_id: int | None) -> List[List[int]]:
        if layer_id is not None and layer_id in self._log2phy_replica_matrix_by_layer:
            return self._log2phy_replica_matrix_by_layer[layer_id]
        return self._log2phy_replica_matrix

    def _replica_count_for_layer(self, layer_id: int | None) -> List[int]:
        if layer_id is not None and layer_id in self._log2phy_replica_count_by_layer:
            return self._log2phy_replica_count_by_layer[layer_id]
        return self._log2phy_replica_count

    def _log2phy_tensor_for_device(self, place, layer_id: int | None) -> paddle.Tensor:
        cache_key = self._cache_key(place, layer_id)
        if cache_key not in self._log2phy_tensor_cache:
            self._log2phy_tensor_cache[cache_key] = paddle.to_tensor(
                self._flat_for_layer(layer_id),
                dtype=paddle.int64,
                place=place,
            )
        return self._log2phy_tensor_cache[cache_key]

    def _log2phy_tensor_for(self, tensor: paddle.Tensor, layer_id: int | None = None) -> paddle.Tensor:
        with self._layout_lock:
            return self._log2phy_tensor_for_device(tensor.place, layer_id)

    def _replica_tensors_for_device(self, place, layer_id: int | None) -> tuple[paddle.Tensor, paddle.Tensor]:
        cache_key = self._cache_key(place, layer_id)
        if cache_key not in self._log2phy_replica_tensor_cache:
            self._log2phy_replica_tensor_cache[cache_key] = paddle.to_tensor(
                self._replica_table_for_layer(layer_id),
                dtype=paddle.int64,
                place=place,
            )
        if cache_key not in self._log2phy_replica_count_tensor_cache:
            self._log2phy_replica_count_tensor_cache[cache_key] = paddle.to_tensor(
                self._replica_count_for_layer(layer_id),
                dtype=paddle.int64,
                place=place,
            )
        return self._log2phy_replica_tensor_cache[cache_key], self._log2phy_replica_count_tensor_cache[cache_key]

    def _replica_tensors_for(self, tensor: paddle.Tensor, layer_id: int | None = None) -> tuple[paddle.Tensor, paddle.Tensor]:
        with self._layout_lock:
            return self._replica_tensors_for_device(tensor.place, layer_id)

    def batch_log2phy(self, topk_idx: paddle.Tensor, layer_id: int | None = None) -> paddle.Tensor:
        """Vectorised logical -> physical conversion for a routing tensor.

        Args:
            topk_idx: ``[num_tokens, top_k]`` logical expert IDs.
        Returns:
            Same shape, physical expert IDs.
        """
        if topk_idx.shape[0] == 0:
            return topk_idx
        orig_shape = topk_idx.shape
        flat_logical = topk_idx.reshape([-1])
        replica_table, replica_count = self._replica_tensors_for(topk_idx, layer_id)
        selected_counts = paddle.index_select(replica_count, flat_logical, axis=0)
        replica_selector = paddle.arange(flat_logical.shape[0], dtype=paddle.int64)
        replica_idx = paddle.remainder(replica_selector, selected_counts)
        gather_idx = paddle.stack([flat_logical, replica_idx], axis=1)
        return paddle.gather_nd(replica_table, gather_idx).reshape(orig_shape)

    def update_from_topology_snapshot(self, snapshot: dict) -> None:
        """Apply a router-generated ready topology snapshot.

        The shape of DeepEP's physical expert space is fixed for v1, so only
        snapshots matching the bootstrap physical size are applied.
        """
        layouts = snapshot.get("expert_layout_by_layer") or []
        if not layouts:
            logger.warning("AFD topology snapshot has no expert_layout_by_layer; skip update.")
            return

        updated_layers = []
        with self._layout_lock:
            for layer_layout in layouts:
                layer_id = int(layer_layout["layer_id"])
                phy2log = [int(v) for v in layer_layout.get("phy2log", [])]
                if len(phy2log) != self.num_physical_experts:
                    logger.warning(
                        "AFD topology layer has incompatible physical size; skip. "
                        f"layer={layer_id}, got={len(phy2log)}, expected={self.num_physical_experts}"
                    )
                    continue

                log2phy: Dict[int, List[int]] = {i: [] for i in range(self.num_logical_experts)}
                for physical_id, logical_id in enumerate(phy2log):
                    if logical_id < 0:
                        continue
                    if logical_id >= self.num_logical_experts:
                        logger.warning(
                            "AFD topology layer has invalid logical expert id; skip entry. "
                            f"layer={layer_id}, physical={physical_id}, logical={logical_id}"
                        )
                        continue
                    log2phy[logical_id].append(physical_id)

                missing = [logical_id for logical_id, physical_ids in log2phy.items() if not physical_ids]
                if missing:
                    logger.warning(
                        "AFD topology layer misses logical experts; keep previous layer layout. "
                        f"layer={layer_id}, missing_count={len(missing)}, sample={missing[:8]}"
                    )
                    continue

                flat = [log2phy[i][0] for i in range(self.num_logical_experts)]
                replica_matrix, replica_count = self._build_replica_table(log2phy)
                self.log2phy_by_layer[layer_id] = log2phy
                self.phy2log_by_layer[layer_id] = phy2log
                self._log2phy_flat_by_layer[layer_id] = flat
                self._log2phy_replica_matrix_by_layer[layer_id] = replica_matrix
                self._log2phy_replica_count_by_layer[layer_id] = replica_count
                updated_layers.append(layer_id)

                for cache_key, cached in list(self._log2phy_tensor_cache.items()):
                    if not cache_key.startswith(f"{layer_id}:"):
                        continue
                    cached.set_value(paddle.to_tensor(flat, dtype=paddle.int64, place=cached.place))
                for cache_key, cached in list(self._log2phy_replica_tensor_cache.items()):
                    if not cache_key.startswith(f"{layer_id}:"):
                        continue
                    cached.set_value(paddle.to_tensor(replica_matrix, dtype=paddle.int64, place=cached.place))
                for cache_key, cached in list(self._log2phy_replica_count_tensor_cache.items()):
                    if not cache_key.startswith(f"{layer_id}:"):
                        continue
                    cached.set_value(paddle.to_tensor(replica_count, dtype=paddle.int64, place=cached.place))

        if updated_layers:
            logger.info(
                "AFDExpertLayout updated from topology snapshot: "
                f"revision={snapshot.get('revision')}, layers={updated_layers[:8]}, "
                f"num_layers={len(updated_layers)}"
            )


@singleton
class AFDDecodeRunner:
    """Decode-phase runner that drives DeepEP dispatch / combine for AFD.

    Created once per worker process (both ATTN and FFN workers create one).
    The underlying ``DeepEPEngine`` is a process-wide singleton so the buffer
    is allocated only once.
    """

    def __init__(self, fd_config, afd_layout: AFDExpertLayout):
        from fastdeploy.config import MoEPhase
        from fastdeploy.model_executor.layers.moe.ep import DeepEPEngine

        self.fd_config = fd_config
        self.device = resolve_afd_worker_device(fd_config)
        self._device_touch_tensor = None
        paddle.device.set_device(self.device)
        self._ensure_device("init")
        self.afd_layout = afd_layout
        self.hidden_size = fd_config.model_config.hidden_size
        self.top_k = fd_config.model_config.num_experts_per_tok
        self.num_physical_experts = afd_layout.num_physical_experts
        self.num_local_physical_experts = afd_layout.num_local_physical_experts
        self._logged_dispatch_device = False

        self.ep_engine = DeepEPEngine(
            num_max_dispatch_tokens_per_rank=fd_config.model_config.num_max_dispatch_tokens_per_rank,
            hidden_size=self.hidden_size,
            num_experts=self.num_physical_experts,
            ep_size=afd_layout.world_size,
            ep_rank=fd_config.parallel_config.expert_parallel_rank,
            splitwise_role=fd_config.scheduler_config.splitwise_role,
            moe_phase=MoEPhase("decode"),
            group=fd_config.parallel_config.ep_group,
            use_internode_ll_two_stage=False,
            top_k=self.top_k,
        )

        logger.info(
            f"AFDDecodeRunner created: physical_experts={self.num_physical_experts}, "
            f"ep_rank={fd_config.parallel_config.expert_parallel_rank}, "
            f"device={self.device}, current_device={paddle.device.get_device()}, "
            f"FLAGS_selected_gpus={os.getenv('FLAGS_selected_gpus')}, "
            f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES')}, "
            f"PADDLE_LOCAL_RANK={os.getenv('PADDLE_LOCAL_RANK')}"
        )

    # ------------------------------------------------------------------
    def _ensure_device(self, callsite: str) -> None:
        current_device = paddle.device.get_device()
        if current_device != self.device:
            logger.warning(
                f"AFDDecodeRunner reset Paddle device before {callsite}: "
                f"current={current_device}, expected={self.device}, "
                f"FLAGS_selected_gpus={os.getenv('FLAGS_selected_gpus')}, "
                f"PADDLE_LOCAL_RANK={os.getenv('PADDLE_LOCAL_RANK')}"
            )
            paddle.device.set_device(self.device)
        # Paddle set_device does not immediately update CUDA runtime current
        # device. A tiny tensor op on the target place makes DeepEP C++ see the
        # same device through cudaGetDevice/current stream.
        self._device_touch_tensor = paddle.empty([0], dtype="int32")

    def _log_dispatch_device_once(self, x, physical_topk_idx, topk_weights) -> None:
        if self._logged_dispatch_device:
            return
        self._logged_dispatch_device = True
        runtime_local_device_id = None
        try:
            runtime_local_device_id = self.ep_engine.deepep_engine.runtime.get_local_device_id()
        except Exception as exc:  # pragma: no cover - diagnostic only
            runtime_local_device_id = f"unavailable: {exc}"
        logger.info(
            "AFD dispatch device check: "
            f"runner_device={self.device}, current_device={paddle.device.get_device()}, "
            f"x_place={getattr(x, 'place', None)}, "
            f"topk_idx_place={getattr(physical_topk_idx, 'place', None)}, "
            f"topk_weights_place={getattr(topk_weights, 'place', None)}, "
            f"deepep_runtime_local_device_id={runtime_local_device_id}, "
            f"FLAGS_selected_gpus={os.getenv('FLAGS_selected_gpus')}, "
            f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES')}, "
            f"PADDLE_LOCAL_RANK={os.getenv('PADDLE_LOCAL_RANK')}"
        )

    def logical_to_physical(self, topk_idx: paddle.Tensor, layer_id: int | None = None) -> paddle.Tensor:
        """Convert router logical expert IDs to AFD physical expert IDs."""
        self._ensure_device("logical_to_physical")
        return self.afd_layout.batch_log2phy(topk_idx, layer_id=layer_id)

    def dispatch_physical(self, x, physical_topk_idx, topk_weights, **kwargs):
        """Low-latency dispatch via DeepEP using physical expert IDs."""
        self._ensure_device("dispatch")
        self._log_dispatch_device_once(x, physical_topk_idx, topk_weights)

        expertwise_scale = kwargs.get("expertwise_scale", None)
        use_fp8 = kwargs.get("use_fp8", False)
        quant_group_size = kwargs.get("quant_group_size", 128)
        use_ue8m0 = kwargs.get("use_ue8m0", False)

        recv_hidden, recv_count, handle, dispatch_hook = (
            self.ep_engine.low_latency_dispatch(
                x, physical_topk_idx, expertwise_scale, use_fp8, quant_group_size, use_ue8m0
            )
        )
        if dispatch_hook is not None:
            dispatch_hook()
        return recv_hidden, recv_count, handle

    def dispatch(self, x, topk_idx, topk_weights, **kwargs):
        """Low-latency dispatch via DeepEP using logical expert IDs."""
        physical_topk_idx = self.logical_to_physical(topk_idx, layer_id=kwargs.pop("layer_id", None))
        return self.dispatch_physical(x, physical_topk_idx, topk_weights, **kwargs)

    def combine(self, ffn_out, physical_topk_idx, topk_weights, handle, **kwargs):
        """Low-latency combine via DeepEP using physical expert IDs."""
        self._ensure_device("combine")
        combined, combine_hook = self.ep_engine.low_latency_combine(
            ffn_out, physical_topk_idx, topk_weights, handle
        )
        if combine_hook is not None:
            combine_hook()
        return combined
