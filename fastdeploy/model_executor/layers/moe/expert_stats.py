"""Optional full routed-expert statistics for pure tensor parallel MoE.

The collector is deliberately a side channel.  It evaluates every routed
expert from the same ``x`` used by the normal top-k path, all-reduces the TP
partial output, and only keeps aggregate statistics.
"""

from __future__ import annotations

import atexit
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
import paddle

from fastdeploy.distributed.communication import tensor_model_parallel_all_reduce


_COLLECTORS: dict[tuple[int, int, int, int, str], "ExpertStatsCollector"] = {}
_EXIT_REGISTERED = False


class ExpertStatsCollector:
    def __init__(self, fd_config):
        model_cfg = fd_config.model_config
        parallel_cfg = fd_config.parallel_config
        self.first_moe_layer = int(getattr(model_cfg, "moe_layer_start_index", 0))
        self.num_layers = max(0, int(model_cfg.num_hidden_layers) - self.first_moe_layer)
        self.num_experts = int(model_cfg.n_routed_experts)
        self.tp_size = int(parallel_cfg.tensor_parallel_size)
        self.tp_rank = int(parallel_cfg.tensor_parallel_rank)
        self.dp_size = int(parallel_cfg.data_parallel_size)
        self.dp_rank = int(parallel_cfg.data_parallel_rank)
        self.enabled = bool(int(os.getenv("FD_MOE_STATS_ENABLE", "0")))
        self.recording = False
        self.output_path = os.getenv("FD_MOE_STATS_PATH", "moe_expert_stats.npz")
        self.token_chunk = max(1, int(os.getenv("FD_MOE_STATS_TOKEN_CHUNK", "16")))
        self.expert_prefix = paddle.arange(1, self.num_experts + 1, dtype="int64")
        self.distance_mask = 1.0 - paddle.eye(self.num_experts, dtype="float32")
        graph_cfg = fd_config.graph_opt_config
        self.max_graph_rows = max(
            int(getattr(graph_cfg, "max_capture_size", 0)),
            int(getattr(graph_cfg, "max_capture_size_prefill", 0)),
        )
        self._current_staged = {}
        self._captured_staged = {}

        shape = [self.num_layers, self.num_experts]
        self.importance_all = paddle.zeros(shape, dtype="float64")
        self.importance_topk = paddle.zeros(shape, dtype="float64")
        self.route_count = paddle.zeros(shape, dtype="int64")
        self.token_count = paddle.zeros([self.num_layers], dtype="int64")
        self.distance_sum = paddle.zeros([self.num_layers, self.num_experts, self.num_experts], dtype="float64")

    def set_recording(self, enabled: bool):
        self.recording = enabled

    def reset(self):
        for tensor in (
            self.importance_all,
            self.importance_topk,
            self.route_count,
            self.token_count,
            self.distance_sum,
        ):
            tensor[...] = 0

    def _layer_index(self, layer_idx: int) -> Optional[int]:
        index = int(layer_idx) - self.first_moe_layer
        return index if 0 <= index < self.num_layers else None

    @staticmethod
    def _scatter_add(target, indices, values):
        target[:] = paddle.scatter_nd_add(target, indices, values)

    @staticmethod
    def _valid_mask(forward_meta, rows: int):
        valid = paddle.ones([rows], dtype="float32")
        token_ids = getattr(forward_meta, "batch_id_per_token", None) if forward_meta is not None else None
        if token_ids is not None:
            valid = (token_ids[:rows] >= 0).cast("float32")
        return valid

    def begin_forward(self):
        self._current_staged = {}

    def stage_tp(
        self,
        layer_idx: int,
        x: paddle.Tensor,
        q: paddle.Tensor,
        topk_weights: paddle.Tensor,
        topk_indices: paddle.Tensor,
        forward_meta,
        layer,
        tp_group,
        expert_method,
    ):
        index = self._layer_index(layer_idx)
        if index is None or not self.enabled or int(x.shape[0]) == 0:
            return
        self._current_staged[index] = (
            paddle.assign(x),
            paddle.assign(q),
            paddle.assign(topk_weights),
            paddle.assign(topk_indices),
            paddle.assign(self._valid_mask(forward_meta, int(x.shape[0]))),
            layer,
            tp_group,
            expert_method,
        )

    def finish_forward(self, actual_rows: int):
        if self._current_staged:
            rows = int(next(iter(self._current_staged.values()))[0].shape[0])
            if not self.recording:
                if rows <= self.max_graph_rows:
                    self._captured_staged[rows] = self._current_staged
                return
            staged = self._current_staged
        else:
            if not self.recording:
                return
            sizes = [rows for rows in self._captured_staged if rows >= actual_rows]
            if not sizes:
                raise RuntimeError(f"no captured MoE statistics buffers cover {actual_rows} tokens")
            staged = self._captured_staged[min(sizes)]

        for index, values in staged.items():
            self._record_staged(index, *values)

    def _evaluate_experts(self, x, layer, tp_group, expert_method):
        """Return one bounded ``[tokens, experts, hidden]`` complete chunk."""
        if getattr(layer, "with_bias", False):
            raise NotImplementedError("full expert statistics with expert bias are not supported")
        quant_type = getattr(expert_method, "moe_quant_type", "w16a16")
        if quant_type not in ("w16a16", ""):
            raise NotImplementedError("full expert statistics currently require unquantized w16a16 experts")

        permuted_x = x.tile([self.num_experts, 1])
        token_ends = self.expert_prefix * x.shape[0]
        out = expert_method.compute_ffn(
            layer,
            permuted_x,
            token_ends,
            None,
            False,
            -1,
            None,
            None,
        ).reshape([self.num_experts, x.shape[0], x.shape[-1]])
        if self.tp_size > 1:
            out = tensor_model_parallel_all_reduce(out.reshape([-1, out.shape[-1]]), tp_group).reshape(out.shape)
        return out.transpose([1, 0, 2]).cast("float32")

    def _record_staged(
        self,
        index: int,
        x: paddle.Tensor,
        q: paddle.Tensor,
        topk_weights: paddle.Tensor,
        topk_indices: paddle.Tensor,
        valid: paddle.Tensor,
        layer,
        tp_group,
        expert_method,
    ):
        """Evaluate all experts from a staged layer input and accumulate metrics."""
        # q is sigmoid(gate(x)) before correction bias and top-k mutation.
        q = q.cast("float32")
        g_all = q / paddle.clip(q.sum(axis=-1, keepdim=True), min=1e-20)
        self.token_count[index] += paddle.sum(valid).cast("int64")
        topk_ids = topk_indices.cast("int64")
        # Process one token chunk at a time so long prefills never retain all
        # 128 expert activations simultaneously.
        for start in range(0, int(x.shape[0]), self.token_chunk):
            end = min(start + self.token_chunk, int(x.shape[0]))
            chunk = self._evaluate_experts(x[start:end], layer, tp_group, expert_method)
            chunk_valid = valid[start:end]
            chunk_q = g_all[start:end]
            norms = paddle.linalg.norm(chunk, axis=-1)
            self.importance_all[index] += (norms * chunk_q * chunk_valid.unsqueeze(-1)).sum(axis=0).cast("float64")

            chunk_ids = topk_ids[start:end]
            chunk_norms = paddle.take_along_axis(norms, chunk_ids, axis=1)
            chunk_values = topk_weights[start:end].cast("float32") * chunk_valid.unsqueeze(-1)
            self._scatter_add(
                self.importance_topk[index],
                chunk_ids.reshape([-1, 1]),
                (chunk_norms * chunk_values).reshape([-1]).cast("float64"),
            )
            route_indices = chunk_ids.reshape([-1, 1])
            route_valid = chunk_valid.unsqueeze(-1).tile([1, chunk_ids.shape[1]]).reshape([-1]).cast("int64")
            self._scatter_add(
                self.route_count[index],
                route_indices,
                paddle.ones([route_indices.shape[0]], dtype="int64") * route_valid,
            )

            # Pairwise Euclidean distance without materializing [B,E,E,H].
            norm_sq = paddle.sum(chunk * chunk, axis=-1)
            dot = paddle.matmul(chunk, chunk.transpose([0, 2, 1]))
            dist_sq = paddle.maximum(
                norm_sq.unsqueeze(-1) + norm_sq.unsqueeze(-2) - 2.0 * dot,
                paddle.zeros_like(dot),
            )
            dist = paddle.sqrt(dist_sq) * self.distance_mask
            self.distance_sum[index] += (dist * chunk_valid.reshape([-1, 1, 1])).sum(axis=0).cast("float64")

    def _payload(self):
        token_count = self.token_count.numpy()
        token_count_safe = np.maximum(token_count, 1)
        distance_sum = self.distance_sum.numpy()
        importance_all = self.importance_all.numpy()
        importance_topk = self.importance_topk.numpy()
        return {
            "distance_sum": distance_sum,
            "distance_mean": distance_sum / token_count_safe[:, None, None],
            "expert_importance_all": importance_all,
            "expert_importance_all_mean": importance_all / token_count_safe[:, None],
            "expert_importance_topk": importance_topk,
            "expert_importance_topk_mean": importance_topk / token_count_safe[:, None],
            "route_count": self.route_count.numpy(),
            "token_count": token_count,
            "layer_ids": np.arange(self.first_moe_layer, self.first_moe_layer + self.num_layers),
            "num_experts": np.asarray(self.num_experts, dtype=np.int64),
            "schema_version": np.asarray(2, dtype=np.int64),
            "metadata": np.asarray(
                json.dumps(
                    {
                        "distance": "sum_t ||Expert_i(x_t)-Expert_j(x_t)||_2",
                        "importance_all": "sum_t q_i/sum(q) * ||Expert_i(x_t)||_2",
                        "importance_topk": "sum_t 1[i in TopK] * g_topk_i * ||Expert_i(x_t)||_2",
                        "correction_bias": "used only for top-k selection",
                        "parallelism": "pure tensor parallel; expert outputs are TP all-reduced",
                        "token_chunk": self.token_chunk,
                        "generated_at_unix": time.time(),
                    }
                )
            ),
        }

    @staticmethod
    def _write_payload(path: Path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as output:
                np.savez_compressed(output, **payload)
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)

    def _output_path(self) -> Path:
        path = Path(self.output_path)
        if self.dp_size <= 1:
            return path
        suffix = path.suffix or ".npz"
        return path.with_name(f"{path.stem}.dp{self.dp_rank}{suffix}")

    def save(self):
        if not self.enabled or self.tp_rank != 0:
            return None
        path = self._output_path()
        self._write_payload(path, self._payload())
        return path


def get_expert_stats_collector(fd_config) -> Optional[ExpertStatsCollector]:
    if not bool(int(os.getenv("FD_MOE_STATS_ENABLE", "0"))):
        return None
    model_cfg = fd_config.model_config
    parallel_cfg = fd_config.parallel_config
    if int(getattr(parallel_cfg, "expert_parallel_size", 1)) != 1:
        return None
    key = (
        int(model_cfg.num_hidden_layers),
        int(model_cfg.n_routed_experts),
        int(parallel_cfg.data_parallel_rank),
        int(parallel_cfg.tensor_parallel_rank),
        os.getenv("FD_MOE_STATS_PATH", "moe_expert_stats.npz"),
    )
    global _EXIT_REGISTERED
    collector = _COLLECTORS.get(key)
    if collector is None:
        collector = ExpertStatsCollector(fd_config)
        _COLLECTORS[key] = collector
    if not _EXIT_REGISTERED:
        atexit.register(save_all_expert_stats)
        _EXIT_REGISTERED = True
    return collector


def reset_all_expert_stats():
    for collector in _COLLECTORS.values():
        collector.reset()


def set_expert_stats_recording(enabled: bool):
    for collector in _COLLECTORS.values():
        collector.set_recording(enabled)


def begin_all_expert_stats_forward():
    for collector in _COLLECTORS.values():
        collector.begin_forward()


def finish_all_expert_stats_forward(actual_rows: int):
    for collector in _COLLECTORS.values():
        collector.finish_forward(actual_rows)


def save_all_expert_stats(raise_on_error: bool = False):
    saved = []
    for collector in _COLLECTORS.values():
        try:
            path = collector.save()
            if path is not None:
                saved.append(str(path))
        except Exception:
            if raise_on_error:
                raise
    return saved
