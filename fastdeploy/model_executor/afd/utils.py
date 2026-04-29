from __future__ import annotations

import os
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Set

import paddle
from paddleformers.utils.log import logger


def resolve_afd_worker_device(fd_config) -> str:
    """Resolve the Paddle device used by the current AFD worker process."""
    if fd_config.device_config.device_type != "cuda":
        return paddle.device.get_device()

    selected_gpus = os.getenv("FLAGS_selected_gpus")
    if selected_gpus:
        selected = [gpu.strip() for gpu in selected_gpus.split(",") if gpu.strip()]
        if len(selected) == 1:
            return f"gpu:{selected[0]}"

    device_ids = str(fd_config.parallel_config.device_ids).split(",")
    local_rank = int(
        os.getenv(
            "PADDLE_LOCAL_RANK",
            fd_config.parallel_config.data_parallel_rank * fd_config.parallel_config.tensor_parallel_size
            + fd_config.parallel_config.tensor_parallel_rank,
        )
    )
    return f"gpu:{local_rank % max(1, len(device_ids))}"


def extract_layer_id_from_weight_name(weight_name: str) -> int | None:
    match = re.search(r"model\.layers\.(\d+)\.", weight_name)
    if match is None:
        return None
    return int(match.group(1))


def loaded_weight_sublayer_name(param_name: str) -> str:
    return re.sub(r"\.(up_gate_proj_weight|down_proj_weight|weight)$", "", param_name)


def record_local_expert(
    experts_by_layer: Dict[int, Dict[int, int]],
    weight_name: str,
    logical_expert_id: int,
    physical_expert_id: int,
    expert_parallel_rank: int,
    experts_per_rank: int,
) -> None:
    layer_id = extract_layer_id_from_weight_name(weight_name)
    if layer_id is None:
        return

    local_expert_id = physical_expert_id - expert_parallel_rank * experts_per_rank
    if 0 <= local_expert_id < experts_per_rank:
        experts_by_layer.setdefault(layer_id, {})[int(logical_expert_id)] = int(local_expert_id)


def build_afd_expert_manifest(
    fd_config,
    world_topology,
    global_rank: int,
    experts_per_rank: int,
    experts_by_layer: Dict[int, Dict[int, int]],
) -> dict:
    register_info = getattr(fd_config, "register_info", {}) or {}
    instance_url = ""
    if register_info.get("host_ip") and register_info.get("port"):
        instance_url = f"http://{register_info['host_ip']}:{register_info['port']}"

    layers = []
    for layer_id in sorted(experts_by_layer):
        experts = [
            {"logical_expert_id": int(logical_id), "local_expert_id": int(local_expert_id)}
            for logical_id, local_expert_id in sorted(experts_by_layer[layer_id].items())
        ]
        layers.append({"layer_id": int(layer_id), "experts": experts})

    return {
        "world_size": world_topology.world_size,
        "attn_ranks": list(world_topology.attn_ranks),
        "ffn_ranks": list(world_topology.ffn_ranks),
        "global_rank": int(global_rank),
        "instance_url": instance_url,
        "experts_per_rank": int(experts_per_rank),
        "layers": layers,
    }


def log_loaded_weight_summary(
    role: str,
    loaded_names: List[str],
    moe_layer_ids: Iterable[int],
    num_layers: int,
) -> None:
    """Debug-only summary of which checkpoint components were consumed."""
    layer_components: Dict[int, Set[str]] = defaultdict(set)
    global_components: Set[str] = set()
    for weight_name in loaded_names:
        component = _component_name(weight_name)
        layer_id = extract_layer_id_from_weight_name(weight_name)
        if layer_id is None:
            global_components.add(component)
        else:
            layer_components[layer_id].add(component)

    moe_layer_ids = set(moe_layer_ids)
    lines = [
        "",
        f"GLM4 AFD loaded weights summary ({role}):",
        f"  total layers: {num_layers}",
        f"  dense layers: {num_layers - len(moe_layer_ids)}",
        f"  MoE layers: {len(moe_layer_ids)}",
    ]

    if global_components:
        lines.append(f"  global components: {', '.join(sorted(global_components))}")

    for layer_id in range(num_layers):
        layer_type = "MoE" if layer_id in moe_layer_ids else "dense"
        components = sorted(layer_components.get(layer_id, set()))
        component_text = ", ".join(components) if components else "(no weights loaded)"
        lines.append(f"  layer {layer_id:>3} [{layer_type:<5}] {component_text}")

    logger.debug("\n".join(lines))


def _component_name(weight_name: str) -> str:
    if ".self_attn.q_norm." in weight_name or ".self_attn.k_norm." in weight_name:
        return "qk_norm"
    if ".self_attn." in weight_name:
        return "self_attn"
    if ".input_layernorm." in weight_name:
        return "input_layernorm"
    if ".post_attention_layernorm." in weight_name:
        return "post_attention_layernorm"
    if ".mlp.shared_experts." in weight_name:
        return "shared_experts"
    if ".mlp.gate.e_score_correction_bias" in weight_name:
        return "gate_bias"
    if ".mlp.gate." in weight_name:
        return "gate"
    if ".mlp.experts." in weight_name:
        return "routed_experts"
    if ".mlp." in weight_name:
        return "mlp"
    if weight_name.startswith("model.embed_tokens"):
        return "embed_tokens"
    if weight_name.startswith("model.norm"):
        return "norm"
    if weight_name.startswith("lm_head"):
        return "lm_head"
    return "other"
