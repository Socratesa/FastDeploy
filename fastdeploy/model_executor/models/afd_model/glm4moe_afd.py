"""
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
"""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Dict, List

import paddle
from paddle import nn
from paddleformers.utils.log import logger

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.afd import AFDDecodeRunner
from fastdeploy.model_executor.forward_meta import ForwardMeta
from fastdeploy.model_executor.graph_optimization.decorator import (
    support_graph_optimization,
)
from fastdeploy.model_executor.layers.embeddings import VocabParallelEmbedding
from fastdeploy.model_executor.layers.lm_head import ParallelLMHead
from fastdeploy.model_executor.layers.moe.moe import FusedMoE, get_moe_scores
from fastdeploy.model_executor.layers.normalization import RMSNorm
from fastdeploy.model_executor.models.glm4_moe import (
    Glm4MoeAttention,
    Glm4MoeMLP,
    rms_norm_func,
)
from fastdeploy.model_executor.models.model_base import ModelCategory, ModelForCasualLM, ModelRegistry
from fastdeploy.worker.experts_manager import RedundantExpertManger

import fastdeploy




# =====================================================================
#  ATTN worker layers
# =====================================================================

class Glm4AFDAttnMoeBlock(nn.Layer):
    """MoE block on the ATTN worker.

    Holds the *gate* (router) and *shared experts*.  Routed expert
    computation is offloaded to FFN workers via DeepEP dispatch / combine.
    """

    def __init__(
        self,
        fd_config: FDConfig,
        layer_id: int,
        prefix: str,
        afd_runner: AFDDecodeRunner,
        redundant_table_manger: RedundantExpertManger = None,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = fd_config.model_config.hidden_size
        self.n_shared_experts = fd_config.model_config.n_shared_experts

        # Routing parameters (needed by get_moe_scores)
        self.top_k = fd_config.model_config.num_experts_per_tok
        self.n_group = fd_config.model_config.n_group
        self.topk_group = fd_config.model_config.topk_group
        self.routed_scaling_factor = fd_config.model_config.routed_scaling_factor
        self.renormalize = fd_config.model_config.norm_topk_prob

        self.afd_runner = afd_runner
        self.redundant_table_manger = redundant_table_manger

        from fastdeploy.model_executor.layers.linear import ReplicatedLinear

        self.gate = ReplicatedLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.gate",
            input_size=self.hidden_size,
            output_size=fd_config.model_config.n_routed_experts,
            with_bias=False,
            skip_quant=True,
            weight_dtype=("float32" if fd_config.model_config.moe_gate_fp32 else ""),
        )
        self.gate.e_score_correction_bias = self.create_parameter(
            shape=[1, fd_config.model_config.n_routed_experts],
            dtype="float32",
            default_initializer=paddle.nn.initializer.Constant(0),
        )

        self.shared_experts = None
        if self.n_shared_experts > 0:
            shared_experts_intermediate_size = (
                self.n_shared_experts * fd_config.model_config.moe_intermediate_size
            )
            self.shared_experts = Glm4MoeMLP(
                fd_config=fd_config,
                intermediate_size=shared_experts_intermediate_size,
                layer_id=layer_id,
                prefix=f"{prefix}.shared_experts",
                reduce_results=True,
            )

    def forward(self, x: paddle.Tensor, forward_meta: ForwardMeta = None) -> paddle.Tensor:
        # --- 1. routing ---
        gate_out = self.gate(x)
        gate_out = gate_out.cast("float32")

        routing_kwargs = {}
        if self.redundant_table_manger is not None:
            (
                _ep_rank_to_expert_id_list,
                expert_id_to_ep_rank_array,
                expert_in_rank_num_list,
                tokens_per_expert_stats_list,
            ) = self.redundant_table_manger.get_ep_rank_to_expert_id_list_by_layer(self.layer_id)
            routing_kwargs = {
                "expert_id_to_ep_rank_array": expert_id_to_ep_rank_array,
                "expert_in_rank_num_list": expert_in_rank_num_list,
                "tokens_per_expert_stats_list": tokens_per_expert_stats_list,
                "redundant_ep_rank_num_plus_one": self.redundant_table_manger.redundant_experts_num + 1,
            }

        _score, topk_weights, topk_idx = get_moe_scores(
            gate_out,
            self.n_group,
            self.topk_group,
            self.top_k,
            self.routed_scaling_factor,
            self.gate.e_score_correction_bias,
            self.renormalize,
            **routing_kwargs,
        )

        # --- 2. dispatch tokens to FFN workers ---
        # EPLB topk returns AFD physical ids when redundant routing is enabled.
        if self.redundant_table_manger is not None:
            physical_topk_idx = topk_idx
        else:
            physical_topk_idx = self.afd_runner.routing_logical_to_physical(topk_idx)
        recv_hidden, _, handle = self.afd_runner.dispatch_physical(
            x, physical_topk_idx, topk_weights,
        )

        # ATTN rank has only phantom experts, produce zero FFN output
        ffn_out = paddle.zeros_like(recv_hidden)

        # --- 3. combine results from FFN workers ---
        routed_out = self.afd_runner.combine(ffn_out, physical_topk_idx, topk_weights, handle)

        # --- 4. shared experts ---
        if self.shared_experts is not None:
            routed_out = routed_out + self.shared_experts(x, forward_meta)

        return routed_out


class Glm4AFDAttnDecoderLayer(nn.Layer):
    def __init__(
        self,
        fd_config: FDConfig,
        prefix: str,
        afd_runner: AFDDecodeRunner,
        redundant_table_manger: RedundantExpertManger = None,
    ) -> None:
        super().__init__()

        layer_id = int(prefix.split(sep=".")[-1])
        self.self_attn = Glm4MoeAttention(
            fd_config=fd_config,
            layer_id=layer_id,
            prefix=f"{prefix}.self_attn",
        )

        if (
            fd_config.model_config.n_routed_experts is not None
            and layer_id >= fd_config.model_config.first_k_dense_replace
        ):
            self.mlp = Glm4AFDAttnMoeBlock(
                fd_config,
                layer_id=layer_id,
                prefix=f"{prefix}.mlp",
                afd_runner=afd_runner,
                redundant_table_manger=redundant_table_manger,
            )
        else:
            self.mlp = Glm4MoeMLP(
                fd_config,
                intermediate_size=fd_config.model_config.intermediate_size,
                layer_id=layer_id,
                prefix=f"{prefix}.mlp",
            )

        self.input_layernorm = RMSNorm(
            fd_config,
            hidden_size=fd_config.model_config.hidden_size,
            eps=fd_config.model_config.rms_norm_eps,
            prefix=f"{prefix}.input_layernorm",
            layer_id=layer_id,
        )
        self.post_attention_layernorm = RMSNorm(
            fd_config,
            hidden_size=fd_config.model_config.hidden_size,
            eps=fd_config.model_config.rms_norm_eps,
            prefix=f"{prefix}.post_attention_layernorm",
            layer_id=layer_id,
        )

    def forward(
        self,
        forward_meta: ForwardMeta,
        hidden_states: paddle.Tensor,
        residual: paddle.Tensor = None,
    ):
        proxy_rmsnorm = rms_norm_func if fastdeploy.envs.FD_USE_PHI_RMSNORM else None
        hidden_states, residual = self.input_layernorm(
            hidden_states, residual_input=residual, forward_meta=forward_meta, proxy_rmsnorm=proxy_rmsnorm
        )
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            forward_meta=forward_meta,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual, proxy_rmsnorm=proxy_rmsnorm)

        hidden_states = self.mlp(hidden_states, forward_meta)

        return hidden_states, residual


@support_graph_optimization
class Glm4AFDAttnModel(nn.Layer):
    def __init__(
        self,
        fd_config: FDConfig,
        afd_runner: AFDDecodeRunner,
        redundant_table_manger: RedundantExpertManger = None,
    ) -> None:
        super().__init__()
        self.num_layers = fd_config.model_config.num_hidden_layers
        fd_config.model_config.pretrained_config.prefix_name = "model"

        self.embed_tokens = VocabParallelEmbedding(
            fd_config,
            num_embeddings=fd_config.model_config.vocab_size,
            embedding_dim=fd_config.model_config.hidden_size,
            params_dtype=paddle.get_default_dtype,
            prefix=f"{fd_config.model_config.pretrained_config.prefix_name}.embed_tokens",
        )
        self.layers = nn.LayerList(
            [
                Glm4AFDAttnDecoderLayer(
                    fd_config,
                    prefix=f"{fd_config.model_config.pretrained_config.prefix_name}.layers.{i}",
                    afd_runner=afd_runner,
                    redundant_table_manger=redundant_table_manger,
                )
                for i in range(self.num_layers)
            ]
        )
        self.norm = RMSNorm(
            fd_config,
            hidden_size=fd_config.model_config.hidden_size,
            eps=fd_config.model_config.rms_norm_eps,
            prefix=f"{fd_config.model_config.pretrained_config.prefix_name}.norm",
        )

    def forward(
        self,
        ids_remove_padding: paddle.Tensor,
        forward_meta: ForwardMeta,
    ):
        hidden_states = self.embed_tokens(ids_remove_padding=ids_remove_padding, forward_meta=forward_meta)

        residual = None

        for layer_id in range(self.num_layers):
            hidden_states, residual = self.layers[layer_id](forward_meta, hidden_states, residual)

        out = self.norm(hidden_states, residual, forward_meta=forward_meta)[0]

        if self.norm.is_last_norm and self.norm.fd_config.parallel_config.use_sequence_parallel_moe:
            out = self.norm.allgather(out, forward_meta.ids_remove_padding.shape[0])

        return out


# =====================================================================
#  FFN worker layers
# =====================================================================

class Glm4AFDFFNMoeBlock(nn.Layer):
    """Routed-expert weights for one MoE layer on the FFN worker.

    ``FusedMoE`` keeps the model's logical expert count for checkpoint name
    matching.  When EPLB is enabled, the redundant table manager supplies the
    inflated AFD physical expert space used by dispatch/combine.
    """

    def __init__(
        self,
        fd_config: FDConfig,
        layer_id: int,
        prefix: str,
        redundant_table_manger: RedundantExpertManger = None,
    ) -> None:
        super().__init__()
        num_experts = fd_config.model_config.n_routed_experts
        if fd_config.afd_config.enable_afd and redundant_table_manger is None:
            num_experts = fd_config.afd_config.afd_num_physical_experts
        self.experts = FusedMoE(
            fd_config,
            hidden_size=fd_config.model_config.hidden_size,
            reduce_results=True,
            renormalize=fd_config.model_config.norm_topk_prob,
            moe_intermediate_size=fd_config.model_config.moe_intermediate_size,
            num_experts=num_experts,
            top_k=fd_config.model_config.num_experts_per_tok,
            topk_method="noaux_tc",
            topk_group=fd_config.model_config.topk_group,
            n_group=fd_config.model_config.n_group,
            routed_scaling_factor=fd_config.model_config.routed_scaling_factor,
            layer_idx=layer_id,
            gate_correction_bias=None,
            redundant_table_manger=redundant_table_manger,
            weight_key_map={
                "up_gate_proj_expert_weight_key": f"{prefix}.experts.{{}}.up_gate_proj.weight",
                "down_proj_expert_weight_key": f"{prefix}.experts.{{}}.down_proj.weight",
            },
            topk_reduce_func=lambda x: x.sum(axis=-1, keepdim=True) + 1e-20,
        )

@support_graph_optimization
class Glm4AFDFFNModel(nn.Layer):
    """Executable FFN participant body for AFD.

    The outer CausalLM owns loading/logits APIs.  This inner layer owns the
    per-layer dispatch -> local expert compute -> combine sequence so it can be
    captured and replayed independently from the ATTN worker graph.
    """

    def __init__(
        self,
        fd_config: FDConfig,
        afd_runner: AFDDecodeRunner,
        moe_layer_ids: List[int],
        redundant_table_manger: RedundantExpertManger = None,
    ) -> None:
        super().__init__()
        self._afd_runner = afd_runner
        self._moe_layer_ids = moe_layer_ids
        self.moe_blocks = nn.LayerDict(
            {
                str(layer_id): Glm4AFDFFNMoeBlock(
                    fd_config=fd_config,
                    layer_id=layer_id,
                    prefix=f"model.layers.{layer_id}.mlp",
                    redundant_table_manger=redundant_table_manger,
                )
                for layer_id in self._moe_layer_ids
            }
        )

        self._hidden_size = fd_config.model_config.hidden_size
        self._top_k = fd_config.model_config.num_experts_per_tok
        self._empty_x = paddle.zeros([0, self._hidden_size], dtype=paddle.get_default_dtype())
        self._empty_topk_idx = paddle.zeros([0, self._top_k], dtype=paddle.int64)
        self._empty_topk_weights = paddle.zeros([0, self._top_k], dtype=paddle.float32)

    def forward(
        self,
        ids_remove_padding: paddle.Tensor = None,
        forward_meta: ForwardMeta = None,
    ) -> paddle.Tensor:
        # ids_remove_padding/forward_meta are graph-shape selectors for the
        # generic GraphOptBackend.  AFD FFN itself originates no tokens.
        for layer_id in self._moe_layer_ids:
            recv_hidden, recv_count, handle = self._afd_runner.dispatch_physical(
                self._empty_x,
                self._empty_topk_idx,
                self._empty_topk_weights,
            )
            ffn_out = self._compute_local_experts(layer_id, recv_hidden, recv_count)
            self._afd_runner.combine(
                ffn_out,
                self._empty_topk_idx,
                self._empty_topk_weights,
                handle,
            )

        return paddle.zeros([1], dtype=paddle.int32)

    def _compute_local_experts(
        self,
        layer_id: int,
        recv_hidden: paddle.Tensor,
        recv_count: paddle.Tensor,
    ) -> paddle.Tensor:
        """Run local routed experts on tokens received from ATTN ranks."""
        layer = self.moe_blocks[str(layer_id)].experts

        dequant_scale = None
        permute_input = recv_hidden
        if isinstance(recv_hidden, (list, tuple)):
            permute_input, dequant_scale = recv_hidden

        num_local_experts = permute_input.shape[0]
        max_num = permute_input.shape[1]
        estimate_total = max_num * num_local_experts

        moe_quant_type = getattr(layer.quant_method, "moe_quant_type", None)
        if moe_quant_type in ["w4a8", "w4afp8"] or layer.with_bias:
            expert_idx_per_token = paddle.arange(num_local_experts, device=permute_input.place)[:, None].tile(
                [1, max_num]
            )
        else:
            expert_idx_per_token = None

        return layer.quant_method.compute_ffn(
            layer,
            permute_input,
            recv_count.cast("int64"),
            expert_idx_per_token,
            True,
            estimate_total,
            dequant_scale,
        )

# =====================================================================
#  ATTN worker CausalLM
# =====================================================================

class Glm4MoeForCausalLM_AFDAttn(ModelForCasualLM):
    def __init__(self, fd_config: FDConfig):
        super().__init__(fd_config)

        self.redundant_table_manger = RedundantExpertManger(
            n_routed_experts=fd_config.model_config.n_routed_experts,
            num_hidden_layers=fd_config.model_config.num_hidden_layers,
            redundant_experts_num=fd_config.eplb_config.redundant_experts_num,
            ep_size=fd_config.parallel_config.expert_parallel_size,
            fd_config=fd_config,
        ) if fd_config.eplb_config.enable_eplb else None

        self._afd_runner = AFDDecodeRunner(fd_config)

        self.model = Glm4AFDAttnModel(
            fd_config,
            afd_runner=self._afd_runner,
            redundant_table_manger=self.redundant_table_manger,
        )

        self.ori_vocab_size = fd_config.model_config.ori_vocab_size

        self.lm_head = ParallelLMHead(
            fd_config,
            embedding_dim=fd_config.model_config.hidden_size,
            num_embeddings=fd_config.model_config.vocab_size,
            prefix="lm_head",
        )

        self._moe_layer_ids = list(
            range(fd_config.model_config.first_k_dense_replace, fd_config.model_config.num_hidden_layers)
        )

    @classmethod
    def name(cls):
        return "Glm4MoeForCausalLM_AFDAttn"

    @paddle.no_grad()
    def load_weights(self, weights_iterator) -> None:
        from fastdeploy.model_executor.utils import (
            default_weight_loader,
            process_weights_after_loading,
        )

        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("up_gate_proj", "gate_proj", "gate"),
            ("up_gate_proj", "up_proj", "up"),
            ("embed_tokens.embeddings", "embed_tokens", None),
            ("lm_head.linear", "lm_head", None),
            ("gate.e_score_correction_bias", "gate.e_score_correction_bias", None),
        ]
        if self.fd_config.model_config.use_qk_norm:
            stacked_params_mapping.append(("qk_norm.q_norm", "q_norm", None))
            stacked_params_mapping.append(("qk_norm.k_norm", "k_norm", None))

        params_dict = dict(self.named_parameters())
        process_weights_after_loading_fn = process_weights_after_loading(dict(self.named_sublayers()), self.fd_config)

        for loaded_weight_name, loaded_weight in weights_iterator:
            if ".mlp.experts." in loaded_weight_name:
                continue

            model_param_name = None
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in loaded_weight_name:
                    continue
                candidate = loaded_weight_name.replace(weight_name, param_name)
                if candidate not in params_dict:
                    continue
                param = params_dict[candidate]
                weight_loader = getattr(param, "weight_loader", default_weight_loader(self.fd_config))
                weight_loader(param, loaded_weight, shard_id)
                model_param_name = candidate
                break

            if model_param_name is None:
                if loaded_weight_name not in params_dict:
                    continue
                param = params_dict[loaded_weight_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader(self.fd_config))
                weight_loader(param, loaded_weight)
                model_param_name = loaded_weight_name

            model_sublayer_name = re.sub(r"\.(up_gate_proj_weight|down_proj_weight|weight)$", "", model_param_name)
            process_weights_after_loading_fn(model_sublayer_name, param)

    @paddle.no_grad()
    def set_state_dict(self, state_dict):
        raise NotImplementedError("AFD models only support loading weights with the default_v1 loader.")

    def forward(self, inputs: Dict, forward_meta: ForwardMeta):
        ids_remove_padding = inputs["ids_remove_padding"]
        return self.model(ids_remove_padding=ids_remove_padding, forward_meta=forward_meta)

    def compute_logits(self, hidden_states: paddle.Tensor, forward_meta: ForwardMeta = None):
        logits = self.lm_head(hidden_states)
        logits = logits.astype(paddle.float32)
        logits[:, self.ori_vocab_size :] = -float("inf")
        return logits

    def empty_input_forward(self, forward_meta):
        return None

    def clear_grpah_opt_backend(self):
        self.model.clear_grpah_opt_backend(fd_config=self.fd_config)

    @paddle.no_grad()
    def update_state_dict(self, state_dict):
        return None


# =====================================================================
#  FFN worker CausalLM
# =====================================================================

class Glm4MoeForCausalLM_AFDFFN(ModelForCasualLM):
    def __init__(self, fd_config: FDConfig):
        super().__init__(fd_config)

        self.redundant_table_manger = RedundantExpertManger(
            n_routed_experts=fd_config.model_config.n_routed_experts,
            num_hidden_layers=fd_config.model_config.num_hidden_layers,
            redundant_experts_num=fd_config.eplb_config.redundant_experts_num,
            ep_size=fd_config.parallel_config.expert_parallel_size,
            fd_config=fd_config,
        ) if fd_config.eplb_config.enable_eplb else None
        self._afd_runner = AFDDecodeRunner(fd_config)

        self._moe_layer_ids = list(
            range(fd_config.model_config.first_k_dense_replace, fd_config.model_config.num_hidden_layers)
        )
        self.model = Glm4AFDFFNModel(
            fd_config,
            afd_runner=self._afd_runner,
            moe_layer_ids=self._moe_layer_ids,
            redundant_table_manger=self.redundant_table_manger,
        )

        capture_sizes = self.fd_config.graph_opt_config.cudagraph_capture_sizes or [1]
        capture_sizes = sorted({int(size) for size in capture_sizes if int(size) > 0}, reverse=True)
        if not capture_sizes:
            capture_sizes = [1]

        self._afd_ffn_graph_metas = []
        for capture_size in capture_sizes:
            ids_remove_padding = paddle.zeros([capture_size], dtype=paddle.int64)
            forward_meta = SimpleNamespace(
                ids_remove_padding=ids_remove_padding,
                exist_prefill=False,
                step_use_cudagraph=True,
            )
            self._afd_ffn_graph_metas.append((ids_remove_padding, forward_meta))
        self._afd_ffn_graph_meta_index = 0
        self._expert_params_mapping = FusedMoE.make_expert_params_mapping(
            num_experts=self.fd_config.model_config.n_routed_experts,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            param_gate_up_proj_name="experts.up_gate_proj_",
            param_down_proj_name="experts.down_proj_",
        )

    @classmethod
    def name(cls):
        return "Glm4MoeForCausalLM_AFDFFN"

    @paddle.no_grad()
    def load_weights(self, weights_iterator):
        from fastdeploy.model_executor.utils import process_weights_after_loading

        params_dict = dict(self.named_parameters())
        process_weights_after_loading_fn = process_weights_after_loading(dict(self.named_sublayers()), self.fd_config)

        for loaded_weight_name, loaded_weight in weights_iterator:
            if ".mlp.experts." not in loaded_weight_name:
                continue

            for param_name, weight_name, expert_id, shard_id in self._expert_params_mapping:
                if weight_name not in loaded_weight_name:
                    continue
                model_param_name = loaded_weight_name.replace(weight_name, param_name)
                model_param_name = model_param_name.replace("model.layers.", "model.moe_blocks.")
                model_param_name = model_param_name.replace(".mlp.experts.", ".experts.")
                if model_param_name not in params_dict:
                    continue
                param = params_dict[model_param_name]

                physical_expert_id = expert_id
                if self.redundant_table_manger is None:
                    physical_expert_id = self.fd_config.afd_config.afd_static_log2phy[expert_id]
                param.weight_loader(param, loaded_weight, shard_id=shard_id, expert_id=physical_expert_id)

                model_sublayer_name = re.sub(
                    r"\.(up_gate_proj_weight|down_proj_weight|weight)$", "", model_param_name
                )
                process_weights_after_loading_fn(model_sublayer_name, param)
                break

    @paddle.no_grad()
    def set_state_dict(self, state_dict):
        raise NotImplementedError("AFD models only support loading weights with the default_v1 loader.")

    def forward(self, inputs: Dict, forward_meta: ForwardMeta):
        should_advance = (
            self.model.use_graph_opt
            and self.fd_config.graph_opt_config.use_cudagraph
            and self._afd_ffn_graph_meta_index < len(self._afd_ffn_graph_metas)
        )
        if should_advance:
            ids_remove_padding, graph_forward_meta = self._afd_ffn_graph_metas[self._afd_ffn_graph_meta_index]
        else:
            ids_remove_padding, graph_forward_meta = self._afd_ffn_graph_metas[0]
        output = self.model(
            ids_remove_padding=ids_remove_padding,
            forward_meta=graph_forward_meta,
        )
        if should_advance:
            self._afd_ffn_graph_meta_index += 1
        return output

    def compute_logits(self, hidden_states, forward_meta=None):
        return None

    def empty_input_forward(self, forward_meta):
        return self.forward(None, forward_meta)

    def clear_grpah_opt_backend(self):
        self.model.clear_grpah_opt_backend(fd_config=self.fd_config)
        self._afd_ffn_graph_meta_index = 0

    @paddle.no_grad()
    def update_state_dict(self, state_dict):
        from fastdeploy.model_executor.utils import process_weights_after_loading

        if isinstance(state_dict, list):
            state_dict = dict(state_dict)
        if not any(key.startswith("model.layers.") and ".mlp.experts." in key for key in state_dict):
            self.model.update_state_dict(state_dict)
            return

        params_dict = dict(self.named_parameters())
        process_weights_after_loading_fn = process_weights_after_loading(dict(self.named_sublayers()), self.fd_config)
        updated_layers = set()

        for loaded_weight_name, loaded_weight in state_dict.items():
            if ".mlp.experts." not in loaded_weight_name:
                continue
            layer_id = int(loaded_weight_name.split(".mlp.experts.", 1)[0].rsplit(".", 1)[-1])
            if layer_id not in self._moe_layer_ids:
                continue
            if layer_id not in updated_layers:
                logger.info(f"Start AFD FFN update layer {layer_id}")
                updated_layers.add(layer_id)
            for param_name, weight_name, expert_id, shard_id in self._expert_params_mapping:
                if weight_name not in loaded_weight_name:
                    continue
                model_param_name = loaded_weight_name.replace(weight_name, param_name)
                model_param_name = model_param_name.replace("model.layers.", "model.moe_blocks.")
                model_param_name = model_param_name.replace(".mlp.experts.", ".experts.")
                if model_param_name not in params_dict:
                    continue
                param = params_dict[model_param_name]
                moe_block = self.model.moe_blocks[str(layer_id)]
                weight_loader = getattr(param, "weight_loader", moe_block.experts.weight_loader)
                physical_expert_id = expert_id
                if self.redundant_table_manger is None:
                    physical_expert_id = self.fd_config.afd_config.afd_static_log2phy[expert_id]
                weight_loader(param, loaded_weight, shard_id=shard_id, expert_id=physical_expert_id)
                model_sublayer_name = re.sub(
                    r"\.(up_gate_proj_weight|down_proj_weight|weight)$", "", model_param_name
                )
                process_weights_after_loading_fn(model_sublayer_name, param)
                break
        for layer_id in sorted(updated_layers):
            logger.info(f"Finish AFD FFN update layer {layer_id}")


ModelRegistry.register_model_class(
    Glm4MoeForCausalLM_AFDAttn,
    architecture="Glm4MoeForCausalLM_AFDAttn",
    module_name="afd_model.glm4moe_afd",
    category=ModelCategory.TEXT_GENERATION,
    primary_use=ModelCategory.TEXT_GENERATION,
)
ModelRegistry.register_model_class(
    Glm4MoeForCausalLM_AFDFFN,
    architecture="Glm4MoeForCausalLM_AFDFFN",
    module_name="afd_model.glm4moe_afd",
    category=ModelCategory.TEXT_GENERATION,
    primary_use=ModelCategory.TEXT_GENERATION,
)
