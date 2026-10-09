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

import numpy as np

__all__ = ["evict_order", "importance_evict_order", "keep_from_order", "gpu_imbalance", "coverage_repair"]


def importance_evict_order(importance: np.ndarray) -> np.ndarray:
    """Rank experts for eviction by importance alone, ignoring similarity.

    This is the exp4 "rebalance" policy: keep the highest-importance experts and
    give up the lowest. Importance is summed over domains, then each layer's
    experts are sorted ascending, so the most expendable (lowest importance) comes
    first and the length-k prefix is the k experts to drop. This matches the
    contract of :func:`evict_order` (same return layout and prefix semantics), so
    every consumer -- ``keep_from_order``, ``reserve_experts`` -- works unchanged.

    Args:
        importance: [D, L, E] importance score per domain, layer and expert.

    Returns:
        order: [L, E] int16, experts in the order they should be given up.
    """
    if importance.ndim != 3:
        raise ValueError(f"importance must be [D, L, E], got {importance.shape}")
    per_layer = importance.astype(np.float64).sum(axis=0)  # [L, E]
    # Ascending importance: the least important expert is the first to drop.
    return np.argsort(per_layer, axis=1, kind="stable").astype(np.int16)


def evict_order(importance: np.ndarray, similarity: np.ndarray):
    """
    Rank logical experts by how expendable they are, per layer, minimising

        Loss(K) = 1/D * sum_d sum_u P[d, u] * (1 - max_{j in K} S[d, u, j])

    The greedy gives up the cheapest expert each round and recomputes; ranking
    once on the full set would give up both halves of a mutually similar pair,
    each replaceable only by the other. The length-k prefix of the result is
    exactly what the same greedy would drop if asked for k, so one order serves
    every surviving-slot count.

    Args:
        importance: [D, L, E] importance score per domain, layer and expert.
        similarity: [D, L, E, E], how well expert j stands in for expert u.
            The diagonal must be 1 so that a kept expert costs nothing.

    Returns:
        order: [L, E] int16, experts in the order they should be given up.
        cost: [L, E] float32, Loss increase per step, non-decreasing along a
            row. The last entry is +inf.
    """
    num_domains, num_layers, num_experts = importance.shape

    order = np.empty((num_layers, num_experts), dtype=np.int16)
    cost = np.empty((num_layers, num_experts), dtype=np.float32)
    for layer in range(num_layers):
        order[layer], cost[layer] = _order_one_layer(importance[:, layer], similarity[:, layer])
    return order, cost


def _order_one_layer(importance: np.ndarray, similarity: np.ndarray):
    """importance [D, E], similarity [D, E, E] (not modified) -> (order, cost)."""
    num_domains, num_experts = importance.shape
    flat_importance = importance.reshape(-1).astype(np.float64)
    # Rows are (domain, expert) pairs, columns are candidate stand-ins.
    work = np.array(similarity, dtype=np.float64).reshape(-1, num_experts)
    rows = np.arange(num_domains * num_experts)

    alive = np.ones(num_experts, dtype=bool)
    order = np.empty(num_experts, dtype=np.int16)
    cost = np.empty(num_experts, dtype=np.float32)

    best_col = work.argmax(axis=1)
    best = work[rows, best_col]
    work[rows, best_col] = -np.inf
    second_col = work.argmax(axis=1)
    second = work[rows, second_col]
    work[rows, best_col] = best
    gain = flat_importance * (best - second)
    delta = np.bincount(best_col, weights=gain, minlength=num_experts)

    for step in range(num_experts - 1):
        victim = int(np.where(alive, delta, np.inf).argmin())
        order[step] = victim
        cost[step] = delta[victim] / num_domains
        alive[victim] = False
        if step == num_experts - 2:
            break
        # Masking the column also clears the diagonal, so a dropped expert stops
        # covering itself.
        work[:, victim] = -np.inf

        affected = np.flatnonzero((best_col == victim) | (second_col == victim))
        if affected.size:
            delta -= np.bincount(best_col[affected], weights=gain[affected], minlength=num_experts)
            sub = work[affected]
            local = np.arange(affected.size)
            best_col[affected] = sub.argmax(axis=1)
            best[affected] = sub[local, best_col[affected]]
            sub[local, best_col[affected]] = -np.inf
            second_col[affected] = sub.argmax(axis=1)
            second[affected] = sub[local, second_col[affected]]
            gain[affected] = flat_importance[affected] * (best[affected] - second[affected])
            delta += np.bincount(best_col[affected], weights=gain[affected], minlength=num_experts)

    order[num_experts - 1] = int(np.flatnonzero(alive)[0])
    cost[num_experts - 1] = np.inf
    return order, cost


def keep_from_order(order: np.ndarray, capacity: int) -> np.ndarray:
    """
    Args:
        order: [L, E] int, from `evict_order`.
        capacity: physical slots per layer, i.e. num_local_physical_experts
            times the number of surviving FFN ranks.

    Returns:
        keep: [L, E] bool, all True when every expert fits.
    """
    if capacity < 1:
        raise ValueError(f"capacity must be at least 1, got {capacity}")
    num_layers, num_experts = order.shape
    keep = np.ones((num_layers, num_experts), dtype=bool)
    num_evict = num_experts - capacity
    if num_evict <= 0:
        return keep
    keep[np.arange(num_layers)[:, None], order[:, :num_evict]] = False
    return keep


def gpu_imbalance(slots: np.ndarray, load: np.ndarray, num_gpus: int) -> float:
    """
    Peak over mean GPU load for one layer. A logical expert's load splits evenly
    over its replicas, matching the kernel's uniform pick among them.

    Args:
        slots: [C] logical expert per surviving slot, ordered by (gpu, slot).
        load: [E] recent token count per logical expert.
        num_gpus: surviving GPUs; len(slots) must divide by it.
    """
    count = np.bincount(slots, minlength=load.shape[0])
    per_gpu = (load[slots] / count[slots]).reshape(num_gpus, -1).sum(axis=1)
    mean = per_gpu.mean()
    return float(per_gpu.max() / mean) if mean > 0 else 1.0


def coverage_repair(
    keep: np.ndarray,
    slots: np.ndarray,
    importance: np.ndarray,
    load: np.ndarray,
    num_gpus: int,
    target_imbalance: float = 1.05,
    max_swap_rounds: int = 400,
):
    """
    Turn a target keep set into the H2D writes that realise it.

    Two phases per layer. First refill coverage, taking slots from experts with a
    spare replica before slots holding the last copy of an expert `keep` drops.
    Then trade slots between the hottest and coldest GPU until the load spread
    is acceptable; a trade costs two writes but no coverage.

    Args:
        keep: [L, E] bool, from `keep_from_order`.
        slots: [L, C] logical expert per surviving slot, ordered by (gpu, slot).
            Not modified.
        importance: [L, E], orders which expert to bring back first and which to
            give up when no spare replica is left.
        load: [L, E] recent token count per logical expert.
        num_gpus: surviving FFN GPUs; C must divide by it.
        target_imbalance: stop trading once peak/mean GPU load is at most this.
        max_swap_rounds: cap on trades per layer.

    Returns:
        moves: list of (layer, slot, expert), in the order they should be
            applied. Coverage comes first, so truncating still leaves a valid
            placement.
        new_slots: [L, C] the placement after every move.
    """
    num_layers, num_experts = keep.shape
    capacity = slots.shape[1]
    if slots.shape[0] != num_layers:
        raise ValueError(f"keep has {num_layers} layers, slots has {slots.shape[0]}")
    if capacity % num_gpus:
        raise ValueError(f"capacity {capacity} is not a multiple of num_gpus {num_gpus}")
    over = keep.sum(axis=1) > capacity
    if over.any():
        raise ValueError(
            f"layer {int(np.flatnonzero(over)[0])} keeps "
            f"{int(keep.sum(axis=1).max())} experts but only has {capacity} slots"
        )

    new_slots = np.array(slots)
    slots_per_gpu = capacity // num_gpus
    gpu_writes = np.zeros(num_gpus, dtype=np.int64)
    for layer in range(num_layers):
        row = new_slots[layer]
        _refill_layer(keep[layer], row, importance[layer], num_gpus, gpu_writes)
        _balance_layer(row, load[layer], num_gpus, target_imbalance, max_swap_rounds)
        np.add.at(gpu_writes, np.flatnonzero(slots[layer] != row) // slots_per_gpu, 1)

    changed = np.nonzero(slots != new_slots)
    moves = [(int(layer), int(slot), int(new_slots[layer, slot])) for layer, slot in zip(*changed)]
    return moves, new_slots


def _refill_layer(
    keep: np.ndarray,
    slots: np.ndarray,
    importance: np.ndarray,
    num_gpus: int,
    gpu_writes: np.ndarray,
) -> None:
    """Bring every kept expert back into `slots` (modified in place)."""
    count = np.bincount(slots, minlength=keep.shape[0])
    incoming = np.flatnonzero(keep & (count == 0))
    incoming = incoming[np.argsort(-importance[incoming])]
    slots_per_gpu = slots.shape[0] // num_gpus
    load = gpu_writes.astype(np.int64).copy()

    for expert in incoming:
        spare = count[slots] >= 2
        donors = np.flatnonzero(spare) if spare.any() else np.flatnonzero(~keep[slots])
        gpus = donors // slots_per_gpu
        # Least loaded GPU, then lowest slot: donors is ascending and argmin takes the
        # first minimum.
        pick = int(np.argmin(load[gpus]))
        slot, gpu = int(donors[pick]), int(gpus[pick])

        count[slots[slot]] -= 1
        slots[slot] = expert
        count[expert] += 1
        load[gpu] += 1


def _balance_layer(
    slots: np.ndarray,
    load: np.ndarray,
    num_gpus: int,
    target: float,
    max_rounds: int,
) -> None:
    """Trade slots between the hottest and coldest GPU (`slots` modified in place)."""
    count = np.bincount(slots, minlength=load.shape[0])
    per_slot = load[slots] / count[slots]
    slots_per_gpu = slots.shape[0] // num_gpus

    for _ in range(max_rounds):
        per_gpu = per_slot.reshape(num_gpus, slots_per_gpu).sum(axis=1)
        mean = per_gpu.mean()
        if mean <= 0 or per_gpu.max() / mean <= target:
            break

        hot, cold = int(per_gpu.argmax()), int(per_gpu.argmin())
        hot_slots = slice(hot * slots_per_gpu, (hot + 1) * slots_per_gpu)
        cold_slots = slice(cold * slots_per_gpu, (cold + 1) * slots_per_gpu)
        # Trading slot i of the hot GPU for slot j of the cold one moves this
        # much load. Refuse trades that overshoot, or the pair oscillates.
        gain = per_slot[hot_slots][:, None] - per_slot[cold_slots][None, :]
        allowed = (gain > 0) & (per_gpu[hot] - gain > per_gpu[cold] + gain - 1e-9)
        if not allowed.any():
            break

        i, j = divmod(int(np.where(allowed, gain, -np.inf).argmax()), slots_per_gpu)
        left, right = hot * slots_per_gpu + i, cold * slots_per_gpu + j
        slots[left], slots[right] = slots[right], slots[left]
        per_slot[left], per_slot[right] = per_slot[right], per_slot[left]


def reserve_experts(
    phy2log: np.ndarray,
    dropped: np.ndarray,
    load: np.ndarray,
    dead_ranks: list,
    local_physical_experts: int,
    max_replicas: int,
) -> np.ndarray:
    """Reserve the experts the down ranks load once they come back.

    Args:
        phy2log: [L, num_physical_experts] int32 with the survivors already placed,
            modified in place.
        dropped: [L, K] int, `evict_order`'s first K columns (most expendable first);
            K is the number of experts the survivors' plan dropped.
        load: [L, E] recent token count per logical expert.
        dead_ranks: global FFN rank ids that are down.
        local_physical_experts: slots per FFN rank.
        max_replicas: replica cap per logical expert.

    Returns:
        phy2log.
    """
    capacity = len(dead_ranks) * local_physical_experts
    if capacity == 0:
        return phy2log

    dead_slots = np.concatenate(
        [np.arange(rank * local_physical_experts, (rank + 1) * local_physical_experts) for rank in dead_ranks]
    )

    # `dropped` lists the most expendable expert first, so reverse it: the dropped experts
    # that matter most get a slot back first.
    picked = dropped[:, ::-1].astype(phy2log.dtype)
    if picked.shape[1] < capacity:
        picked = np.concatenate(
            [picked, _hot_replicas(phy2log, picked, load, capacity - picked.shape[1], max_replicas)],
            axis=1,
        )

    # Deal the picks round-robin over the dead ranks.
    num_layers = picked.shape[0]
    dealt = picked.reshape(num_layers, local_physical_experts, len(dead_ranks)).transpose(0, 2, 1)
    phy2log[:, dead_slots] = dealt.reshape(num_layers, -1)
    return phy2log


def _hot_replicas(phy2log: np.ndarray, picked: np.ndarray, load: np.ndarray, want: int, max_replicas: int):
    """[L, want] hottest experts per layer, repeated when there are fewer experts than slots."""
    num_layers, num_experts = load.shape
    count = np.zeros((num_layers, num_experts), dtype=np.int64)
    placed = phy2log >= 0
    np.add.at(count, (np.repeat(np.arange(num_layers), phy2log.shape[1])[placed.ravel()], phy2log[placed]), 1)
    count[np.arange(num_layers)[:, None], picked] += 1  # the picked experts already hold one copy

    hot = np.argsort(-load, axis=1)
    rows = np.arange(num_layers)[:, None]
    taken = np.zeros((num_layers, num_experts), dtype=bool)
    taken[rows, picked] = True

    # Fast path: the hottest experts nobody picked that still have room, one copy each.
    free = ~taken[rows, hot] & (count[rows, hot] < max_replicas)
    free &= np.cumsum(free, axis=1) <= want
    if free.sum(axis=1).min() == want:
        return hot[free].reshape(num_layers, want).astype(phy2log.dtype)

    # Fewer distinct experts than slots (several ranks down at once): repeat the hottest
    # ones that still have room, so no slot is left at -1.
    chosen = np.empty((num_layers, want), dtype=phy2log.dtype)
    for layer in range(num_layers):
        # Experts nobody picked first, then the picked ones, hottest first in both groups.
        ranking = np.concatenate([hot[layer][~taken[layer][hot[layer]]], hot[layer][taken[layer][hot[layer]]]])
        room = max_replicas - count[layer]
        picks = []
        for expert in ranking:
            if len(picks) == want:
                break
            picks.extend([int(expert)] * min(int(room[expert]), want - len(picks)))
        if len(picks) < want:
            raise ValueError(f"layer {layer}: cannot reserve {want} experts within {max_replicas} copies")
        chosen[layer] = picks
    return chosen
