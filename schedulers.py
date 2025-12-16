"""Scheduling algorithms for LTE/5G simulator."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class ChannelState:
    rates: np.ndarray  # shape (users, slots, rbs) achievable rate or quality
    user_groups: Optional[List[str]] = None
    traffic_classes: Optional[List[str]] = None


@dataclass
class QoSState:
    queue: np.ndarray
    hol_delay: np.ndarray
    avg_rate: np.ndarray


@dataclass
class AllocationResult:
    allocation: np.ndarray  # shape (users, slots, rbs)
    per_slot_throughput: np.ndarray  # same shape as allocation (rate served per rb)
    metadata: Dict[str, np.ndarray] = field(default_factory=dict)


EPS = 1e-9


def _flatten_allocation(per_slot_allocation: List[np.ndarray]) -> np.ndarray:
    if not per_slot_allocation:
        return np.zeros((0, 0))
    stacked = np.stack(per_slot_allocation, axis=1)
    return stacked


def rr_schedule(num_users: int, num_rbs: int, num_slots: int) -> AllocationResult:
    allocations = []
    throughput = []
    for slot in range(num_slots):
        alloc = np.zeros((num_users, num_rbs))
        for rb in range(num_rbs):
            user = (slot * num_rbs + rb) % num_users
            alloc[user, rb] = 1
        allocations.append(alloc)
        throughput.append(np.zeros_like(alloc))
    return AllocationResult(_flatten_allocation(allocations), _flatten_allocation(throughput))


def pf_schedule(num_users: int, num_rbs: int, num_slots: int, channel: ChannelState, beta: float = 0.9) -> AllocationResult:
    avg_rate = np.ones(num_users)
    allocations = []
    throughput = []
    for slot in range(num_slots):
        slot_alloc = np.zeros((num_users, num_rbs))
        slot_thr = np.zeros_like(slot_alloc)
        current_rates = channel.rates[:, slot]
        for rb in range(num_rbs):
            metric = current_rates[:, rb] / (avg_rate + EPS)
            user = int(np.argmax(metric))
            slot_alloc[user, rb] = 1
            slot_thr[user, rb] = current_rates[user, rb]
            avg_rate[user] = beta * avg_rate[user] + (1 - beta) * current_rates[user, rb]
        allocations.append(slot_alloc)
        throughput.append(slot_thr)
    return AllocationResult(_flatten_allocation(allocations), _flatten_allocation(throughput))


def mt_schedule(num_users: int, num_rbs: int, num_slots: int, channel: ChannelState) -> AllocationResult:
    allocations = []
    throughput = []
    for slot in range(num_slots):
        slot_alloc = np.zeros((num_users, num_rbs))
        slot_thr = np.zeros_like(slot_alloc)
        current_rates = channel.rates[:, slot]
        winners = np.argmax(current_rates, axis=0)
        for rb, user in enumerate(winners):
            slot_alloc[user, rb] = 1
            slot_thr[user, rb] = current_rates[user, rb]
        allocations.append(slot_alloc)
        throughput.append(slot_thr)
    return AllocationResult(_flatten_allocation(allocations), _flatten_allocation(throughput))


def wrr_schedule(num_users: int, num_rbs: int, num_slots: int, weights: Sequence[int]) -> AllocationResult:
    if len(weights) != num_users:
        raise ValueError("Weights length must match number of users")
    expanded: List[int] = []
    for user, weight in enumerate(weights):
        expanded.extend([user] * max(1, int(weight)))
    if not expanded:
        expanded = list(range(num_users))
    allocations = []
    throughput = []
    idx = 0
    for _ in range(num_slots):
        slot_alloc = np.zeros((num_users, num_rbs))
        slot_thr = np.zeros_like(slot_alloc)
        for rb in range(num_rbs):
            user = expanded[idx % len(expanded)]
            slot_alloc[user, rb] = 1
            idx += 1
        allocations.append(slot_alloc)
        throughput.append(slot_thr)
    return AllocationResult(_flatten_allocation(allocations), _flatten_allocation(throughput), metadata={"weights": np.array(weights)})


def exp_pf_schedule(
    num_users: int,
    num_rbs: int,
    num_slots: int,
    channel: ChannelState,
    rng: np.random.Generator,
    traffic_classes: Sequence[str],
    arrival_rate: float = 1.0,
    queue_limit: int = 100,
) -> AllocationResult:
    avg_rate = np.ones(num_users)
    queue = np.zeros(num_users)
    hol_delay = np.zeros(num_users)
    allocations = []
    throughput = []
    hol_history: List[np.ndarray] = []

    class_alpha = {"voip": 1.2, "video": 0.8, "be": 0.4}
    delay_norm = {"voip": 4.0, "video": 8.0, "be": 12.0}
    for slot in range(num_slots):
        slot_alloc = np.zeros((num_users, num_rbs))
        slot_thr = np.zeros_like(slot_alloc)
        current_rates = channel.rates[:, slot]
        arrivals = rng.poisson(arrival_rate, size=num_users)
        queue = np.minimum(queue + arrivals, queue_limit)
        hol_delay += queue > 0
        hol_history.append(hol_delay.copy())

        for rb in range(num_rbs):
            cls = np.array(traffic_classes)
            alpha_vec = np.vectorize(class_alpha.get)(cls)
            norm_vec = np.vectorize(delay_norm.get)(cls)
            metric = np.exp(alpha_vec * (hol_delay / (norm_vec + EPS))) * (current_rates[:, rb] / (avg_rate + EPS))
            user = int(np.argmax(metric))
            slot_alloc[user, rb] = 1
            served = min(queue[user], max(1, int(np.round(current_rates[user, rb]))))
            queue[user] = max(0, queue[user] - served)
            hol_delay[user] = queue[user] > 0
            slot_thr[user, rb] = current_rates[user, rb]
            avg_rate[user] = 0.9 * avg_rate[user] + 0.1 * current_rates[user, rb]
        allocations.append(slot_alloc)
        throughput.append(slot_thr)

    metadata = {"hol_delay": hol_delay.copy(), "queue": queue.copy(), "avg_rate": avg_rate.copy()}
    if hol_history:
        metadata["hol_history"] = np.stack(hol_history, axis=0)
    return AllocationResult(_flatten_allocation(allocations), _flatten_allocation(throughput), metadata=metadata)


def get_scheduler(name: str):
    name = name.lower()
    if name == "rr":
        return rr_schedule
    if name == "pf":
        return pf_schedule
    if name in {"mt", "max_cqi"}:
        return mt_schedule
    if name == "wrr":
        return wrr_schedule
    if name == "exp_pf":
        return exp_pf_schedule
    raise ValueError(f"Unknown scheduler {name}")
