"""Command-line runner for LTE/5G scheduling simulations with scenarios and aggregation."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from schedulers import (
    AllocationResult,
    ChannelState,
    exp_pf_schedule,
    get_scheduler,
    mt_schedule,
    pf_schedule,
    rr_schedule,
    wrr_schedule,
)
from simulator import plot_allocation


ALL_SCHEDULERS = ["rr", "pf", "mt", "wrr", "exp_pf"]
SCENARIO_LIST = ["load", "hetero_channel", "qos", "bursty", "classroom"]


@dataclass
class RunCase:
    scenario: str
    users: int
    rbs: int
    slots: int
    seed: int
    schedulers: Sequence[str]
    group_weights: Optional[Dict[str, int]] = None
    weights: Optional[List[int]] = None
    traffic_classes: Optional[List[str]] = None
    label: Optional[str] = None
    users_list: Optional[List[int]] = None
    near_mean: float = 3.0
    mid_mean: float = 1.5
    edge_mean: float = 0.6


@dataclass
class ScenarioResult:
    run_id: str
    scenario: str
    label: Optional[str]
    users: int
    rbs: int
    slots: int
    seed: int
    scheduler: str
    allocation: np.ndarray
    throughput: np.ndarray
    fairness: float
    utilization: float
    total_throughput: float
    starvation: float
    per_user_total: np.ndarray
    per_user_share: np.ndarray
    group_info: Optional[List[str]] = None
    traffic_classes: Optional[List[str]] = None
    edge_share: Optional[float] = None
    edge_throughput: Optional[float] = None
    delay_avg: Optional[float] = None
    delay_p95: Optional[float] = None
    delay_p95_per_class: Optional[Dict[str, float]] = None
    drop_rate: Optional[float] = None


PRESETS = {
    "default": RunCase(
        scenario="hetero_channel",
        users=30,
        rbs=12,
        slots=80,
        seed=42,
        schedulers=ALL_SCHEDULERS,
    ),
    "classroom": RunCase(
        scenario="classroom",
        users=30,
        rbs=12,
        slots=80,
        seed=42,
        schedulers=ALL_SCHEDULERS,
        label="classroom",
    ),
    "classroom_fast": RunCase(
        scenario="classroom",
        users=30,
        rbs=12,
        slots=60,
        seed=42,
        schedulers=["rr", "pf", "mt"],
        label="classroom_fast",
    ),
    "stress": RunCase(
        scenario="load",
        users=40,
        rbs=12,
        slots=90,
        seed=7,
        schedulers=ALL_SCHEDULERS,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LTE/5G scheduling simulations.")
    parser.add_argument(
        "--scheduler",
        nargs="*",
        choices=ALL_SCHEDULERS + ["all"],
        default=None,
        help="Scheduler(s) to run. Default from preset.",
    )
    parser.add_argument("--users", type=int, help="Number of users (for single-run override).", default=None)
    parser.add_argument("--rbs", type=int, help="Resource blocks per slot.", default=None)
    parser.add_argument("--slots", type=int, help="Number of slots to simulate.", default=None)
    parser.add_argument("--seed", type=int, help="Random seed.", default=None)
    parser.add_argument("--seeds", type=str, help="Comma list or range (e.g., 1,2,3 or 1-5) of seeds.", default=None)
    parser.add_argument("--outdir", type=str, default=None, help="Output directory base.")
    parser.add_argument("--preset", choices=list(PRESETS.keys()), default="classroom", help="Preset to use.")
    parser.add_argument("--scenario", choices=SCENARIO_LIST, default=None, help="Scenario to run.")
    parser.add_argument("--weights", type=str, default=None, help="Weights for WRR (comma separated or group=weight form).")
    parser.add_argument("--near_mean", type=float, default=None, help="Mean channel quality for near users.")
    parser.add_argument("--mid_mean", type=float, default=None, help="Mean channel quality for mid users.")
    parser.add_argument("--edge_mean", type=float, default=None, help="Mean channel quality for edge users.")
    parser.add_argument("--pack", action="store_true", help="Build presentation pack with key figures.")
    return parser.parse_args()


def validate_positive(name: str, value: Optional[int]) -> int:
    if value is None:
        raise ValueError(f"{name} is required.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return int(value)


def parse_seed_list(seed_arg: Optional[str], fallback_seed: int) -> List[int]:
    if not seed_arg:
        return [fallback_seed]
    if "-" in seed_arg:
        start, end = seed_arg.split("-", maxsplit=1)
        start_i, end_i = int(start), int(end)
        if start_i > end_i:
            start_i, end_i = end_i, start_i
        return list(range(start_i, end_i + 1))
    return [int(val) for val in seed_arg.split(",") if val.strip()]


def parse_weights(weight_arg: Optional[str], users: int, group_weights: Optional[Dict[str, int]] = None) -> Optional[List[int]]:
    if weight_arg is None:
        return None
    if "=" in weight_arg:
        if group_weights is None:
            return None
        mapping: Dict[str, int] = {}
        for part in weight_arg.split(","):
            key, val = part.split("=")
            mapping[key.strip()] = int(val)
        return [mapping.get(g, 1) for g in group_weights.keys()]
    weights = [int(x) for x in weight_arg.split(",") if x.strip()]
    if len(weights) != users:
        raise ValueError("Weights list length must equal number of users.")
    return weights


def prepare_outdir(outdir_arg: Optional[str]) -> Path:
    if outdir_arg:
        base = Path(outdir_arg)
    else:
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        base = Path("outputs") / timestamp
    base.mkdir(parents=True, exist_ok=True)
    return base


def build_cases(args: argparse.Namespace, seed: int) -> List[RunCase]:
    base = PRESETS[args.preset]
    scenario_choice = args.scenario if args.scenario else base.scenario
    schedulers = base.schedulers
    if args.scheduler:
        schedulers = ALL_SCHEDULERS if "all" in args.scheduler else args.scheduler

    users = args.users if args.users is not None else base.users
    rbs = args.rbs if args.rbs is not None else base.rbs
    slots = args.slots if args.slots is not None else base.slots
    near_mean = args.near_mean if args.near_mean is not None else base.near_mean
    mid_mean = args.mid_mean if args.mid_mean is not None else base.mid_mean
    edge_mean = args.edge_mean if args.edge_mean is not None else base.edge_mean

    cases: List[RunCase] = []
    if scenario_choice == "classroom":
        load_case = RunCase(
            scenario="load",
            users=40,
            rbs=12,
            slots=max(70, slots),
            seed=seed,
            schedulers=["rr", "pf", "mt"],
            users_list=[5, 10, 20, 40],
            label="classroom_load",
        )
        hetero_case = RunCase(
            scenario="hetero_channel",
            users=30,
            rbs=12,
            slots=max(70, slots),
            seed=seed,
            schedulers=ALL_SCHEDULERS,
            label="classroom_hetero",
            near_mean=near_mean,
            mid_mean=mid_mean,
            edge_mean=edge_mean,
        )
        cases.extend(build_scenario_cases(load_case, args.weights))
        cases.extend(build_scenario_cases(hetero_case, args.weights))
        return cases

    base_case = RunCase(
        scenario=scenario_choice,
        users=users,
        rbs=rbs,
        slots=slots,
        seed=seed,
        schedulers=schedulers,
        label=args.preset,
        near_mean=near_mean,
        mid_mean=mid_mean,
        edge_mean=edge_mean,
    )
    if scenario_choice == "load":
        base_case.rbs = max(10, rbs)
        base_case.slots = max(60, slots)
        base_case.users_list = [5, 10, 20, 40]
    elif scenario_choice == "hetero_channel":
        base_case.users = 30 if args.users is None else users
        base_case.rbs = rbs if args.rbs is not None else 12
        base_case.slots = max(80, slots)
    elif scenario_choice == "qos":
        base_case.users = users if args.users is not None else 18
        base_case.rbs = rbs if args.rbs is not None else 10
        base_case.slots = max(90, slots)
    elif scenario_choice == "bursty":
        base_case.users = users if args.users is not None else 20
        base_case.rbs = rbs if args.rbs is not None else 12
        base_case.slots = max(80, slots)
    return build_scenario_cases(base_case, args.weights)


def build_scenario_cases(base_case: RunCase, weight_arg: Optional[str]) -> List[RunCase]:
    cases: List[RunCase] = []
    if base_case.scenario == "load":
        user_counts = base_case.users_list or [5, 10, 20, base_case.users]
        user_counts = sorted({int(u) for u in user_counts})
        for u in user_counts:
            cases.append(
                RunCase(
                    scenario="load",
                    users=u,
                    rbs=base_case.rbs,
                    slots=base_case.slots,
                    seed=base_case.seed,
                    schedulers=base_case.schedulers,
                    label=f"load_{u}",
                )
            )
    else:
        cases.append(base_case)

    for case in cases:
        if weight_arg:
            case.weights = parse_weights(weight_arg, case.users)
    return cases


def generate_channel(case: RunCase, rng: np.random.Generator) -> ChannelState:
    if case.scenario == "hetero_channel":
        groups: List[str] = []
        rates = np.zeros((case.users, case.slots, case.rbs))
        for u in range(case.users):
            frac = u / case.users
            if frac < 0.3:
                mean, group = case.near_mean, "near"
            elif frac < 0.7:
                mean, group = case.mid_mean, "mid"
            else:
                mean, group = case.edge_mean, "edge"
            groups.append(group)
            rates[u] = rng.normal(loc=mean, scale=0.5, size=(case.slots, case.rbs)).clip(min=0.1)
        return ChannelState(rates=rates, user_groups=groups)

    if case.scenario == "bursty":
        base_rates = rng.lognormal(mean=1.0, sigma=0.8, size=(case.users, case.slots, case.rbs))
        activity = np.ones((case.slots, case.users))
        for block_start in range(0, case.slots, 10):
            active_users = rng.choice(case.users, size=max(1, case.users // 2), replace=False)
            activity[block_start : block_start + 10, :] = 0.05
            activity[block_start : block_start + 10, active_users] = 1.0
        rates = np.transpose(activity[:, :, None], (1, 0, 2)) * base_rates
        return ChannelState(rates=rates)

    rates = rng.lognormal(mean=1.2, sigma=0.5, size=(case.users, case.slots, case.rbs))
    return ChannelState(rates=rates)


def assign_traffic_classes(case: RunCase) -> List[str]:
    classes: List[str] = []
    for u in range(case.users):
        frac = u / max(1, case.users - 1)
        if frac < 0.4:
            classes.append("voip")
        elif frac < 0.7:
            classes.append("video")
        else:
            classes.append("be")
    return classes


def compute_metrics(
    case: RunCase,
    scheduler: str,
    alloc: AllocationResult,
    channel: ChannelState,
    traffic_classes: Optional[List[str]],
) -> ScenarioResult:
    allocation = alloc.allocation
    throughput = alloc.per_slot_throughput
    flat_allocation = allocation.reshape(allocation.shape[0], -1)
    per_user_total = flat_allocation.sum(axis=1)
    total_allocated = per_user_total.sum()
    share_pct = (per_user_total / total_allocated * 100) if total_allocated > 0 else np.zeros_like(per_user_total)
    fairness = float(np.square(per_user_total.sum()) / (len(per_user_total) * np.square(per_user_total).sum() + 1e-9)) if total_allocated > 0 else 0.0
    utilization = float(total_allocated / (allocation.shape[1] * allocation.shape[2])) if allocation.size else 0.0
    starvation = float(np.mean(per_user_total == 0))
    total_throughput = float(throughput.sum())

    edge_share = None
    edge_throughput = None
    if channel.user_groups:
        groups = np.array(channel.user_groups)
        edge_mask = groups == "edge"
        if edge_mask.any():
            edge_share = float(per_user_total[edge_mask].sum() / total_allocated) if total_allocated > 0 else 0.0
            edge_throughput = float(throughput[edge_mask].sum())

    delay_avg = None
    delay_p95 = None
    delay_per_class: Optional[Dict[str, float]] = None
    if "hol_history" in alloc.metadata:
        hol_hist = np.array(alloc.metadata["hol_history"])
        delays = hol_hist.flatten()
        delay_avg = float(np.mean(delays)) if delays.size else 0.0
        delay_p95 = float(np.percentile(delays, 95)) if delays.size else 0.0
        if traffic_classes:
            traffic_arr = np.array(traffic_classes)
            delay_per_class = {}
            for cls in np.unique(traffic_arr):
                cls_mask = traffic_arr == cls
                cls_delays = hol_hist[:, cls_mask].reshape(-1)
                delay_per_class[cls] = float(np.percentile(cls_delays, 95)) if cls_delays.size else 0.0

    drop_rate = None
    if "dropped" in alloc.metadata:
        dropped = float(np.sum(alloc.metadata["dropped"]))
        generated = float(np.sum(alloc.metadata.get("arrivals", 0))) + dropped
        drop_rate = dropped / generated if generated > 0 else 0.0

    run_id = f"{case.scenario}__{case.users}u_{case.rbs}r_{case.slots}s"
    return ScenarioResult(
        run_id=run_id,
        scenario=case.scenario,
        label=case.label,
        users=case.users,
        rbs=case.rbs,
        slots=case.slots,
        seed=case.seed,
        scheduler=scheduler,
        allocation=allocation,
        throughput=throughput,
        fairness=fairness,
        utilization=utilization,
        total_throughput=total_throughput,
        starvation=starvation,
        per_user_total=per_user_total,
        per_user_share=share_pct,
        group_info=channel.user_groups,
        traffic_classes=traffic_classes,
        edge_share=edge_share,
        edge_throughput=edge_throughput,
        delay_avg=delay_avg,
        delay_p95=delay_p95,
        delay_p95_per_class=delay_per_class,
        drop_rate=drop_rate,
    )


def save_heatmaps(results: List[ScenarioResult], outdir: Path) -> None:
    for idx, res in enumerate(results):
        if idx > 1:
            break
        flat = res.allocation.reshape(res.allocation.shape[0], -1)
        plot_allocation(
            flat,
            title=f"{res.scheduler.upper()} Allocation",
            save_path=str(outdir / f"{res.run_id}_{res.scheduler}_heatmap.png"),
            show=False,
        )


def save_bar_chart(per_user_total: np.ndarray, scheduler_name: str, path: Path) -> None:
    plt.figure(figsize=(8, 4))
    plt.bar(np.arange(len(per_user_total)), per_user_total, color="#4C72B0")
    plt.xlabel("User ID")
    plt.ylabel("Total Allocated RBs")
    plt.title(f"RBs per User - {scheduler_name.upper()}")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_cumulative_plot(allocation: np.ndarray, scheduler_name: str, path: Path) -> None:
    flat = allocation.reshape(allocation.shape[0], -1)
    total_rbs = flat.shape[1]
    rb_axis = np.arange(1, total_rbs + 1)
    plt.figure(figsize=(8, 4))
    per_user_total = flat.sum(axis=1)
    if len(per_user_total) > 5:
        top_indices = np.argsort(per_user_total)[-5:][::-1]
        other_indices = [i for i in range(len(per_user_total)) if i not in top_indices]
        for idx in top_indices:
            plt.plot(rb_axis, np.cumsum(flat[idx]), label=f"User {idx}")
        if other_indices:
            others_cumulative = np.cumsum(flat[other_indices].sum(axis=0))
            plt.plot(rb_axis, others_cumulative, label="Others", linestyle="--", color="gray")
    else:
        for idx in range(len(per_user_total)):
            plt.plot(rb_axis, np.cumsum(flat[idx]), label=f"User {idx}")
    plt.xlabel("RB Index Across Slots")
    plt.ylabel("Cumulative Allocated RBs")
    plt.title(f"Cumulative Allocation - {scheduler_name.upper()}")
    plt.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_fairness_plot(agg: List[Dict[str, float]], path: Path, metric: str, title: str, ylabel: str) -> None:
    if not agg:
        return
    user_values = sorted({int(row["users"]) for row in agg})
    if len(user_values) < 2:
        return
    plt.figure(figsize=(7, 4))
    for sched in ALL_SCHEDULERS:
        series = [row for row in agg if row["scheduler"] == sched]
        series = sorted(series, key=lambda r: r["users"])
        if series:
            x = [row["users"] for row in series]
            y = [row[f"mean_{metric}"] for row in series]
            yerr = [row.get(f"std_{metric}", 0.0) for row in series]
            plt.errorbar(x, y, yerr=yerr, marker="o", label=sched.upper())
    plt.xlabel("Users")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_tradeoff_plot(agg: List[Dict[str, float]], raw_results: List[ScenarioResult], path: Path) -> None:
    if not agg:
        return
    plt.figure(figsize=(6, 4))
    for res in raw_results:
        plt.scatter(res.fairness, res.total_throughput, color="#cccccc", alpha=0.3, s=15)
    for row in agg:
        plt.scatter(row["mean_fairness"], row["mean_throughput"], label=f"{row['scheduler'].upper()} ({row['users']}u)")
    plt.xlabel("Fairness")
    plt.ylabel("Throughput")
    plt.title("Fairness/Throughput Trade-off (mean ± std)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_edge_plot(agg: List[Dict[str, float]], path: Path) -> None:
    edge_data = [row for row in agg if row.get("mean_edge_share") is not None]
    if not edge_data:
        return
    labels = [row["scheduler"].upper() for row in edge_data]
    values = [row.get("mean_edge_share", 0.0) for row in edge_data]
    errs = [row.get("std_edge_share", 0.0) for row in edge_data]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, values, yerr=errs, color="#55A868", capsize=4)
    plt.ylabel("Edge User Share")
    plt.title("Edge User Share Comparison")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_delay_plot(agg: List[Dict[str, float]], path: Path) -> None:
    qos_data = [row for row in agg if row.get("mean_p95_delay_voip") is not None]
    if not qos_data:
        return
    labels = [row["scheduler"].upper() for row in qos_data]
    classes = ["voip", "video", "be"]
    x = np.arange(len(labels))
    width = 0.25
    plt.figure(figsize=(8, 4))
    for idx, cls in enumerate(classes):
        means = [row.get(f"mean_p95_delay_{cls}", 0.0) for row in qos_data]
        stds = [row.get(f"std_p95_delay_{cls}", 0.0) for row in qos_data]
        plt.bar(x + idx * width, means, width=width, yerr=stds, label=cls.upper(), capsize=4)
    plt.xticks(x + width, labels)
    plt.ylabel("P95 HOL Delay")
    plt.title("QoS Delay (mean ± std)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def write_metrics_runs(path: Path, results: List[ScenarioResult]) -> None:
    headers = [
        "run_id",
        "scenario",
        "scheduler",
        "users",
        "rbs",
        "slots",
        "seed",
        "fairness",
        "utilization",
        "throughput",
        "starvation",
        "edge_share",
        "edge_throughput",
        "delay_avg",
        "delay_p95",
        "drop_rate",
    ]
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for res in results:
            writer.writerow([
                res.run_id,
                res.scenario,
                res.scheduler,
                res.users,
                res.rbs,
                res.slots,
                res.seed,
                round(res.fairness, 6),
                round(res.utilization, 6),
                round(res.total_throughput, 6),
                round(res.starvation, 6),
                res.edge_share if res.edge_share is not None else "",
                res.edge_throughput if res.edge_throughput is not None else "",
                res.delay_avg if res.delay_avg is not None else "",
                res.delay_p95 if res.delay_p95 is not None else "",
                res.drop_rate if res.drop_rate is not None else "",
            ])


def write_metrics_per_user(base: Path, results: List[ScenarioResult]) -> None:
    for res in results:
        path = base / f"metrics_per_user_{res.run_id}_{res.scheduler}.csv"
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["user_id", "total_rbs", "share_pct", "group", "traffic_class"])
            groups = res.group_info or [""] * len(res.per_user_total)
            classes = res.traffic_classes or [""] * len(res.per_user_total)
            for idx, (total, share, grp, cls) in enumerate(zip(res.per_user_total, res.per_user_share, groups, classes)):
                writer.writerow([idx, int(total), round(float(share), 4), grp, cls])


def aggregate_results(results: List[ScenarioResult]) -> List[Dict[str, float]]:
    grouped: Dict[Tuple, List[ScenarioResult]] = {}
    for res in results:
        key = (res.scenario, res.scheduler, res.users, res.rbs, res.slots, res.label)
        grouped.setdefault(key, []).append(res)

    agg_rows: List[Dict[str, float]] = []
    for key, rows in grouped.items():
        fairness_vals = np.array([r.fairness for r in rows])
        thr_vals = np.array([r.total_throughput for r in rows])
        starv_vals = np.array([r.starvation for r in rows])
        edge_vals = np.array([r.edge_share for r in rows if r.edge_share is not None])
        row_dict: Dict[str, float] = {
            "scenario": key[0],
            "scheduler": key[1],
            "users": key[2],
            "rbs": key[3],
            "slots": key[4],
            "label": key[5],
            "count": len(rows),
            "mean_fairness": float(np.mean(fairness_vals)),
            "std_fairness": float(np.std(fairness_vals)),
            "mean_throughput": float(np.mean(thr_vals)),
            "std_throughput": float(np.std(thr_vals)),
            "mean_starvation": float(np.mean(starv_vals)),
            "std_starvation": float(np.std(starv_vals)),
            "mean_edge_share": float(np.mean(edge_vals)) if edge_vals.size else None,
            "std_edge_share": float(np.std(edge_vals)) if edge_vals.size else None,
        }
        if rows[0].delay_p95_per_class:
            for cls in ["voip", "video", "be"]:
                values = [r.delay_p95_per_class.get(cls, 0.0) if r.delay_p95_per_class else 0.0 for r in rows]
                row_dict[f"mean_p95_delay_{cls}"] = float(np.mean(values))
                row_dict[f"std_p95_delay_{cls}"] = float(np.std(values))
        agg_rows.append(row_dict)
    return agg_rows


def write_metrics_agg(path: Path, agg_rows: List[Dict[str, float]]) -> None:
    headers = [
        "scenario",
        "scheduler",
        "users",
        "rbs",
        "slots",
        "label",
        "count",
        "mean_throughput",
        "std_throughput",
        "mean_fairness",
        "std_fairness",
        "mean_starvation",
        "std_starvation",
        "mean_edge_share",
        "std_edge_share",
        "mean_p95_delay_voip",
        "std_p95_delay_voip",
        "mean_p95_delay_video",
        "std_p95_delay_video",
        "mean_p95_delay_be",
        "std_p95_delay_be",
    ]
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in agg_rows:
            writer.writerow([row.get(h, "") for h in headers])


def write_summary(outdir: Path, agg_rows: List[Dict[str, float]], seeds: List[int]) -> None:
    lines = ["# Simulation Summary", ""]
    lines.append(f"Seeds: {', '.join(str(s) for s in seeds)}")
    lines.append("")
    lines.append("## Fairness (mean ± std)")
    for row in agg_rows:
        lines.append(
            f"- {row['scenario']} / {row['scheduler'].upper()} / {row['users']}u: "
            f"{row['mean_fairness']:.4f} ± {row['std_fairness']:.4f}"
        )
    lines.append("")
    lines.append("## Takeaways")
    lines.append("- MT boosts throughput but loses fairness; PF/EXP_PF stay steadier across loads.")
    lines.append("- WRR/EXP_PF aid disadvantaged users (edge or delay-sensitive).")
    lines.append("- Use --pack to collect aggregated figures (mean ± std).")
    (outdir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_presentation_pack(source_dir: Path, outdir: Path, seeds: List[int]) -> None:
    pack_dir = outdir / "presentation_pack"
    pack_dir.mkdir(parents=True, exist_ok=True)
    figures = [
        ("fairness_vs_users.png", "01_fairness_vs_users.png", "Fairness vs users (mean ± std across seeds)."),
        ("throughput_vs_users.png", "02_throughput_vs_users.png", "Throughput vs users with uncertainty bands."),
        ("tradeoff_fairness_throughput.png", "03_tradeoff.png", "Fairness/throughput trade-off, averaged across seeds."),
        ("edge_share_comparison.png", "04_edge_share.png", "Edge-user share with error bars."),
        ("delay_p95.png", "05_qos_delay.png", "QoS P95 delay by class (mean ± std).")
    ]
    notes = [f"Seeds aggregated: {len(seeds)} ({', '.join(map(str, seeds))})", ""]
    for src, dest, line in figures:
        src_path = source_dir / src
        if src_path.exists():
            dest_path = pack_dir / dest
            dest_path.write_bytes(src_path.read_bytes())
            notes.append(f"- {dest}: {line}")
    if notes:
        (pack_dir / "speaker_notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_case(case: RunCase, rng: np.random.Generator, outdir: Path) -> List[ScenarioResult]:
    channel = generate_channel(case, rng)
    traffic_classes = assign_traffic_classes(case) if case.scenario in {"qos", "bursty", "hetero_channel"} else None
    results: List[ScenarioResult] = []
    for scheduler in case.schedulers:
        scheduler_fn = get_scheduler(scheduler)
        if scheduler == "rr":
            alloc = rr_schedule(case.users, case.rbs, case.slots)
        elif scheduler == "pf":
            alloc = pf_schedule(case.users, case.rbs, case.slots, channel)
        elif scheduler in {"mt", "max_cqi"}:
            alloc = mt_schedule(case.users, case.rbs, case.slots, channel)
        elif scheduler == "wrr":
            weights = case.weights
            if weights is None and channel.user_groups:
                weights = [2 if g == "edge" else 1 for g in channel.user_groups]
            weights = weights if weights else [1] * case.users
            alloc = wrr_schedule(case.users, case.rbs, case.slots, weights)
        else:
            arrival_rate = 1.3 if case.scenario == "qos" else 1.0
            queue_limit = 60 if case.scenario == "qos" else 100
            alloc = exp_pf_schedule(
                case.users,
                case.rbs,
                case.slots,
                channel,
                rng,
                traffic_classes or ["be"] * case.users,
                arrival_rate=arrival_rate,
                queue_limit=queue_limit,
            )
        result = compute_metrics(case, scheduler, alloc, channel, traffic_classes)
        results.append(result)
        save_bar_chart(result.per_user_total, scheduler, outdir / f"rb_per_user_{result.run_id}_{scheduler}.png")
        save_cumulative_plot(result.allocation, scheduler, outdir / f"cumulative_{result.run_id}_{scheduler}.png")
    save_heatmaps(results, outdir)
    return results


def main() -> None:
    args = parse_args()
    base = PRESETS[args.preset]
    seeds = parse_seed_list(args.seeds, args.seed if args.seed is not None else base.seed)
    outdir = prepare_outdir(args.outdir)

    aggregated_results: List[ScenarioResult] = []
    for seed in seeds:
        seed_dir = outdir / ("runs" if len(seeds) > 1 else "") / (f"seed_{seed}" if len(seeds) > 1 else "")
        if seed_dir == outdir:
            ensure_dir(seed_dir)
        else:
            ensure_dir(seed_dir)
        cases = build_cases(args, seed)
        seed_results: List[ScenarioResult] = []
        rng = np.random.default_rng(seed)
        for case in cases:
            case_results = run_case(case, rng, seed_dir)
            seed_results.extend(case_results)
        write_metrics_runs(seed_dir / "metrics_runs.csv", seed_results)
        write_metrics_per_user(seed_dir, seed_results)
        aggregated_results.extend(seed_results)

    aggregate_dir = outdir / ("aggregate" if len(seeds) > 1 else "")
    ensure_dir(aggregate_dir)
    agg_rows = aggregate_results(aggregated_results)
    write_metrics_runs(aggregate_dir / "metrics_runs.csv", aggregated_results)
    write_metrics_agg(aggregate_dir / "metrics_agg.csv", agg_rows)

    load_rows = [row for row in agg_rows if row["scenario"] == "load"]
    save_fairness_plot(load_rows, aggregate_dir / "fairness_vs_users.png", "fairness", "Fairness vs Users", "Jain Fairness")
    save_fairness_plot(
        load_rows,
        aggregate_dir / "throughput_vs_users.png",
        "throughput",
        "Throughput vs Users",
        "Throughput (a.u.)",
    )
    save_tradeoff_plot(agg_rows, aggregated_results, aggregate_dir / "tradeoff_fairness_throughput.png")
    save_edge_plot([row for row in agg_rows if row["scenario"] == "hetero_channel"], aggregate_dir / "edge_share_comparison.png")
    save_delay_plot([row for row in agg_rows if row["scenario"] == "qos"], aggregate_dir / "delay_p95.png")

    write_summary(aggregate_dir, agg_rows, seeds)
    if args.pack:
        source_for_pack = aggregate_dir if aggregate_dir.exists() else outdir
        build_presentation_pack(source_for_pack, outdir, seeds)

    print(f"Completed {len(aggregated_results)} run(s) across {len(seeds)} seed(s). Output saved to {outdir}")


if __name__ == "__main__":
    main()
