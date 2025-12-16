"""Command-line runner for LTE/5G scheduling demo with multiple scenarios."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

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


PRESETS = {
    "default": RunCase(
        scenario="hetero_channel",
        users=12,
        rbs=18,
        slots=40,
        seed=42,
        schedulers=ALL_SCHEDULERS,
    ),
    "classroom": RunCase(
        scenario="hetero_channel",
        users=10,
        rbs=14,
        slots=30,
        seed=42,
        schedulers=ALL_SCHEDULERS,
        label="classroom",
    ),
    "stress": RunCase(
        scenario="load",
        users=40,
        rbs=24,
        slots=80,
        seed=7,
        schedulers=ALL_SCHEDULERS,
    ),
}


SCENARIO_LIST = ["load", "hetero_channel", "qos", "bursty"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LTE/5G scheduling simulations.")
    parser.add_argument("--scheduler", nargs="*", choices=ALL_SCHEDULERS + ["all"], default=None,
                        help="Scheduler(s) to run. Default from preset.")
    parser.add_argument("--users", type=int, help="Number of users (for single-run override).", default=None)
    parser.add_argument("--rbs", type=int, help="Resource blocks per slot.", default=None)
    parser.add_argument("--slots", type=int, help="Number of slots to simulate.", default=None)
    parser.add_argument("--seed", type=int, help="Random seed.", default=None)
    parser.add_argument("--outdir", type=str, default=None, help="Output directory base.")
    parser.add_argument("--preset", choices=list(PRESETS.keys()), default="classroom", help="Preset to use.")
    parser.add_argument("--scenario", choices=SCENARIO_LIST, default=None, help="Scenario to run.")
    parser.add_argument("--weights", type=str, default=None, help="Weights for WRR (comma separated or group=weight form).")
    parser.add_argument("--pack", action="store_true", help="Build presentation pack with key figures.")
    return parser.parse_args()


def validate_positive(name: str, value: Optional[int]) -> int:
    if value is None:
        raise ValueError(f"{name} is required.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return int(value)


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
        return [mapping.get(g, 1) for g in group_weights.keys()]  # placeholder unused path
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


def build_cases(args: argparse.Namespace) -> List[RunCase]:
    base = PRESETS[args.preset]
    schedulers = base.schedulers
    if args.scheduler:
        if "all" in args.scheduler:
            schedulers = ALL_SCHEDULERS
        else:
            schedulers = args.scheduler
    users = args.users if args.users is not None else base.users
    rbs = args.rbs if args.rbs is not None else base.rbs
    slots = args.slots if args.slots is not None else base.slots
    seed = args.seed if args.seed is not None else base.seed
    scenario = args.scenario if args.scenario else base.scenario
    label = base.label
    return build_scenario_cases(scenario, users, rbs, slots, seed, schedulers, args.weights, label)


@dataclass
class ScenarioResult:
    run_id: str
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
    drop_rate: Optional[float] = None


def build_scenario_cases(scenario: str, users: int, rbs: int, slots: int, seed: int, schedulers: Sequence[str], weight_arg: Optional[str], label: Optional[str]) -> List[RunCase]:
    if scenario == "load":
        user_counts = [5, 10, 20, users]
        cases = [RunCase(scenario="load", users=u, rbs=rbs, slots=slots, seed=seed, schedulers=schedulers, label=f"load_{u}") for u in user_counts]
    elif scenario == "hetero_channel":
        cases = [RunCase(scenario="hetero_channel", users=users, rbs=rbs, slots=slots, seed=seed, schedulers=schedulers, label=label or "hetero")]
    elif scenario == "qos":
        cases = [RunCase(scenario="qos", users=users, rbs=rbs, slots=slots, seed=seed, schedulers=schedulers, label=label or "qos")]
    else:
        cases = [RunCase(scenario="bursty", users=users, rbs=rbs, slots=slots, seed=seed, schedulers=schedulers, label=label or "bursty")]
    for case in cases:
        if weight_arg:
            case.weights = parse_weights(weight_arg, case.users)
    return cases


def generate_channel(case: RunCase, rng: np.random.Generator) -> ChannelState:
    if case.scenario == "hetero_channel":
        groups = []
        rates = np.zeros((case.users, case.slots, case.rbs))
        for u in range(case.users):
            if u < case.users * 0.3:
                mean, group = 6, "near"
            elif u < case.users * 0.7:
                mean, group = 3, "mid"
            else:
                mean, group = 1.5, "edge"
            groups.append(group)
            rates[u] = rng.normal(loc=mean, scale=0.5, size=(case.slots, case.rbs)).clip(min=0.2)
        return ChannelState(rates=rates, user_groups=groups)
    if case.scenario == "bursty":
        rates = rng.lognormal(mean=1.0, sigma=0.8, size=(case.users, case.slots, case.rbs))
        return ChannelState(rates=rates)
    rates = rng.lognormal(mean=1.2, sigma=0.5, size=(case.users, case.slots, case.rbs))
    return ChannelState(rates=rates)


def assign_traffic_classes(case: RunCase) -> List[str]:
    classes = []
    for u in range(case.users):
        if u % 3 == 0:
            classes.append("voip")
        elif u % 3 == 1:
            classes.append("video")
        else:
            classes.append("be")
    return classes


def compute_metrics(run_id: str, scheduler: str, alloc: AllocationResult, channel: ChannelState, traffic_classes: Optional[List[str]]) -> ScenarioResult:
    allocation = alloc.allocation
    throughput = alloc.per_slot_throughput
    flat_allocation = allocation.reshape(allocation.shape[0], -1)
    per_user_total = flat_allocation.sum(axis=1)
    total_rbs = flat_allocation.shape[1]
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
    if "hol_history" in alloc.metadata and traffic_classes:
        hol_hist = alloc.metadata["hol_history"]
        delays = hol_hist.flatten()
        delay_avg = float(np.mean(delays)) if delays.size else 0.0
        delay_p95 = float(np.percentile(delays, 95)) if delays.size else 0.0
    drop_rate = None
    return ScenarioResult(
        run_id=run_id,
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
        drop_rate=drop_rate,
    )


def save_heatmaps(results: List[ScenarioResult], outdir: Path) -> None:
    for idx, res in enumerate(results):
        if idx > 1:
            continue
        flat = res.allocation.reshape(res.allocation.shape[0], -1)
        plot_allocation(flat, title=f"{res.scheduler.upper()} Allocation", save_path=str(outdir / f"{res.run_id}_{res.scheduler}_heatmap.png"), show=False)


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


def save_fairness_plot(metrics: List[ScenarioResult], path: Path) -> None:
    plt.figure(figsize=(7, 4))
    by_sched: Dict[str, List[float]] = {}
    by_users: List[int] = []
    for res in metrics:
        by_sched.setdefault(res.scheduler, []).append(res.fairness)
        by_users.append(int(res.allocation.shape[1]))
    users_axis = sorted(set(by_users))
    for sched in ALL_SCHEDULERS:
        vals = [res.fairness for res in metrics if res.scheduler == sched]
        x = [res.allocation.shape[0] for res in metrics if res.scheduler == sched]
        if vals:
            plt.plot(x, vals, marker="o", label=sched.upper())
    plt.xlabel("Users")
    plt.ylabel("Jain Fairness")
    plt.title("Fairness vs Users")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_throughput_plot(metrics: List[ScenarioResult], path: Path) -> None:
    plt.figure(figsize=(7, 4))
    for sched in ALL_SCHEDULERS:
        vals = [res.total_throughput for res in metrics if res.scheduler == sched]
        x = [res.allocation.shape[0] for res in metrics if res.scheduler == sched]
        if vals:
            plt.plot(x, vals, marker="s", label=sched.upper())
    plt.xlabel("Users")
    plt.ylabel("Throughput (a.u.)")
    plt.title("Throughput vs Users")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_tradeoff_plot(metrics: List[ScenarioResult], path: Path) -> None:
    plt.figure(figsize=(6, 4))
    for res in metrics:
        plt.scatter(res.fairness, res.total_throughput, label=res.scheduler.upper())
    plt.xlabel("Fairness")
    plt.ylabel("Throughput")
    plt.title("Fairness/Throughput Trade-off")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_edge_plot(metrics: List[ScenarioResult], path: Path) -> None:
    edge_data = [res for res in metrics if res.edge_share is not None]
    if not edge_data:
        return
    labels = [res.scheduler.upper() for res in edge_data]
    values = [res.edge_share for res in edge_data]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, values, color="#55A868")
    plt.ylabel("Edge User Share")
    plt.title("Edge User Share Comparison")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_delay_plot(metrics: List[ScenarioResult], path: Path) -> None:
    qos_data = [res for res in metrics if res.delay_p95 is not None]
    if not qos_data:
        return
    labels = [res.scheduler.upper() for res in qos_data]
    values = [res.delay_p95 for res in qos_data]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, values, color="#C44E52")
    plt.ylabel("P95 HOL Delay")
    plt.title("QoS Delay (p95)")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def write_metrics_runs(path: Path, results: List[ScenarioResult], cases: List[RunCase]) -> None:
    headers = [
        "run_id",
        "scenario",
        "scheduler",
        "users",
        "rbs",
        "slots",
        "fairness",
        "utilization",
        "throughput",
        "starvation",
        "edge_share",
        "edge_throughput",
        "delay_avg",
        "delay_p95",
    ]
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for res in results:
            writer.writerow([
                res.run_id,
                res.run_id.split("__")[0],
                res.scheduler,
                res.allocation.shape[0],
                res.allocation.shape[2],
                res.allocation.shape[1],
                round(res.fairness, 6),
                round(res.utilization, 6),
                round(res.total_throughput, 6),
                round(res.starvation, 6),
                res.edge_share if res.edge_share is not None else "",
                res.edge_throughput if res.edge_throughput is not None else "",
                res.delay_avg if res.delay_avg is not None else "",
                res.delay_p95 if res.delay_p95 is not None else "",
            ])


def write_metrics_per_user(base: Path, results: List[ScenarioResult]) -> None:
    for res in results:
        path = base / f"metrics_per_user_{res.scheduler}.csv"
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["user_id", "total_rbs", "share_pct", "group", "traffic_class"])
            groups = res.group_info or [""] * len(res.per_user_total)
            classes = res.traffic_classes or [""] * len(res.per_user_total)
            for idx, (total, share, grp, cls) in enumerate(zip(res.per_user_total, res.per_user_share, groups, classes)):
                writer.writerow([idx, int(total), round(float(share), 4), grp, cls])


def write_summary(outdir: Path, cases: List[RunCase], results: List[ScenarioResult]) -> None:
    lines = ["# Simulation Summary", "", "## Parameters"]
    for case in cases:
        lines.append(f"- Scenario: {case.scenario} | Users={case.users}, RBs={case.rbs}, Slots={case.slots}, Seed={case.seed}")
    lines.append("")
    lines.append("## Fairness (Jain)")
    for res in results:
        lines.append(f"- {res.run_id} / {res.scheduler.upper()}: {res.fairness:.4f}")
    lines.append("")
    lines.append("## Takeaways")
    lines.append("- MT pushes throughput but can reduce fairness; WRR can elevate protected users.")
    lines.append("- EXP/PF balances delay urgency with proportional fairness for QoS traffic.")
    lines.append("- Edge-user share highlights fairness gaps in heterogeneous channels.")
    lines.append("- Use --pack to collect numbered figures for presentation.")
    (outdir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_presentation_pack(outdir: Path) -> None:
    pack_dir = outdir / "presentation_pack"
    if not pack_dir.exists():
        return
    figures = [
        ("fairness_vs_users.png", "01_fairness_vs_users.png", "Fairness drops for MT when load increases; PF/EXP_PF stay steadier."),
        ("throughput_vs_users.png", "02_throughput_vs_users.png", "MT leads throughput, while RR/WRR trade throughput for equity."),
        ("tradeoff_fairness_throughput.png", "03_tradeoff.png", "Scatter shows fairness/throughput balance across schedulers."),
        ("edge_share_comparison.png", "04_edge_share.png", "Edge-user share improves with WRR weights."),
        ("delay_p95.png", "05_qos_delay.png", "EXP/PF reduces delay tails for VoIP/Video."),
    ]
    notes = []
    for src, dest, line in figures:
        src_path = outdir / src
        if src_path.exists():
            dest_path = pack_dir / dest
            dest_path.write_bytes(src_path.read_bytes())
            notes.append(f"- {dest}: {line}")
    if notes:
        (pack_dir / "speaker_notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")


def ensure_pack_dir(outdir: Path, pack: bool) -> None:
    if pack:
        (outdir / "presentation_pack").mkdir(parents=True, exist_ok=True)


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
            weights = case.weights if case.weights else [1] * case.users
            alloc = wrr_schedule(case.users, case.rbs, case.slots, weights)
        else:
            alloc = exp_pf_schedule(case.users, case.rbs, case.slots, channel, rng, traffic_classes or ["be"] * case.users)
        run_id = f"{case.scenario}__{case.users}u_{case.rbs}r_{case.slots}s"
        result = compute_metrics(run_id, scheduler, alloc, channel, traffic_classes)
        results.append(result)
    return results


def main() -> None:
    args = parse_args()
    cases = build_cases(args)
    outdir = prepare_outdir(args.outdir)
    ensure_pack_dir(outdir, args.pack)

    all_results: List[ScenarioResult] = []

    for case in cases:
        rng = np.random.default_rng(case.seed)
        case_results = run_case(case, rng, outdir)
        all_results.extend(case_results)
        for res in case_results:
            save_bar_chart(res.per_user_total, res.scheduler, outdir / f"rb_per_user_{res.run_id}_{res.scheduler}.png")
            save_cumulative_plot(res.allocation, res.scheduler, outdir / f"cumulative_{res.run_id}_{res.scheduler}.png")
    save_heatmaps(all_results, outdir)
    write_metrics_runs(outdir / "metrics_runs.csv", all_results, cases)
    write_metrics_per_user(outdir, all_results)
    save_fairness_plot(all_results, outdir / "fairness_vs_users.png")
    save_throughput_plot(all_results, outdir / "throughput_vs_users.png")
    save_tradeoff_plot(all_results, outdir / "tradeoff_fairness_throughput.png")
    save_edge_plot(all_results, outdir / "edge_share_comparison.png")
    save_delay_plot(all_results, outdir / "delay_p95.png")
    write_summary(outdir, cases, all_results)
    build_presentation_pack(outdir)
    print(f"Completed {len(cases)} scenario(s). Output saved to {outdir}")


if __name__ == "__main__":
    main()
