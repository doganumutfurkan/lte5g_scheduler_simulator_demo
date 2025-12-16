"""Command-line runner for LTE/5G scheduling demo."""
import argparse
import csv
import datetime as dt
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from simulator import Scheduler, plot_allocation


PRESETS = {
    "default": {"users": 8, "rbs": 12, "slots": 30, "seed": 42, "scheduler": "all"},
    "classroom": {"users": 10, "rbs": 15, "slots": 40, "seed": 42, "scheduler": "all"},
    "stress": {"users": 40, "rbs": 25, "slots": 120, "seed": 1, "scheduler": "all"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LTE/5G scheduling simulations.")
    parser.add_argument("--scheduler", choices=["rr", "pf", "all"], default=None,
                        help="Scheduler to run (default is taken from preset).")
    parser.add_argument("--users", type=int, default=None, help="Number of users (UEs).")
    parser.add_argument("--rbs", type=int, default=None, help="Number of resource blocks per slot.")
    parser.add_argument("--slots", type=int, default=None, help="Number of time slots to simulate.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for deterministic runs.")
    parser.add_argument("--outdir", type=str, default=None, help="Optional output directory.")
    parser.add_argument("--preset", choices=list(PRESETS.keys()), default="default",
                        help="Preset configuration to start from.")
    return parser.parse_args()


def validate_positive(name: str, value: int) -> int:
    if value is None:
        raise ValueError(f"{name} is required.")
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def build_config(args: argparse.Namespace) -> Dict[str, int]:
    config = PRESETS[args.preset].copy()
    for field, arg_name in [("users", "users"), ("rbs", "rbs"), ("slots", "slots"), ("seed", "seed"), ("scheduler", "scheduler")]:
        override = getattr(args, arg_name)
        if override is not None:
            config[field] = override
    config["users"] = validate_positive("users", int(config["users"]))
    config["rbs"] = validate_positive("rbs", int(config["rbs"]))
    config["slots"] = validate_positive("slots", int(config["slots"]))
    config["seed"] = int(config["seed"])
    if config.get("scheduler") not in {"rr", "pf", "all"}:
        raise ValueError("scheduler must be one of {rr, pf, all}.")
    return config


def prepare_outdir(outdir_arg: str) -> Path:
    if outdir_arg:
        base = Path(outdir_arg)
    else:
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        base = Path("outputs") / timestamp
    base.mkdir(parents=True, exist_ok=True)
    return base


def run_allocation(scheduler_name: str, num_users: int, num_rb: int, num_slots: int) -> np.ndarray:
    scheduler = Scheduler(num_users=num_users, num_rb=num_rb)
    allocations: List[np.ndarray] = []
    for _ in range(num_slots):
        if scheduler_name == "rr":
            alloc = scheduler.round_robin()
        else:
            alloc = scheduler.proportional_fair()
        allocations.append(alloc)
    if not allocations:
        return np.zeros((num_users, num_rb))
    return np.concatenate(allocations, axis=1) if len(allocations) > 1 else allocations[0]


def compute_metrics(allocation: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
    per_user_total = allocation.sum(axis=1)
    total_rbs = per_user_total.sum()
    if total_rbs == 0:
        share_pct = np.zeros_like(per_user_total)
        fairness = 0.0
        utilization = 0.0
    else:
        share_pct = (per_user_total / total_rbs) * 100
        denominator = len(per_user_total) * np.square(per_user_total).sum()
        fairness = float(np.square(per_user_total.sum()) / denominator) if denominator > 0 else 0.0
        utilization = float(total_rbs / allocation.shape[1]) if allocation.shape[1] > 0 else 0.0
    return per_user_total, share_pct, fairness, utilization


def save_metrics_csv(path: Path, per_user_total: np.ndarray, share_pct: np.ndarray) -> None:
    with path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["user_id", "total_allocated_rbs", "share_pct"])
        for idx, (total, share) in enumerate(zip(per_user_total, share_pct)):
            writer.writerow([idx, int(total), round(float(share), 4)])


def save_heatmap(allocation: np.ndarray, scheduler_name: str, path: Path) -> None:
    title = f"{scheduler_name.upper()} Allocation Heatmap"
    plot_allocation(allocation, title=title, save_path=str(path), show=False)


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
    total_rbs = allocation.shape[1]
    per_user_total = allocation.sum(axis=1)
    rb_axis = np.arange(1, total_rbs + 1)
    plt.figure(figsize=(8, 4))
    if len(per_user_total) > 5:
        top_indices = np.argsort(per_user_total)[-5:][::-1]
        other_indices = [i for i in range(len(per_user_total)) if i not in top_indices]
        for idx in top_indices:
            plt.plot(rb_axis, np.cumsum(allocation[idx]), label=f"User {idx}")
        if other_indices:
            others_cumulative = np.cumsum(allocation[other_indices].sum(axis=0))
            plt.plot(rb_axis, others_cumulative, label="Others", linestyle="--", color="gray")
    else:
        for idx in range(len(per_user_total)):
            plt.plot(rb_axis, np.cumsum(allocation[idx]), label=f"User {idx}")
    plt.xlabel("RB Index Across Slots")
    plt.ylabel("Cumulative Allocated RBs")
    plt.title(f"Cumulative Allocation - {scheduler_name.upper()}")
    plt.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_fairness_comparison(fairness_map: Dict[str, float], path: Path) -> None:
    labels = list(fairness_map.keys())
    values = [fairness_map[label] for label in labels]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, values, color=["#4C72B0", "#55A868"])
    plt.ylim(0, 1.05)
    plt.ylabel("Jain's Fairness Index")
    plt.title("Fairness Comparison")
    for idx, value in enumerate(values):
        plt.text(idx, value + 0.02, f"{value:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def write_summary(outdir: Path, config: Dict[str, int], fairness: Dict[str, float], utilization: Dict[str, float], shares: Dict[str, np.ndarray]) -> None:
    summary_path = outdir / "summary.md"
    schedulers_list = [name.upper() for name in fairness.keys()]
    lines = [
        "# Simulation Summary",
        f"- Output directory: `{outdir}`",
        f"- Schedulers run: {', '.join(schedulers_list)}",
        f"- Users: {config['users']}",
        f"- Resource Blocks per slot: {config['rbs']}",
        f"- Slots: {config['slots']}",
        f"- Seed: {config['seed']}",
        "",
        "## Fairness (Jain's index)",
    ]
    for name, value in fairness.items():
        lines.append(f"- {name.upper()}: {value:.4f}")
    lines.append("")
    lines.append("## RB Utilization")
    for name, value in utilization.items():
        lines.append(f"- {name.upper()}: {value * 100:.2f}% of available RBs")
    lines.append("")
    lines.append("## Interpretation")

    bullet_points = []
    if "rr" in fairness and "pf" in fairness:
        if fairness["pf"] >= fairness["rr"]:
            bullet_points.append("Proportional Fair achieved higher or equal fairness than Round Robin in this scenario.")
        else:
            bullet_points.append("Round Robin achieved higher fairness than Proportional Fair in this scenario.")
    bullet_points.append("All RBs were allocated; utilization reflects how evenly users shared them.")
    for sched, share_vector in shares.items():
        top_user = int(np.argmax(share_vector)) if len(share_vector) > 0 else -1
        top_share = float(share_vector[top_user]) if len(share_vector) > 0 else 0.0
        bullet_points.append(f"Top user share under {sched.upper()}: User {top_user} at {top_share:.2f}% of RBs.")
        if len(bullet_points) >= 4:
            break

    lines.extend([f"- {point}" for point in bullet_points[:4]])

    with summary_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def print_summary(config: Dict[str, int], fairness: Dict[str, float]) -> None:
    print("Simulation complete.")
    print(f"Users={config['users']} | RBs per slot={config['rbs']} | Slots={config['slots']} | Seed={config['seed']}")
    for name, value in fairness.items():
        print(f"Fairness ({name.upper()}): {value:.4f}")


def main() -> None:
    args = parse_args()
    config = build_config(args)
    outdir = prepare_outdir(args.outdir)

    np.random.seed(config["seed"])

    schedulers_to_run = [config["scheduler"]] if config["scheduler"] != "all" else ["rr", "pf"]

    fairness_values: Dict[str, float] = {}
    utilization_values: Dict[str, float] = {}
    share_values: Dict[str, np.ndarray] = {}

    for sched_name in schedulers_to_run:
        np.random.seed(config["seed"])
        allocation = run_allocation(sched_name, config["users"], config["rbs"], config["slots"])
        per_user_total, share_pct, fairness, utilization = compute_metrics(allocation)
        fairness_values[sched_name] = fairness
        utilization_values[sched_name] = utilization
        share_values[sched_name] = share_pct

        save_heatmap(allocation, sched_name, outdir / f"{sched_name}_heatmap.png")
        save_metrics_csv(outdir / f"metrics_{sched_name}.csv", per_user_total, share_pct)
        save_bar_chart(per_user_total, sched_name, outdir / f"rb_per_user_{sched_name}.png")
        save_cumulative_plot(allocation, sched_name, outdir / f"cumulative_rb_{sched_name}.png")

    if len(fairness_values) > 1:
        save_fairness_comparison(fairness_values, outdir / "fairness_comparison.png")

    write_summary(outdir, config, fairness_values, utilization_values, share_values)
    print_summary(config, fairness_values)


if __name__ == "__main__":
    main()
