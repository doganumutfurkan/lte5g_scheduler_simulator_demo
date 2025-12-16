# Repo Inspection Notes
- Entrypoint: `simulator.py` can be run directly; under `if __name__ == "__main__"` it runs both schedulers and plots allocations.
- Scheduling logic: `Scheduler` class in `simulator.py` implements `round_robin` and `proportional_fair` returning allocation matrices shaped `(num_users, num_rb)` with binary assignments.
- Results storage: allocations are simple NumPy arrays (per-run), not persisted; `Scheduler.user_rates` tracks proportional fair priorities.
- Plotting: `plot_allocation` in `simulator.py` generates heatmaps using Matplotlib, saving via `utils.save_plot` (which ensures directories exist) and displaying with `plt.show()`.
- Plan: keep scheduling methods unchanged, add a CLI wrapper to run repeated slots, compute metrics, and emit presentation-ready plots/summary into timestamped `outputs/` folders without adding dependencies beyond NumPy/Matplotlib/stdlib.
