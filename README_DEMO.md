# LTE/5G Scheduling Demo (Classroom Edition)

## Windows Quick Start
1. Open **Command Prompt** or **PowerShell** in this folder.
2. Create and activate a virtual environment:
   - `python -m venv .venv`
   - `\.venv\\Scripts\\activate` (Command Prompt) or `source .venv/Scripts/activate` (PowerShell)
3. Install dependencies:
   - `pip install -r requirements.txt`
4. Run a preset demo:
   - `python run_sim.py --preset classroom --pack`

> Tested with Python 3.10+; relies only on NumPy, Matplotlib, and the standard library.

## 12-Minute Classroom Flow (live demo)
1. **Load sweep (RR vs PF vs MT, with uncertainty):** `python run_sim.py --scenario load --scheduler rr pf mt --seeds 1-3 --pack`
2. **Heterogeneous channels (all 5 schedulers):** `python run_sim.py --scenario hetero_channel --scheduler all --seeds 1,2,3 --near_mean 3.0 --mid_mean 1.5 --edge_mean 0.6 --pack`
3. **QoS delay stress (EXP/PF focus):** `python run_sim.py --scenario qos --scheduler rr pf exp_pf --seeds 1-3 --pack`
4. **Fast classroom preset:** `python run_sim.py --preset classroom_fast --pack`

Each command creates a timestamped folder under `outputs/<timestamp>/`. Multi-seed runs store raw metrics under `runs/seed_<N>/`, aggregate plots/tables under `aggregate/`, and (when `--pack` is used) a numbered `presentation_pack/` with aggregated figures and notes.

## Troubleshooting (Windows)
- **Matplotlib backend warnings:** The runner forces a non-GUI backend; rerun the command after activation if you see backend errors.
- **Long paths:** If you hit Windows path limits, move the repo to a short path like `C:\demo`.
- **Activation issues:** Ensure you run the terminal as an administrator if `activate` scripts are blocked; or run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in PowerShell.
- **Missing Python:** Install Python from https://www.python.org/downloads/ and select "Add Python to PATH" during setup.
