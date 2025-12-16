# LTE/5G Scheduling Demo (Classroom Edition)

## Windows Quick Start
1. Open **Command Prompt** or **PowerShell** in this folder.
2. Create and activate a virtual environment:
   - `python -m venv .venv`
   - `\.venv\\Scripts\\activate` (Command Prompt) or `source .venv/Scripts/activate` (PowerShell)
3. Install dependencies:
   - `pip install -r requirements.txt`
4. Run a preset demo:
   - `python run_sim.py --preset classroom`

> Tested with Python 3.10+; relies only on NumPy, Matplotlib, and the standard library.

## 12-Minute Classroom Flow (live demo)
1. **Baseline fairness vs throughput (RR vs PF):** `python run_sim.py --scheduler rr pf --preset classroom`
2. **Max Throughput collapse:** `python run_sim.py --scheduler mt --scenario load --users 20 --rbs 12 --slots 25`
3. **WRR edge boost:** `python run_sim.py --scheduler wrr --scenario hetero_channel --weights 2,2,3,3,3,1,1,1,1,1`
4. **QoS delay rescue (EXP/PF):** `python run_sim.py --scheduler exp_pf --scenario qos --users 12 --slots 40 --pack`

Each command creates a timestamped folder under `outputs/` containing heatmaps (first runs only), per-user bar charts, cumulative plots, fairness/throughput trade-off figures, metrics CSVs, and a numbered `presentation_pack/` with speaker notes when `--pack` is used.

## Troubleshooting (Windows)
- **Matplotlib backend warnings:** The runner forces a non-GUI backend; rerun the command after activation if you see backend errors.
- **Long paths:** If you hit Windows path limits, move the repo to a short path like `C:\demo`.
- **Activation issues:** Ensure you run the terminal as an administrator if `activate` scripts are blocked; or run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in PowerShell.
- **Missing Python:** Install Python from https://www.python.org/downloads/ and select "Add Python to PATH" during setup.
