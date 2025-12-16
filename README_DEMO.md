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

## 10-Minute Classroom Flow (live demo)
1. **Baseline (default preset):** `python run_sim.py`
2. **Change load:** `python run_sim.py --scheduler all --users 12 --rbs 25 --slots 60 --seed 42`
3. **Stress view:** `python run_sim.py --preset stress`

Each command creates a timestamped folder under `outputs/` containing heatmaps, per-user bar charts, cumulative plots, fairness comparison, metrics CSVs, and `summary.md` with talking points.

## Troubleshooting (Windows)
- **Matplotlib backend warnings:** The runner forces a non-GUI backend; rerun the command after activation if you see backend errors.
- **Long paths:** If you hit Windows path limits, move the repo to a short path like `C:\demo`.
- **Activation issues:** Ensure you run the terminal as an administrator if `activate` scripts are blocked; or run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in PowerShell.
- **Missing Python:** Install Python from https://www.python.org/downloads/ and select "Add Python to PATH" during setup.
