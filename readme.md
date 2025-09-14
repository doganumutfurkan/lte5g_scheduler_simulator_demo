# LTE/5G Resource Scheduling Simulator

A Python-based simulator for **LTE/5G radio resource scheduling**, demonstrating how resource blocks (RBs) are allocated to multiple users using different scheduling algorithms. The project includes visualizations to help understand network behavior and fairness.

## 🚀 Features

- **Round-Robin (RR) Scheduling**: Allocates resource blocks sequentially to all users.
- **Proportional Fair (PF) Scheduling**: Balances efficiency and fairness by prioritizing users who have received fewer resources.
- **Visualization**: Heatmaps showing which user gets which resource block.
- **Extensible**: Add new scheduling algorithms or change network parameters easily.
- **Cross-Platform**: Runs on any system with Python, NumPy, and Matplotlib.

---

## 📂 Project Structure

```bash
lte5g-scheduler-simulator/
│── simulator.py        # Main simulator
│── utils.py            # Helper functions for plotting/saving
│── data/               # Save simulation data
│── plots/              # Automatically saved plots
│── README.md
│── requirements.txt    # Python dependencies
```

## 🛠️ Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/lte5g-scheduler-simulator.git
cd lte5g-scheduler-simulator
```

2. Install dependencies::

```bash
pip install -r requirements.txt
```

## 🎮 Usage

Run the simulator:

```bash
python simulator.py
```

## 📌 Future Improvements

- Add **additional scheduling algorithms** (Max C/I, Weighted Fair, etc.)
- Introduce **dynamic channel conditions**
- Simulate **multiple cells and interference**
- Export simulation results to **CSV or Excel** for further analysis
- Create a **GUI version** for interactive simulation
