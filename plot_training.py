"""Plot training learning curves (no app.py changes).

Reads:
  models/saved/ppo_history.csv  (timestep, avg_reward, avg_discount, avg_waste)
  models/saved/xgboost_metrics.json
Writes PNGs to models/saved/plots/ for review.

Usage:
    python -m models.plot_training
"""
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
SAVED = BASE / "models" / "saved"
PLOTS = SAVED / "plots"


def plot_ppo_history(path=None, out=None):
    import pandas as pd
    path = Path(path or SAVED / "ppo_history.csv")
    if not path.exists():
        print(f"No history yet at {path}. Train first: python -m models.ppo_agent")
        return None
    df = pd.read_csv(path)
    PLOTS.mkdir(parents=True, exist_ok=True)
    out = Path(out or PLOTS / "ppo_learning_curve.png")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(df["timestep"], df["avg_reward"], label="Avg reward")
    axes[0].set_ylabel("Reward (Rs)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(df["timestep"], df["avg_discount"], color="orange", label="Avg discount %")
    axes[1].set_ylabel("Discount %")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(df["timestep"], df["avg_waste"], color="red", label="Avg waste (units)")
    axes[2].set_ylabel("Waste")
    axes[2].set_xlabel("Timesteps")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    fig.suptitle("PPO Learning Curve — reward should rise, waste should fall")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved {out} ({len(df)} points)")
    return out


def main():
    p = plot_ppo_history()
    if p:
        print("Done. Open the PNG to see learning.")
        print("TensorBoard (if --tensorboard used): tensorboard --logdir runs/ppo")


if __name__ == "__main__":
    main()
