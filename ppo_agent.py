"""EcoPrice PPO Discount Agent.

Trains a PPO (Proximal Policy Optimization) agent to decide optimal
discount percentages for perishable products to minimize waste while
maximizing revenue.

Uses Stable-Baselines3 PPO with the custom RetailPricingEnv.

Usage:
    # Train the PPO agent:
    python -m models.ppo_agent

    # Train with custom steps:
    python -m models.ppo_agent --timesteps 200000

    # From other modules:
    from models.ppo_agent import load_agent, get_discount_suggestion
    agent = load_agent()
    suggestion = get_discount_suggestion(agent, product_state)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models" / "saved"
AGENT_PATH = MODEL_DIR / "ppo_discount_agent"
TRAINING_LOG = MODEL_DIR / "ppo_training_log.json"
HISTORY_CSV = MODEL_DIR / "ppo_history.csv"
TENSORBOARD_DIR = BASE_DIR / "runs" / "ppo"


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_agent(total_timesteps=100_000, seed=42, tensorboard=False):
    """Train PPO agent on the RetailPricingEnv.

    Logging (no app.py changes):
    - terminal: Avg Reward/Discount/Waste every 10k steps
    - CSV: models/saved/ppo_history.csv (step, avg_reward, avg_discount, avg_waste)
    - TensorBoard (optional): runs/ppo/ — use --tensorboard flag,
      view with `tensorboard --logdir runs/ppo`
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from models.environment import RetailPricingEnv

    print("=" * 60)
    print("  EcoPrice -- PPO Discount Agent Training")
    print("=" * 60)

    # Create environment
    env = RetailPricingEnv(
        lambda_spoilage=5.0,
        gamma_discount=2.0,
        max_discount=0.30,
        price_elasticity=2.5,
    )

    print(f"\nEnvironment: RetailPricingEnv")
    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space}")
    print(f"  Lambda (spoilage): {env.lambda_spoilage}")
    print(f"  Gamma (discount): {env.gamma_discount}")
    print(f"  Max discount: {env.max_discount * 100:.0f}%")
    print(f"  Price elasticity: {env.price_elasticity}")

    # PPO hyperparameters
    ppo_params = {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,       # entropy bonus for exploration
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "verbose": 1,
        "seed": seed,
    }

    print(f"\nPPO Hyperparameters:")
    for k, v in ppo_params.items():
        if k != "verbose":
            print(f"  {k}: {v}")

    print(f"\nTraining for {total_timesteps:,} timesteps...")
    print("-" * 60)

    # Custom callback for logging
    class RewardLogger(BaseCallback):
        def __init__(self):
            super().__init__()
            self.episode_rewards = []
            self.episode_discounts = []
            self.episode_waste = []
            self.history = []  # (timestep, avg_reward, avg_discount, avg_waste)
            self.log_interval = 10000

        def _on_step(self):
            # Collect info from completed episodes
            for info in self.locals.get("infos", []):
                if "discount" in info:
                    self.episode_discounts.append(info["discount"])
                    self.episode_waste.append(info["units_wasted"])
                    self.episode_rewards.append(info.get("raw_reward", 0))

            if self.num_timesteps % self.log_interval == 0 and self.episode_rewards:
                recent = self.episode_rewards[-1000:]
                recent_disc = self.episode_discounts[-1000:]
                recent_waste = self.episode_waste[-1000:]
                row = (self.num_timesteps, float(np.mean(recent)),
                       float(np.mean(recent_disc)), float(np.mean(recent_waste)))
                self.history.append(row)
                print(
                    f"  Step {self.num_timesteps:>8,} | "
                    f"Avg Reward: {np.mean(recent):>10.1f} | "
                    f"Avg Discount: {np.mean(recent_disc):>5.1f}% | "
                    f"Avg Waste: {np.mean(recent_waste):>5.1f} units"
                )
            return True

    logger = RewardLogger()

    # Train (TensorBoard optional, needs `pip install tensorboard`)
    tb_log = str(TENSORBOARD_DIR) if tensorboard else None
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs={
            "net_arch": dict(pi=[128, 128], vf=[128, 128]),  # separate actor-critic networks
        },
        tensorboard_log=tb_log,
        **ppo_params,
    )
    model.learn(total_timesteps=total_timesteps, callback=logger)

    # Save
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(AGENT_PATH))

    # Save training log
    training_log = {
        "total_timesteps": total_timesteps,
        "final_avg_reward": round(float(np.mean(logger.episode_rewards[-5000:])), 2) if logger.episode_rewards else 0,
        "final_avg_discount": round(float(np.mean(logger.episode_discounts[-5000:])), 1) if logger.episode_discounts else 0,
        "final_avg_waste": round(float(np.mean(logger.episode_waste[-5000:])), 2) if logger.episode_waste else 0,
        "total_episodes": len(logger.episode_rewards),
        "lambda_spoilage": env.lambda_spoilage,
        "gamma_discount": env.gamma_discount,
        "max_discount": env.max_discount,
        "price_elasticity": env.price_elasticity,
    }
    with open(TRAINING_LOG, "w") as f:
        json.dump(training_log, f, indent=2)

    # Save per-interval history for plotting (no app.py changes)
    import csv
    with open(HISTORY_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestep", "avg_reward", "avg_discount", "avg_waste"])
        w.writerows(logger.history)
    print(f"  History CSV: {HISTORY_CSV} ({len(logger.history)} rows)")
    if tb_log:
        print(f"  TensorBoard: tensorboard --logdir {tb_log}")

    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    print(f"  Agent saved to: {AGENT_PATH}")
    print(f"  Total episodes: {len(logger.episode_rewards):,}")
    print(f"  Final avg reward: {training_log['final_avg_reward']:.2f}")
    print(f"  Final avg discount: {training_log['final_avg_discount']:.1f}%")
    print(f"  Final avg waste: {training_log['final_avg_waste']:.2f} units")

    return model


# ---------------------------------------------------------------------------
# Load / Predict
# ---------------------------------------------------------------------------
def load_agent(path=None):
    """Load a trained PPO agent from disk."""
    from stable_baselines3 import PPO
    path = path or str(AGENT_PATH)
    if not Path(path + ".zip").exists() and not Path(path).exists():
        raise FileNotFoundError(
            f"No trained agent at {path}. Run `python -m models.ppo_agent` first."
        )
    model = PPO.load(path)
    print(f"PPO agent loaded from {path}")
    return model


def get_discount_suggestion(agent, product_state: dict) -> dict:
    """Get a discount suggestion from the PPO agent for a product.

    Args:
        agent: Trained PPO model
        product_state: Dict with keys:
            stock, days_to_expiry, predicted_demand, price,
            temperature, humidity, is_weekend, is_holiday,
            days_to_festival, rolling_avg_7d

    Returns:
        Dict with: discount_pct, new_price, confidence, reason
    """
    obs = np.array([
        product_state.get("stock", 20),
        product_state.get("days_to_expiry", 3),
        product_state.get("predicted_demand", 5),
        product_state.get("price", 50),
        product_state.get("temperature", 25),
        product_state.get("humidity", 60),
        product_state.get("is_weekend", 0),
        product_state.get("is_holiday", 0),
        product_state.get("days_to_festival", 8),
        product_state.get("rolling_avg_7d", 5),
    ], dtype=np.float32)

    action, _states = agent.predict(obs, deterministic=True)
    discount = float(np.clip(action[0], 0, 0.30))
    discount_pct = round(discount * 100, 1)

    price = product_state.get("price", 50)
    new_price = round(price * (1 - discount), 2)

    # Generate confidence based on how "urgent" the situation is
    days_left = product_state.get("days_to_expiry", 3)
    stock = product_state.get("stock", 20)
    demand = product_state.get("predicted_demand", 5)
    surplus = stock - demand

    if days_left <= 1 and surplus > 5:
        confidence = min(95, 80 + discount_pct)
        reason = f"URGENT: {days_left}d to expiry, {int(surplus)} units surplus"
    elif days_left <= 2 and surplus > 3:
        confidence = min(90, 70 + discount_pct)
        reason = f"Expiring soon ({days_left}d), demand ({demand:.0f}) below stock ({stock:.0f})"
    elif surplus > 10:
        confidence = min(85, 60 + discount_pct)
        reason = f"Overstocked by {int(surplus)} units, discount boosts turnover"
    elif discount_pct < 5:
        confidence = 60
        reason = "Demand matches stock, minimal discount needed"
    else:
        confidence = min(80, 55 + discount_pct)
        reason = f"Moderate surplus ({int(surplus)} units), discount helps clear"

    return {
        "discount_pct": discount_pct,
        "new_price": new_price,
        "original_price": price,
        "confidence": min(int(confidence), 98),
        "reason": reason,
        "days_to_expiry": days_left,
        "stock": int(stock),
        "predicted_demand": round(demand, 1),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_agent(agent, n_episodes=1000):
    """Evaluate trained agent performance."""
    from models.environment import RetailPricingEnv

    env = RetailPricingEnv()
    results = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        action, _ = agent.predict(obs, deterministic=True)
        _, reward, _, _, info = env.step(action)
        results.append(info)

    # Aggregate
    avg_discount = np.mean([r["discount"] for r in results])
    avg_waste = np.mean([r["units_wasted"] for r in results])
    avg_revenue = np.mean([r["revenue"] for r in results])
    avg_reward = np.mean([r["raw_reward"] for r in results])

    # Compare with no-discount baseline
    baseline_results = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        _, reward, _, _, info = env.step(np.array([0.0]))
        baseline_results.append(info)

    base_waste = np.mean([r["units_wasted"] for r in baseline_results])
    base_revenue = np.mean([r["revenue"] for r in baseline_results])
    base_reward = np.mean([r["raw_reward"] for r in baseline_results])

    print("\n" + "=" * 60)
    print("  PPO Agent Evaluation")
    print("=" * 60)
    print(f"  Episodes: {n_episodes:,}")
    print(f"\n  {'Metric':<25s} {'PPO Agent':>12s} {'No Discount':>12s} {'Improvement':>12s}")
    print(f"  {'-'*61}")
    print(f"  {'Avg Reward':<25s} {avg_reward:>12.1f} {base_reward:>12.1f} {(avg_reward-base_reward):>+12.1f}")
    print(f"  {'Avg Revenue':<25s} {avg_revenue:>12.1f} {base_revenue:>12.1f} {(avg_revenue-base_revenue):>+12.1f}")
    print(f"  {'Avg Waste (units)':<25s} {avg_waste:>12.1f} {base_waste:>12.1f} {(avg_waste-base_waste):>+12.1f}")
    print(f"  {'Avg Discount %':<25s} {avg_discount:>11.1f}% {'0.0':>11s}% {'':>12s}")
    waste_reduction = ((base_waste - avg_waste) / (base_waste + 1e-6)) * 100
    print(f"\n  Waste Reduction: {waste_reduction:.1f}%")

    return {
        "ppo_reward": round(float(avg_reward), 2),
        "ppo_revenue": round(float(avg_revenue), 2),
        "ppo_waste": round(float(avg_waste), 2),
        "ppo_discount": round(float(avg_discount), 1),
        "baseline_reward": round(float(base_reward), 2),
        "baseline_revenue": round(float(base_revenue), 2),
        "baseline_waste": round(float(base_waste), 2),
        "waste_reduction_pct": round(float(waste_reduction), 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train EcoPrice PPO Discount Agent")
    parser.add_argument("--timesteps", type=int, default=100_000, help="Training timesteps")
    parser.add_argument("--eval", action="store_true", help="Evaluate only (skip training)")
    parser.add_argument("--tensorboard", action="store_true", help="Log to runs/ppo for TensorBoard")
    args = parser.parse_args()

    if args.eval:
        agent = load_agent()
        evaluate_agent(agent)
    else:
        agent = train_agent(total_timesteps=args.timesteps, tensorboard=args.tensorboard)
        print("\n\nRunning evaluation...")
        evaluate_agent(agent, n_episodes=2000)
        try:
            from models.plot_training import plot_ppo_history
            plot_ppo_history()
        except Exception as e:
            print(f"(plot skipped: {e})")

    print("\nDone!")


if __name__ == "__main__":
    main()
