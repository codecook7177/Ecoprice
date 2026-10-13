"""EcoPrice Retail Pricing Environment (Gymnasium).

A custom gym environment that simulates a perishable product on the shelf.
The PPO agent observes the product state and decides what discount to apply.

State (observation):
    [stock, days_to_expiry, predicted_demand, price, temperature,
     humidity, is_weekend, is_holiday, days_to_festival, rolling_avg_7d]

Action:
    Continuous discount in [0, max_discount] (default 0–0.30 i.e. 0–30%)

Reward:
    R = revenue_from_sales - lambda * spoilage_penalty - gamma * discount_penalty

    revenue_from_sales = price * (1 - discount) * units_sold
    spoilage_penalty   = price * units_wasted
    discount_penalty   = gamma * discount^2 * units_sold   (prevents wild discounts)

    where:
        units_sold   = min(stock, demand_after_discount)
        demand_boost = 1 + elasticity * discount   (discounts increase demand)
        demand_after_discount = predicted_demand * demand_boost
        units_wasted = max(0, stock - units_sold) if days_to_expiry <= 1

Episode:
    Each episode is a single pricing decision for one product snapshot.
    The env randomly samples a product state from the training data.
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from pathlib import Path


class RetailPricingEnv(gym.Env):
    """Single-step retail pricing environment for perishable goods."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data_path=None,
        lambda_spoilage=5.0,
        gamma_discount=2.0,
        max_discount=0.30,
        price_elasticity=2.5,
        max_stock=50,
    ):
        super().__init__()

        # Load the feature matrix for sampling realistic states
        data_path = data_path or (
            Path(__file__).resolve().parent.parent / "data" / "feature_matrix.csv"
        )
        self.df = pd.read_csv(data_path)

        # Hyperparameters (matching the EcoPrice reward function from PDF)
        self.lambda_spoilage = lambda_spoilage   # spoilage penalty weight
        self.gamma_discount = gamma_discount     # discount penalty weight
        self.max_discount = max_discount         # maximum allowable discount
        self.price_elasticity = price_elasticity # how much demand increases per % discount
        self.max_stock = max_stock

        # Observation: [stock, days_to_expiry, predicted_demand, price,
        #               temperature, humidity, is_weekend, is_holiday,
        #               days_to_festival, rolling_avg_7d]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, -10, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([max_stock, 10, max_stock, 50000, 50, 100, 1, 1, 8, 15], dtype=np.float32),
        )

        # Action: single continuous value = discount fraction [0, max_discount]
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([max_discount], dtype=np.float32),
        )

        self.state = None
        self._current_row = None

    def _sample_state(self):
        """Sample a realistic product state from the dataset."""
        row = self.df.sample(1).iloc[0]
        self._current_row = row

        stock = np.random.randint(5, self.max_stock + 1)
        predicted_demand = float(row["quantity"])
        price = float(row["price"])
        days_to_expiry = int(row["days_to_expiry"])
        temperature = float(row["temperature"])
        humidity = float(row["humidity"])
        is_weekend = float(row["is_weekend"])
        is_holiday = float(row["is_holiday"])
        days_to_festival = float(row["days_to_festival"])
        rolling_avg = float(row["rolling_avg_7d"])

        return np.array([
            stock, days_to_expiry, predicted_demand, price,
            temperature, humidity, is_weekend, is_holiday,
            days_to_festival, rolling_avg
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self._sample_state()
        return self.state, {}

    def step(self, action):
        discount = float(np.clip(action[0], 0.0, self.max_discount))

        stock = self.state[0]
        days_to_expiry = self.state[1]
        predicted_demand = self.state[2]
        price = self.state[3]

        # Demand increases with discount (price elasticity)
        demand_boost = 1.0 + self.price_elasticity * discount
        actual_demand = predicted_demand * demand_boost

        # Add some stochasticity (real world is noisy)
        noise = np.random.normal(1.0, 0.1)
        actual_demand = max(0, actual_demand * noise)

        # Units sold = min(stock, demand)
        units_sold = min(stock, actual_demand)

        # Revenue from discounted sales
        discounted_price = price * (1.0 - discount)
        revenue = discounted_price * units_sold

        # Spoilage: unsold units that expire (only if expiring today/tomorrow)
        remaining = stock - units_sold
        if days_to_expiry <= 1:
            units_wasted = remaining
        else:
            units_wasted = 0.0

        spoilage_cost = self.lambda_spoilage * price * units_wasted

        # Discount penalty (prevents agent from always giving max discount)
        discount_penalty = self.gamma_discount * (discount ** 2) * units_sold * price

        # Final reward
        reward = revenue - spoilage_cost - discount_penalty

        # Normalize reward to roughly [-1, 1] range for stable training
        reward_normalized = np.clip(reward / (price * stock + 1e-6), -5.0, 5.0)

        # Episode info for logging
        info = {
            "discount": round(discount * 100, 1),
            "units_sold": round(units_sold, 1),
            "units_wasted": round(units_wasted, 1),
            "revenue": round(revenue, 2),
            "spoilage_cost": round(spoilage_cost, 2),
            "discount_penalty": round(discount_penalty, 2),
            "raw_reward": round(reward, 2),
            "stock": round(stock, 0),
            "days_to_expiry": round(days_to_expiry, 0),
            "predicted_demand": round(predicted_demand, 1),
            "price": round(price, 2),
        }

        # Single-step episode (terminated after one decision)
        terminated = True
        truncated = False

        return self.state, float(reward_normalized), terminated, truncated, info

    def render(self):
        if self.state is None:
            return
        print(
            f"Stock: {self.state[0]:.0f} | "
            f"Expiry: {self.state[1]:.0f}d | "
            f"Demand: {self.state[2]:.1f} | "
            f"Price: Rs.{self.state[3]:.0f} | "
            f"Temp: {self.state[4]:.1f}C | "
            f"Weekend: {'Y' if self.state[6] else 'N'} | "
            f"Holiday: {'Y' if self.state[7] else 'N'}"
        )


# Register the environment with Gymnasium
gym.register(
    id="RetailPricing-v0",
    entry_point="models.environment:RetailPricingEnv",
)


if __name__ == "__main__":
    # Quick test
    env = RetailPricingEnv()
    obs, info = env.reset()
    print("Environment created successfully!")
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    env.render()

    # Take a random action
    action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(action)
    print(f"\nAction (discount): {action[0]*100:.1f}%")
    print(f"Reward: {reward:.4f}")
    print(f"Info: {info}")
    print("\nEnvironment test passed!")
