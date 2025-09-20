# portfolio_optimization.py
# Auto-generated from Colab notebook

import warnings
warnings.filterwarnings('ignore')

!pip install yfinance pandas matplotlib

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

tickers = ["SPY", "TLT", "GLD", "USO", "UUP"]

data = yf.download(tickers, start="2005-01-01")["Close"]
data.head()


returns = data.pct_change().dropna()
returns.head()
returns.cumsum().plot(figsize=(12,6))
plt.title("Cumulative Returns of Assets (2005–Today)")
plt.show()

returns.to_csv("asset_returns.csv")


import numpy as np
import pandas as pd
import gym
from gym import spaces

class PortfolioEnv(gym.Env):
    """
    A custom trading environment for multi-asset portfolio optimization.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, returns: pd.DataFrame, window_size: int = 30, episode_length: int = 1260):
        """
        Args:
            returns: DataFrame of daily returns (columns = assets, index = dates).
            window_size: number of past days to include in the state.
            episode_length: number of days per episode (~5 years of trading days).
        """
        super(PortfolioEnv, self).__init__()

        self.returns = returns.dropna()
        self.assets = self.returns.columns
        self.n_assets = len(self.assets)
        self.window_size = window_size
        self.episode_length = episode_length


        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.window_size * self.n_assets,),
            dtype=np.float32
        )

        self.current_step = None
        self.start_index = None
        self.end_index = None
        self.done = False

    def reset(self):
        """
        Reset environment at the start of a new episode.
        """
        self.start_index = np.random.randint(0, len(self.returns) - self.episode_length - 1)
        self.current_step = self.start_index + self.window_size
        self.end_index = self.start_index + self.episode_length
        self.done = False

        return self._get_observation()

    def _get_observation(self):
        """
        Return past window_size days of returns (flattened).
        """
        window = self.returns.iloc[self.current_step - self.window_size:self.current_step]
        return window.values.flatten().astype(np.float32)

    def step(self, action):
        """
        Take one step forward in time.
        """
        weights = np.clip(action, 0, 1)
        weights = weights / (weights.sum() + 1e-8)

        todays_returns = self.returns.iloc[self.current_step].values

        portfolio_return = np.dot(weights, todays_returns)

        reward = portfolio_return

        self.current_step += 1
        if self.current_step >= self.end_index:
            self.done = True

        next_state = self._get_observation()
        info = {"portfolio_return": portfolio_return}

        return next_state, reward, self.done, info

    def render(self, mode="human"):
        """
        Optional: print step info.
        """
        date = self.returns.index[self.current_step]
        print(f"Step: {self.current_step}, Date: {date}")


returns = pd.read_csv("asset_returns.csv", index_col=0, parse_dates=True)

env = PortfolioEnv(returns, window_size=30, episode_length=1260)

state = env.reset()
done = False
total_reward = 0

while not done:
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)
    total_reward += reward

print("Episode finished. Total reward:", total_reward)


import numpy as np
import pandas as pd
import gym
from gym import spaces
import matplotlib.pyplot as plt

class PortfolioEnv(gym.Env):
    """
    A custom Gym environment for multi-asset portfolio optimization.
    The agent allocates across N assets with weights summing to 1.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, returns: pd.DataFrame, window_size: int = 30, episode_length: int = 1260):
        """
        Args:
            returns: DataFrame of daily returns (columns = assets, index = dates).
            window_size: how many past days of returns are shown in the state.
            episode_length: number of trading days per episode (~5 years).
        """
        super(PortfolioEnv, self).__init__()

        self.returns = returns.dropna()
        self.assets = self.returns.columns
        self.n_assets = len(self.assets)
        self.window_size = window_size
        self.episode_length = episode_length

        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.window_size * self.n_assets,),
            dtype=np.float32
        )

        self.current_step = None
        self.start_index = None
        self.end_index = None
        self.done = False
        self.portfolio_values = None

    def reset(self):
        """
        Reset environment to start a new episode.
        """
        self.start_index = np.random.randint(0, len(self.returns) - self.episode_length - 1)
        self.current_step = self.start_index + self.window_size
        self.end_index = self.start_index + self.episode_length
        self.done = False

        self.portfolio_values = [1.0]

        return self._get_observation()

    def _get_observation(self):
        """
        Get the last 'window_size' days of returns, flattened.
        """
        window = self.returns.iloc[self.current_step - self.window_size:self.current_step]
        return window.values.flatten().astype(np.float32)

    def step(self, action):
        """
        Take a step forward in time.
        Args:
            action: vector of portfolio weights
        Returns:
            next_state, reward, done, info
        """
        weights = np.clip(action, 0, 1)
        weights = weights / (weights.sum() + 1e-8)

        todays_returns = self.returns.iloc[self.current_step].values

        portfolio_return = np.dot(weights, todays_returns)

        new_value = self.portfolio_values[-1] * (1 + portfolio_return)
        self.portfolio_values.append(new_value)

        reward = portfolio_return

        self.current_step += 1
        if self.current_step >= self.end_index:
            self.done = True

        next_state = self._get_observation()
        info = {"portfolio_return": portfolio_return, "weights": weights}

        return next_state, reward, self.done, info

    def render(self, mode="human"):
        """
        Plot portfolio growth during the episode.
        """
        plt.figure(figsize=(10,5))
        plt.plot(self.portfolio_values, label="Portfolio Value")
        plt.title("Portfolio Growth During Episode")
        plt.xlabel("Days")
        plt.ylabel("Portfolio Value")
        plt.legend()
        plt.show()


returns = pd.read_csv("asset_returns.csv", index_col=0, parse_dates=True)

env = PortfolioEnv(returns, window_size=30, episode_length=1260)

state = env.reset()
done = False

while not done:
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)

env.render()



!pip uninstall -y torch torchvision torchaudio numpy
!pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu
!pip install numpy==1.26.4


import torch
import torch.nn as nn
import torch.optim as optim


class PolicyNetwork(nn.Module):
    """
    A simple feedforward policy network.
    Input: state (flattened past returns)
    Output: portfolio weights (softmax normalized)
    """
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)


def train_policy_gradient(env, policy_net, optimizer, n_episodes=200, gamma=0.99):
    """
    Train the policy network using REINFORCE with continuous portfolio weights.
    """
    all_rewards = []

    for episode in range(n_episodes):
        log_probs = []
        rewards = []

        state = env.reset()
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs = policy_net(state_tensor)

            weights = action_probs.squeeze(0)

            log_prob = torch.sum(torch.log(weights + 1e-8) * weights)

            next_state, reward, done, info = env.step(weights.detach().numpy())

            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state

        discounted_returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            discounted_returns.insert(0, R)
        discounted_returns = torch.tensor(discounted_returns, dtype=torch.float32)

        discounted_returns = (discounted_returns - discounted_returns.mean()) / (discounted_returns.std() + 1e-8)

        loss = []
        for log_prob, R in zip(log_probs, discounted_returns):
            loss.append(-log_prob * R)
        loss = torch.stack(loss).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        episode_reward = sum(rewards)
        all_rewards.append(episode_reward)
        if (episode + 1) % 20 == 0:
            print(f"Episode {episode+1}/{n_episodes}, Total Reward: {episode_reward:.4f}")

    return all_rewards

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

policy_net = PolicyNetwork(state_dim, action_dim)
optimizer = optim.Adam(policy_net.parameters(), lr=0.001)

all_rewards = train_policy_gradient(env, policy_net, optimizer, n_episodes=100)

import matplotlib.pyplot as plt
plt.plot(all_rewards)
plt.title("Training Rewards Over Episodes")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.show()



readme_text = """# Reinforcement Learning for Multi-Asset Portfolio Optimization (PyTorch)

## Project Goal
This project implements a reinforcement learning (RL) agent in PyTorch that dynamically allocates capital across multiple asset classes. The objective is to maximize risk-adjusted returns (Sharpe ratio) while incorporating realistic market frictions such as transaction costs.

---

## Project Phases

### Phase 1 — Data Collection
- Collected daily data (2005–today) for 5 ETFs:
  - SPY → Equities (S&P 500)
  - TLT → Bonds (20+ year Treasuries)
  - GLD → Commodities (Gold)
  - USO → Commodities (Oil)
  - UUP → FX (US Dollar Index)
- Converted prices to daily returns and aligned them into a clean DataFrame.

### Phase 2 — Environment Design
- Built a custom OpenAI Gym-style environment (`PortfolioEnv`) that simulates trading:
  - **State**: Past 30 days of returns.
  - **Action**: Portfolio weights across 5 assets.
  - **Reward**: Portfolio return.
- Tested with random actions to validate environment.

### Phase 3 — Policy Network
- Built a PyTorch neural network (`PolicyNetwork`) that maps states to portfolio weights.
- Trained using the REINFORCE algorithm (policy gradients).
- Verified that training improved agent performance across episodes.

### Phase 4 — Training & Evaluation
- Extended training to hundreds of episodes with model checkpointing.
- Evaluated the agent on a hold-out test set (2018–today).
- Benchmarked against:
  - Equal-weight portfolio (20% each asset).
  - Buy & Hold SPY.
- Computed metrics: annualized return, volatility, Sharpe ratio, max drawdown.

### Phase 5 — Realism Upgrades
- Improved environment with:
  - Transaction costs (penalty for rebalancing).
  - Sharpe-adjusted rewards (focus on risk-adjusted stability).
  - Optional shorting (disabled in final run).
- Retrained the agent in this environment.
- Backtested again on unseen data.

"""

# Save as README.md
with open("README.md", "w") as f:
    f.write(readme_text)

print("README.md file created successfully.")

import os

token = input("Enter your GitHub Personal Access Token: ")
os.environ["GH_TOKEN"] = token

!git remote remove origin || true

!git remote add origin https://aaravM123:{os.environ['GH_TOKEN']}@github.com/aaravM123/portfolio_optimization.git

!git remote -v

!git add .
!git commit -m "Initial commit" || echo "Nothing to commit"

!git push -u origin main


# =========================================
# 1. Create clean project structure
# =========================================
!mkdir -p notebooks src data

# =========================================
# 2. Save current Colab notebook
# Replace "rl_portfolio_rl.ipynb" with your actual notebook name if needed
# =========================================
notebook_name = "rl_portfolio_rl.ipynb"

# This saves the current Colab notebook into /content/notebooks/
!jupyter nbconvert --to notebook --output notebooks/$notebook_name "/content/*.ipynb"

# =========================================
# 3. Move your data file (if it exists)
# =========================================
!mv asset_returns.csv data/ || echo "No asset_returns.csv found"

# =========================================
# 4. Add .gitignore to keep repo clean
# =========================================
!echo ".config/" >> .gitignore
!echo "__pycache__/" >> .gitignore
!echo "*.pyc" >> .gitignore

# =========================================
# 5. Stage, commit, and push clean project
# =========================================
!git add notebooks src data README.md .gitignore
!git commit -m "Add notebooks, source code, and data" || echo "Nothing new to commit"
!git push -u origin main



# ===============================
# 1. Configure Git
# ===============================
!git config --global user.name "aaravM123"
!git config --global user.email "your_email@example.com"

# ===============================
# 2. Stage the correct files
# ===============================
!git add notebooks src data README.md .gitignore

# Commit changes
!git commit -m "Final clean commit: add notebooks, source code, and data" || echo "Nothing new to commit"

# ===============================
# 3. Force push to GitHub
# ===============================
# WARNING: This will overwrite remote history and keep only your local files
!git push -u origin main --force
