# Reinforcement Learning for Multi-Asset Portfolio Optimization (PyTorch)

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

### Phase 6 — Extensions (Future Work)
- Possible improvements:
  - Allow shorting.
  - Add more asset classes (crypto, EM equities, corporate bonds).
  - Include macroeconomic features (CPI, unemployment, VIX).
  - Upgrade RL algorithm (PPO, A2C, DDPG).
  - Build regime-switching agents.

---

## Final Results (2018–Today)

Portfolio values (start = 1.0):

![Backtest Results](e2a72162-dae8-4f9a-a013-546a33d9c58b.png)

- Buy & Hold SPY: ~2.8x growth  
- RL Agent (Phase 5): ~1.8x growth  
- Equal Weight: ~1.7x growth  

---

## Key Takeaways
- The RL agent performed close to equal-weight but underperformed SPY on raw returns.
- The project demonstrates:
  - Building custom RL environments for finance.
  - Implementing policy gradient reinforcement learning in PyTorch.
  - Evaluating strategies using quant metrics (Sharpe, volatility, drawdown).
- Even if performance is below SPY, the full pipeline represents a complete RL research framework.

---

## Repository Structure
rl-portfolio-optimization/
│── data/
│ └── asset_returns.csv
│
│── notebooks/
│ ├── 01_phase1_data.ipynb
│ ├── 02_phase2_env.ipynb
│ ├── 03_phase3_basic_agent.ipynb
│ ├── 04_phase4_training_eval.ipynb
│ ├── 05_phase5_enhanced_env.ipynb
│ └── 06_phase5_backtest.ipynb
│
│── src/
│ ├── portfolio_env.py
│ ├── policy_network.py
│ ├── train.py
│ ├── backtest.py
│
│── README.md

pgsql
Copy code

---

## Learning Outcomes
- How to design a custom RL environment for portfolio optimization.
- How to implement and train policy gradient agents in PyTorch.
- How to evaluate portfolio strategies with financial metrics.
- Gained experience with the full ML-to-finance pipeline.
