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

---

## Key Takeaways
- The RL agent performed close to equal-weight but underperformed SPY on raw returns.
- The project demonstrates:
  - Building custom RL environments for finance.
  - Implementing policy gradient reinforcement learning in PyTorch.
  - Evaluating strategies using quant metrics (Sharpe, volatility, drawdown).
- Even if performance is below SPY, the full pipeline represents a complete RL research framework.


