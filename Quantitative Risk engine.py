import datetime
import numpy as np
import pandas as pd
import scipy.optimize as sco
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

# 1. Pipeline: Ingest Historical Data
TICKERS = ["SPY", "QQQ", "TLT", "GLD", "EEM"] # Equities, Tech, Treasuries, Gold, EM
START_DATE = "2018-01-01"
END_DATE = datetime.date.today().strftime("%Y-%m-%d")

print(f"Downloading historical data for {TICKERS}...")
raw_data = yf.download(TICKERS, start=START_DATE, end=END_DATE, progress=False)
price_data = raw_data["Adj Close"] if "Adj Close" in raw_data else raw_data["Close"]
returns = price_data.pct_change().dropna()

mean_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252

# 2. Portfolio Optimization (Maximum Sharpe Ratio)
def portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate=0.04):
    port_return = np.sum(mean_returns * weights)
    port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = (port_return - risk_free_rate) / port_std
    return port_return, port_std, sharpe

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.04):
    return -portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)[2]

num_assets = len(TICKERS)
constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1} # Weights sum to 100%
bounds = tuple((0.0, 1.0) for _ in range(num_assets)) # Long-only
init_guess = num_assets * [1.0 / num_assets]

opt_results = sco.minimize(
    neg_sharpe_ratio,
    init_guess,
    args=(mean_returns, cov_matrix),
    method="SLSQP",
    bounds=bounds,
    constraints=constraints
)

opt_weights = opt_results.x
opt_return, opt_vol, opt_sharpe = portfolio_performance(opt_weights, mean_returns, cov_matrix)

print("\n--- Optimized Maximum Sharpe Allocation ---")
for ticker, w in zip(TICKERS, opt_weights):
    print(f"{ticker}: {w * 100:.2f}%")
print(f"Expected Annual Return: {opt_return * 100:.2f}%")
print(f"Annualized Volatility: {opt_vol * 100:.2f}%")
print(f"Sharpe Ratio (Rf=4%): {opt_sharpe:.2f}")

# 3. Risk & Value-at-Risk Engine
portfolio_daily_returns = returns.dot(opt_weights)
alpha = 0.01 # 99% Confidence Level
portfolio_value = 1_000_000 # $1,000,000 Portfolio Base

# Parametric VaR
z_score = 2.326 # 99% Z-score
param_var_pct = z_score * portfolio_daily_returns.std()
param_var_dollar = portfolio_value * param_var_pct

# Historical VaR & CVaR (Expected Shortfall)
hist_var_pct = -np.percentile(portfolio_daily_returns, 100 * alpha)
hist_var_dollar = portfolio_value * hist_var_pct
cvar_pct = -portfolio_daily_returns[portfolio_daily_returns <= -hist_var_pct].mean()
cvar_dollar = portfolio_value * cvar_pct

# Monte Carlo VaR (50,000 Runs)
mc_sims = 50_000
mc_daily_vol = np.sqrt(np.dot(opt_weights.T, np.dot(returns.cov(), opt_weights)))
simulated_returns = np.random.normal(portfolio_daily_returns.mean(), mc_daily_vol, mc_sims)
mc_var_pct = -np.percentile(simulated_returns, 100 * alpha)
mc_var_dollar = portfolio_value * mc_var_pct

print("\n--- 1-Day Risk Metrics ($1,000,000 Portfolio) ---")
print(f"Historical VaR (99%): ${hist_var_dollar:,.2f} ({hist_var_pct*100:.2f}%)")
print(f"Parametric VaR (99%): ${param_var_dollar:,.2f} ({param_var_pct*100:.2f}%)")
print(f"Monte Carlo VaR (99%): ${mc_var_dollar:,.2f} ({mc_var_pct*100:.2f}%)")
print(f"Conditional VaR / CVaR: ${cvar_dollar:,.2f} ({cvar_pct*100:.2f}%)")

# 4. Generate Visual Risk Tearsheet
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot A: Cumulative Wealth Curve vs Benchmark
cum_port = (1 + portfolio_daily_returns).cumprod()
cum_spy = (1 + returns["SPY"]).cumprod()
axes[0, 0].plot(cum_port.index, cum_port, label="Optimized Multi-Asset Portfolio", color="#1f77b4", lw=2)
axes[0, 0].plot(cum_spy.index, cum_spy, label="SPY Benchmark", color="#7f7f7f", linestyle="--", alpha=0.7)
axes[0, 0].set_title("Historical Cumulative Performance vs SPY", fontsize=12, fontweight="bold")
axes[0, 0].set_ylabel("Growth of $1")
axes[0, 0].legend()

# Plot B: Asset Correlation Heatmap
sns.heatmap(returns.corr(), annot=True, cmap="vlag", fmt=".2f", ax=axes[0, 1], cbar=False)
axes[0, 1].set_title("Cross-Asset Correlation Matrix", fontsize=12, fontweight="bold")

# Plot C: Drawdown Chart
drawdown = (cum_port - cum_port.cummax()) / cum_port.cummax()
axes[1, 0].fill_between(drawdown.index, drawdown, 0, color="#d62728", alpha=0.4)
axes[1, 0].plot(drawdown.index, drawdown, color="#d62728", lw=1)
axes[1, 0].set_title(f"Historical Drawdown Profile (Max: {drawdown.min()*100:.2f}%)", fontsize=12, fontweight="bold")
axes[1, 0].set_ylabel("Drawdown %")

# Plot D: Daily Returns Distribution & 99% VaR Threshold
sns.histplot(portfolio_daily_returns, bins=80, kde=True, ax=axes[1, 1], color="#2ca02c", stat="density")
axes[1, 1].axvline(-hist_var_pct, color="red", linestyle="--", lw=2, label=f"99% 1-Day VaR: {hist_var_pct*100:.2f}%")
axes[1, 1].axvline(-cvar_pct, color="darkred", linestyle=":", lw=2, label=f"99% CVaR: {cvar_pct*100:.2f}%")
axes[1, 1].set_title("Daily Return Distribution & Tail Risk Cutoffs", fontsize=12, fontweight="bold")
axes[1, 1].set_xlabel("Daily Return")
axes[1, 1].legend()

plt.tight_layout()
plt.savefig("risk_tearsheet.png", dpi=300)
print("\nTearsheet successfully generated and saved as 'risk_tearsheet.png'.")