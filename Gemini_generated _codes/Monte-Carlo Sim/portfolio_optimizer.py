"""
Portfolio Optimization & Monte Carlo Simulation Tool
----------------------------------------------------
Author: [PEHC]
Date: 2026-01-25
Description:
    This script downloads historical financial data, optimizes a portfolio using
    Modern Portfolio Theory (Mean-Variance Optimization), and runs a
    Monte Carlo simulation to project future wealth accumulation via DCA.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.optimize import minimize
from typing import List, Tuple

# --- CONFIGURATION ---
TICKERS = ['SPYM', 'QQQ', 'VEA', 'TLT', 'GLDM']  # SPYM = S&P 500
TARGET_VOLATILITY = 0.14  # 14% Annualized Risk Target
RISK_FREE_RATE = 0.035  # 3.5% Risk Free Rate
MONTHLY_CONTRIBUTION = 185  # USD
INITIAL_CAPITAL = 1230  # USD
SIM_YEARS = 5
NUM_SIMULATIONS = 1000000


def get_data(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Downloads adjusted close prices from Yahoo Finance."""
    print(f"Downloading data for: {tickers}...")
    try:
        data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            prices = data['Close']
        else:
            prices = data['Close'] if 'Close' in data.columns else data
        return prices.dropna()
    except Exception as e:
        print(f"Error parsing data: {e}")
        return pd.DataFrame()


def optimize_portfolio(mean_returns: pd.Series, cov_matrix: pd.DataFrame) -> np.ndarray:
    """
    Calculates optimal weights to maximize return for a specific volatility target.
    Constraints: Sum of weights = 100%, Volatility <= Target.
    """
    num_assets = len(mean_returns)

    def get_portfolio_return(weights):
        return -np.sum(mean_returns * weights) * 252

    def check_volatility(weights):
        annual_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        return TARGET_VOLATILITY - annual_vol

    cons = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Sum = 1.0
        {'type': 'ineq', 'fun': check_volatility}  # Vol constraint
    )

    # Asset Bounds: Min 2%, Max 70% per asset for diversification
    bounds = tuple((0.02, 0.70) for _ in range(num_assets))

    init_guess = [1 / num_assets] * num_assets

    result = minimize(
        get_portfolio_return,
        init_guess,
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        tol=1e-6
    )
    return result.x


def run_monte_carlo(mu: float, sigma: float, years: int, n_sims: int) -> Tuple[np.ndarray, float]:
    """Runs a Geometric Brownian Motion simulation with monthly contributions."""
    np.random.seed(42)  # For reproducible results
    n_days = 252 * years
    dt = 1 / 252

    drift = mu - 0.5 * (sigma ** 2)
    Z = np.random.normal(0, 1, (n_days, n_sims))
    daily_returns = np.exp(drift * dt + sigma * np.sqrt(dt) * Z)

    paths = np.zeros((n_days + 1, n_sims))
    paths[0] = INITIAL_CAPITAL

    total_invested = INITIAL_CAPITAL
    for t in range(1, n_days + 1):
        paths[t] = paths[t - 1] * daily_returns[t - 1]
        if t % 21 == 0:
            paths[t] += MONTHLY_CONTRIBUTION
            if t < n_days:
                total_invested += MONTHLY_CONTRIBUTION

    return paths, total_invested


def main():
    # 1. Data Ingestion
    prices = get_data(TICKERS, start_date='2020-01-01', end_date='2026-01-01')
    returns = prices.pct_change().dropna()

    # 2. Optimization
    print("\n--- Optimizing Portfolio ---")
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    weights = optimize_portfolio(mean_returns, cov_matrix)

    # Metrics
    opt_ret = np.sum(mean_returns * weights) * 252
    opt_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    sharpe = (opt_ret - RISK_FREE_RATE) / opt_vol

    print(f"\nTarget Volatility: {TARGET_VOLATILITY:.1%}")
    print(f"Expected Return:   {opt_ret:.1%}")
    print(f"Sharpe Ratio:      {sharpe:.2f}")
    print("-" * 30)
    print("OPTIMIZED ALLOCATION:")

    # Display Allocation
    asset_dict = dict(zip(returns.columns, weights))
    sorted_assets = sorted(asset_dict.items(), key=lambda x: x[1], reverse=True)
    for ticker, weight in sorted_assets:
        print(f"{ticker:<6} : {weight:>6.1%}")

    # 3. Simulation
    print(f"\n--- Running {NUM_SIMULATIONS} Monte Carlo Scenarios ---")
    price_paths, invested = run_monte_carlo(opt_ret, opt_vol, SIM_YEARS, NUM_SIMULATIONS)

    ending_values = price_paths[-1]

    # Calculate CAGR & CI
    sim_cagrs = (ending_values / invested) ** (1 / SIM_YEARS) - 1
    mean_cagr = np.mean(sim_cagrs)
    ci_lower_ret = np.percentile(sim_cagrs, 2.5)
    ci_upper_ret = np.percentile(sim_cagrs, 97.5)

    print(f"Total Invested: ${invested:,.0f}")
    print(f"Mean Outcome:   ${np.mean(ending_values):,.0f}")
    print(f"95% Worst Case: ${np.percentile(ending_values, 5):,.0f}")

    # 4. VISUALIZATION (Professional Style)

    # --- CHART 1: SPAGHETTI PLOT ---
    plt.figure(figsize=(12, 7))
    colormap = plt.get_cmap('viridis')
    norm = plt.Normalize(vmin=np.min(ending_values), vmax=np.max(ending_values))

    # Plot a subset of paths (e.g., 200) to avoid lag, colored by final outcome
    indices = np.random.choice(NUM_SIMULATIONS, 200, replace=False)
    for i in indices:
        plt.plot(price_paths[:, i], color=colormap(norm(ending_values[i])), alpha=0.4, linewidth=0.8)

    plt.plot(np.mean(price_paths, axis=1), color='red', linewidth=3, linestyle='--', label='Mean Path')

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=plt.gca(), label='Ending Portfolio Value ($)')

    plt.title(f'Monte Carlo: {SIM_YEARS}-Year Outlook (DCA Strategy)', fontsize=14, fontweight='bold')
    plt.xlabel('Trading Days')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()

    # --- CHART 2: DISTRIBUTION WITH CI LINES ---
    plt.figure(figsize=(12, 7))
    weights_hist = np.ones_like(sim_cagrs) / len(sim_cagrs)
    plt.hist(sim_cagrs, bins=70, weights=weights_hist, color='#198964', edgecolor='white', alpha=0.85)

    # Vertical Lines for Mean and CI
    plt.axvline(mean_cagr, color='red', linewidth=3, label=f'Mean Return: {mean_cagr:.1%}')
    plt.axvline(ci_lower_ret, color='blue', linestyle='--', linewidth=2.5,
                label=f'95% Lower Limit: {ci_lower_ret:.1%}')
    plt.axvline(ci_upper_ret, color='blue', linestyle='--', linewidth=2.5, label=f'95% Upper Limit: {ci_upper_ret:.1%}')

    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    plt.title('Return Distribution with Risk Limits (95% CI)', fontsize=14, fontweight='bold')
    plt.xlabel('Annual Return (CAGR)')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()