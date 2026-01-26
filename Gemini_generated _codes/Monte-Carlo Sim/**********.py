import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


def run_final_super_sim():
    print("--- Starting FINAL 'Super-Portfolio' Simulation ---")

    # 1. DEFINE THE 5 ASSETS

    core_stocks = ['SPYM', 'QQQ']
    new_stocks = ['VEA']
    bonds = ['TLT']
    gold = ['GLDM']
    all_stocks = core_stocks + new_stocks
    all_tickers = all_stocks + bonds + gold
    n_stocks = len(all_stocks)
    n_bonds = len(bonds)
    n_gold = len(gold)
    all_tickers = core_stocks + new_stocks + bonds + gold
    n_assets = len(all_tickers)

    print(f"Downloading data for {n_assets} assets...")
    data = yf.download(all_tickers, start='2020-01-01', end='2026-01-01', auto_adjust=True, progress=False)

    try:
        if isinstance(data.columns, pd.MultiIndex):
            prices = data['Close']
        else:
            prices = data['Close'] if 'Close' in data.columns else data
    except:
        prices = data
    prices = prices.dropna()
    returns = prices.pct_change().dropna()
    print("Data Ready.\n")

    # 2. OPTIMIZATION (Target Vol 14% + Diversity)
    print("Optimizing: Target Volatility 14% with Broad Diversification...")

    cov_matrix = returns.cov()
    mean_returns = returns.mean()
    rf_rate = 0.035
    TARGET_VOL = 0.14

    # Objective: Maximize Return
    def get_return(weights):
        return -np.sum(mean_returns * weights) * 252

    # Constraint: Volatility <= 14%
    def check_volatility(weights):
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        return TARGET_VOL - port_vol

    cons = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'ineq', 'fun': check_volatility},
        # Macro Guardrails
        {'type': 'ineq', 'fun': lambda x: np.sum(x[:n_stocks]) - 0.50},  # Stocks Min 50%
        {'type': 'ineq', 'fun': lambda x: 0.96 - np.sum(x[:n_stocks])},  # Stocks Max 96%
        {'type': 'ineq', 'fun': lambda x: np.sum(x[n_stocks : n_stocks+n_bonds]) - 0.02},  # Bonds Min 2%
        {'type': 'ineq', 'fun': lambda x: np.sum(x[-n_gold]) - 0.02}  # Gold Min 2%
    ]

    # BOUNDS: Force 3% Min on ALL Stocks (Diversity) "No stock left behind"
    bounds = []
    for i in range(n_assets):
        if i < n_stocks:
            # It is a Stock: Force 3% Minimum
            bounds.append((0.03, 0.96))
        elif i < n_stocks + n_bonds:
            # It is a Bond: least 2% (but macro rule forces group total to 15%)
            bounds.append((0.02, 0.02))
        else:
            # It is Gold: least 2% (but macro rule forces group total to 5%)
            bounds.append((0.02, 0.02))
    bounds = tuple(bounds)

    init_guess = [1 / n_assets] * n_assets
    res = minimize(get_return, init_guess, method='SLSQP', bounds=bounds, constraints=cons, tol=1e-10)
    final_weights = res.x

    # 3. PRINT WEIGHTS
    portfolio_weights = dict(zip(all_tickers, final_weights))
    print("\n" + "=" * 55)
    print(f"{'FINAL SUPER-PORTFOLIO WEIGHTS':^55}")
    print("=" * 55)
    print(f"{'Ticker':<10} {'Asset Type':<15} {'Weight (%)':<12} {'Value ($100k)'}")
    print("-" * 55)

    total_val = 100000
    for t in all_tickers:
        if t in core_stocks:
            a_type = "Core Stock"
        elif t in new_stocks:
            a_type = "New Recruit"
        elif t in bonds:
            a_type = "Bond"
        else:
            a_type = "Gold"

        w = portfolio_weights[t]
        if w > 0.00000000000000000001:
            print(f"{t:<10} {a_type:<15} {w:>10.2%}      ${total_val * w:,.0f}")

    print("=" * 55)

    # 4. MONTE CARLO SIMULATION
    print("\nRunning 1000000 Simulations...")
    n_sims = 1000000
    n_days = 252 * 5

    port_ret = np.dot(final_weights, returns.mean())
    port_cov = np.dot(final_weights.T, np.dot(returns.cov(), final_weights))
    port_std = np.sqrt(port_cov)

    drift = port_ret - 0.5 * port_cov
    Z = np.random.normal(0, 1, (n_days, n_sims))
    daily_growth = np.exp(drift + port_std * Z)

    price_paths = np.zeros((n_days + 1, n_sims))
    price_paths[0] = total_val
    for t in range(1, n_days + 1):
        price_paths[t] = price_paths[t - 1] * daily_growth[t - 1]

    ending_values = price_paths[-1]

    # Metrics
    sim_cagrs = (ending_values / total_val) ** (1 / 5) - 1
    mean_cagr = np.mean(sim_cagrs)
    sharpe = (mean_cagr - 0.035) / (port_std * np.sqrt(252))

    # 95% Confidence Intervals
    ci_lower_val = np.percentile(ending_values, 2.5)
    ci_upper_val = np.percentile(ending_values, 97.5)
    ci_lower_ret = np.percentile(sim_cagrs, 2.5)
    ci_upper_ret = np.percentile(sim_cagrs, 97.5)

    print("\n" + "=" * 50)
    print(f"{'PREDICTED PERFORMANCE (5 Years)':^50}")
    print("=" * 50)
    print(f"Mean Annual Return:   {mean_cagr:.2%}")
    print(f"Annual Volatility:    {port_std * np.sqrt(252):.2%}")
    print(f"Sharpe Ratio:         {sharpe:.2f}")
    print("-" * 50)
    print(f"95% CI Lower (Worst Case):  ${ci_lower_val:,.0f} (CAGR: {ci_lower_ret:.2%})")
    print(f"95% CI Upper (Best Case):   ${ci_upper_val:,.0f} (CAGR: {ci_upper_ret:.2%})")
    print("=" * 50)

    # 5. VISUALIZATION

    # --- CHART 1: SPAGHETTI PLOT ---
    plt.figure(figsize=(12, 7))
    colormap = plt.get_cmap('viridis')
    norm = plt.Normalize(vmin=np.min(ending_values), vmax=np.max(ending_values))
    indices = np.random.choice(n_sims, 200, replace=False)
    for i in indices:
        plt.plot(price_paths[:, i], color=colormap(norm(ending_values[i])), alpha=0.4, linewidth=0.8)
    plt.plot(np.mean(price_paths, axis=1), color='red', linewidth=3, linestyle='--', label='Mean Path')
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=plt.gca(), label='Ending Portfolio Value ($)')
    plt.title('Monte Carlo: 5-Year Outlook (Super-Portfolio)', fontsize=14, fontweight='bold')
    plt.xlabel('Trading Days')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.savefig('final_spaghetti.png', dpi=300)
    plt.show()

    # --- CHART 2: DISTRIBUTION WITH CI LINES ---
    plt.figure(figsize=(12, 7))
    weights = np.ones_like(sim_cagrs) / len(sim_cagrs)
    plt.hist(sim_cagrs, bins=70, weights=weights, color='#198964', edgecolor='white', alpha=0.85)

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
    plt.show()


if __name__ == "__main__":
    run_final_super_sim()