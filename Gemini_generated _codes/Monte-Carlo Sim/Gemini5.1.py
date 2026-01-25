import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


def run_colorful_elite_sim():
    # ==========================================
    # 1. SETUP & DATA
    # ==========================================
    print("--- Starting Final Elite Simulation (Colorful Mode) ---")

    stock_tickers = ['GOOGL', 'JPM', 'PEP', 'UNH', 'ASML', 'LVMUY', 'NVS', 'TTE', 'SIEGY']
    dom_bonds = ['BGRN', 'LQD']
    bond_etfs = ['TLT', 'IEF']
    gold = ['GLD']
    all_tickers = stock_tickers + dom_bonds + bond_etfs + gold

    print(f"Downloading data for {len(all_tickers)} assets...")
    data = yf.download(all_tickers, start='2020-01-01', end='2025-01-01', auto_adjust=True, progress=False)

    try:
        prices = data['Close']
    except KeyError:
        prices = data
    prices = prices.dropna()
    returns = prices.pct_change().dropna()
    print("Data Ready.\n")

    # ==========================================
    # 2. OPTIMIZE STOCK BUCKET (MAX SHARPE)
    # ==========================================
    print("Optimizing Stock Weights for Maximum Efficiency...")

    stock_ret = returns[stock_tickers]
    stock_cov = stock_ret.cov()
    mean_stock_ret = stock_ret.mean()
    rf_rate = 0.035

    # Objective: Maximize Sharpe Ratio
    def get_negative_sharpe(weights):
        port_ret = np.sum(mean_stock_ret * weights) * 252
        port_vol = np.sqrt(np.dot(weights.T, np.dot(stock_cov, weights))) * np.sqrt(252)
        return -(port_ret - rf_rate) / port_vol

    n_stocks = len(stock_tickers)
    # Constraints: Sum = 1.0, Max 40% per stock
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0.02, 0.40) for _ in range(n_stocks))
    init_guess = [1 / n_stocks] * n_stocks

    res = minimize(get_negative_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints, tol=1e-10)
    best_stock_weights = res.x

    # ==========================================
    # 3. BUILD TOTAL PORTFOLIO (80% Stocks)
    # ==========================================
    # 80% Stocks (Optimized)
    final_stock_w = best_stock_weights * 0.80

    # 7.5% Domestic Bonds (3.75% each)
    w_dom = np.array([0.0375, 0.0375])

    # 7.5% Bond ETFs (3.75% each)
    w_etf = np.array([0.0375, 0.0375])

    # 5% Gold
    w_gold = np.array([0.05])

    final_weights = np.concatenate([final_stock_w, w_dom, w_etf, w_gold])
    portfolio_weights = dict(zip(all_tickers, final_weights))

    # --- PRINT WEIGHTS ---
    print("\n" + "=" * 55)
    print(f"{'FINAL OPTIMIZED WEIGHTS (80% Aggressive)':^55}")
    print("=" * 55)
    print(f"{'Ticker':<10} {'Asset Type':<15} {'Weight (%)':<12} {'Value ($100k)'}")
    print("-" * 55)

    total_val = 100000
    for t in all_tickers:
        if t in stock_tickers:
            a_type = "Elite Stock"
        elif t in dom_bonds:
            a_type = "Yield Bond"
        elif t in bond_etfs:
            a_type = "Safety Bond"
        else:
            a_type = "Gold"

        w = portfolio_weights[t]
        print(f"{t:<10} {a_type:<15} {w:>10.2%}      ${total_val * w:,.0f}")
    print("-" * 55)

    # ==========================================
    # 4. MONTE CARLO SIMULATION
    # ==========================================
    print("\nRunning 1000000 Simulations...")
    n_sims = 1000000
    n_years = 5
    n_days = 252 * n_years

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
    sim_cagrs = (ending_values / total_val) ** (1 / n_years) - 1
    mean_cagr = np.mean(sim_cagrs)
    sharpe = (mean_cagr - 0.035) / (port_std * np.sqrt(252))

    # --- 95% CONFIDENCE INTERVALS ---
    ci_lower_val = np.percentile(ending_values, 2.5)
    ci_upper_val = np.percentile(ending_values, 97.5)

    print("\n" + "=" * 50)
    print(f"{'PREDICTED PERFORMANCE (5 Years)':^50}")
    print("=" * 50)
    print(f"Mean Annual Return:   {mean_cagr:.2%}")
    print(f"Sharpe Ratio:         {sharpe:.2f}")
    print(f"Annual Volatility:      {port_std * np.sqrt(252):.2%}")
    print("-" * 50)
    print(f"95% CI Lower (Worst Case):  ${ci_lower_val:,.0f}")
    print(f"95% CI Upper (Best Case):   ${ci_upper_val:,.0f}")
    print("=" * 50)

    # ==========================================
    # 5. VISUALIZATION
    # ==========================================

    # --- CHART 1: Colorful Spaghetti Plot ---
    plt.figure(figsize=(12, 7))

    # Robust colormap access
    colormap = plt.get_cmap('viridis')
    norm = plt.Normalize(vmin=np.min(ending_values), vmax=np.max(ending_values))

    # Plot 200 random paths
    indices = np.random.choice(n_sims, 200, replace=False)
    for i in indices:
        plt.plot(price_paths[:, i], color=colormap(norm(ending_values[i])), alpha=0.4, linewidth=0.8)

    # Plot Mean Path (Bright Red)
    plt.plot(np.mean(price_paths, axis=1), color='red', linewidth=3, linestyle='--', label='Mean Path')

    # Add Colorbar (FIXED LINE BELOW)
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=plt.gca(), label='Ending Portfolio Value ($)')

    plt.title(f'Monte Carlo: 5-Year Outlook (n={n_sims})', fontsize=14, fontweight='bold')
    plt.xlabel('Trading Days')
    plt.ylabel('Portfolio Value ($)')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.2)
    plt.savefig('colorful_spaghetti.png', dpi=1200)
    plt.show()

    # --- CHART 2: Annualized Return Histogram (Frequency vs %) ---
    plt.figure(figsize=(12, 7))

    weights = np.ones_like(sim_cagrs) / len(sim_cagrs)
    plt.hist(sim_cagrs, bins=70, weights=weights, color='#2980b9', edgecolor='white', alpha=0.85)

    plt.axvline(mean_cagr, color='red', linestyle='-', linewidth=3, label=f'Mean Return: {mean_cagr:.1%}')

    # Format Axes as Percentages
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    plt.title('Probability Distribution of Annual Returns', fontsize=14, fontweight='bold')
    plt.xlabel('Annual Return (CAGR)')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.savefig('return_histogram.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    run_colorful_elite_sim()