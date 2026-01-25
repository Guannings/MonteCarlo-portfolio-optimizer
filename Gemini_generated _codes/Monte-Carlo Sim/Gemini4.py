import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def run_target_vol_portfolio():
    # ==========================================
    # 1. SETUP & DATA DOWNLOAD
    # ==========================================
    print("--- Starting Target Volatility Optimization ---")

    # Define Assets
    stock_tickers = ['GOOGL', 'JPM', 'PEP', 'UNH', 'ASML', 'LVMUY', 'NVS', 'TTE', 'SIEGY']
    dom_bonds = ['BGRN', 'LQD']  # 17.5% Allocation
    bond_etfs = ['TLT', 'IEF']  # 7.5% Allocation
    gold = ['GLD']  # 5% Allocation

    all_tickers = stock_tickers + dom_bonds + bond_etfs + gold

    print(f"Downloading data for {len(all_tickers)} assets...")
    start_date = '2020-01-01'
    end_date = '2025-01-01'

    data = yf.download(all_tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)

    try:
        prices = data['Close']
    except KeyError:
        prices = data

    prices = prices.dropna()
    returns = prices.pct_change().dropna()

    # ==========================================
    # 2. OPTIMIZE: MAX RETURN with VOLATILITY CAP
    # ==========================================
    print("\nOptimizing Stocks: Max Return with 15% Volatility Cap...")

    stock_ret = returns[stock_tickers]
    stock_cov = stock_ret.cov()
    mean_stock_ret = stock_ret.mean()

    # TARGET VOLATILITY CEILING (The "Speed Limit")
    # 0.15 = 15% Annual Volatility
    TARGET_VOL = 0.15

    # Objective: Maximize Return (Minimize Negative Return)
    def get_negative_return(weights):
        # Annualized Return
        port_ret = np.sum(mean_stock_ret * weights) * 252
        return -port_ret

    # Constraint: Portfolio Volatility must be <= TARGET_VOL
    def check_volatility(weights):
        # Annualized Volatility
        port_vol = np.sqrt(np.dot(weights.T, np.dot(stock_cov, weights))) * np.sqrt(252)
        # Return positive if under limit, negative if over
        return TARGET_VOL - port_vol

    n_stocks = len(stock_tickers)

    # Define Constraints Dictionary
    constraints = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Sum = 100%
        {'type': 'ineq', 'fun': check_volatility}  # Volatility <= 15%
    )

    # Bounds: Min 2%, Max 35% per stock (Balanced)
    bounds = tuple((0.02, 0.35) for _ in range(n_stocks))
    init_guess = [1 / n_stocks] * n_stocks

    res = minimize(get_negative_return, init_guess, method='SLSQP', bounds=bounds, constraints=constraints, tol=1e-10)

    if not res.success:
        print("Warning: Optimization struggled to hit target. Using best guess.")

    opt_stock_weights = res.x

    # ==========================================
    # 3. CONSTRUCT TOTAL PORTFOLIO
    # ==========================================
    # Scale stocks to 70%
    final_stock_w = opt_stock_weights * 0.70

    # Assign Fixed Weights
    w_dom = 0.175 / 2
    w_etf = 0.075 / 2
    w_gold = 0.05

    final_dom_w = np.array([w_dom, w_dom])
    final_etf_w = np.array([w_etf, w_etf])
    final_gold_w = np.array([w_gold])

    final_weights = np.concatenate([final_stock_w, final_dom_w, final_etf_w, final_gold_w])
    portfolio_weights = dict(zip(all_tickers, final_weights))

    # --- PRINT WEIGHTS ---
    print("\n" + "=" * 55)
    print(f"{'BALANCED (TARGET VOL) PORTFOLIO WEIGHTS':^55}")
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

    # Metrics
    ending_values = price_paths[-1]
    expected_cagr = (np.mean(ending_values) / total_val) ** (1 / 5) - 1
    sharpe = (expected_cagr - 0.035) / (port_std * np.sqrt(252))
    ci_lower = np.percentile(ending_values, 2.5)
    ci_upper = np.percentile(ending_values, 97.5)
    var_95 = total_val - np.percentile(ending_values, 5)

    print("\n" + "=" * 55)
    print(f"{'SIMULATION RESULTS (5 YEARS)':^55}")
    print("=" * 55)
    print(f"Annual Return (CAGR):   {expected_cagr:.2%}")
    print(f"Annual Volatility:      {port_std * np.sqrt(252):.2%}")
    print(f"Sharpe Ratio:           {sharpe:.2f}")
    print(f"Value at Risk (5%):     ${var_95:,.2f}")
    print("=" * 55)

    # ==========================================
    # 5. VISUALIZATION
    # ==========================================
    plt.figure(figsize=(10, 6))
    plt.plot(price_paths[:, :20], lw=1)
    plt.plot(price_paths.mean(axis=1), 'r--', label='Mean Path', lw=3)
    plt.title('Monte-Carlo: Balanced (Target Vol) Portfolio')
    plt.xlabel('Trading Days')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('balanced_paths.png')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.hist(ending_values, bins=50, edgecolor='black', alpha=0.7)
    plt.title('Distribution of Ending Portfolio Values')
    plt.axvline(np.mean(ending_values), color='r', linestyle='dashed', linewidth=2, label='Mean')
    plt.axvline(ci_lower, color='orange', linestyle='dashed', linewidth=2, label='95% CI')
    plt.axvline(ci_upper, color='orange', linestyle='dashed', linewidth=2)
    plt.legend()
    plt.savefig('balanced_dist.png')
    plt.show()


if __name__ == "__main__":
    run_target_vol_portfolio()