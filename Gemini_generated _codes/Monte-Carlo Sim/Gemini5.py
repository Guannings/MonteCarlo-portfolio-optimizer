import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def run_elite_portfolio_simulation():
    # ==========================================
    # 1. SETUP & DATA DOWNLOAD
    # ==========================================
    print("--- Starting Portfolio Simulation ---")

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
    # 2. MINIMUM VARIANCE OPTIMIZATION (STOCKS)
    # ==========================================
    print("\nCalculating Optimal Weights for Stocks (MVP Mode)...")

    stock_ret = returns[stock_tickers]
    stock_cov = stock_ret.cov()

    # Goal: Minimize Annualized Volatility of the 70% stock bucket
    def get_stock_volatility(weights):
        daily_var = np.dot(weights.T, np.dot(stock_cov, weights))
        return np.sqrt(daily_var) * np.sqrt(252)

    n_stocks = len(stock_tickers)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0.02, 1.0) for _ in range(n_stocks))  # 2% floor within bucket
    init_guess = [1 / n_stocks] * n_stocks

    res = minimize(get_stock_volatility, init_guess, method='SLSQP', bounds=bounds, constraints=constraints, tol=1e-10)
    opt_stock_sub_weights = res.x

    # ==========================================
    # 3. CONSTRUCT TOTAL PORTFOLIO WEIGHTS
    # ==========================================
    # Scale stocks to 70% of the WHOLE portfolio
    final_stock_w = opt_stock_sub_weights * 0.80

    # Assign Fixed Weights
    w_dom = 0.075 / 2
    w_etf = 0.075 / 2
    w_gold = 0.05

    final_dom_w = np.array([w_dom, w_dom])
    final_etf_w = np.array([w_etf, w_etf])
    final_gold_w = np.array([w_gold])

    # Combined Array (Order must match 'all_tickers')
    final_weights = np.concatenate([final_stock_w, final_dom_w, final_etf_w, final_gold_w])
    portfolio_weights = dict(zip(all_tickers, final_weights))

    # --- PRINT THE WEIGHT BREAKDOWN ---
    print("\n" + "=" * 55)
    print(f"{'DETAILED PORTFOLIO WEIGHTS':^55}")
    print("=" * 55)
    print(f"{'Ticker':<10} {'Asset Type':<15} {'Weight (%)':<12} {'Value ($)'}")
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
    print(f"{'TOTAL':<10} {'':<15} {1.0:>10.2%}      ${total_val:,.0f}")

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

    print("\n" + "=" * 55)
    print(f"{'SIMULATION RESULTS (5 YEARS)':^55}")
    print("=" * 55)
    print(f"Annual Return (CAGR):   {expected_cagr:.2%}")
    print(f"Annual Volatility:      {port_std * np.sqrt(252):.2%}")
    print(f"Sharpe Ratio:           {sharpe:.2f}")
    print(f"95% CI Lower:           ${ci_lower:,.0f}")
    print(f"95% CI Upper:           ${ci_upper:,.0f}")
    print("=" * 55)

    # ==========================================
    # 5. VISUALIZATION
    # ==========================================
    plt.figure(figsize=(10, 6))
    plt.plot(price_paths[:, :20], lw=1)
    plt.plot(price_paths.mean(axis=1), 'r--', label='Mean Path', lw=3)
    plt.title('Monte-Carlo Simulation: Elite Hybrid Portfolio')
    plt.xlabel('Trading Days')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.hist(ending_values, bins=50, edgecolor='black', alpha=0.7)
    plt.title('Distribution of Ending Portfolio Values')
    plt.axvline(np.mean(ending_values), color='r', linestyle='dashed', linewidth=2, label='Mean')
    plt.axvline(ci_lower, color='orange', linestyle='dashed', linewidth=2, label='95% CI')
    plt.axvline(ci_upper, color='orange', linestyle='dashed', linewidth=2)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    run_elite_portfolio_simulation()