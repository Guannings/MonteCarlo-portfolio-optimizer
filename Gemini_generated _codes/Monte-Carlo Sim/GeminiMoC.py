import numpy as np
import yfinance as yf
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def run_elite_portfolio_simulation():
    # ==========================================
    # 1. SETUP & DATA DOWNLOAD
    # ==========================================
    print("--- Starting Portfolio Simulation ---")

    # Define Assets by Bucket
    stock_tickers = ['GOOGL', 'JPM', 'PEP', 'UNH', 'ASML', 'LVMUY', 'NVS', 'TTE', 'SIEGY']
    dom_bonds = ['BGRN', 'LQD']  # 17.5% Allocation Bucket
    bond_etfs = ['TLT', 'IEF']  # 7.5% Allocation Bucket
    gold = ['GLD']  # 5% Allocation Bucket

    all_tickers = stock_tickers + dom_bonds + bond_etfs + gold

    print(f"Downloading data for {len(all_tickers)} assets...")
    start_date = '2020-01-01'
    end_date = '2025-01-01'

    # Download data
    data = yf.download(all_tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)

    try:
        prices = data['Close']
    except KeyError:
        prices = data

    prices = prices.dropna()
    returns = prices.pct_change().dropna()
    print("Data download complete.")

    # ==========================================
    # 2. OPTIMIZATION (STOCKS ONLY)
    # ==========================================
    print("\nOptimizing 'Elite Stocks' for Minimum Variance...")

    stock_ret = returns[stock_tickers]
    stock_cov = stock_ret.cov()

    # Minimize Annualized Volatility
    def get_portfolio_volatility(weights):
        daily_var = np.dot(weights.T, np.dot(stock_cov, weights))
        return np.sqrt(daily_var) * np.sqrt(252)

    n_stocks = len(stock_tickers)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0.02, 1.0) for _ in range(n_stocks))
    init_guess = [1 / n_stocks] * n_stocks

    res = minimize(get_portfolio_volatility, init_guess, method='SLSQP', bounds=bounds, constraints=constraints,
                   tol=1e-10)
    opt_stock_weights = res.x

    # ==========================================
    # 3. BUILD FINAL PORTFOLIO
    # ==========================================
    # Scale stocks to 70%
    final_stock_w = opt_stock_weights * 0.70

    # Assign Fixed Weights to others
    w_dom = 0.175 / 2
    w_etf = 0.075 / 2
    w_gold = 0.05

    final_dom_w = np.array([w_dom, w_dom])
    final_etf_w = np.array([w_etf, w_etf])
    final_gold_w = np.array([w_gold])

    final_weights = np.concatenate([final_stock_w, final_dom_w, final_etf_w, final_gold_w])
    portfolio_weights = dict(zip(all_tickers, final_weights))

    print("\n" + "=" * 40)
    print(f"{'FINAL PORTFOLIO COMPOSITION':^40}")
    print("=" * 40)
    for t in all_tickers:
        print(f"{t:<8} {portfolio_weights[t]:>7.2%}")

    # ==========================================
    # 4. MONTE CARLO SIMULATION
    # ==========================================
    print("\nRunning 1000000 Simulations...")

    n_sims = 1000000
    n_years = 5
    n_days = 252 * n_years
    initial_capital = 100000

    port_ret = np.dot(final_weights, returns.mean())
    port_cov = np.dot(final_weights.T, np.dot(returns.cov(), final_weights))
    port_std = np.sqrt(port_cov)

    drift = port_ret - 0.5 * port_cov
    Z = np.random.normal(0, 1, (n_days, n_sims))
    daily_growth = np.exp(drift + port_std * Z)

    price_paths = np.zeros((n_days + 1, n_sims))
    price_paths[0] = initial_capital

    for t in range(1, n_days + 1):
        price_paths[t] = price_paths[t - 1] * daily_growth[t - 1]

    # Metrics
    ending_values = price_paths[-1]
    expected_cagr = (np.mean(ending_values) / initial_capital) ** (1 / n_years) - 1
    if port_std > 0:
        sharpe = (expected_cagr - 0.035) / (port_std * np.sqrt(252))
    else:
        sharpe = 0

    var_95 = initial_capital - np.percentile(ending_values, 5)

    print("-" * 40)
    print(f"Expected Annual Return: {expected_cagr:.2%}")
    print(f"Sharpe Ratio:           {sharpe:.2f}")
    print(f"Value at Risk (5%):     ${var_95:,.2f}")
    print("-" * 40)

    # ==========================================
    # 5. VISUALIZATION (YOUR REQUESTED STYLE)
    # ==========================================

    # Prepare variables to match your code snippet
    time_horizon_years = n_years
    mean_ending_value = np.mean(ending_values)
    ci_lower = np.percentile(ending_values, 2.5)
    ci_upper = np.percentile(ending_values, 97.5)

    # --- YOUR PLOTS ---
    plt.figure(figsize=(10, 6))
    plt.plot(price_paths[:, :50], lw=1)
    plt.plot(price_paths.mean(axis=1), 'r--', label='Mean Path', lw=3)
    plt.title(f'Monte-Carlo Simulation: US/EU Portfolio ({time_horizon_years} Years)')
    plt.xlabel('Trading Days')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('simple_paths.png')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.hist(ending_values, bins=50, edgecolor='black', alpha=0.7)
    plt.title('Distribution of Ending Portfolio Values')
    plt.axvline(mean_ending_value, color='r', linestyle='dashed', linewidth=2, label='Mean')
    plt.axvline(ci_lower, color='orange', linestyle='dashed', linewidth=2, label='95% CI')
    plt.axvline(ci_upper, color='orange', linestyle='dashed', linewidth=2)
    plt.legend()
    plt.savefig('simple_distribution.png')
    plt.show()


if __name__ == "__main__":
    run_elite_portfolio_simulation()