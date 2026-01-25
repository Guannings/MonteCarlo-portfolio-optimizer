import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


def run_target_vol_global():
    print("--- Starting Global 'Target Volatility (12%)' Optimization ---")

    # 1. SETUP
    stock_tickers = ['GOOGL', 'JPM', 'PEP', 'UNH', 'ASML', 'LVMUY', 'NVS', 'TTE', 'SIEGY']
    dom_bonds = ['BGRN', 'LQD']
    bond_etfs = ['TLT', 'IEF']
    gold = ['GLD']
    all_tickers = stock_tickers + dom_bonds + bond_etfs + gold

    print(f"Downloading data for {len(all_tickers)} assets...")
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

    # 2. GLOBAL OPTIMIZATION (TARGET VOLATILITY)
    print("Finding the highest return for 12% Risk...")

    cov_matrix = returns.cov()
    mean_returns = returns.mean()
    rf_rate = 0.035

    # TARGET: 12% Annual Volatility
    TARGET_VOL = 0.12

    # Objective: Maximize Return (Minimize Negative Return)
    def get_negative_return(weights):
        port_ret = np.sum(mean_returns * weights) * 252
        return -port_ret

    # Constraint: Volatility must be <= 12%
    def check_volatility(weights):
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        return TARGET_VOL - port_vol  # Positive if safe, negative if risky

    n_assets = len(all_tickers)
    init_guess = [1 / n_assets] * n_assets

    # Constraints dictionary
    cons = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Sum = 100%
        {'type': 'ineq', 'fun': check_volatility}  # Vol <= 12%
    ]

    # Add Guardrails (Flexible but Balanced)
    # Stocks: 40% - 75%
    cons.append({'type': 'ineq', 'fun': lambda x: np.sum(x[:9]) - 0.40})
    cons.append({'type': 'ineq', 'fun': lambda x: 0.75 - np.sum(x[:9])})

    # Bonds: Min 15%
    cons.append({'type': 'ineq', 'fun': lambda x: np.sum(x[9:13]) - 0.15})

    # Gold: 5% - 15%
    cons.append({'type': 'ineq', 'fun': lambda x: np.sum(x[13:]) - 0.05})
    cons.append({'type': 'ineq', 'fun': lambda x: 0.15 - np.sum(x[13:])})

    bounds = tuple((0.00, 0.25) for _ in range(n_assets))

    res = minimize(get_negative_return, init_guess, method='SLSQP', bounds=bounds, constraints=cons, tol=1e-10)
    final_weights = res.x

    if not res.success:
        print("Warning: Optimization struggled to hit exact target. Using closest match.")

    # 3. PRINT WEIGHTS
    portfolio_weights = dict(zip(all_tickers, final_weights))
    print("\n" + "=" * 55)
    print(f"{'TARGET VOLATILITY (12%) PORTFOLIO':^55}")
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
        if w > 0.001:
            print(f"{t:<10} {a_type:<15} {w:>10.2%}      ${total_val * w:,.0f}")

    w_stocks_total = np.sum(final_weights[:9])
    w_bonds_total = np.sum(final_weights[9:13])
    w_gold_total = final_weights[13]

    print("-" * 55)
    print(f"MACRO SPLIT: Stocks {w_stocks_total:.1%} | Bonds {w_bonds_total:.1%} | Gold {w_gold_total:.1%}")
    print("=" * 55)

    # 4. MONTE CARLO
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
    sim_cagrs = (ending_values / total_val) ** (1 / 5) - 1
    mean_cagr = np.mean(sim_cagrs)
    sharpe = (mean_cagr - 0.035) / (port_std * np.sqrt(252))

    ci_lower = np.percentile(ending_values, 2.5)
    ci_upper = np.percentile(ending_values, 97.5)

    print("\n" + "=" * 50)
    print(f"{'PREDICTED PERFORMANCE (5 Years)':^50}")
    print("=" * 50)
    print(f"Mean Annual Return:   {mean_cagr:.2%}")
    print(f"Annual Volatility:    {port_std * np.sqrt(252):.2%}")
    print(f"Sharpe Ratio:         {sharpe:.2f}")
    print("-" * 50)
    print(f"95% CI Lower:         ${ci_lower:,.0f}")
    print(f"95% CI Upper:         ${ci_upper:,.0f}")
    print("=" * 50)

    # 5. VISUALIZATION
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
    plt.title('Target Volatility (12%) Portfolio: 5-Year Outlook')
    plt.xlabel('Trading Days')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.show()

    plt.figure(figsize=(12, 7))
    weights = np.ones_like(sim_cagrs) / len(sim_cagrs)
    plt.hist(sim_cagrs, bins=70, weights=weights, color='#f39c12', edgecolor='white', alpha=0.85)
    plt.axvline(mean_cagr, color='red', linewidth=3, label=f'Mean Return: {mean_cagr:.1%}')
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.title('Return Distribution (Balanced Growth)')
    plt.xlabel('Annual Return (CAGR)')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.show()


if __name__ == "__main__":
    run_target_vol_global()