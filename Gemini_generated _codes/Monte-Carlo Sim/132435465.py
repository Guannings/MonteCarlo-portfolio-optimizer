import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


def run_defensive_replica_sim():
    print("--- Starting 'Defensive Barbell' Simulation (60% SCHD / 30% QQQM / 10% LVMUY) ---")

    # 1. SETUP
    # Using QQQ as proxy for QQQM
    tickers = ['SPLG', 'TLT', 'GLDM']
    weights = np.array([0.75, 0.15, 0.1])

    print(f"Downloading data for {tickers}...")
    data = yf.download(tickers, start='2021-01-01', end='2026-01-01', auto_adjust=True, progress=False)

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

    # 2. CALCULATE PORTFOLIO STATS
    cov_matrix = returns.cov()
    mean_returns = returns.mean()
    rf_rate = 0.035

    port_ret = np.sum(mean_returns * weights) * 252
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    sharpe = (port_ret - rf_rate) / port_vol

    print("=" * 50)
    print(f"{'HISTORICAL METRICS (2021/1/1-2026/1/1)':^50}")
    print("=" * 50)
    print(f"Annual Return:       {port_ret:.2%}")
    print(f"Annual Volatility:   {port_vol:.2%} ")
    print(f"Sharpe Ratio:        {sharpe:.2f}")
    print("=" * 50)

    # 3. MONTE CARLO
    n_sims = 1000000
    n_days = 252 * 5
    initial_capital = 100000

    drift = port_ret - 0.5 * (port_vol ** 2)
    Z = np.random.normal(0, 1, (n_days, n_sims))
    daily_growth = np.exp(drift / 252 + (port_vol / np.sqrt(252)) * Z)

    price_paths = np.zeros((n_days + 1, n_sims))
    price_paths[0] = initial_capital
    for t in range(1, n_days + 1):
        price_paths[t] = price_paths[t - 1] * daily_growth[t - 1]

    ending_values = price_paths[-1]

    # 4. RESULTS
    sim_cagrs = (ending_values / initial_capital) ** (1 / 5) - 1
    mean_cagr = np.mean(sim_cagrs)

    ci_lower_val = np.percentile(ending_values, 2.5)
    ci_lower_ret = np.percentile(sim_cagrs, 2.5)
    ci_upper_val = np.percentile(ending_values, 97.5)
    ci_upper_ret = np.percentile(sim_cagrs, 97.5)
    print("\n" + "=" * 50)
    print(f"{'PROJECTED FUTURE PERFORMANCE (5 Years)':^50}")
    print("=" * 50)
    print(f"Mean CAGR (Return):   {mean_cagr:.2%}")
    print(f"Worst Case (95% CI):  ${ci_lower_val:,.0f} (CAGR: {ci_lower_ret:.2%})")
    print(f"Best Case (95% CI): ${ci_upper_val:,.0f} (CAGR: {ci_upper_ret:.2%})")
    print("=" * 50)

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
    run_defensive_replica_sim()