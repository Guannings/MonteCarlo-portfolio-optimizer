import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


def run_separate_charts_sim():
    print("--- Starting FINAL Simulation (Separate Charts Mode) ---")

    # 1. SETUP & DATA
    tickers = ['SPYM', 'QQQ', 'VEA', 'TLT', 'GLDM']
    print(f"Downloading data for {tickers}...")
    data = yf.download(tickers, start='2020-01-01', end='2026-01-08', auto_adjust=True, progress=False)

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

    # 2. OPTIMIZATION
    cov_matrix = returns.cov()
    mean_returns_series = returns.mean()
    TARGET_VOL = 0.14

    def get_return(weights):
        return -np.sum(mean_returns_series * weights) * 252

    def check_volatility(weights):
        return TARGET_VOL - np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)

    # STRICT BOUNDS (10% Safety Cap)
    bounds = (
        (0.10, 0.60),  # SPLG
        (0.10, 0.60),  # QQQ
        (0.05, 0.35),  # VEA
        (0.02, 0.10),  # TLT
        (0.02, 0.10)  # GLD
    )
    cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}, {'type': 'ineq', 'fun': check_volatility}]
    init_guess = [0.35, 0.35, 0.10, 0.10, 0.10]

    res = minimize(get_return, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
    opt_weights = res.x

    # 3. PRINT SHOPPING LIST
    print("=" * 50)
    print(f"{'YOUR AGGRESSIVE SHOPPING LIST':^50}")
    print("=" * 50)
    ticker_map = dict(zip(tickers, opt_weights))
    sorted_tickers = sorted(ticker_map.items(), key=lambda item: item[1], reverse=True)
    total_budget = 185
    for t, w in sorted_tickers:
        display = "QQQ" if t == "QQQ" else t
        print(f"{display:<6} {w:>8.1%}  |  Buy ${total_budget * w:>6.2f} / month")

    # Optimized Stats
    opt_ret = np.sum(mean_returns_series * opt_weights) * 252
    opt_vol = np.sqrt(np.dot(opt_weights.T, np.dot(cov_matrix, opt_weights))) * np.sqrt(252)
    print("-" * 50)
    print(f"Expected Annual Return:     {opt_ret:.2%}")
    print(f"Expected Annual Volatility: {opt_vol:.2%}")
    print("=" * 50)

    # 4. SIMULATION SETUP
    n_sims = 1000000
    n_years = 5
    n_days = 252 * n_years

    drift = opt_ret - 0.5 * (opt_vol ** 2)
    Z = np.random.normal(0, 1, (n_days, n_sims))
    daily_growth = np.exp(drift / 252 + (opt_vol / np.sqrt(252)) * Z)

    # --- SIM A: LUMP SUM (For Return Distribution) ---
    lump_paths = np.zeros((n_days + 1, n_sims))
    lump_paths[0] = 100000
    for t in range(1, n_days + 1):
        lump_paths[t] = lump_paths[t - 1] * daily_growth[t - 1]

    # Calculate SCALAR Metrics
    ending_lump = lump_paths[-1]
    cagrs = (ending_lump / 100000) ** (1 / n_years) - 1
    mean_cagr = float(np.mean(cagrs))
    worst_cagr = float(np.percentile(cagrs, 5))
    best_cagr = float(np.percentile(cagrs, 95))

    final_weights = res.x
    port_cov = np.dot(final_weights.T, np.dot(returns.cov(), final_weights))
    port_std = np.sqrt(port_cov)
    sharpe = (mean_cagr - 0.035) / (port_std * np.sqrt(252))
    # --- SIM B: DCA REALITY (For Wealth Spaghetti) ---
    dca_paths = np.zeros((n_days + 1, n_sims))
    initial_capital = 1230
    dca_paths[0] = initial_capital
    for t in range(1, n_days + 1):
        dca_paths[t] = dca_paths[t - 1] * daily_growth[t - 1]
        if t % 21 == 0:
            dca_paths[t] += total_budget

    ending_dca = dca_paths[-1]
    total_invested = initial_capital + (total_budget * 12 * n_years)
    avg_dca = float(np.mean(ending_dca))

    print("\n" + "=" * 50)
    print(f"{'5-YEAR FORECAST':^50}")
    print("=" * 50)
    print(f"Avg DCA Balance:          ${avg_dca:,.0f}")
    print(f"Total Invested:           ${total_invested:,.0f}")
    print(f"Net Profit:               ${avg_dca - total_invested:,.0f}")
    print("-" * 50)
    print(f"Mean Annual Return:       {mean_cagr:.2%}")
    print(f"Sharpe Ratio:             {sharpe:.2f}")
    print(f"95% Worst Case Return:    {worst_cagr:.2%}")
    print(f"95% Best Case Return:     {best_cagr:.2%}")
    print("=" * 50)

    # 5. VISUALIZATION (SEPARATE PLOTS)

    # --- CHART 1: SPAGHETTI PLOT (WEALTH) ---
    plt.figure(figsize=(10, 6))  # Create Figure 1
    plt.plot(dca_paths[:, :200], color='#1BA0C9', alpha=0.1)
    plt.plot(np.mean(dca_paths, axis=1), color='red', linewidth=3, label='Average Path')

    cash_line = np.linspace(initial_capital, total_invested, n_days + 1)
    plt.plot(cash_line, color='black', linestyle='--', linewidth=2, label='Cash Savings')

    plt.title('DCA Wealth Growth ($185/mo)', fontsize=14, fontweight='bold')
    plt.ylabel('Portfolio Value ($)')
    plt.xlabel('Trading Days')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()  # Show Chart 1 immediately

    # --- CHART 2: DISTRIBUTION PLOT (RETURN) ---
    plt.figure(figsize=(10, 6))  # Create Figure 2
    weights = np.ones_like(cagrs) / len(cagrs)
    plt.hist(cagrs, bins=100, weights=weights, color='#198964', edgecolor='white', alpha=0.85)

    # Vertical Lines
    plt.axvline(mean_cagr, color='red', linewidth=3, label=f'Mean Return: {mean_cagr:.1%}')
    plt.axvline(worst_cagr, color='blue', linestyle='--', linewidth=2, label=f'95% Worst Case: {worst_cagr:.1%}')
    plt.axvline(best_cagr, color='blue', linestyle='--', linewidth=2, label=f'95% Best Case: {best_cagr:.1%}')

    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    plt.title('Distribution of Annualized Returns (CAGR)', fontsize=14, fontweight='bold')
    plt.xlabel('Annual Return (CAGR)')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()  # Show Chart 2 immediately


if __name__ == "__main__":
    run_separate_charts_sim()