import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


def plot_simulation_matched_frontier():
    print("--- Generating Efficient Frontier (Matched to Simulation) ---")

    # 1. SETUP
    stock_tickers = ['GOOGL', 'JPM', 'PEP', 'UNH', 'ASML', 'LVMUY', 'NVS', 'TTE', 'SIEGY']
    dom_bonds = ['BGRN', 'LQD']
    bond_etfs = ['TLT', 'IEF']
    gold = ['GLD']
    all_tickers = stock_tickers + dom_bonds + bond_etfs + gold

    # YOUR EXACT WEIGHTS (From the Diversified Target Vol Run)
    my_weights = np.array([
        0.0300, 0.1022, 0.2500, 0.1444, 0.0300, 0.0653, 0.0300, 0.0300, 0.0682,  # Stocks
        0.0245, 0.0374,  # Dom Bonds
        0.1253, 0.0128,  # ETF Bonds
        0.0500  # Gold
    ])

    print("Downloading data...")
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

    cov_matrix = returns.cov()
    mean_returns = returns.mean()
    rf_rate = 0.035

    # 2. GENERATE FRONTIER (ARITHMETIC MEAN - MATCHES SIMULATION AVERAGE)
    print("Calculating Standard Efficient Frontier...")

    def get_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)

    def get_return(weights):
        # Arithmetic Mean (Matches the 'Mean' of Monte Carlo)
        return -np.sum(mean_returns * weights) * 252

    n_assets = len(all_tickers)
    bounds = tuple((0.0, 1.0) for _ in range(n_assets))
    target_vols = np.linspace(0.05, 0.25, 50)

    frontier_vol = []
    frontier_ret = []

    for tv in target_vols:
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: get_volatility(x) - tv})
        res = minimize(get_return, [1 / n_assets] * n_assets, method='SLSQP', bounds=bounds, constraints=cons)

        if res.success:
            frontier_vol.append(tv)
            frontier_ret.append(-res.fun)

    # 3. CALCULATE YOUR PORTFOLIO STATS
    my_ret = np.sum(mean_returns * my_weights) * 252
    my_vol = np.sqrt(np.dot(my_weights.T, np.dot(cov_matrix, my_weights))) * np.sqrt(252)
    my_sharpe = (my_ret - rf_rate) / my_vol

    print(f"\nYOUR PORTFOLIO STATS (Matched):\nReturn: {my_ret:.2%}\nVolatility: {my_vol:.2%}\nSharpe: {my_sharpe:.2f}")

    # 4. FIND TANGENCY PORTFOLIO (MAX SHARPE)
    # This point defines the angle of the Capital Market Line
    max_sharpe_idx = np.argmax([(r - rf_rate) / v for r, v in zip(frontier_ret, frontier_vol)])
    tan_vol = frontier_vol[max_sharpe_idx]
    tan_ret = frontier_ret[max_sharpe_idx]
    tan_sharpe = (tan_ret - rf_rate) / tan_vol

    # 5. PLOT
    plt.figure(figsize=(12, 8))

    # A. Frontier Curve
    plt.plot(frontier_vol, frontier_ret, 'b-', linewidth=3, label='Efficient Frontier')

    # B. Capital Market Line (CML)
    # y = rf + Sharpe * x
    cml_x = np.linspace(0, 0.25, 100)
    cml_y = rf_rate + tan_sharpe * cml_x
    plt.plot(cml_x, cml_y, 'r--', linewidth=2, label='Capital Market Line (Optimal Trade-off)')

    # C. Points
    plt.scatter(tan_vol, tan_ret, color='green', s=150, zorder=10,
                label=f'Max Sharpe Portfolio (Ratio: {tan_sharpe:.2f})')

    # Your Portfolio
    plt.scatter(my_vol, my_ret, color='gold', s=350, marker='*', edgecolors='black', zorder=15,
                label=f'Your Portfolio (Sharpe: {my_sharpe:.2f})')

    # Risk Free
    plt.scatter(0, rf_rate, color='black', s=100, marker='s', zorder=10, label='Risk-Free Rate (3.5%)')

    plt.title('Modern Portfolio Theory: Your Portfolio vs. The Optimal Line', fontsize=14, fontweight='bold')
    plt.xlabel('Annual Volatility (Risk)')
    plt.ylabel('Expected Annual Return (Arithmetic)')
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.legend(loc='upper left', frameon=True, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 0.25)
    plt.ylim(0, 0.25)
    plt.show()


if __name__ == "__main__":
    plot_simulation_matched_frontier()