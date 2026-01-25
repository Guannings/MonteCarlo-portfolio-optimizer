import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


def plot_final_truth_chart():
    print("--- Starting Final 'Truth' Comparison ---")

    # 1. SETUP ASSETS
    # Your 20-Asset Super Portfolio
    my_assets = ['GOOGL', 'JPM', 'PEP', 'UNH', 'ASML', 'LVMUY', 'NVS', 'TTE', 'SIEGY',
                 'ACN', 'COST', 'MCD', 'BRK-B', 'SONY', 'ITOCY',
                 'BGRN', 'LQD', 'TLT', 'IEF', 'GLD']

    # The Benchmark
    benchmark = ['SPY']
    all_tickers = my_assets + benchmark

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

    # 2. DEFINE PLAYERS
    rf_rate = 0.035

    # PLAYER A: YOU (Hardcoded from your Simulation)
    # We trust the simulation result because it captures the rebalancing premium
    my_ret = 0.1719
    my_vol = 0.1400
    my_sharpe = 0.98
    print(f"YOUR PORTFOLIO (Simulated): Return {my_ret:.2%} | Vol {my_vol:.2%} | Sharpe {my_sharpe:.2f}")

    # PLAYER B: THE MARKET (Calculated from Data)
    spy_ret = returns['SPY'].mean() * 252
    spy_vol = returns['SPY'].std() * np.sqrt(252)
    spy_sharpe = (spy_ret - rf_rate) / spy_vol
    print(f"MARKET (SPY):               Return {spy_ret:.2%} | Vol {spy_vol:.2%} | Sharpe {spy_sharpe:.2f}")

    # 3. GENERATE EFFICIENT FRONTIER (The Context)
    print("Calculating Efficient Frontier Curve...")
    asset_returns = returns[my_assets]
    cov_matrix = asset_returns.cov()
    mean_returns = asset_returns.mean()
    n_assets = len(my_assets)

    def get_vol(w):
        return np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))) * np.sqrt(252)

    def get_ret(w):
        return -np.sum(mean_returns * w) * 252

    bounds = tuple((0.0, 1.0) for _ in range(n_assets))
    target_vols = np.linspace(0.05, 0.25, 50)
    frontier_rets = []
    frontier_vols = []

    for tv in target_vols:
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: get_vol(x) - tv})
        res = minimize(get_ret, [1 / n_assets] * n_assets, bounds=bounds, constraints=cons, method='SLSQP')
        if res.success:
            frontier_rets.append(-res.fun)
            frontier_vols.append(tv)

    # 4. CALCULATE CML (Tangency Line)
    # Find the theoretical Max Sharpe point on the curve
    max_idx = np.argmax([(r - rf_rate) / v for r, v in zip(frontier_rets, frontier_vols)])
    tan_ret = frontier_rets[max_idx]
    tan_vol = frontier_vols[max_idx]
    tan_sharpe = (tan_ret - rf_rate) / tan_vol

    cml_x = np.linspace(0, 0.25, 100)
    cml_y = rf_rate + tan_sharpe * cml_x

    # 5. PLOT
    plt.figure(figsize=(12, 8))

    # A. The Arena (Frontier & CML)
    plt.plot(frontier_vols, frontier_rets, 'b-', linewidth=2, alpha=0.5, label='Efficient Frontier (20 Assets)')
    plt.plot(cml_x, cml_y, 'r--', linewidth=2, label='Capital Market Line (Optimal)')

    # B. The Players
    # YOU (Gold Star)
    plt.scatter(my_vol, my_ret, color='gold', s=500, marker='*', edgecolors='black', zorder=15,
                label=f'You (Simulated)\nSharpe: {my_sharpe:.2f}')

    # MARKET (Purple Square)
    plt.scatter(spy_vol, spy_ret, color='purple', s=150, marker='s', edgecolors='black', zorder=10,
                label=f'S&P 500 (Actual)\nSharpe: {spy_sharpe:.2f}')

    # Tangency Point (Green Dot)
    plt.scatter(tan_vol, tan_ret, color='green', s=100, zorder=10, label='Tangency Portfolio')

    # C. Formatting
    plt.title('Final Verdict: Your Super-Portfolio vs. The Market', fontsize=14, fontweight='bold')
    plt.xlabel('Annual Volatility (Risk)')
    plt.ylabel('Annual Return')

    # Annotate the Winner
    if my_sharpe > spy_sharpe:
        plt.annotate('WINNER\n(Higher Efficiency)',
                     xy=(my_vol, my_ret), xytext=(my_vol - 0.04, my_ret + 0.02),
                     arrowprops=dict(facecolor='black', shrink=0.05), fontsize=11, fontweight='bold')

    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.legend(loc='upper left', frameon=True, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 0.25)
    plt.ylim(0, 0.30)
    plt.show()


if __name__ == "__main__":
    plot_final_truth_chart()