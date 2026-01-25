import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


def run_dca_comparison():
    print("--- Starting DCA 'Pain vs. Gain' Comparison ---")

    # INPUTS
    initial_capital = 1230  # 40,000 TWD
    monthly_contrib = 185  # 6,000 TWD
    years = 10
    n_sims = 1000000
    months = years * 12

    # 1. THE "EASY" REPLICA (SPLG/TLT/GLDM)
    mu_easy = 0.1472
    vol_easy = 0.1272

    # 2. THE "HARD" SUPER-PORTFOLIO (20 Assets)
    mu_hard = 0.1719
    vol_hard = 0.1400

    # SIMULATION FUNCTION
    def simulate_wealth(mu, vol):
        mu_monthly = mu / 12
        vol_monthly = vol / np.sqrt(12)
        paths = np.zeros((months + 1, n_sims))
        paths[0] = initial_capital

        for t in range(1, months + 1):
            Z = np.random.normal(0, 1, n_sims)
            growth = np.exp((mu_monthly - 0.5 * vol_monthly ** 2) + vol_monthly * Z)
            paths[t] = (paths[t - 1] * growth) + monthly_contrib
        return paths

    print("Simulating Easy Path...")
    paths_easy = simulate_wealth(mu_easy, vol_easy)

    print("Simulating Hard Path...")
    paths_hard = simulate_wealth(mu_hard, vol_hard)

    # RESULTS
    avg_easy = np.mean(paths_easy[-1])
    avg_hard = np.mean(paths_hard[-1])
    total_invested = initial_capital + (monthly_contrib * months)

    print("=" * 50)
    print(f"{'THE 10-YEAR PAYOFF':^50}")
    print("=" * 50)
    print(f"Total Cash Invested:       ${total_invested:,.0f}")
    print("-" * 50)
    print(f"1. EASY PORTFOLIO (3 ETFs)")
    print(f"   Final Value:            ${avg_easy:,.0f}")
    print(f"   Profit:                 ${avg_easy - total_invested:,.0f}")
    print("-" * 50)
    print(f"2. HARD PORTFOLIO (20 Assets)")
    print(f"   Final Value:            ${avg_hard:,.0f}")
    print(f"   Profit:                 ${avg_hard - total_invested:,.0f}")
    print("-" * 50)
    print(f"THE 'SWEAT EQUITY' BONUS:  +${avg_hard - avg_easy:,.0f}")
    print("=" * 50)

    # PLOT
    plt.figure(figsize=(10, 6))
    plt.plot(np.mean(paths_hard, axis=1), color='#e74c3c', linewidth=3, label='Super-Portfolio (20 Assets)')
    plt.plot(np.mean(paths_easy, axis=1), color='#2ecc71', linewidth=3, label='Easy Replica (3 ETFs)')
    plt.plot(np.linspace(initial_capital, total_invested, months + 1), color='gray', linestyle='--',
             label='Cash Savings')

    plt.fill_between(range(months + 1), np.mean(paths_easy, axis=1), np.mean(paths_hard, axis=1), color='gray',
                     alpha=0.1)

    plt.title('Is the Hard Work Worth It? (10 Year Projection)', fontsize=14, fontweight='bold')
    plt.xlabel('Months')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    run_dca_comparison()