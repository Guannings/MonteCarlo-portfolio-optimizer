import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


def run_dca_simulation():
    print("--- Starting DCA Simulation (Your Real Situation) ---")

    # 1. YOUR REALITY INPUTS
    initial_capital = 1230  # 40,000 TWD
    monthly_contrib = 185  # 6,000 TWD
    years = 10  # Let's look at 10 years (5 is too short for compounding to explode)

    # Portfolio Stats (From your previous 12% Vol run)
    mu_annual = 0.1472  # 14.72% Return
    vol_annual = 0.1272  # 12.72% Volatility

    # Simulation Settings
    n_sims = 5000
    months = years * 12
    dt = 1 / 12  # Monthly steps

    # 2. RUN SIMULATION
    # drift per month
    mu_monthly = mu_annual / 12
    vol_monthly = vol_annual / np.sqrt(12)

    # Arrays to store path
    # wealth_paths[month, sim]
    wealth_paths = np.zeros((months + 1, n_sims))
    total_invested = np.zeros(months + 1)

    wealth_paths[0] = initial_capital
    total_invested[0] = initial_capital

    # Loop through months
    for t in range(1, months + 1):
        # 1. Update Total Invested (Principal)
        total_invested[t] = total_invested[t - 1] + monthly_contrib

        # 2. Grow Previous Balance
        # Random shock
        Z = np.random.normal(0, 1, n_sims)
        growth_factor = np.exp((mu_monthly - 0.5 * vol_monthly ** 2) + vol_monthly * Z)

        # New Balance = (Old Balance * Growth) + New Contribution
        wealth_paths[t] = (wealth_paths[t - 1] * growth_factor) + monthly_contrib

    # 3. RESULTS
    final_values = wealth_paths[-1]
    avg_final = np.mean(final_values)
    total_principal = total_invested[-1]

    # Percentiles
    worst_case = np.percentile(final_values, 5)  # 5th percentile
    best_case = np.percentile(final_values, 95)  # 95th percentile

    print("=" * 50)
    print(f"{'REALITY CHECK (10 Years)':^50}")
    print("=" * 50)
    print(f"GIVEN RETURN RATE PER YEAR: 14.72%")
    print(f"Total Cash You Saved:    ${total_principal:,.0f}")
    print("-" * 50)
    print(f"Portfolio Value (Avg):   ${avg_final:,.0f}")
    print(f"Total Profit (Avg):      ${avg_final - total_principal:,.0f}")
    print(f"Return on Cost:          {(avg_final / total_principal) - 1:.1%}")
    print("-" * 50)
    print(f"Worst Case (Bad Luck):   ${worst_case:,.0f}")
    print(f"Best Case (Good Luck):   ${best_case:,.0f}")
    print("=" * 50)

    # 4. PLOT
    plt.figure(figsize=(10, 6))

    # Plot Mean Path
    plt.plot(np.mean(wealth_paths, axis=1), color='green', linewidth=3, label='Portfolio Value (Avg)')

    # Plot "Mattress Money" (Just saving cash)
    plt.plot(total_invested, color='gray', linestyle='--', linewidth=2, label='Cash Under Mattress (0% Return)')

    # Plot Worst/Best Cone
    plt.fill_between(range(months + 1),
                     np.percentile(wealth_paths, 5, axis=1),
                     np.percentile(wealth_paths, 95, axis=1),
                     color='green', alpha=0.1, label='95% Outcome Range')

    plt.title(f'The Power of $185/Month (10 Year Projection)', fontsize=14, fontweight='bold')
    plt.xlabel('Months')
    plt.ylabel('Total Value ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    run_dca_simulation()