import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


def run_monte_carlo():
    # --- Step 1: Define Parameters (UPDATED) ---
    # We added 'TLT' (US Treasury Bonds) to the list
    tickers = ['^GSPC', '^STOXX50E', 'TLT']

    # We adjusted weights to be 40% US Stocks, 20% EU Stocks, 40% Bonds
    # This 60/40 Stock/Bond split is a classic "balanced" strategy.
    weights = np.array([0.40, 0.20, 0.40])

    start_date = '2015-01-01'
    end_date = '2025-01-01'

    initial_portfolio_value = 100000

    # Explicit casting for type safety
    num_simulations = int(5000)
    time_horizon_years = int(5)
    trading_days_per_year = int(252)
    total_steps = int(time_horizon_years * trading_days_per_year)

    # --- Step 2: Get & Clean Data ---
    print(f"Downloading data for {tickers}...")

    # auto_adjust=True gives us the total return (price + dividends)
    raw_data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)

    # Robust data extraction
    try:
        prices = raw_data['Close']
    except KeyError:
        prices = raw_data

    # Drop missing data
    prices = prices.dropna()
    print("Data download successful.")

    # Calculate logarithmic returns
    log_returns = np.log(prices / prices.shift(1)).dropna()

    # Calculate historical Mean (Drift) and Covariance Matrix
    avg_daily_return = log_returns.mean()
    cov_matrix = log_returns.cov()

    # --- Step 3: Run Monte-Carlo Simulation ---
    print(f"Running {num_simulations} simulations...")

    # Calculate Portfolio Expected Return and Volatility
    port_daily_return = np.sum(weights * avg_daily_return)
    port_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    port_std_dev = np.sqrt(port_variance)

    # Pre-compute Drift
    drift = (port_daily_return - 0.5 * port_variance)

    # Generate random Z-scores
    Z = np.random.normal(0.0, 1.0, (total_steps, num_simulations))

    # Calculate Daily Returns
    daily_returns = np.exp(drift + port_std_dev * Z)

    # Accumulate returns
    price_paths = np.zeros_like(daily_returns)
    price_paths[0] = initial_portfolio_value

    for t in range(1, total_steps):
        price_paths[t] = price_paths[t - 1] * daily_returns[t]

    # --- Step 4: Analyze Results ---
    ending_values = price_paths[-1, :]
    mean_ending_value = np.mean(ending_values)

    ci_lower = np.percentile(ending_values, 2.5)
    ci_upper = np.percentile(ending_values, 97.5)

    var_95_threshold = np.percentile(ending_values, 5)
    var_amount = initial_portfolio_value - var_95_threshold

    # If we made money in the bottom 5%, VaR is effectively 0 (no loss)
    if var_amount < 0:
        var_amount = 0

    print(f"\n--- Simulation Results ({time_horizon_years} Years) ---")
    print(f"Portfolio Mix:      40% US Stocks, 20% EU Stocks, 40% Bonds (TLT)")
    print(f"Initial Investment:  ${initial_portfolio_value:,.2f}")
    print(f"Expected Mean Value: ${mean_ending_value:,.2f}")
    print(f"95% Conf. Interval:  ${ci_lower:,.2f} — ${ci_upper:,.2f}")
    print(f"Value at Risk (5%):  ${var_amount:,.2f}")

    # --- Step 5: Visualize ---
    # Chart 1: Price Paths
    plt.figure(figsize=(10, 6))
    plt.plot(price_paths[:, :20], lw=1)
    plt.plot(price_paths.mean(axis=1), 'r--', label='Mean Path', lw=3)
    plt.title(f'Monte-Carlo: Diversified Portfolio (Stocks + Bonds)')
    plt.xlabel('Trading Days')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('monte_carlo_safe_paths.png', dpi=300)
    plt.show()

    # Chart 2: Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(ending_values, bins=50, edgecolor='black', alpha=0.7)
    plt.title('Distribution of Ending Portfolio Values (With Bonds)')
    plt.axvline(initial_portfolio_value, color='k', linestyle='-', linewidth=1, label='Initial $100k')
    plt.axvline(mean_ending_value, color='r', linestyle='dashed', linewidth=2, label=f'Mean: ${mean_ending_value:,.0f}')
    plt.axvline(ci_lower, color='orange', linestyle='dashed', linewidth=2, label=f'95% CI Lower: ${ci_lower:,.0f}')
    plt.axvline(ci_upper, color='orange', linestyle='dashed', linewidth=2)
    plt.legend()
    plt.savefig('monte_carlo_safe_dist.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    run_monte_carlo()