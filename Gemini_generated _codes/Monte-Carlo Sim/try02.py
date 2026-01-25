import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


def run_monte_carlo():
    # --- Step 1: Define Parameters ---
    tickers = ['^GSPC', '^STOXX50E']
    weights = np.array([0.6, 0.4])

    start_date = '2015-01-01'
    end_date = '2025-01-01'

    initial_portfolio_value = 100000

    # Explicitly cast to int for strict type checkers
    num_simulations = int(5000)
    time_horizon_years = int(5)
    trading_days_per_year = int(252)
    total_steps = int(time_horizon_years * trading_days_per_year)

    # --- Step 2: Get & Clean Data (Simplified) ---
    print(f"Downloading data for {tickers}...")

    # Auto_adjust=True puts the adjusted price into the 'Close' column
    raw_data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)

    # ROBUST DATA EXTRACTION:
    # Instead of checking levels, we just try to grab 'Close'.
    # If it's a MultiIndex, raw_data['Close'] returns the dataframe of tickers.
    try:
        prices = raw_data['Close']
    except KeyError:
        # If 'Close' isn't a top-level key, the data might be flat or formatted differently.
        # We assume the dataframe itself contains the price data.
        prices = raw_data

    # Ensure we drop missing data
    prices = prices.dropna()
    print("Data download successful.")

    # Calculate logarithmic returns
    # Using specific numpy float type to avoid type warnings
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
    # Using 0.0 and 1.0 (floats) to satisfy type checker
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
    if var_amount < 0: var_amount = 0

    print(f"\n--- Simulation Results ({time_horizon_years} Years) ---")
    print(f"Initial Investment:  ${initial_portfolio_value:,.2f}")
    print(f"Expected Mean Value: ${mean_ending_value:,.2f}")
    print(f"95% Conf. Interval:  ${ci_lower:,.2f} — ${ci_upper:,.2f}")
    print(f"Value at Risk (5%):  ${var_amount:,.2f}")

    # --- Step 5: Visualize ---
    plt.figure(figsize=(10, 6))
    plt.plot(price_paths[:, :20], lw=1)
    plt.plot(price_paths.mean(axis=1), 'r--', label='Mean Path', lw=3)
    plt.title(f'Monte-Carlo Simulation: US/EU Portfolio ({time_horizon_years} Years)')
    plt.xlabel('Trading Days')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.hist(ending_values, bins=50, edgecolor='black', alpha=0.7)
    plt.title('Distribution of Ending Portfolio Values')
    plt.axvline(mean_ending_value, color='r', linestyle='dashed', linewidth=2, label='Mean')
    plt.axvline(ci_lower, color='orange', linestyle='dashed', linewidth=2, label='95% CI')
    plt.axvline(ci_upper, color='orange', linestyle='dashed', linewidth=2)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    run_monte_carlo()