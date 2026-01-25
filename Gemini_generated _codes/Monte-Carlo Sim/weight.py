import numpy as np
import yfinance as yf
from scipy.optimize import minimize


def optimize_weights():
    # --- 1. Define Your "Elite" List ---
    # US Stocks: GOOGL, JPM, PEP, UNH
    # EU Stocks (ADRs in USD): ASML, LVMUY (LVMH), NVS (Novartis), TTE (Total), SIEGY (Siemens)
    tickers = ['GOOGL', 'JPM', 'PEP', 'UNH', 'ASML', 'LVMUY', 'NVS', 'TTE', 'SIEGY']

    start_date = '2020-01-01'
    end_date = '2026-01-01'

    print(f"Downloading data for {len(tickers)} stocks...")
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)['Close']

    # Calculate daily returns and covariance
    returns = data.pct_change().dropna()
    cov_matrix = returns.cov()

    # --- 2. Define the "Minimum Variance" Function ---
    def calculate_volatility(weights):
        # Formula: Variance = Weights Transposed * Covariance Matrix * Weights
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        return np.sqrt(portfolio_variance)  # We minimize Standard Deviation

    # --- 3. Set Constraints ---
    num_assets = len(tickers)
    init_guess = [1 / num_assets] * num_assets  # Start with equal weights

    # Constraint 1: Weights must sum to 100%
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    # Constraint 2: "The Floor Rule" from the PDF (Page 40)
    # Every stock must have at least 2% weight (0.02), max 100% (1.0)
    bounds = tuple((0.02, 1.0) for _ in range(num_assets))

    # --- 4. Run Optimization ---
    print("Optimizing for Minimum Variance...")
    result = minimize(calculate_volatility, init_guess,
                      method='SLSQP', bounds=bounds, constraints=constraints)

    # --- 5. Output Results ---
    optimal_weights = result.x

    print("\n--- Optimized 'Minimum Variance' Weights ---")
    print(f"{'Ticker':<8} {'Weight':<10} {'Value ($10k Investment)'}")
    print("-" * 40)

    for ticker, weight in zip(tickers, optimal_weights):
        print(f"{ticker:<8} {weight:>7.1%}      ${10000 * weight:,.0f}")

    print("-" * 40)
    print(f"Expected Annual Volatility: {result.fun * np.sqrt(252):.1%}")


if __name__ == "__main__":
    optimize_weights()