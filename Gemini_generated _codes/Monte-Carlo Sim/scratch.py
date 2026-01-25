import numpy as np
from scipy.stats import norm

# Define inputs
initial_investment = 10000
cash_flows = [2000, 3000, 4000]
discount_rate = 0.05
num_scenarios = 10000

# Define parameters for probability distribution (normal)
mean = 3000  # Replace with actual mean value
sigma = 500  # Replace with actual standard deviation value

probability_distribution = norm(loc=mean, scale=sigma)  # Replace with actual parameters

# Generate random scenarios
np.random.seed(42)
scenarios = np.random.normal(size=(num_scenarios, len(cash_flows)), loc=cash_flows, scale=sigma)

# Compute NPV for each scenario
npvs = []
for scenario in scenarios:
    npv = -initial_investment + sum(scenario / (1 + discount_rate)**(np.arange(len(scenario)) + 1))
    npvs.append(npv)

# Aggregate results
expected_npv = np.mean(npvs)
std_dev = np.std(npvs)

print(f"Expected NPV: {expected_npv:.2f}")
print(f"Standard Deviation of NPV: {std_dev:.2f}")

import matplotlib.pyplot as plt

plt.hist(npvs, bins=50, density=True)
plt.xlabel("NPV")
plt.ylabel("Probability Density")
plt.title("Histogram of NPV Scenarios")
plt.show()
