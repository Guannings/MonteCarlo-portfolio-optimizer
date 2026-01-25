# ⚠️ HARDWARE WARNING: 
  **HIGH COMPUTATIONAL LOAD**

**PLEASE READ BEFORE RUNNING:**
This script is currently configured to run **1,000,000 (1 Million)** Monte Carlo simulations. This is an extreme stress test intended for high-performance workstations.

* **System Requirements:** Minimum **32GB RAM** and a multi-core processor (e.g., Ryzen 7 / Core i7 or better).
* **Risk:** Running this on a standard office laptop or non-gaming PC (8GB/16GB RAM) will likely cause a **Memory Overflow (OOM)**, resulting in a system freeze or crash.

**Recommendation for Standard Users:**
Before running the script, open `portfolio_optimizer.py` and **find the configuration line: NUM_SIMULATIONS = 1000000** and **change it to 10000 or a smaller number of your choice.**

==============================================================================================

# ⚠️ Disclaimer and Terms of Use
**1. Educational Purpose Only**

This software is for educational and research purposes only. It is not intended to be a source of financial advice, and the authors are not registered financial advisors. The algorithms, simulations, and optimization techniques implemented herein are demonstrations of theoretical concepts (Modern Portfolio Theory, Geometric Brownian Motion) and should not be construed as a recommendation to buy, sell, or hold any specific security or asset class.

**2. No Financial Advice**

Nothing in this repository constitutes professional financial, legal, or tax advice. Investment decisions should be made based on your own research and consultation with a qualified financial professional. The strategies modeled in this software may not be suitable for your specific financial situation, risk tolerance, or investment goals.

**3. Risk of Loss**

All investments involve risk, including the possible loss of principal.

Past Performance: Historical returns and volatility data used in these simulations are not indicative of future results.

Simulation Limitations: Monte Carlo simulations are probabilistic models based on assumptions that may not reflect real-world market conditions, black swan events, or liquidity crises.

Market Data: Data fetched from third-party APIs (e.g., Yahoo Finance) may be delayed, inaccurate, or incomplete.

**4. "As-Is" Software Warranty**

This software is provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and non-infringement. In no event shall the authors or copyright holders be liable for any claim, damages, or other liability, whether in an action of contract, tort, or otherwise, arising from, out of, or in connection with the software or the use or other dealings in the software.

By using this software, you agree to assume all risks associated with your investment decisions and release the author from any liability regarding your financial outcomes.
