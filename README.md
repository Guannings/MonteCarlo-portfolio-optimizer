# ⚠️ HARDWARE WARNING: 
  **HIGH COMPUTATIONAL LOAD**

**PLEASE READ BEFORE RUNNING:**
This script is currently configured to run **1,000,000 (1 Million)** Monte Carlo simulations. This is an extreme stress test intended for high-performance workstations.

* **System Requirements:** Minimum **32GB RAM** and a multi-core processor (e.g., Ryzen 7 / Core i7 or better).
* **Risk:** Running this on a standard office laptop or non-gaming PC (8GB/16GB RAM) will likely cause a **Memory Overflow (OOM)**, resulting in a system freeze or crash.

**Recommendation for Standard Users:**
Before running the script, open `portfolio_optimizer.py` and **find the configuration line: NUM_SIMULATIONS = 1000000** and **change it to 10000 or a smaller number of your choice.**

====================================================================================

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

**4. "AS-IS" SOFTWARE WARRANTY**

THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

**BY USING THIS SOFTWARE, YOU AGREE TO ASSUME ALL RISKS ASSOCIATED WITH YOUR INVESTMENT DECISIONS AND RELEASE THE AUTHOR FROM ANY LIABILITY REGARDING YOUR FINANCIAL OUTCOMES.**

====================================================================================

# Project Case Study: Quantitative Portfolio Optimization Engine
**1. Executive Summary**

This project is a quantitative analysis tool designed to construct a mathematically optimal investment portfolio. Unlike traditional "rule of thumb" investing (e.g., the 60/40 split), this model utilizes Modern Portfolio Theory (MPT) and Convex Optimization to solve for the highest possible risk-adjusted return (Sharpe Ratio).

The system targets a specific volatility profile (14% annualized risk) and validates the strategy by simulating 10,000 potential market futures over a 5-year horizon using Geometric Brownian Motion (GBM).

**2. The Problem Statement**

The author wanted to apply dynamic, data-driven modeling to personal asset management to answer three key questions:

a. Efficiency: Can we mathematically beat the standard market index by optimizing asset weights?

b. Risk Management: What is the statistical probability of loss over a 5-year period?

c. Scenario Planning: How does a "Dollar Cost Averaging" (DCA) strategy perform under thousands of different market conditions?

**3. Methodology & Asset Selection**

The model constructs a Portfolio using five distinct asset classes, each playing a specific role in the risk/reward ecosystem.

a. The Growth Engines (Risk-On):

* SPYM (S&P 500): Provides core exposure to the US large-cap economy.

* QQQ (Nasdaq 100): Captures high-growth technology sector momentum.

* VEA (Developed Markets): mitigating single-country geopolitical risk.

b. The Hedges (Risk-Off):

* TLT (Long-Term Treasuries): Acts as a deflationary hedge and negative-correlation asset during crashes.

* GLDM (Gold): Provides a hedge against currency devaluation and inflationary pressure.

**4. The Optimization Engine (The "Brain")**

The core of the project is a Python-based optimization algorithm (scipy.optimize). Instead of guessing weights, the model solves a mathematical problem:

"Find the exact combination of these 5 assets that maximizes return, subject to the constraint that total Portfolio Volatility cannot exceed 14%."

This process plots the Efficient Frontier—the curve representing the best possible returns for any given level of risk. The algorithm forces the portfolio to sit on this line, ensuring capital efficiency.

Constraints Applied: To prevent over-concentration (e.g., putting 100% into Tech), the model enforces "guardrails":

a. No single asset can exceed 70%.

b. Every asset must have at least a 2% allocation to ensure diversification.

c. The sum of weights must strictly equal 100%.

**5. Monte Carlo Simulation (The "Stress Test")**

Historical data is limited—it only shows us one version of the past. To understand the future, the author implemented a Monte Carlo Simulation.

This engine generates 1,000,000 theoretical future market paths based on the statistical properties (drift and standard deviation) of the optimized portfolio.

In the real world, we often deal with uncertainty. This simulation moves beyond "average returns" to look at tail risks—specifically, the 95% Value at Risk (VaR). It answers the critical question: "In the worst 5% of possible futures, is the portfolio still solvent?"

**6. Technology Stack & AI Integration**

This project utilizes a modern, AI-assisted workflow:

a. Language: Python (NumPy for matrix math, Pandas for data handling).

b. Data Source: Yahoo Finance API (Real-time adjusted close prices).

c. AI Implementation: Large Language Models (LLMs) were utilized to accelerate syntax generation and debug complex optimization constraints, allowing focus to remain on financial logic and architectural design rather than boilerplate coding.

**7. Key Findings**

The model rejected a balanced approach in favor of an aggressive growth strategy suitable for a long-time horizon:

a. Optimal Allocation: Heavily weighted toward SPYM (~60%) and QQQ (~23%), with minor allocations to hedges.

b. Projected Outcome: The simulation predicts a mean annualized return (CAGR) significantly outperforming a standard savings account, with a 95% probability of positive returns over a 5-year period.

====================================================================================

# Acknowledgments & Methodology 
This project was developed using an AI-Assisted Workflow. 

The conceptual architecture, financial strategy, asset selection, and risk constraints were designed by the author. 

The underlying Python syntax and library implementation were generated via Large Language Models (Gemini), demonstrating a modern approach to rapid prototyping and financial modeling where technical execution is accelerated by AI. 

**This approach focuses on leveraging AI as a force multiplier for rapid prototyping and complex quantitative modeling.**
