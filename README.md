# ⚠️ HARDWARE WARNING:
  **HIGH COMPUTATIONAL LOAD**

**PLEASE READ BEFORE RUNNING:**

This script is currently configured to run **1,000,000 (1 Million)** Monte Carlo simulations. This is an extreme stress test intended for high-performance workstations.

* **System Requirements:** Minimum **32GB RAM** and a multi-core processor (e.g., Ryzen 7 / Core i7 or better).
* **Risk:** Running this on a standard office laptop or non-gaming PC (8GB/16GB RAM) will likely cause a **Memory Overflow (OOM)**, resulting in a system freeze or crash.

**Recommendation for Standard Users:**

If you were to run the script after carefully reading and agreed with the "⚠️ Disclaimer and Terms of Use" below, **before running**, open `latest_code.py` and **find the configuration line: `n_sims = 1000000`** and **change it to 10000 or a smaller number of your choice.**

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

**THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.**

**BY USING THIS SOFTWARE, YOU AGREE TO ASSUME ALL RISKS ASSOCIATED WITH YOUR INVESTMENT DECISIONS AND RELEASE THE AUTHOR FROM ANY LIABILITY REGARDING YOUR FINANCIAL OUTCOMES.**

====================================================================================

# Project Case Study: Quantitative Portfolio Optimization Engine
**1. Executive Summary**

This project is a quantitative analysis tool designed to construct a mathematically optimal investment portfolio. Unlike traditional "rule of thumb" investing (e.g., the 60/40 split), this model utilizes Modern Portfolio Theory (MPT) and Convex Optimization to solve for the highest possible return within a defined risk budget.

The system targets a specific volatility profile (18% annualized risk) and validates the strategy by simulating 1,000,000 potential market futures over a 5-year horizon using Geometric Brownian Motion (GBM).

**2. The Problem Statement**

The author wanted to apply dynamic, data-driven modeling to personal asset management to answer three key questions:

a. Efficiency: Can we mathematically beat the standard market index by optimizing asset weights?

b. Risk Management: What is the statistical probability of loss over a 5-year period?

c. Scenario Planning: How does a lump-sum growth strategy perform under a million different simulated market conditions?

**3. Methodology & Asset Selection**

The model constructs a portfolio using six assets across four categories, each playing a specific role in the risk/reward ecosystem.

a. The Equity Growth Engine (Risk-On)

* SPYM (SPDR Portfolio S&P 500 ETF): Acts as the core anchor, providing broad exposure to the 500 largest US-listed companies.

* QQQ (Invesco QQQ Trust): Targets high-growth technology and innovation sectors to capture market-leading momentum.

* VEA (Vanguard FTSE Developed Markets ETF): Serves as a diversifier to mitigate single-country risk by providing exposure to established markets outside the US, such as Europe and Japan.

b. The Stability Layer (Risk-Off)

* AGG (iShares Core U.S. Aggregate Bond ETF): Provides a safety anchor and income through high-quality US investment-grade bonds, stabilizing the portfolio during equity volatility.

* GLDM (SPDR Gold MiniShares Trust): Functions as a classic inflation hedge and "store of value" during periods of currency devaluation or geopolitical uncertainty.

c. The Digital Frontier (High-Alpha Speculation)

* BTC-USD (Bitcoin): Introduced as a digital store of value and speculative growth asset.

Risk Guardrail: To maintain professional risk standards, this model implements a strict **8% maximum cap** on cryptocurrency allocation to prevent excessive exposure to digital asset volatility.

**4. The Optimization Engine (The "Brain")**

The core of the project is a Python-based optimization algorithm (`scipy.optimize.minimize` with the SLSQP method). Instead of guessing weights, the model solves a mathematical problem:

*"Find the exact combination of these 6 assets that maximizes return, subject to the constraint that total Portfolio Volatility cannot exceed 18%."*

The optimizer iterates through possible weight combinations and converges on the allocation that delivers the highest expected annual return while staying within the volatility ceiling.

**4a. How SLSQP Works Under the Hood**

SLSQP stands for **Sequential Least Squares Quadratic Programming**. That name is a mouthful, so let's break down what it actually does in plain language, and then walk through every formula the code uses.

**The Big Idea (No Math Yet)**

Imagine you have 6 dials — one for each asset. Each dial controls what percentage of your money goes into that asset. Your goal: twist the dials until you get the highest possible return, BUT you are not allowed to twist them into a combination that makes the portfolio too risky (volatile). You also can't set any dial below its minimum or above its maximum, and all six dials must add up to exactly 100%.

You could try random combinations for hours, but SLSQP is the algorithm that does this systematically. It starts with an initial guess (equal weights — ~16.7% each), evaluates how good that guess is, figures out which direction to nudge the dials, nudges them, checks again, and repeats — getting closer to the best answer each time. It stops when the improvement becomes negligibly small (less than `0.0000000001`, the tolerance set in the code).

**Step-by-Step: The Math Behind Each Piece**

Below are the actual formulas used in the code. No prior math knowledge is assumed.

---

**Formula 1: Expected Portfolio Return**

```
Portfolio Return = (w1 × r1 + w2 × r2 + w3 × r3 + ... + w6 × r6) × 252
```

What each symbol means:
- `w1, w2, ... w6` = the weight (percentage) allocated to each of the 6 assets. For example, if SPYM gets 60% of your money, then w1 = 0.60.
- `r1, r2, ... r6` = the average *daily* return of each asset, calculated from historical price data. For example, if SPYM went up by an average of 0.05% per day, then r1 = 0.0005.
- `× 252` = there are roughly 252 trading days in a year. Multiplying by 252 scales the tiny daily number up to an annual return that is easier to understand.

So this formula is really just a *weighted average*. If you put 60% into something that returns 12% a year and 40% into something that returns 5% a year, your blended return is `(0.60 × 12%) + (0.40 × 5%) = 9.2%`.

In the code, this is written as:
```python
-np.sum(mean_returns * weights) * 252
```

Why the minus sign? Because `scipy.optimize.minimize` can only *minimize* things. If we want to *maximize* return, we flip it: minimizing `-return` is the same as maximizing `return`. Think of it like this: minimizing your *losses* is the same as maximizing your *gains*.

---

**Formula 2: Portfolio Volatility (Risk)**

```
Portfolio Volatility = √(wᵀ × Σ × w) × √252
```

This one looks scary, but here's what it means in plain English:

- **Volatility** = how wildly the portfolio's value swings up and down. High volatility = big swings = more risk.
- `w` = the list of weights (same as above), written as a column of numbers.
- `wᵀ` = the same list of weights, but flipped sideways into a row. (The `ᵀ` just means "transpose" — flip the column into a row.)
- `Σ` (Sigma) = the **covariance matrix**. This is a table that captures two things:
  - How much each asset swings on its own (the diagonal of the table).
  - How much each *pair* of assets moves together (the off-diagonal cells). If stocks and gold tend to move in opposite directions, their covariance is negative — and combining them *reduces* overall risk. This is the mathematical foundation of diversification.
- `√(...)` = square root. Covariance is measured in "squared" units, so we take the square root to get back to a percentage that makes intuitive sense.
- `× √252` = just like returns, we scale from daily to annual. For volatility, the scaling factor is the square root of 252 (≈ 15.87), not 252 itself. This is because risk grows with the *square root* of time, not linearly — a fundamental property of random processes.

**A concrete analogy:** Imagine you are mixing paints. Each paint (asset) has its own "intensity" (volatility). But when you mix them, the result is not just the average intensity — it depends on whether the colors *reinforce* each other (positive covariance, like mixing two reds → even redder) or *cancel* each other out (negative covariance, like mixing red + green → muted brown). The covariance matrix is the lookup table that tells you how every pair interacts.

In the code, this is:
```python
port_vol = √(wᵀ · Σ · w) × √252
```
And the constraint checks: `18% - port_vol ≥ 0` (i.e., volatility must stay at or below 18%).

---

**Formula 3: The Sharpe Ratio**

```
Sharpe Ratio = (Portfolio Return - Risk-Free Rate) / Portfolio Volatility
```

This answers a simple question: **"How much extra return am I getting for each unit of risk I'm taking?"**

- **Portfolio Return** = the annualized return from Formula 1.
- **Risk-Free Rate** = what you would earn with zero risk (e.g., a government savings account or Treasury bill). The code uses 3.5% (0.035).
- **Portfolio Volatility** = the annualized risk from Formula 2.

If a portfolio returns 15% with 18% volatility and the risk-free rate is 3.5%, the Sharpe Ratio = `(15% - 3.5%) / 18% = 0.64`. A higher Sharpe Ratio means you are being better compensated for the risk you are taking. A Sharpe Ratio below 0 means you would be better off in a savings account.

---

**How SLSQP Uses These Formulas**

Now that we know what each formula calculates, here is what SLSQP does with them:

1. **Start:** Set all 6 weights to equal (≈16.7% each).
2. **Evaluate:** Plug the weights into Formula 1 (return) and Formula 2 (volatility). Check if all rules are satisfied (weights sum to 1, volatility ≤ 18%, each weight within its min/max bounds).
3. **Approximate:** Build a simplified "curved surface" (quadratic approximation) of the return formula near the current weights. This surface is easier to do math on than the real thing.
4. **Solve the sub-problem:** On this simplified surface, find the direction that improves the return the most while keeping all constraints satisfied. This gives a set of tiny adjustments: "increase SPYM by 0.3%, decrease AGG by 0.1%," etc.
5. **Step:** Apply those adjustments to the weights.
6. **Repeat:** Go back to step 2 with the new weights. Each loop gets closer to the optimum.
7. **Stop:** When the improvement from one loop to the next is smaller than `0.0000000001` (the tolerance), declare the current weights as the optimal solution.

The key advantage of SLSQP over simpler methods: it respects all the rules (constraints) *during* the search, not as an afterthought. It never proposes a solution where weights don't add up to 100% or where volatility exceeds 18% — those are baked into every step.

**Why SLSQP and not other solvers?** Portfolio optimization is a *constrained* problem: the weights must sum to exactly 1.0, each weight must stay within its bounds, and the portfolio volatility must stay under a ceiling. Many popular optimizers (e.g., gradient descent, Adam) are designed for *unconstrained* problems and would require penalty terms or workarounds to handle these rules. SLSQP handles equality constraints (weights sum to 1), inequality constraints (volatility ≤ 18%, stock allocation ≥ 50%), and box bounds (each weight between its min and max) natively in a single unified framework. This makes it both simpler to set up and more numerically reliable for this class of problem.

Constraints Applied: The model enforces layered "guardrails" to shape a realistic allocation:

a. **Equities (SPYM, QQQ, VEA):** Each stock must hold at least 3%. Combined equity allocation is bounded between 50% and 96%.

b. **Bonds (AGG):** Locked at a minimum 2% allocation to guarantee a baseline stability layer.

c. **Gold (GLDM):** Locked at a minimum 2% allocation to guarantee a baseline inflation hedge.

d. **Crypto (BTC-USD):** Bounded between 3% and 8%, allowing speculative upside while capping tail risk.

e. **The sum of all weights must strictly equal 100%.**

Because bonds and gold are effectively pinned at their minimums (2% each) and crypto is capped at 8%, the optimizer funnels the vast majority of the remaining weight (~88%) into the three equity ETFs. This is by design: the guardrails ensure that hedges are always present, but the optimizer is free to aggressively pursue returns within the equity bucket.

**5. How Weights Are Output**

After optimization, the script prints a detailed allocation table showing each asset's ticker, category (Core Stock, New Recruit, Bond, Gold, or Crypto), its optimized weight as a percentage, and the dollar value that weight represents on a $100,000 portfolio. Assets are grouped by their role so the user can immediately see the equity-heavy tilt vs. the hedge and crypto layers.

The script also produces a **Monthly Investment Breakdown** at the end. Given a user-defined monthly contribution (in TWD), it multiplies each asset's optimized weight by the monthly budget to show exactly how much to allocate per asset each month. This translates the abstract percentage weights into a concrete, actionable investment plan.

**6. Monte Carlo Simulation (The "Stress Test")**

Historical data is limited—it only shows us one version of the past. To understand the range of possible futures, the author implemented a Monte Carlo Simulation.

This engine generates **1,000,000** theoretical future market paths over 5 years. Each path uses Geometric Brownian Motion (GBM), starting from a $100,000 lump-sum investment and growing it daily based on the portfolio's statistical properties (drift and standard deviation).

The simulation reports the **95% Confidence Interval**: it calculates the 2.5th and 97.5th percentile outcomes across all one million paths, showing both the near-worst-case and near-best-case scenarios. It also reports the mean CAGR, annualized volatility, and Sharpe Ratio of the simulated outcomes.

Two visualizations are produced:

a. **Spaghetti Plot:** 200 randomly sampled paths are plotted, color-coded by final portfolio value (using a viridis colormap), with the mean path overlaid in red.

b. **Return Distribution Histogram:** A histogram of all simulated CAGRs with vertical lines marking the mean return and the 95% confidence bounds.

**7. Technology Stack & AI Integration**

This project utilizes a modern, AI-assisted workflow:

a. Language: Python (NumPy for matrix math, Pandas for data handling).

b. Data Source: Yahoo Finance API (Real-time adjusted close prices).

c. AI Implementation: Large Language Models (LLMs) were utilized to accelerate syntax generation and debug complex optimization constraints, allowing focus to remain on financial logic and architectural design rather than boilerplate coding.

**8. Key Findings**

The model rejected a balanced approach in favor of an aggressive growth strategy suitable for a long-time horizon:

a. Optimal Allocation: The guardrails lock bonds, gold, and crypto at relatively small fixed allocations, leaving the optimizer free to concentrate heavily on equities (SPYM, QQQ, VEA). The result is a growth-tilted portfolio with minimal but guaranteed hedge exposure.

b. Projected Outcome: The simulation predicts a mean annualized return (CAGR) significantly outperforming a standard savings account, with a 95% confidence interval indicating the range of plausible outcomes over a 5-year period.

====================================================================================

# Acknowledgments & Methodology
**The conceptual architecture, financial strategy, asset selection, and risk constraints were designed by the author.**

This project was developed using an AI-Assisted Workflow.

The underlying Python syntax and library implementation were generated via Large Language Models (Gemini), demonstrating a modern approach to rapid prototyping and financial modeling where technical execution is accelerated by AI.

**This approach focuses on leveraging AI as a force multiplier for rapid prototyping and complex quantitative modeling.**

====================================================================================
