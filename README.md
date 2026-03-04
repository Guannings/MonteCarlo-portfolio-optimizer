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

**Every Math Formula in This Code, Explained From Scratch**

Below is every formula the code uses, listed in the order they appear. No prior math knowledge is assumed. If you know what addition and multiplication are, you can follow this.

---

**Formula 1: Daily Return** (line 38)

```
Daily Return = (today's price - yesterday's price) / yesterday's price
```

This measures how much an asset's price moved in one day, expressed as a fraction. If a stock was $100 yesterday and $102 today, the daily return is `(102 - 100) / 100 = 0.02`, which is 2%. If it dropped to $97, the return is `(97 - 100) / 100 = -0.03`, or -3%.

The code computes this for every single day in the dataset, for all 6 assets at once. The result is a big table: each row is a date, each column is an asset, and every cell is that asset's return on that day.

```python
returns = prices.pct_change().dropna()
```

`pct_change()` is just a shortcut that does the subtraction and division for you. `.dropna()` removes the first row (which has no "yesterday" to compare to).

---

**Formula 2: Mean Daily Return** (line 45)

```
Mean Daily Return = (Day 1 return + Day 2 return + ... + Day N return) / N
```

This is just the ordinary average you learned in school. Add up all the daily returns for one asset, divide by how many days there are. It tells you: "On a typical day, this asset goes up (or down) by roughly this much."

For example, if SPYM had returns of +1%, -0.5%, +0.8% over 3 days, the mean daily return is `(1% + (-0.5%) + 0.8%) / 3 = 0.43%`.

The code does this for all 6 assets in one line:

```python
mean_returns = returns.mean()
```

---

**Formula 3: Covariance Matrix** (line 44)

```
Covariance between A and B = average of [(A's return - A's mean) × (B's return - B's mean)]
```

This one needs a bit more explanation. **Covariance** measures whether two assets tend to move in the same direction or opposite directions.

- On a given day, if Asset A is *above* its average AND Asset B is also *above* its average, the product of those two deviations is *positive*.
- If Asset A is *above* average but Asset B is *below* average, the product is *negative*.
- Average all those products across every day in the dataset. If the result is positive, the two assets tend to move together. If negative, they tend to move in opposite directions.

The **covariance matrix** is a table that computes this for *every possible pair* of assets. With 6 assets, it's a 6×6 table. The diagonal cells (e.g., SPYM vs. SPYM) are just the **variance** — how much that asset swings on its own. The off-diagonal cells tell you how each pair interacts.

Why does this matter? Because if you combine two assets that move in opposite directions, their swings partially cancel out. This is the math behind the idea of diversification — and it's the reason the optimizer can find combinations that have *less* total risk than any single asset alone.

```python
cov_matrix = returns.cov()
```

---

**Formula 4: Expected Portfolio Return (Objective Function)** (line 51)

```
Portfolio Return = (w1 × r1 + w2 × r2 + w3 × r3 + ... + w6 × r6) × 252
```

What each symbol means:
- `w1, w2, ... w6` = the weight (percentage) allocated to each of the 6 assets. For example, if SPYM gets 60% of your money, then w1 = 0.60.
- `r1, r2, ... r6` = the mean daily return of each asset (from Formula 2).
- `× 252` = there are roughly 252 trading days in a year. Multiplying by 252 scales the tiny daily number up to an annual return that is easier to understand.

This formula is really just a **weighted average**. If you put 60% into something that returns 12% a year and 40% into something that returns 5% a year, your blended return is `(0.60 × 12%) + (0.40 × 5%) = 9.2%`.

In the code:
```python
-np.sum(mean_returns * weights) * 252
```

**Why the minus sign?** Because `scipy.optimize.minimize` can only *minimize* things. If we want to *maximize* return, we flip it: minimizing `-return` is the same as maximizing `return`. Think of it like this: minimizing your *losses* is the same as maximizing your *gains*.

---

**Formula 5: Portfolio Volatility (Risk Constraint)** (line 55)

```
Portfolio Volatility = √(wᵀ × Σ × w) × √252
```

This looks scary, so let's break it down piece by piece:

- `w` = the list of 6 weights, written as a vertical column of numbers.
- `wᵀ` = the same list, but flipped sideways into a horizontal row. (The `ᵀ` just means "transpose" — flip the column into a row.)
- `Σ` (Sigma) = the covariance matrix from Formula 3.

**What the multiplication `wᵀ × Σ × w` actually does:** It takes every pair of assets, multiplies their covariance by both of their weights, and adds it all up. The result is a single number that captures the total risk of the *combined* portfolio — accounting for both how much each asset swings individually AND how much they cancel each other out (or reinforce each other).

**Concrete analogy:** Imagine you are mixing paints. Each paint (asset) has its own "intensity" (volatility). But when you mix them, the result is not just the average intensity — it depends on whether the colors *reinforce* each other (positive covariance, like mixing two reds → even redder) or *cancel* each other out (negative covariance, like mixing red + green → muted brown). The covariance matrix is the lookup table that tells you how every pair interacts.

- `√(...)` = square root. The multiplication above gives us **variance** (risk squared). Taking the square root converts it back to **standard deviation** — a percentage that makes intuitive sense.
- `× √252` = just like returns, we scale from daily to annual. But for volatility, the scaling factor is the **square root** of 252 (≈ 15.87), not 252 itself. Why? Because risk grows with the *square root* of time, not linearly. This is a fundamental property of random processes — if you flip a coin twice as many times, you don't get twice as much deviation from 50/50, you get about 1.41× as much (√2 ≈ 1.41).

In the code:
```python
port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
```

The constraint then checks: `18% - port_vol ≥ 0`. In plain English: "the portfolio's volatility must stay at or below 18%."

---

**Formula 6: Sum-to-One Constraint** (line 59)

```
w1 + w2 + w3 + w4 + w5 + w6 = 1.0
```

This one is simple: all your money must go somewhere. If you put 60% in stocks, 2% in bonds, 2% in gold, and 8% in crypto, the remaining 28% must also be allocated to other stocks. You can't invest 110% of your money, and you can't leave 10% under the mattress (in this model).

```python
{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
```

The `'eq'` means "equality constraint." The function `np.sum(x) - 1` must equal zero — i.e., the sum must be exactly 1.

---

**Formula 7: Portfolio Daily Return (for Monte Carlo)** (line 123)

```
Portfolio Daily Return = w1 × r1 + w2 × r2 + ... + w6 × r6
```

This is the same weighted average as Formula 4, but **without** the `× 252`. Here we need the daily return as-is because the Monte Carlo simulation steps through the portfolio day by day. The code calls this `port_ret`.

```python
port_ret = np.dot(final_weights, returns.mean())
```

`np.dot` is just a compact way to multiply each weight by its corresponding return and add them all up.

---

**Formula 8: Portfolio Variance** (line 124)

```
Portfolio Variance = wᵀ × Σ × w
```

This is the same matrix multiplication as inside Formula 5, but **without** the square root and **without** the `× √252`. It gives us the raw daily variance of the portfolio — a number we need as an ingredient for the next two formulas.

```python
port_cov = np.dot(final_weights.T, np.dot(returns.cov(), final_weights))
```

---

**Formula 9: Portfolio Standard Deviation** (line 125)

```
Portfolio Std Dev = √(Portfolio Variance)
```

Square root of Formula 8. This is the daily standard deviation — how much the portfolio typically swings per day, measured in the same units as returns (a decimal fraction).

```python
port_std = np.sqrt(port_cov)
```

---

**Formula 10: GBM Drift** (line 127)

```
Drift = Portfolio Daily Return - 0.5 × Portfolio Variance
```

This is where the simulation's core math begins. **Geometric Brownian Motion (GBM)** is the standard mathematical model for how stock prices move. It assumes prices take a "random walk" that trends upward (or downward) over time.

The **drift** is the *average direction* the portfolio tends to move each day. But why subtract `0.5 × variance`? This is called the **Itô correction** (named after mathematician Kiyosi Itô), and it's the single most unintuitive part of the entire codebase. Here's the plain-English reason:

Imagine a stock that goes +10% one day and -10% the next day. You might think you're back to even, but you're not:
- Start with $100. Go up 10% → $110.
- Go down 10% → $99. You **lost** $1.

This asymmetry exists because percentage gains and losses are not symmetric — a 10% loss requires an 11.1% gain to recover. The bigger the swings (variance), the worse this "volatility drag" gets. The `- 0.5 × variance` term corrects for this drag so that the simulation accurately represents how compounding works in real markets. Without it, the simulation would systematically overestimate returns.

```python
drift = port_ret - 0.5 * port_cov
```

---

**Formula 11: Random Shocks (Z-scores)** (line 128)

```
Z = a random number drawn from a Normal Distribution with mean 0 and standard deviation 1
```

A **Normal Distribution** (also called a "bell curve") is the classic shape where most values cluster near the middle and extreme values are rare. Think of it like human heights: most people are near average, a few are very tall or very short, and almost nobody is 8 feet tall.

Here, each random number represents **one day's "surprise"** — the unpredictable part of the market. A Z of 0 means "average day, nothing surprising." A Z of +2 means "a really good day" (roughly 2.3% chance). A Z of -3 means "a terrible day" (roughly 0.1% chance).

The code generates a massive grid: 1,260 days (252 × 5 years) by 1,000,000 simulations. That's over **1.2 billion** random numbers.

```python
Z = np.random.normal(0, 1, (n_days, n_sims))
```

---

**Formula 12: Daily Growth Factor (GBM Core)** (line 129)

```
Daily Growth = e^(Drift + Standard Deviation × Z)
```

This is the heart of the simulation — the formula that converts yesterday's randomness into today's price movement.

- `e` = Euler's number (≈ 2.71828), a mathematical constant. Using `e^(something)` ensures that prices can never go negative (since `e^(anything)` is always positive), which is how real stock prices work.
- `Drift` = the average daily tendency from Formula 10.
- `Standard Deviation × Z` = the random "shock" from Formula 11, scaled by how volatile the portfolio is. A more volatile portfolio has bigger shocks.

**Example:** If drift = 0.0004 and std dev = 0.01, then:
- On a boring day (Z = 0): growth = `e^(0.0004 + 0.01 × 0)` = `e^(0.0004)` ≈ 1.0004, meaning the portfolio goes up 0.04%.
- On a great day (Z = +2): growth = `e^(0.0004 + 0.01 × 2)` = `e^(0.0204)` ≈ 1.0206, meaning the portfolio goes up about 2%.
- On a crash day (Z = -3): growth = `e^(0.0004 + 0.01 × (-3))` = `e^(-0.0296)` ≈ 0.9708, meaning the portfolio drops about 3%.

```python
daily_growth = np.exp(drift + port_std * Z)
```

---

**Formula 13: Path Simulation (Compounding)** (line 134)

```
Today's Value = Yesterday's Value × Today's Growth Factor
```

This is how compound growth works. Each day, you take what you had yesterday and multiply it by that day's growth factor (from Formula 12). Do this 1,260 times in a row and you get one complete 5-year price path.

The code does this for all 1,000,000 simulations simultaneously:

```python
price_paths[0] = 100000  # Start with $100,000
for t in range(1, n_days + 1):
    price_paths[t] = price_paths[t - 1] * daily_growth[t - 1]
```

After this loop, each of the 1,000,000 columns is a different "alternate universe" — one possible version of what the portfolio's value could look like over 5 years.

---

**Formula 14: CAGR (Compound Annual Growth Rate)** (line 139)

```
CAGR = (Ending Value / Starting Value) ^ (1 / Number of Years) - 1
```

CAGR answers: **"If the portfolio grew at a perfectly steady rate each year, what would that rate be?"** It smooths out all the daily ups and downs into a single annual number.

- `Ending Value / Starting Value` = how much your money multiplied. If you started with $100,000 and ended with $200,000, this ratio is 2.0 (your money doubled).
- `^ (1 / 5)` = the "fifth root" (since this is a 5-year simulation). This finds the per-year multiplier. The fifth root of 2.0 is about 1.149.
- `- 1` = converts the multiplier into a percentage. 1.149 - 1 = 0.149, or 14.9% per year.

**Example:** $100,000 grows to $180,000 over 5 years. CAGR = `(180,000 / 100,000)^(1/5) - 1` = `1.8^0.2 - 1` ≈ 12.5% per year.

```python
sim_cagrs = (ending_values / total_val) ** (1 / 5) - 1
```

This is computed for each of the 1,000,000 simulations, producing 1,000,000 different CAGRs — one for each "alternate universe."

---

**Formula 15: Sharpe Ratio** (line 141)

```
Sharpe Ratio = (Mean CAGR - Risk-Free Rate) / Annualized Volatility
```

This answers: **"How much extra return am I getting for each unit of risk I'm taking?"**

- **Mean CAGR** = the average CAGR across all 1,000,000 simulations.
- **Risk-Free Rate** = what you would earn with zero risk (e.g., a government savings account). The code uses 3.5% (0.035).
- **Annualized Volatility** = the daily standard deviation (Formula 9) scaled up to annual: `port_std × √252`.

**Example:** If mean CAGR = 15%, risk-free rate = 3.5%, and annual volatility = 18%, then Sharpe = `(15% - 3.5%) / 18% = 0.64`. A higher Sharpe means you are being better compensated for the risk. A Sharpe below 0 means you would be better off in a savings account.

```python
sharpe = (mean_cagr - 0.035) / (port_std * np.sqrt(252))
```

---

**Formula 16: Annualized Volatility** (line 153)

```
Annual Volatility = Daily Standard Deviation × √252
```

This converts the daily standard deviation (a tiny number like 0.01) into an annual percentage (like 15.87%). Same `× √252` scaling explained in Formula 5.

```python
port_std * np.sqrt(252)
```

---

**Formula 17: 95% Confidence Interval (Percentiles)** (lines 144–147)

```
Lower Bound = the value below which only 2.5% of outcomes fall
Upper Bound = the value below which 97.5% of outcomes fall
```

Imagine sorting all 1,000,000 ending values from smallest to largest. The **2.5th percentile** is the value at position 25,000 (2.5% of the way through the list) — only 2.5% of simulations did worse than this. The **97.5th percentile** is at position 975,000 — only 2.5% did better.

Together, these two values capture the middle 95% of all outcomes, giving you a range: "In 95 out of 100 possible futures, the portfolio ends up somewhere between $X and $Y."

```python
ci_lower_val = np.percentile(ending_values, 2.5)
ci_upper_val = np.percentile(ending_values, 97.5)
ci_lower_ret = np.percentile(sim_cagrs, 2.5)
ci_upper_ret = np.percentile(sim_cagrs, 97.5)
```

The code computes this for both the raw dollar values and the CAGRs.

---

**Formula 18: Monthly Allocation** (line 227)

```
Amount Per Asset = Monthly Budget × Weight
```

The simplest formula in the codebase. If your monthly budget is 8,000 TWD and SPYM's optimized weight is 60%, then you put `8,000 × 0.60 = 4,800 TWD` into SPYM that month.

```python
investment_per_asset = monthly_investment_twd * w
```

---

**How SLSQP Uses Formulas 4, 5, and 6**

Now that we know what each formula calculates, here is what SLSQP does with Formulas 4 (return), 5 (volatility), and 6 (sum-to-one) to find the optimal weights:

1. **Start:** Set all 6 weights to equal (≈16.7% each).
2. **Evaluate:** Plug the weights into Formula 4 (return) and Formula 5 (volatility). Check if all rules are satisfied (weights sum to 1, volatility ≤ 18%, each weight within its min/max bounds).
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

This engine generates **1,000,000** theoretical future market paths over 5 years. Each path uses Geometric Brownian Motion (GBM), starting from a $100,000 lump-sum investment and growing it daily based on the portfolio's statistical properties. The math behind the simulation is fully explained in Section 4a:

- **Formulas 7–9** compute the portfolio's daily return, variance, and standard deviation from the optimized weights.
- **Formula 10 (Drift)** adjusts the expected return downward to account for volatility drag (the Itô correction).
- **Formula 11 (Random Shocks)** generates 1.26 billion random numbers representing daily market surprises.
- **Formula 12 (GBM Core)** combines drift and randomness into a daily growth factor.
- **Formula 13 (Compounding)** chains 1,260 days together to produce each complete 5-year price path.
- **Formula 14 (CAGR)** converts each path's final value into a single annualized return.
- **Formulas 15–16 (Sharpe, Annualized Volatility)** summarize the risk-adjusted performance.
- **Formula 17 (95% CI)** identifies the 2.5th and 97.5th percentile outcomes.

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
