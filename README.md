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

**Formula 1: Daily Return** ([line 38](https://github.com/Guannings/MonteCarlo-Portfolio-Optimizer/blob/49b0e61/Gemini_generated%20_codes/Monte-Carlo%20Sim/latest_code.py#L38))

$$r_t = \frac{P_t - P_{t-1}}{P_{t-1}}$$

This measures how much an asset's price moved in one day, expressed as a fraction. If a stock was $100 yesterday and $102 today, the daily return is `(102 - 100) / 100 = 0.02`, which is 2%. If it dropped to $97, the return is `(97 - 100) / 100 = -0.03`, or -3%.

The code computes this for every single day in the dataset, for all 6 assets at once. The result is a big table: each row is a date, each column is an asset, and every cell is that asset's return on that day.

```python
returns = prices.pct_change().dropna()
```

`pct_change()` is just a shortcut that does the subtraction and division for you. `.dropna()` removes the first row (which has no "yesterday" to compare to).

---

**Formula 2: Mean Daily Return** ([line 45](https://github.com/Guannings/MonteCarlo-Portfolio-Optimizer/blob/49b0e61/Gemini_generated%20_codes/Monte-Carlo%20Sim/latest_code.py#L45))

$$\mu = \frac{1}{N}\sum_{i=1}^{N} r_i$$

This is just the ordinary average you learned in school. Add up all the daily returns for one asset, divide by how many days there are. It tells you: "On a typical day, this asset goes up (or down) by roughly this much."

For example, if SPYM had returns of +1%, -0.5%, +0.8% over 3 days, the mean daily return is `(1% + (-0.5%) + 0.8%) / 3 = 0.43%`.

The code does this for all 6 assets in one line:

```python
mean_returns = returns.mean()
```

---

**Formula 3: Covariance Matrix** ([line 44](https://github.com/Guannings/MonteCarlo-Portfolio-Optimizer/blob/49b0e61/Gemini_generated%20_codes/Monte-Carlo%20Sim/latest_code.py#L44))

$$\text{Cov}(A, B) = \frac{1}{N}\sum_{i=1}^{N}(r_i^A - \mu^A)(r_i^B - \mu^B)$$

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

**Formula 4: Expected Portfolio Return (Objective Function)** ([lines 50–51](https://github.com/Guannings/MonteCarlo-Portfolio-Optimizer/blob/49b0e61/Gemini_generated%20_codes/Monte-Carlo%20Sim/latest_code.py#L50-L51))

$$R_p = \left(\sum_{i=1}^{6} w_i \cdot \mu_i\right) \times 252$$

What each symbol means:
- $w_1, w_2, \ldots, w_6$ = the weight (percentage) allocated to each of the 6 assets. For example, if SPYM gets 60% of your money, then $w_1 = 0.60$.
- $\mu_1, \mu_2, \ldots, \mu_6$ = the mean daily return of each asset (from Formula 2).
- $\times 252$ = there are roughly 252 trading days in a year. Multiplying by 252 scales the tiny daily number up to an annual return that is easier to understand.

This formula is really just a **weighted average**. If you put 60% into something that returns 12% a year and 40% into something that returns 5% a year, your blended return is `(0.60 × 12%) + (0.40 × 5%) = 9.2%`.

In the code:
```python
-np.sum(mean_returns * weights) * 252
```

**Why the minus sign?** Because `scipy.optimize.minimize` can only *minimize* things. If we want to *maximize* return, we flip it: minimizing `-return` is the same as maximizing `return`. Think of it like this: minimizing your *losses* is the same as maximizing your *gains*.

---

**Formula 5: Portfolio Volatility (Risk Constraint)** ([lines 54–56](https://github.com/Guannings/MonteCarlo-Portfolio-Optimizer/blob/49b0e61/Gemini_generated%20_codes/Monte-Carlo%20Sim/latest_code.py#L54-L56))

$$\sigma_p = \sqrt{\mathbf{w}^\top \Sigma \mathbf{w}}  \times \sqrt{252}$$

This looks scary, so let's break it down piece by piece:

- $\mathbf{w}$ = the list of 6 weights, written as a vertical column of numbers.
- $\mathbf{w}^\top$ = the same list, but flipped sideways into a horizontal row. (The $\top$ just means "transpose" — flip the column into a row.)
- $\Sigma$ (Sigma) = the covariance matrix from Formula 3.

**What the multiplication $\mathbf{w}^\top \Sigma \mathbf{w}$ actually does:** It takes every pair of assets, multiplies their covariance by both of their weights, and adds it all up. The result is a single number that captures the total risk of the *combined* portfolio — accounting for both how much each asset swings individually AND how much they cancel each other out (or reinforce each other).

**Concrete analogy:** Imagine you are mixing paints. Each paint (asset) has its own "intensity" (volatility). But when you mix them, the result is not just the average intensity — it depends on whether the colors *reinforce* each other (positive covariance, like mixing two reds → even redder) or *cancel* each other out (negative covariance, like mixing red + green → muted brown). The covariance matrix is the lookup table that tells you how every pair interacts.

- `√(...)` = square root. The multiplication above gives us **variance** (risk squared). Taking the square root converts it back to **standard deviation** — a percentage that makes intuitive sense.
- `× √252` = just like returns, we scale from daily to annual. But for volatility, the scaling factor is the **square root** of 252 (≈ 15.87), not 252 itself. Why? Because risk grows with the *square root* of time, not linearly. This is a fundamental property of random processes — if you flip a coin twice as many times, you don't get twice as much deviation from 50/50, you get about 1.41× as much (√2 ≈ 1.41).

In the code:
```python
port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
```

The constraint then checks: `18% - port_vol ≥ 0`. In plain English: "the portfolio's volatility must stay at or below 18%."

---

**Formula 6: Sum-to-One Constraint** ([line 59](https://github.com/Guannings/MonteCarlo-Portfolio-Optimizer/blob/49b0e61/Gemini_generated%20_codes/Monte-Carlo%20Sim/latest_code.py#L59))

$$\sum_{i=1}^{6} w_i = 1$$

This one is simple: all your money must go somewhere. If you put 60% in stocks, 2% in bonds, 2% in gold, and 8% in crypto, the remaining 28% must also be allocated to other stocks. You can't invest 110% of your money, and you can't leave 10% under the mattress (in this model).

```python
{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
```

The `'eq'` means "equality constraint." The function `np.sum(x) - 1` must equal zero — i.e., the sum must be exactly 1.

---

**Formula 7: Portfolio Daily Return (for Monte Carlo)** ([line 123](https://github.com/Guannings/MonteCarlo-Portfolio-Optimizer/blob/49b0e61/Gemini_generated%20_codes/Monte-Carlo%20Sim/latest_code.py#L123))

$$R_{p,\text{daily}} = \sum_{i=1}^{6} w_i \cdot \mu_i$$

This is the same weighted average as Formula 4, but **without** the `× 252`. Here we need the daily return as-is because the Monte Carlo simulation steps through the portfolio day by day. The code calls this `port_ret`.

```python
port_ret = np.dot(final_weights, returns.mean())
```

`np.dot` is just a compact way to multiply each weight by its corresponding return and add them all up.

---

**Formula 8: Portfolio Variance** ([line 124](https://github.com/Guannings/MonteCarlo-Portfolio-Optimizer/blob/49b0e61/Gemini_generated%20_codes/Monte-Carlo%20Sim/latest_code.py#L124))

$$\sigma_p^2 = \mathbf{w}^\top \Sigma \mathbf{w}$$

This is the same matrix multiplication as inside Formula 5, but **without** the square root and **without** the `× √252`. It gives us the raw daily variance of the portfolio — a number we need as an ingredient for the next two formulas.

```python
port_cov = np.dot(final_weights.T, np.dot(returns.cov(), final_weights))
```

---

**Formula 9: Portfolio Standard Deviation** ([line 125](https://github.com/Guannings/MonteCarlo-Portfolio-Optimizer/blob/49b0e61/Gemini_generated%20_codes/Monte-Carlo%20Sim/latest_code.py#L125))

$$\sigma_p = \sqrt{\sigma_p^2}$$

Square root of Formula 8. This is the daily standard deviation — how much the portfolio typically swings per day, measured in the same units as returns (a decimal fraction).

```python
port_std = np.sqrt(port_cov)
```

---

**Formula 10: GBM Drift** ([line 127](https://github.com/Guannings/MonteCarlo-Portfolio-Optimizer/blob/49b0e61/Gemini_generated%20_codes/Monte-Carlo%20Sim/latest_code.py#L127))

$$\text{drift} = \mu_p - \tfrac{1}{2}\sigma_p^2$$

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

**Formula 11: Random Shocks (Z-scores)** ([line 128](https://github.com/Guannings/MonteCarlo-Portfolio-Optimizer/blob/49b0e61/Gemini_generated%20_codes/Monte-Carlo%20Sim/latest_code.py#L128))

$$Z \sim \mathcal{N}(0, 1)$$

A **Normal Distribution** (also called a "bell curve") is the classic shape where most values cluster near the middle and extreme values are rare. Think of it like human heights: most people are near average, a few are very tall or very short, and almost nobody is 8 feet tall.

Here, each random number represents **one day's "surprise"** — the unpredictable part of the market. A Z of 0 means "average day, nothing surprising." A Z of +2 means "a really good day" (roughly 2.3% chance). A Z of -3 means "a terrible day" (roughly 0.1% chance).

The code generates a massive grid: 1,260 days (252 × 5 years) by 1,000,000 simulations. That's over **1.2 billion** random numbers.

```python
Z = np.random.normal(0, 1, (n_days, n_sims))
```

---

**Formula 12: Daily Growth Factor (GBM Core)** ([line 129](https://github.com/Guannings/MonteCarlo-Portfolio-Optimizer/blob/49b0e61/Gemini_generated%20_codes/Monte-Carlo%20Sim/latest_code.py#L129))

$$G_t = e^{\text{drift} + \sigma_p \cdot Z_t}$$

This is the heart of the simulation — the formula that converts yesterday's randomness into today's price movement.

- $e$ = Euler's number ($\approx 2.71828$), a mathematical constant. Using $e^{(\cdot)}$ ensures that prices can never go negative (since $e^x > 0$ for all $x$), which is how real stock prices work.
- $\text{drift}$ = the average daily tendency from Formula 10.
- $\sigma_p \cdot Z$ = the random "shock" from Formula 11, scaled by how volatile the portfolio is. A more volatile portfolio has bigger shocks.

**Example:** If drift = 0.0004 and $\sigma_p$ = 0.01, then:
- On a boring day ($Z = 0$): growth = $e^{0.0004} \approx 1.0004$, meaning the portfolio goes up 0.04%.
- On a great day ($Z = +2$): growth = $e^{0.0204} \approx 1.0206$, meaning the portfolio goes up about 2%.
- On a crash day ($Z = -3$): growth = $e^{-0.0296} \approx 0.9708$, meaning the portfolio drops about 3%.

```python
daily_growth = np.exp(drift + port_std * Z)
```

---

**Formula 13: Path Simulation (Compounding)** ([lines 131–134](https://github.com/Guannings/MonteCarlo-Portfolio-Optimizer/blob/49b0e61/Gemini_generated%20_codes/Monte-Carlo%20Sim/latest_code.py#L131-L134))

$$S_t = S_{t-1} \times G_t$$

This is how compound growth works. Each day, you take what you had yesterday and multiply it by that day's growth factor (from Formula 12). Do this 1,260 times in a row and you get one complete 5-year price path.

The code does this for all 1,000,000 simulations simultaneously:

```python
price_paths[0] = 100000  # Start with $100,000
for t in range(1, n_days + 1):
    price_paths[t] = price_paths[t - 1] * daily_growth[t - 1]
```

After this loop, each of the 1,000,000 columns is a different "alternate universe" — one possible version of what the portfolio's value could look like over 5 years.

---

**Formula 14: CAGR (Compound Annual Growth Rate)** ([line 139](https://github.com/Guannings/MonteCarlo-Portfolio-Optimizer/blob/49b0e61/Gemini_generated%20_codes/Monte-Carlo%20Sim/latest_code.py#L139))

$$\text{CAGR} = \left(\frac{S_T}{S_0}\right)^{1/T} - 1$$

CAGR answers: **"If the portfolio grew at a perfectly steady rate each year, what would that rate be?"** It smooths out all the daily ups and downs into a single annual number.

- $S_T / S_0$ = how much your money multiplied. If you started with \$100,000 and ended with \$200,000, this ratio is 2.0 (your money doubled).
- Raising to the power $1/5$ = the "fifth root" (since this is a 5-year simulation). This finds the per-year multiplier. The fifth root of 2.0 is about 1.149.
- $- 1$ = converts the multiplier into a percentage. $1.149 - 1 = 0.149$, or 14.9% per year.

**Example:** \$100,000 grows to \$180,000 over 5 years. CAGR $= (1.8)^{0.2} - 1 \approx 12.5\%$ per year.

```python
sim_cagrs = (ending_values / total_val) ** (1 / 5) - 1
```

This is computed for each of the 1,000,000 simulations, producing 1,000,000 different CAGRs — one for each "alternate universe."

---

**Formula 15: Sharpe Ratio** ([line 141](https://github.com/Guannings/MonteCarlo-Portfolio-Optimizer/blob/49b0e61/Gemini_generated%20_codes/Monte-Carlo%20Sim/latest_code.py#L141))

$$S = \frac{\bar{R}_p - R_f}{\sigma_p \cdot \sqrt{252}}$$

This answers: **"How much extra return am I getting for each unit of risk I'm taking?"**

- $\bar{R}_p$ = the mean CAGR across all 1,000,000 simulations.
- $R_f$ = the risk-free rate — what you would earn with zero risk (e.g., a government savings account). The code uses 3.5% (0.035).
- $\sigma_p \cdot \sqrt{252}$ = the daily standard deviation (Formula 9) scaled up to annual.

**Example:** If mean CAGR = 15%, risk-free rate = 3.5%, and annual volatility = 18%, then $S = \frac{0.15 - 0.035}{0.18} = 0.64$. A higher Sharpe means you are being better compensated for the risk. A Sharpe below 0 means you would be better off in a savings account.

```python
sharpe = (mean_cagr - 0.035) / (port_std * np.sqrt(252))
```

---

**Formula 16: Annualized Volatility** ([line 153](https://github.com/Guannings/MonteCarlo-Portfolio-Optimizer/blob/49b0e61/Gemini_generated%20_codes/Monte-Carlo%20Sim/latest_code.py#L153))

$$\sigma_{\text{annual}} = \sigma_{\text{daily}} \times \sqrt{252}$$

This converts the daily standard deviation (a tiny number like 0.01) into an annual percentage (like 15.87%). Same `× √252` scaling explained in Formula 5.

```python
port_std * np.sqrt(252)
```

---

**Formula 17: 95% Confidence Interval (Percentiles)** ([lines 144–147](https://github.com/Guannings/MonteCarlo-Portfolio-Optimizer/blob/49b0e61/Gemini_generated%20_codes/Monte-Carlo%20Sim/latest_code.py#L144-L147))

$$\text{CI}_{95\%} = \big[ P_{2.5}, \quad P_{97.5} \big]$$

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

**Formula 18: Monthly Allocation** ([line 227](https://github.com/Guannings/MonteCarlo-Portfolio-Optimizer/blob/49b0e61/Gemini_generated%20_codes/Monte-Carlo%20Sim/latest_code.py#L227))

$$A_i = B \times w_i$$

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

# Appendix: Full Mathematical Derivations

This appendix derives every core formula used in the code from first principles. Each derivation starts from the simplest possible foundation and builds up step by step. Cross-references to the corresponding code lines in [`latest_code.py`](https://github.com/Guannings/MonteCarlo-Portfolio-Optimizer/blob/49b0e61/Gemini_generated%20_codes/Monte-Carlo%20Sim/latest_code.py) are provided throughout.

---

## A.1 — Daily Return

**Code reference:** [line 38](https://github.com/Guannings/MonteCarlo-Portfolio-Optimizer/blob/49b0e61/Gemini_generated%20_codes/Monte-Carlo%20Sim/latest_code.py#L38)

**Starting point:** We have a time series of daily closing prices for an asset: P₀, P₁, P₂, ..., Pₙ.

**Goal:** Measure how much the price changed each day, as a proportion of the previous day's price.

**Derivation:**

The **simple return** on day $t$ is defined as:

$$r_t = \frac{P_t - P_{t-1}}{P_{t-1}}$$

This can be rewritten as:

$$r_t = \frac{P_t}{P_{t-1}} - 1$$

Both forms say the same thing: "the new price divided by the old price, minus 1." If the price went from $100 to $103, the return is `103/100 - 1 = 0.03` (3%). If it dropped to $95, the return is `95/100 - 1 = -0.05` (-5%).

**Why use returns instead of raw prices?** Because returns are *scale-independent*. A $1 move on a $10 stock (10% return) is a much bigger deal than a $1 move on a $500 stock (0.2% return). Returns let us compare assets of different price levels on equal footing.

---

## A.2 — Mean (Expected) Daily Return

**Code reference:** [line 45](https://github.com/Guannings/MonteCarlo-Portfolio-Optimizer/blob/49b0e61/Gemini_generated%20_codes/Monte-Carlo%20Sim/latest_code.py#L45)

**Starting point:** A series of daily returns r₁, r₂, ..., rₙ for one asset.

**Goal:** Estimate the asset's "typical" daily performance.

**Derivation:**

The **arithmetic mean** is:

$$\mu = \frac{1}{N}\sum_{i=1}^{N} r_i$$

In plain English: add up all N daily returns, then divide by N. This gives the average daily return.

**Example with 5 days:** Returns are +2%, -1%, +0.5%, +1.5%, -0.5%.

$$\mu = \frac{0.02 + (-0.01) + 0.005 + 0.015 + (-0.005)}{5} = \frac{0.025}{5} = 0.005 \quad(\text{i.e., } 0.5\%\text{ per day})$$

**Why the arithmetic mean?** The code uses it as an input to the optimizer (Formula 4) and the Monte Carlo simulation (Formula 7). It represents the *expected value* — the return you would predict for any given future day if you had no other information. It is the simplest unbiased estimator of the true daily expected return.

---

## A.3 — Covariance and the Covariance Matrix

**Code reference:** [line 44](https://github.com/Guannings/MonteCarlo-Portfolio-Optimizer/blob/49b0e61/Gemini_generated%20_codes/Monte-Carlo%20Sim/latest_code.py#L44)

**Starting point:** Daily returns for two assets, A and B: (rᴬ₁, rᴮ₁), (rᴬ₂, rᴮ₂), ..., (rᴬₙ, rᴮₙ).

**Goal:** Measure whether A and B tend to move in the same direction or opposite directions.

### Step 1: Variance (one asset)

First, consider just one asset A. Its **variance** measures how spread out its returns are around its mean:

$$\text{Var}(A) = \frac{1}{N}\sum_{i=1}^{N}\left(r_i^A - \mu^A\right)^2$$

- `rᴬᵢ - μᴬ` = how far day i's return was from the average. This is called the **deviation**.
- Squaring it makes all deviations positive (so above-average and below-average days don't cancel out) and penalizes large deviations more than small ones.
- Averaging all the squared deviations gives the variance.

**Example:** If an asset's returns are +3%, -1%, +2%, with mean = +1.33%, then:
$$\text{Var} = \frac{(3-1.33)^2 + (-1-1.33)^2 + (2-1.33)^2}{3} = \frac{2.79 + 5.43 + 0.45}{3} = 2.89$$

### Step 2: Standard Deviation

Variance is in "squared" units, which is hard to interpret. The **standard deviation** is simply the square root of variance, which puts it back into the same units as the original returns:

$$\sigma^A = \sqrt{\text{Var}(A)}$$

In our example: `σ = √2.89 ≈ 1.70%`. This tells us "on a typical day, the asset's return deviates about 1.70 percentage points from its average."

### Step 3: Covariance (two assets)

Now bring in a second asset B. The **covariance** between A and B is:

$$\text{Cov}(A, B) = \frac{1}{N}\sum_{i=1}^{N}\left(r_i^A - \mu^A\right)\left(r_i^B - \mu^B\right)$$

Instead of squaring one asset's deviation, we *multiply* A's deviation by B's deviation on the same day.

- If both are above average on the same day: `(+) × (+) = (+)`. Positive contribution.
- If both are below average on the same day: `(-) × (-) = (+)`. Also positive.
- If one is above and the other below: `(+) × (-) = (-)`. Negative contribution.

If the average of all these products is **positive**, the two assets tend to move together. If **negative**, they tend to move in opposite directions (this is what makes them good diversifiers).

**Important special case:** `Cov(A, A) = Var(A)`. An asset's covariance with *itself* is just its own variance.

### Step 4: The Full Covariance Matrix (Σ)

With 6 assets, we compute covariance for every possible pair. This gives us a 6×6 table (matrix), denoted **Σ** (capital Sigma):

```
        SPYM     QQQ      VEA      AGG      GLDM     BTC
SPYM  [ Var(S)   Cov(S,Q) Cov(S,V) Cov(S,A) Cov(S,G) Cov(S,B) ]
QQQ   [ Cov(Q,S) Var(Q)   Cov(Q,V) Cov(Q,A) Cov(Q,G) Cov(Q,B) ]
VEA   [ Cov(V,S) Cov(V,Q) Var(V)   Cov(V,A) Cov(V,G) Cov(V,B) ]
AGG   [ Cov(A,S) Cov(A,Q) Cov(A,V) Var(A)   Cov(A,G) Cov(A,B) ]
GLDM  [ Cov(G,S) Cov(G,Q) Cov(G,V) Cov(G,A) Var(G)   Cov(G,B) ]
BTC   [ Cov(B,S) Cov(B,Q) Cov(B,V) Cov(B,A) Cov(B,G) Var(B)   ]
```

Properties of this matrix:
- The **diagonal** entries are each asset's own variance.
- The matrix is **symmetric**: `Cov(A,B) = Cov(B,A)` (the order doesn't matter).
- There are `6 × 6 = 36` cells, but because of symmetry, only `6 + 15 = 21` are unique (6 variances + 15 unique pairs).

---

## A.4 — Expected Portfolio Return

**Code reference:** [lines 50–51](https://github.com/Guannings/MonteCarlo-Portfolio-Optimizer/blob/49b0e61/Gemini_generated%20_codes/Monte-Carlo%20Sim/latest_code.py#L50-L51)

**Starting point:** 6 assets with mean daily returns μ₁, μ₂, ..., μ₆, and weights w₁, w₂, ..., w₆.

**Goal:** Calculate the expected daily return of the combined portfolio, then annualize it.

### Step 1: Daily portfolio return

On any given day t, the portfolio return is the weighted sum of individual asset returns:

$$R_{p,t} = w_1 r_{1,t} + w_2 r_{2,t} + \cdots + w_6 r_{6,t}$$

This is because if you put 60% into Asset 1 and it goes up 2%, that contributes `0.60 × 2% = 1.2%` to your overall portfolio return. Add up contributions from all 6 assets and you get the total portfolio return for that day.

### Step 2: Expected value

The **expected value** (mean) of a sum is the sum of the expected values:

$$E[R_p] = w_1 \mu_1 + w_2 \mu_2 + \cdots + w_6 \mu_6 = \sum_{i=1}^{6} w_i \mu_i$$

This is a basic property of expectations in probability theory, and it holds regardless of how the assets are correlated.

### Step 3: Annualization

To convert from daily to annual, multiply by the number of trading days:

$$E[R_{p,\text{annual}}] = E[R_{p,\text{daily}}] \times 252$$

**Why 252?** Stock markets are open approximately 252 days per year (365 days minus weekends and holidays). If you expect to earn 0.05% per day, your annual expectation is `0.05% × 252 = 12.6%`.

**Why simple multiplication works for returns:** This is an approximation. It assumes that daily returns are small enough that compounding effects are negligible at the daily level. For typical daily returns of 0.01%–0.10%, this approximation is extremely accurate.

### Step 4: The negation trick

Since `scipy.optimize.minimize` can only minimize, the code negates the return:

$$\text{Objective} = -\left(\sum_{i=1}^{6} w_i \mu_i\right) \times 252$$

Minimizing `-R` is mathematically identical to maximizing `R`.

---

## A.5 — Portfolio Volatility (The Matrix Formula)

**Code reference:** [lines 54–56](https://github.com/Guannings/MonteCarlo-Portfolio-Optimizer/blob/49b0e61/Gemini_generated%20_codes/Monte-Carlo%20Sim/latest_code.py#L54-L56)

**Starting point:** Weights w₁...w₆, covariance matrix Σ.

**Goal:** Calculate the portfolio's overall volatility (standard deviation of returns).

This is the most important derivation in the entire project, because it captures *why diversification works mathematically*.

### Step 1: Start with 2 assets

Let's first derive this for a simple 2-asset portfolio, then generalize.

Portfolio return on day t:
$$R_p = w_1 r_1 + w_2 r_2$$

The **variance of a sum** of random variables is NOT just the sum of variances. The full rule from probability theory is:

$$\text{Var}(aX + bY) = a^2 \text{Var}(X) + b^2 \text{Var}(Y) + 2ab  \text{Cov}(X, Y)$$

Applying this to our portfolio (where $a = w_1$, $b = w_2$, $X = r_1$, $Y = r_2$):

$$\text{Var}(R_p) = w_1^2  \text{Var}(r_1) + w_2^2  \text{Var}(r_2) + 2  w_1 w_2  \text{Cov}(r_1, r_2)$$

**This is the key insight:** The portfolio's risk depends not just on each asset's individual risk (`Var(r₁)`, `Var(r₂)`), but also on how they interact (`Cov(r₁,r₂)`). If the covariance is negative (they move in opposite directions), the third term *reduces* the total variance. This is why mixing negatively-correlated assets reduces risk.

### Step 2: Expand to N assets

For N assets, the same logic extends:

$$\text{Var}(R_p) = \sum_{i=1}^{N}\sum_{j=1}^{N} w_i  w_j  \text{Cov}(r_i, r_j)$$

This is a double sum: for every pair of assets (i, j), multiply their weights together and multiply by their covariance, then add everything up. Note that when i = j, `Cov(rᵢ,rᵢ) = Var(rᵢ)`, so the individual variances are included as special cases.

### Step 3: Write it as a matrix multiplication

The double sum above is exactly what matrix multiplication does. If we define:

- **w** = column vector of weights: `[w₁, w₂, ..., w₆]ᵀ`
- **Σ** = the 6×6 covariance matrix

Then:

$$\text{Var}(R_p) = \mathbf{w}^\top \Sigma \mathbf{w}$$

**Why this works:** The matrix multiplication `Σw` produces a vector where each entry is a weighted sum of covariances. Then `wᵀ` (the weights as a row) multiplies and sums those entries, giving the single number we need. It's a compact notation for the double sum.

**Concrete 2-asset example:** Let $\mathbf{w} = [0.6, 0.4]$ and:

$$\Sigma = \begin{bmatrix} 0.04 & 0.01 \\ 0.01 & 0.02 \end{bmatrix}$$

$$\Sigma\mathbf{w} = \begin{bmatrix} 0.04 \times 0.6 + 0.01 \times 0.4 \\ 0.01 \times 0.6 + 0.02 \times 0.4 \end{bmatrix} = \begin{bmatrix} 0.028 \\ 0.014 \end{bmatrix}$$

$$\mathbf{w}^\top\Sigma\mathbf{w} = 0.6 \times 0.028 + 0.4 \times 0.014 = 0.0168 + 0.0056 = 0.0224$$

### Step 4: From variance to standard deviation

$$\sigma_{p,\text{daily}} = \sqrt{\mathbf{w}^\top \Sigma \mathbf{w}}$$

The square root converts variance (squared units) back to standard deviation (same units as returns).

In our example: `σ = √0.0224 ≈ 0.1497`, or about 14.97% daily.

### Step 5: Annualization — why √252, not 252

**Claim:** If daily returns are independent, then:
$$\text{Var}_{\text{annual}} = 252 \times \text{Var}_{\text{daily}}, \qquad \sigma_{\text{annual}} = \sqrt{252} \times\sigma_{\text{daily}}$$

**Proof:** The annual return is approximately the sum of 252 daily returns. A fundamental property of independent random variables states:

$$\text{Var}(X_1 + X_2 + \cdots + X_{252}) = \text{Var}(X_1) + \text{Var}(X_2) + \cdots + \text{Var}(X_{252})$$

If each day has the same variance $\sigma^2$:

$$\text{Var}_{\text{annual}} = 252 \times \sigma^2$$

Taking the square root of both sides:

$$\sigma_{\text{annual}} = \sqrt{252 \times \sigma^2} = \sqrt{252} \times\sigma \approx 15.87 \times \sigma$$

**Intuition:** Risk doesn't grow as fast as time. If you flip a coin 4 times instead of 1, you don't get 4× more deviation from 50/50 — you get 2× more (√4 = 2). The same applies to portfolio returns.

### Final formula:

$$\sigma_{p,\text{annual}} = \sqrt{\mathbf{w}^\top \Sigma \mathbf{w}}  \times \sqrt{252}$$

The constraint in the code checks that this value does not exceed 18%.

---

## A.6 — Geometric Brownian Motion (GBM)

**Code reference:** [lines 127–134](https://github.com/Guannings/MonteCarlo-Portfolio-Optimizer/blob/49b0e61/Gemini_generated%20_codes/Monte-Carlo%20Sim/latest_code.py#L127-L134)

This is the mathematical model used to simulate how the portfolio's value evolves over time. GBM is the standard model in quantitative finance (it underpins the Black-Scholes option pricing formula).

### Step 1: The continuous-time model

GBM assumes that the portfolio value S follows this **stochastic differential equation** (SDE):

$$\frac{dS}{S} = \mu  dt + \sigma  dW$$

In words: "The percentage change in portfolio value over a tiny time interval dt equals:
- a deterministic drift `μ dt` (the expected return), plus
- a random shock `σ dW` (volatility times a random Brownian increment)."

`dW` is a **Wiener process** increment — a random draw from a Normal distribution with mean 0 and variance dt.

### Step 2: Solving the SDE — where the Itô correction comes from

To simulate prices, we need to solve for S(t). Using **Itô's Lemma** (the stochastic calculus version of the chain rule), we apply a change of variables by taking the natural logarithm of S:

Let $Y = \ln(S)$. By Itô's Lemma:

$$dY = d(\ln S) = \frac{1}{S}  dS - \frac{1}{2}\frac{1}{S^2}(dS)^2$$

The `- (1/2)(1/S²)(dS)²` term is what makes stochastic calculus different from ordinary calculus. In ordinary calculus, `(dt)²` is negligible. But in stochastic calculus, `(dW)²` behaves like `dt` (this is called the **quadratic variation** of Brownian motion), so this term does NOT vanish.

Substituting $dS = S(\mu  dt + \sigma  dW)$:

$$(dS)^2 = S^2(\mu  dt + \sigma  dW)^2 \approx S^2 \sigma^2  dt \qquad\text{(keeping only the }dt\text{-order term)}$$

Therefore:

$$dY = \left(\mu - \tfrac{1}{2}\sigma^2\right)dt + \sigma  dW$$

The `- ½σ²` is the **Itô correction**. It emerges naturally from the mathematics of stochastic calculus and is NOT an arbitrary adjustment.

### Step 3: Integrating over one time step

Over a discrete time step Δt (one trading day, so Δt = 1/252):

$$\ln S_t - \ln S_{t-1} = \left(\mu - \tfrac{1}{2}\sigma^2\right)\Delta t + \sigma\sqrt{\Delta t} Z$$

where $Z \sim \mathcal{N}(0,1)$ is a standard normal random variable.

Exponentiating both sides:

$$\frac{S_t}{S_{t-1}} = \exp\left[\left(\mu - \tfrac{1}{2}\sigma^2\right)\Delta t + \sigma\sqrt{\Delta t} Z\right]$$

Which gives us the **price evolution formula**:

$$S_t = S_{t-1} \times \exp\left[\left(\mu - \tfrac{1}{2}\sigma^2\right)\Delta t + \sigma\sqrt{\Delta t} Z\right]$$

### Step 4: Simplification used in the code

The code uses daily time steps where Δt = 1/252. However, since the input parameters (port_ret and port_std) are already in daily units (not annualized), the code simplifies to Δt = 1:

$$\text{drift} = \mu_p - \tfrac{1}{2}\sigma_p^2 \qquad G_t = e^{\text{drift} + \sigma_p Z} \qquad S_t = S_{t-1} \times G_t$$

### Why the Itô correction matters — numerical proof

Without the correction (using `drift = port_ret` instead of `port_ret - 0.5 × port_cov`):

Suppose daily return μ = 0.05% and daily variance σ² = 0.01%. Over 252 days, the naive expected growth factor would be:

$$E\left[e^{\mu \times 252}\right] = e^{0.0005 \times 252} = e^{0.126} \approx 1.1342 \quad(13.42\%\text{ annual})$$

But the actual expected growth of $e^{\mu + \sigma Z}$ is:

$$E\left[e^{\mu + \sigma Z}\right] = e^{\mu + \frac{1}{2}\sigma^2} \qquad\text{(property of log-normal distribution)}$$

So over 252 days, the compounded expected value is:

$$E\left[\frac{S_{252}}{S_0}\right] = e^{\left(\mu + \frac{1}{2}\sigma^2\right) \times 252}$$

This is *higher* than $e^{\mu \times 252}$ — the simulation would overestimate returns. By subtracting $\frac{1}{2}\sigma^2$ from the drift, we ensure:

$$E\left[\frac{S_{252}}{S_0}\right] = e^{\left(\mu - \frac{1}{2}\sigma^2 + \frac{1}{2}\sigma^2\right) \times 252} = e^{\mu \times 252}$$

The correction makes the simulation's expected growth rate match the true expected return.

---

## A.7 — The Normal Distribution and Z-Scores

**Code reference:** [line 128](https://github.com/Guannings/MonteCarlo-Portfolio-Optimizer/blob/49b0e61/Gemini_generated%20_codes/Monte-Carlo%20Sim/latest_code.py#L128)

The **Normal (Gaussian) Distribution** is defined by its probability density function:

$$f(z) = \frac{1}{\sqrt{2\pi}} e^{-z^2/2}$$

Key properties:
- **Mean = 0**: the distribution is centered at zero.
- **Standard deviation = 1**: about 68% of values fall between -1 and +1.
- **Symmetry**: the curve is perfectly symmetric around 0.
- **Tail probabilities:**
  - ~95.4% of values fall within ±2 standard deviations.
  - ~99.7% of values fall within ±3 standard deviations.
  - Values beyond ±4 occur about 0.006% of the time.

**Why does finance use the Normal distribution?** The **Central Limit Theorem** states that the sum of many independent random effects (news events, trades, sentiment shifts) tends toward a Normal distribution, regardless of the distribution of each individual effect. Daily stock returns, being the aggregate of millions of trades, approximately follow this pattern.

**Limitation:** Real market returns have "fatter tails" than the Normal distribution predicts — extreme events (crashes, bubbles) occur more often than the bell curve would suggest. GBM with Normal shocks will underestimate the probability of extreme outcomes.

---

## A.8 — CAGR (Compound Annual Growth Rate)

**Code reference:** [line 139](https://github.com/Guannings/MonteCarlo-Portfolio-Optimizer/blob/49b0e61/Gemini_generated%20_codes/Monte-Carlo%20Sim/latest_code.py#L139)

**Starting point:** The compound interest formula.

### Step 1: Compound interest

If you invest an amount P₀ and it grows at a constant annual rate g for T years:

$$P_T = P_0 \times (1 + g)^T$$

### Step 2: Solving for g

We know $P_0$ (starting value) and $P_T$ (ending value). We want to find $g$. Rearranging:

$$(1 + g)^T = \frac{P_T}{P_0}$$

Take the T-th root of both sides (equivalently, raise to the power $1/T$):

$$1 + g = \left(\frac{P_T}{P_0}\right)^{1/T}$$

Subtract 1:

$$g = \left(\frac{P_T}{P_0}\right)^{1/T} - 1$$

This is the **CAGR formula**. It answers: "what constant annual rate would turn P₀ into Pₜ over T years?"

### Step 3: Application in the code

With P₀ = $100,000, T = 5 years, and each simulation producing a different Pₜ:

$$\text{CAGR} = \left(\frac{\text{ending\_value}}{100{,}000}\right)^{1/5} - 1$$

**Worked example:**
- Start: $100,000. End: $207,893.
- Ratio: `207,893 / 100,000 = 2.07893`
- Fifth root: `2.07893^0.2 = 1.1576`
- CAGR: `1.1576 - 1 = 0.1576` = 15.76% per year

This means $100,000 growing at a steady 15.76% for 5 years would reach $207,893.

---

## A.9 — Sharpe Ratio

**Code reference:** [line 141](https://github.com/Guannings/MonteCarlo-Portfolio-Optimizer/blob/49b0e61/Gemini_generated%20_codes/Monte-Carlo%20Sim/latest_code.py#L141)

**Starting point:** William Sharpe's 1966 insight that return alone is meaningless without considering risk.

### Derivation

The Sharpe Ratio is defined as:

$$S = \frac{R_p - R_f}{\sigma_p}$$

Where:
- `Rₚ` = the portfolio's expected return (annualized).
- `Rᶠ` = the risk-free rate (the return from a "zero-risk" investment like government bonds). In the code: 3.5%.
- `σₚ` = the portfolio's annualized standard deviation (volatility).

**What the numerator means:** `Rₚ - Rᶠ` is the **excess return** — the extra return you earn *above* what you could get risk-free. If you earn 15% but could get 3.5% risk-free, your excess return is 11.5%. This is the reward for taking risk.

**What the denominator means:** `σₚ` is the risk you took to earn that excess return.

**What the ratio means:** Return per unit of risk. If Sharpe = 0.64, you earned 0.64% excess return for every 1% of volatility you endured.

**Interpretation benchmarks:**
- `S < 0` — you did worse than the risk-free rate. Should have just bought bonds.
- `S = 0 to 0.5` — poor to mediocre risk-adjusted return.
- `S = 0.5 to 1.0` — good.
- `S = 1.0 to 2.0` — very good.
- `S > 2.0` — exceptional (rare for long-term strategies).

---

## A.10 — Annualized Volatility (The √T Rule)

**Code reference:** [line 153](https://github.com/Guannings/MonteCarlo-Portfolio-Optimizer/blob/49b0e61/Gemini_generated%20_codes/Monte-Carlo%20Sim/latest_code.py#L153)

### Full derivation of the √T scaling rule

**Claim:** If daily returns are independent and identically distributed with variance σ²daily, then the variance of the T-day return is T × σ²daily, and the standard deviation of the T-day return is √T × σdaily.

**Proof:**

Let R₁, R₂, ..., Rₜ be T independent daily returns, each with variance σ².

The T-day return (approximately) is:

$$R_{\text{total}} = R_1 + R_2 + \cdots + R_T$$

By the **independence property of variance**:

$$\text{Var}(R_{\text{total}}) = \text{Var}(R_1) + \text{Var}(R_2) + \cdots + \text{Var}(R_T)$$

(Note: this is ONLY valid because the returns are independent. If they were correlated, there would be covariance terms.)

Since each $\text{Var}(R_i) = \sigma^2$:

$$\text{Var}(R_{\text{total}}) = T \times \sigma^2$$

Taking the square root:

$$\sigma_{\text{total}} = \sqrt{T \times \sigma^2} = \sqrt{T} \times\sigma$$

For annualization: T = 252 trading days, so:

$$\sigma_{\text{annual}} = \sqrt{252} \times\sigma_{\text{daily}} \approx 15.87 \times \sigma_{\text{daily}}$$

**Concrete example:** If daily std dev = 1%, then annual std dev ≈ 15.87%. NOT 252% (which is what you'd get if risk scaled linearly).

---

## A.11 — Percentiles and the 95% Confidence Interval

**Code reference:** [lines 144–147](https://github.com/Guannings/MonteCarlo-Portfolio-Optimizer/blob/49b0e61/Gemini_generated%20_codes/Monte-Carlo%20Sim/latest_code.py#L144-L147)

### Definition

The **p-th percentile** of a dataset is the value below which p% of the data falls.

Given N = 1,000,000 simulation outcomes sorted from smallest to largest:

$$\text{Position of the }p\text{-th percentile} = \left\lceil \frac{p}{100} \times N \right\rceil$$

- The **2.5th percentile** is at position `⌈0.025 × 1,000,000⌉ = 25,000`. This means 25,000 simulations (out of 1,000,000) ended up below this value — only the worst 2.5%.
- The **97.5th percentile** is at position `⌈0.975 × 1,000,000⌉ = 975,000`. Only the best 2.5% of simulations exceeded this value.

### Why 2.5% and 97.5%?

Together they form the **95% confidence interval**: the middle 95% of outcomes fall between these two values. This is a standard statistical convention — 95% is wide enough to be reliable but narrow enough to be informative.

$$\text{95\% CI} = \big[ P_{2.5}, \quad P_{97.5} \big]$$

**Interpretation:** "We are 95% confident that the true outcome will fall within this range." Or equivalently: "In 950,000 out of 1,000,000 simulated futures, the portfolio ended up between $X and $Y."

### The relationship to the Normal distribution

For a Normal distribution with mean μ and standard deviation σ:

$$P_{2.5} \approx \mu - 1.96\sigma, \qquad P_{97.5} \approx \mu + 1.96\sigma$$

The factor 1.96 comes from the inverse of the Normal cumulative distribution function. However, the code does not assume normality — it directly computes percentiles from the raw simulation data, which is more robust.

---

## A.12 — Summary: How All Formulas Connect

The formulas in this codebase form two pipelines:

**Pipeline 1: Optimization (finding the best weights)**
```
Raw Prices
  → Formula 1 (Daily Returns)
    → Formula 2 (Mean Returns) ──────────────┐
    → Formula 3 (Covariance Matrix) ─────────┤
                                              ▼
                                    Formula 4 (Objective: maximize return)
                                    Formula 5 (Constraint: cap volatility)
                                    Formula 6 (Constraint: weights sum to 1)
                                              │
                                              ▼
                                         SLSQP Optimizer
                                              │
                                              ▼
                                      Optimal Weights [w₁...w₆]
```

**Pipeline 2: Monte Carlo Simulation (stress-testing the weights)**
```
Optimal Weights
  → Formula 7 (Portfolio daily return)
  → Formula 8 (Portfolio variance)
  → Formula 9 (Portfolio std dev)
    → Formula 10 (GBM drift with Itô correction)
    → Formula 11 (1.26 billion random shocks)
      → Formula 12 (Daily growth factors)
        → Formula 13 (Compound 1,260 days × 1M paths)
          → Formula 14 (CAGR for each path)
            → Formula 15 (Sharpe Ratio)
            → Formula 16 (Annualized Volatility)
            → Formula 17 (95% Confidence Interval)

Formula 18 (Monthly Allocation) is a standalone calculation using the optimal weights.
```

====================================================================================

# Acknowledgments & Methodology
**The conceptual architecture, financial strategy, asset selection, and risk constraints were designed by the author.**

This project was developed using an AI-Assisted Workflow.

The underlying Python syntax and library implementation were generated via Large Language Models (Gemini), demonstrating a modern approach to rapid prototyping and financial modeling where technical execution is accelerated by AI.

**This approach focuses on leveraging AI as a force multiplier for rapid prototyping and complex quantitative modeling.**

====================================================================================
