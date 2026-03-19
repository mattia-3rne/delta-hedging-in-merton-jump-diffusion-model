# Quantitative Simulation Framework for Delta Hedging in Jump-Diffusion Environments

## 1. Abstract
This repository presents a quantitative framework for evaluating the efficacy of dynamic delta hedging strategies within both a standard Black-Scholes-Merton (BSM) environment and a Merton Jump-Diffusion market model. By simulating stochastic asset trajectories and applying discrete-time hedging formulations to European options, we demonstrate both analytically and empirically how the introduction of discontinuous price jumps disrupts the idealized replicating portfolio. Furthermore, the analysis highlights the inherent discretization risk that emerges when continuous-time continuous-state models are approximated using finite trading intervals.

---

## 2. Theoretical Background

### 2.1 The Hedging Hypothesis and Market Completeness
The foundational premise of the Black-Scholes-Merton no-arbitrage pricing framework is that derivative securities can be perfectly replicated via a continuously adjusted, self-financing portfolio comprising the underlying asset and a risk-free bond. Delta hedging attempts to immunize this portfolio against directional market risk by maintaining an aggregate sensitivity (Delta) to infinitesimal price movements of exactly zero. Under standard BSM assumptions, continuous trading and continuous price paths ensure perfect replication, thereby rendering the market "complete." 

Crucially, this theoretical construct relies on the assumption of a **frictionless market**. This implies an environment devoid of transaction costs, bid-ask spreads, and taxes, alongside the assumption that assets are perfectly divisible and short selling is unrestricted.

### 2.2 Discrete Approximation Risk
While the underlying mathematics of the BSM model assume continuous-time portfolio rebalancing, empirical trading—and the programmatic simulation implemented herein—operates across discrete temporal intervals. For instance, the baseline configuration simulates $2520$ trading steps across a $10$-year horizon. Because the hedge is adjusted at finite intervals, the portfolio inherently assumes exposure to asset price volatility between rebalancing nodes. This discretization risk systematically explains the residual variance from a theoretically perfect hedge, even under pure Geometric Brownian Motion dynamics.

### 2.3 Market Dynamics: Diffusion vs. Jump-Diffusion
Empirical asset returns frequently exhibit heavy tails and excess kurtosis that violate the standard assumptions of continuous diffusion models.
* **Geometric Brownian Motion:** The canonical assumption wherein asset prices evolve continuously over time, driven by a constant drift and Wiener process.
* **Merton Jump-Diffusion:** To better capture empirical stylized facts, such as sudden market crashes or macroeconomic shocks, the Merton model superimposes a Poisson jump process onto the continuous diffusion process. 

### 2.4 The Breakdown of Delta Hedging in Jump-Diffusion Models
Delta hedging is intrinsically a localized risk management technique; it is predicated upon the continuity of asset paths and only protects against infinitesimal, continuous price diffusions. The introduction of jump dynamics fundamentally violates this localized assumption. 

When a jump occurs, the underlying asset experiences an instantaneous, discontinuous translation from one price level to another. Because the transition is instantaneous, dynamic rebalancing during the jump is mathematically and practically impossible. Furthermore, jump-diffusion models introduce an orthogonal source of randomness alongside the standard Brownian motion. Because the hedger still only possesses a single underlying asset to trade, it becomes impossible to simultaneously span both sources of stochastic variation. Consequently, the market becomes fundamentally "incomplete." A purely delta-neutral portfolio is structurally deficient against jump risk, leading to significant, unhedged variance in the terminal Profit and Loss (P&L) distribution.

### 2.5 Financial Instruments
The primary instruments utilized for the replication strategy in this pipeline are standard European options:
* **European Call Option:** Confers the right, but not the obligation, to purchase the underlying asset at a predetermined strike price at maturity.
* **European Put Option:** Confers the right to sell the underlying asset at a predetermined strike price at maturity.

---

## 3. Mathematical Framework

### 3.1 Asset Generation and Discretization
The continuous asset price $S(t)$ is modeled as a stochastic process, integrated computationally over discrete time steps $dt$.

**Standard GBM:**
The log-return process is governed purely by continuous drift and diffusion. Simulated paths are generated via the exact discretized solution:

$$S_{t+dt} = S_t \exp\left(\left(r - \frac{1}{2}\sigma^2\right)dt + \sigma \sqrt{dt} Z\right)$$

Where $Z \sim \mathcal{N}(0,1)$, $r$ represents the risk-free interest rate, and $\sigma$ denotes the constant volatility parameter.

**Merton Jump-Diffusion:**
This specification introduces a Poisson process $N$ characterized by an expected jump intensity $\lambda$. The jump magnitudes are modeled as log-normally distributed random variables with mean $\mu_j$ and volatility $\sigma_j$. 

To preserve the no-arbitrage condition and ensure the asset appreciates at the risk-free rate under the equivalent martingale measure, the deterministic drift is adjusted via a compensator term $k = \exp(\mu_j + 0.5 \sigma_j^2) - 1$. 

$$S_{t+dt} = S_t \exp\left(\left(r - \lambda k - \frac{1}{2}\sigma^2\right)dt + \sigma \sqrt{dt} Z + \sum_{i=1}^{N_{dt}} Y_i\right)$$

### 3.2 Pricing and the Derivation of Delta
The framework employs the standard analytic solutions for BSM option pricing to extract the required Greeks for dynamic rebalancing. The price of a European Call $C$ is formulated as:

$$C = S \Phi(d_1) - K \exp(-rT) \Phi(d_2)$$

Where:
$$d_1 = \frac{\ln(S/K) + (r + 0.5 \sigma^2) T}{\sigma \sqrt{T}}$$
$$d_2 = d_1 - \sigma \sqrt{T}$$

**Deriving the Delta ($\Delta$):**
Delta represents the first-order partial derivative of the option valuation with respect to the underlying asset's price: $\Delta = \frac{\partial C}{\partial S}$.
Applying the chain rule to the call pricing function yields:

$$\frac{\partial C}{\partial S} = \Phi(d_1) + S \phi(d_1) \frac{\partial d_1}{\partial S} - K \exp(-rT) \phi(d_2) \frac{\partial d_2}{\partial S}$$

Because the structural relationship between $d_1$ and $d_2$ is linear with respect to $S$, their partial derivatives evaluating the sensitivity to the underlying asset are identical:

$$\frac{\partial d_1}{\partial S} = \frac{\partial d_2}{\partial S} = \frac{1}{S \sigma \sqrt{T}}$$

Furthermore, a fundamental mathematical identity within the Black-Scholes partial differential equation establishes that:

$$S \phi(d_1) = K \exp(-rT) \phi(d_2)$$

Where $\phi(\cdot)$ denotes the probability density function of the standard normal distribution. By substituting these derived relationships back into the expanded partial derivative equation, the subsequent terms perfectly negate one another. This algebraic cancellation isolates the cumulative distribution function, yielding the final, simplified expressions for the theoretical deltas:

$$\Delta_{call} = \Phi(d_1)$$
$$\Delta_{put} = \Phi(d_1) - 1$$

The programmatic implementation of these analytic solutions also integrates robust handling for time-boundary conditions, ensuring $\Delta$ evaluates to discrete states ($1.0$ or $0.0$) as the time to maturity $T \le 1e-10$.

---

## 4. Pipeline Architecture

The computational framework is structured as a sequential, vectorized pipeline to efficiently evaluate the discrete hedging strategy across thousands of Monte Carlo paths.

| Phase | Process | Methodological Details |
| :--- | :--- | :--- |
| **1** | **Market Simulation** | Simulates stochastic price trajectories $S(t)$ via exact discretization. Accommodates both pure diffusion (GBM) and compound Poisson jump processes. Initializes underlying assets over the defined trading horizon. |
| **2** | **Analytic Valuation** | Evaluates the continuous-time BSM option pricing formulations and extracts the theoretical first-order sensitivities ($\Delta$) at every discrete temporal node. Enforces boundary conditions at terminal maturity ($T \le 1e-10$). |
| **3** | **Strategy Execution** | Simulates the self-financing replicating portfolio. Iteratively rebalances the underlying asset position by $\Delta_t$ at each discrete time step, continuously compounding the residual capital in the risk-free bank account to derive the terminal unhedged P&L. |

---

## Getting Started

### Prerequisites
* Python 3.8+
* Jupyter Environment

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/mattia-3rne/delta-hedging-in-merton-jump-diffusion-model.git
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Pipeline**:
    Open the notebooks in order:
    1.  `01_market_simulation.ipynb`
    2.  `02_delta_hedging.ipynb`

---

## Repository Structure

### Notebooks
* `notebooks/01_market_simulation.ipynb`: Generates simulated paths.
* `notebooks/02_delta_hedging.ipynb`: Handling the delta-hedging.

### Source Code
* `src/stochastic_engines.py`: Engines for path generation.
* `src/config_loader.py`: Functions for parsing YAML.
* `src/analytics.py`: Quantitative formulations.

### Configuration & Data
* `requirements.txt`: List of the Python dependecies.
* `config/`: Central parameter configuration.
* `data/`: Directory structure for storing data.