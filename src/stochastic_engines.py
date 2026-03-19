import numpy as np

class GBMModel:
    def __init__(self, config):
        self.S0 = config.market_params.S0
        self.r = config.market_params.r
        self.sigma = config.market_params.sigma
        self.T = config.market_params.T
        self.dt = config.market_params.dt
        self.n_paths = config.simulation.n_paths
        self.seed = config.simulation.seed
        
        # Number of time steps
        self.n_steps = int(self.T / self.dt)

    def generate_paths(self):
        np.random.seed(self.seed)
        
        # Standard normal increments for Brownian Motion
        Z = np.random.standard_normal((self.n_paths, self.n_steps))
        
        # Calculating daily returns
        drift = (self.r - 0.5 * self.sigma**2) * self.dt
        diffusion = self.sigma * np.sqrt(self.dt) * Z
        daily_returns = np.exp(drift + diffusion)
        
        # Creating paths array
        paths = np.zeros((self.n_paths, self.n_steps + 1))
        paths[:, 0] = self.S0
        
        # Price levels
        paths[:, 1:] = self.S0 * np.cumprod(daily_returns, axis=1)
        
        return paths


class MertonJumpDiffusion:
    def __init__(self, config):
        self.S0 = config.market_params.S0
        self.r = config.market_params.r
        self.sigma = config.market_params.sigma
        self.T = config.market_params.T
        self.dt = config.market_params.dt
        self.n_paths = config.simulation.n_paths
        self.seed = config.simulation.seed
        
        # Jump parameters
        self.lmbda = config.merton_params.lmbda
        self.mu_j = config.merton_params.mu_j
        self.sigma_j = config.merton_params.sigma_j
        
        self.n_steps = int(self.T / self.dt)

    def generate_paths(self):
        np.random.seed(self.seed)
        
        Z = np.random.standard_normal((self.n_paths, self.n_steps))
        N = np.random.poisson(self.lmbda * self.dt, (self.n_paths, self.n_steps))
        log_jumps = np.random.normal(self.mu_j, self.sigma_j, (self.n_paths, self.n_steps)) * N
        
        k = np.exp(self.mu_j + 0.5 * self.sigma_j**2) - 1
        
        drift = (self.r - self.lmbda * k - 0.5 * self.sigma**2) * self.dt
        diffusion = self.sigma * np.sqrt(self.dt) * Z
        log_returns = drift + diffusion + log_jumps
        
        paths = np.zeros((self.n_paths, self.n_steps + 1))
        paths[:, 0] = np.log(self.S0)
        paths[:, 1:] = np.log(self.S0) + np.cumsum(log_returns, axis=1)
        
        return np.exp(paths)