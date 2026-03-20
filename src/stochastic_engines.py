import numpy as np

class GBMModel:
    def __init__(self, config):
        self.S0 = config.market_params.S0
        self.r = config.market_params.r
        self.sigma = config.market_params.sigma
        self.T_sim = config.simulation.T_sim
        self.n_paths = config.simulation.n_paths
        self.seed = config.simulation.seed
        self.n_steps = config.market_params.n_steps
        self.dt = self.T_sim / self.n_steps
        self.rng = np.random.default_rng(self.seed)

    def generate_paths(self):
        Z = self.rng.standard_normal((self.n_paths, self.n_steps))
        
        drift = (self.r - 0.5 * self.sigma**2) * self.dt
        diffusion = self.sigma * np.sqrt(self.dt) * Z
        log_returns = drift + diffusion
        
        paths = np.zeros((self.n_paths, self.n_steps + 1))
        paths[:, 0] = np.log(self.S0)
        paths[:, 1:] = np.log(self.S0) + np.cumsum(log_returns, axis=1)
        
        return np.exp(paths)


class MertonJumpDiffusion:
    def __init__(self, config):
        self.S0 = config.market_params.S0
        self.r = config.market_params.r
        self.sigma = config.market_params.sigma
        self.T_sim = config.simulation.T_sim
        self.n_paths = config.simulation.n_paths
        self.seed = config.simulation.seed
        
        self.lmbda = config.merton_params.lmbda
        self.mu_j = config.merton_params.mu_j
        self.sigma_j = config.merton_params.sigma_j
        self.n_steps = config.market_params.n_steps
        self.dt = self.T_sim / self.n_steps
        self.rng = np.random.default_rng(self.seed)
        

    def generate_paths(self):
        Z = self.rng.standard_normal((self.n_paths, self.n_steps))
        N = self.rng.poisson(self.lmbda * self.dt, (self.n_paths, self.n_steps))
        log_jumps = self.rng.normal(self.mu_j * N, self.sigma_j * np.sqrt(N))
        
        k = np.exp(self.mu_j + 0.5 * self.sigma_j**2) - 1
        
        drift = (self.r - self.lmbda * k - 0.5 * self.sigma**2) * self.dt
        diffusion = self.sigma * np.sqrt(self.dt) * Z
        log_returns = drift + diffusion + log_jumps
        
        paths = np.zeros((self.n_paths, self.n_steps + 1))
        paths[:, 0] = np.log(self.S0)
        paths[:, 1:] = np.log(self.S0) + np.cumsum(log_returns, axis=1)
        
        return np.exp(paths)