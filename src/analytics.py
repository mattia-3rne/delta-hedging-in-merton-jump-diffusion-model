from scipy.stats import norm
import numpy as np

VALID_OPTION_TYPES = ("call", "put")


def _validate_option_type(option_type: str) -> str:
    normalized = str(option_type).lower()
    if normalized not in VALID_OPTION_TYPES:
        raise ValueError(f"Unsupported option_type='{option_type}'. Expected one of {VALID_OPTION_TYPES}.")
    return normalized


def bsm_price(S, K, T, r, sigma, option_type: str = "call"):
    option_type = _validate_option_type(option_type)

    if T <= 0:
        if option_type == "call":
            return np.maximum(S - K, 0)
        elif option_type == "put":
            return np.maximum(K - S, 0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return price


def bsm_delta(S, K, T, r, sigma, option_type: str = "call"):
    option_type = _validate_option_type(option_type)

    if T <= 0:
        if option_type == "call":
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    if option_type == "call":
        return norm.cdf(d1)
    elif option_type == "put":
        return norm.cdf(d1) - 1