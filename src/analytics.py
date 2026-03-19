from scipy.stats import norm
import numpy as np

VALID_OPTION_TYPES = ("call", "put")


def _validate_option_type(option_type: str) -> str:
    """Normalize and validate the option type."""
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
        return 0.0

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    if option_type == "call":
        return norm.cdf(d1)
    elif option_type == "put":
        return norm.cdf(d1) - 1


def bsm_gamma(S, K, T, r, sigma):
    if T <= 0:
        return 0.0

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def bsm_vega(S, K, T, r, sigma):
    if T <= 0:
        return 0.0

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)


def bsm_theta(S, K, T, r, sigma, option_type: str = "call"):
    option_type = _validate_option_type(option_type)

    if T <= 0:
        return 0.0

    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    first = -(S * norm.pdf(d1) * sigma) / (2.0 * sqrtT)
    if option_type == "call":
        second = -r * K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        second = r * K * np.exp(-r * T) * norm.cdf(-d2)
    return first + second