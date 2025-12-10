"""
Normalization utilities used by the Bitcoin trading environment.
Based on FinRL scaling approach (power-of-two downscaling).
"""

import numpy as np


TECH_SCALE_FACTORS = np.array([
    2**-1,   # macd
    2**-15,  # boll_ub
    2**-15,  # boll_lb
    2**-6,   # rsi_30
    2**-6,   # dx_30
    2**-15,  # close_30_sma
    2**-15,  # close_60_sma
], dtype=np.float32)


def normalize_state(balance, price_vec, tech_vec, turbulence_vec, signal_vec, holdings):
    """
    Apply FinRL-style normalization to all components of the state.

    Parameters
    ----------
    balance : float
        Current USD balance.
    price_vec : np.ndarray
        Price features for the current timestep.
    tech_vec : np.ndarray
        Technical indicator features.
    turbulence_vec : np.ndarray
        Turbulence & VIX features.
    signal_vec : np.ndarray
        Strategy signal features (already numeric).
    holdings : float
        BTC position size (can be positive or negative).

    Returns
    -------
    np.ndarray
        The normalized state vector.
    """

    # Account balance
    norm_balance = balance * 2**-18

    # Price features
    norm_price = price_vec * 2**-15

    # Technical indicators - per-index scaling
    norm_tech = tech_vec * TECH_SCALE_FACTORS

    # Turbulence & VIX
    norm_turbulence = ???

    # Strategy signals
    norm_signal = signal_vec  # Keep as-is (0/1 encoding)

    # Holdings (BTC position)
    norm_holdings = holdings * 2**-4

    # Final state vector
    state = np.hstack(
        (norm_balance, norm_price, norm_tech, norm_turbulence, norm_signal, norm_holdings)
    ).astype(np.float32)

    return state