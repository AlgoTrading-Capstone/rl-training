"""
Normalization utilities used by the Bitcoin trading environment.
Based on FinRL scaling approach (power-of-two downscaling).
"""

import numpy as np


def normalize_state(balance, price_vec, tech_vec, signal_vec, holdings):
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

    # Technical indicators: we can scale all indicators uniformly
    # or individually if needed. FinRL often uses per-index scaling.
    # For now: uniform scaling similar to price.
    norm_tech = tech_vec * 2**-15

    # Strategy signals (LONG/SHORT/HOLD/FLAT → numeric)
    norm_signal = signal_vec * 2**-1

    # Holdings (BTC position)
    norm_holdings = holdings * 2**-4

    # Final state vector
    state = np.hstack(
        (norm_balance, norm_price, norm_tech, norm_signal, norm_holdings)
    ).astype(np.float32)

    return state