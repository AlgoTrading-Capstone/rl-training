"""
Normalization utilities used by the Bitcoin trading environment.
"""

import numpy as np
from config import INITIAL_BALANCE, MAX_POSITION_BTC, ENABLE_TURBULENCE,EXTERNAL_ASSETS




def normalize_state(balance, price_vec, tech_vec, turbulence_vec, signal_vec, holdings):
    """
    Normalization for Bitcoin RL Environment.

    Logic:
    - Prices (OHLC)    -> Divided by current Close Price (Relative scaling).
    - Volume           -> Log scaled.
    - Price Indicators -> Divided by current Close Price.
    - Oscillators      -> Divided by 100.
    - Turbulence/VIX   -> Scaled & squashed with tanh.
    - Signals          -> Passed as-is (Binary 0/1).
    """

    # ---------------------------------------------------------
    # Setup Reference Price (Current Close)
    # ---------------------------------------------------------
    # We take Close (index 3) as the reference anchor.
    reference_price = price_vec[3]

    # Safety check to avoid division by zero
    if reference_price <= 0:
        reference_price = 1.0

    # ---------------------------------------------------------
    # Balance Normalization
    # ---------------------------------------------------------
    norm_balance = balance / INITIAL_BALANCE

    # ---------------------------------------------------------
    # Price & Volume Normalization
    # ---------------------------------------------------------
    # Normalize OHLC relative to Close (results will be ~1.0)
    norm_prices = price_vec[:4] / reference_price

    # Volume: Log scale handles massive spikes better than linear division
    # log1p(vol) / 20.0 keeps typical crypto volume in range [0, 1]
    vol = price_vec[4]
    norm_vol = np.log1p(vol) / 20.0

    norm_price_features = np.hstack((norm_prices, norm_vol))

    # ---------------------------------------------------------
    # Technical Indicators Normalization
    # ---------------------------------------------------------
    norm_tech = np.zeros_like(tech_vec, dtype=np.float32)

    # GROUP A: Price Levels (SMA, Bollinger)
    # Directly relative to price.
    norm_tech[1] = tech_vec[1] / reference_price  # boll_ub
    norm_tech[2] = tech_vec[2] / reference_price  # boll_lb
    norm_tech[5] = tech_vec[5] / reference_price  # sma_30
    norm_tech[6] = tech_vec[6] / reference_price  # sma_60

    # GROUP B: Price Differences (MACD)
    # Much smaller than price. To bring them into a similar range- multiply by 50.
    norm_tech[0] = (tech_vec[0] / reference_price) * 50.0

    # GROUP C: Oscillators (RSI, DX)
    # Native range: 0 to 100. Divide by 100.
    norm_tech[3] = tech_vec[3] / 100.0  # rsi
    norm_tech[4] = tech_vec[4] / 100.0  # dx

    # ---------------------------------------------------------
    # Turbulence & VIX Normalization
    # ---------------------------------------------------------
    norm_turb_list = []
    idx = 0

    # If turbulence is enabled, take the first element (if exists)
    if ENABLE_TURBULENCE and idx < len(turbulence_vec):
        turb_val = float(turbulence_vec[idx])
        # Typical range ~[0.005, 0.05] - scale by 20, squash with tanh
        norm_turb_list.append(np.tanh(turb_val * 20.0))
        idx += 1

    # If VIX is enabled, take the next element (if exists)
    if any(asset.get('enabled', False) and asset.get('col_name') == 'vix' for asset in EXTERNAL_ASSETS) and idx < len(turbulence_vec):
        vix_val = float(turbulence_vec[idx])
        # Typical range [10, 80] - divide by 100, squash with tanh
        norm_turb_list.append(np.tanh(vix_val / 100.0))
        idx += 1

    norm_turbulence = np.array(norm_turb_list, dtype=np.float32)

    # ---------------------------------------------------------
    # Signal Normalization
    # ---------------------------------------------------------
    # Already binary 0/1
    norm_signal = signal_vec

    # ---------------------------------------------------------
    # Holdings Normalization
    # ---------------------------------------------------------
    if MAX_POSITION_BTC > 0:
        norm_holdings = holdings / MAX_POSITION_BTC
    else:
        norm_holdings = 0.0

    # ---------------------------------------------------------
    # Combine State
    # ---------------------------------------------------------
    # Stack all features into a single 1D vector
    state = np.hstack(
        (
            [norm_balance],
            norm_price_features,
            norm_tech,
            norm_turbulence,
            norm_signal,
            [norm_holdings]
        )
    ).astype(np.float32) # Cast to float32: Standard input format for Deep Learning models

    return state


def inverse_normalize_state(
    *,
    state_norm: np.ndarray,
    price_vec: np.ndarray,
    tech_vec: np.ndarray,
    turbulence_vec: np.ndarray,
    signal_vec: np.ndarray,
) -> dict:
    """
    Inverse normalization for human-readable backtesting.

    IMPORTANT:
    - This function is intended ONLY for logging / analysis.
    - It does NOT guarantee perfect numerical inversion (especially for tanh-squashed features).

    Returns:
        dict with readable, de-normalized values grouped by domain.
    """

    result = {}

    # ---------------------------------------------------------
    # Reference price (Close)
    # ---------------------------------------------------------
    close_price = float(price_vec[3])
    if close_price <= 0:
        close_price = 1.0

    idx = 0

    # ---------------------------------------------------------
    # Balance
    # ---------------------------------------------------------
    norm_balance = state_norm[idx]
    idx += 1
    result["balance"] = float(norm_balance * INITIAL_BALANCE)

    # ---------------------------------------------------------
    # Price (OHLC + Volume)
    # ---------------------------------------------------------
    # OHLC normalized relative to close
    norm_open, norm_high, norm_low, _ = state_norm[idx : idx + 4]
    idx += 4

    result["open"] = float(norm_open * close_price)
    result["high"] = float(norm_high * close_price)
    result["low"] = float(norm_low * close_price)
    result["close"] = close_price

    # Volume: inverse of log1p(vol) / 20
    norm_vol = state_norm[idx]
    idx += 1
    result["volume"] = float(np.expm1(norm_vol * 20.0))

    # ---------------------------------------------------------
    # Technical Indicators (RAW, readable)
    # ---------------------------------------------------------
    tech_raw = {}

    # NOTE:
    # Indices here MUST match normalize_state exactly.
    # We intentionally reconstruct from state_norm, not tech_vec.

    # MACD (scaled by 50 and relative to price)
    macd_norm = state_norm[idx]
    tech_raw["macd"] = float((macd_norm / 50.0) * close_price)
    idx += 1

    # Bollinger Upper
    boll_ub_norm = state_norm[idx]
    tech_raw["boll_ub"] = float(boll_ub_norm * close_price)
    idx += 1

    # Bollinger Lower
    boll_lb_norm = state_norm[idx]
    tech_raw["boll_lb"] = float(boll_lb_norm * close_price)
    idx += 1

    # RSI
    rsi_norm = state_norm[idx]
    tech_raw["rsi"] = float(rsi_norm * 100.0)
    idx += 1

    # DX
    dx_norm = state_norm[idx]
    tech_raw["dx"] = float(dx_norm * 100.0)
    idx += 1

    # SMA 30
    sma30_norm = state_norm[idx]
    tech_raw["sma_30"] = float(sma30_norm * close_price)
    idx += 1

    # SMA 60
    sma60_norm = state_norm[idx]
    tech_raw["sma_60"] = float(sma60_norm * close_price)
    idx += 1

    result["indicators"] = tech_raw

    # ---------------------------------------------------------
    # Turbulence / VIX (approx inverse of tanh)
    # ---------------------------------------------------------
    turb_raw = {}

    if ENABLE_TURBULENCE:
        norm_turb = state_norm[idx]
        idx += 1
        # inverse tanh
        turb_raw["turbulence"] = float(np.arctanh(np.clip(norm_turb, -0.999, 0.999)) / 20.0)

    if any(asset.get('enabled', False) and asset.get('col_name') == 'vix' for asset in EXTERNAL_ASSETS):
        norm_vix = state_norm[idx]
        idx += 1
        turb_raw["vix"] = float(np.arctanh(np.clip(norm_vix, -0.999, 0.999)) * 100.0)

    result["turbulence"] = turb_raw

    # ---------------------------------------------------------
    # Strategies (binary, as-is)
    # ---------------------------------------------------------
    num_strategy_vals = len(signal_vec)
    result["strategies_raw"] = signal_vec[:num_strategy_vals].tolist()
    idx += num_strategy_vals

    # ---------------------------------------------------------
    # Holdings
    # ---------------------------------------------------------
    norm_holdings = state_norm[idx]
    if MAX_POSITION_BTC > 0:
        result["holdings"] = float(norm_holdings * MAX_POSITION_BTC)
    else:
        result["holdings"] = 0.0

    return result