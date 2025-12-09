# Strategy Adaptation Guardrails

Guidelines for LLMs adapting trading strategies to this project's framework.

## Prime Directive

**DO NOT modify the core trading logic. Not even a comma.**

The original strategy logic must remain 100% identical. Any change to indicator calculations, signal conditions, or thresholds will invalidate the strategy and break the system.

---

## What to Keep (NEVER CHANGE)

| Component | Example | Why |
|-----------|---------|-----|
| Indicator calculations | `ta.ATR(df, timeperiod=14) * 2.0` | Core logic |
| Signal conditions | `close_change > atr.shift(1)` | Entry/exit rules |
| Timeframe & resampling | `resample_to_interval(df, 180)` | Strategy design |
| Periods & multipliers | `timeperiod=14`, `* 2.0` | Tuned parameters |
| Shift operations | `.shift(1)`, `.diff()` | Prevents look-ahead bias |

---

## What to Remove

| Component | Reason |
|-----------|--------|
| `leverage()` | Handled by risk management layer |
| `custom_stake_amount()` | Handled by position sizing layer |
| `adjust_trade_position()` | Handled by meta-strategy engine |
| `populate_exit_trend()` | Handled by meta-strategy engine |
| `minimal_roi`, `stoploss` | Handled by risk management layer |
| Freqtrade decorators | Framework-specific |
| `metadata` parameter | Not used |

---

## Required Interface

### Class Structure

```python
from strategies.base_strategy import BaseStrategy, SignalType, StrategyRecommendation

class YourStrategy(BaseStrategy):
    
    MIN_CANDLES_REQUIRED = <calculated_value>
    
    def __init__(self):
        super().__init__(
            name="StrategyName",
            description="Brief description",
            timeframe="1h",  # Base timeframe expected
            lookback_hours=<calculated_value>
        )
    
    def _calculate_indicators(self, df: DataFrame) -> DataFrame:
        # EXACT copy of original populate_indicators() logic
        pass
    
    def _generate_signal(self, df: DataFrame) -> SignalType:
        # EXACT copy of original populate_entry_trend() logic
        # Return SignalType instead of setting columns
        pass
    
    def run(self, df: pd.DataFrame, timestamp: datetime) -> StrategyRecommendation:
        # Validation + call helpers + return recommendation
        pass
```

### Signal Mapping

| Original (Freqtrade) | Target (Our System) |
|----------------------|---------------------|
| `enter_long = 1` | `return SignalType.LONG` |
| `enter_short = 1` | `return SignalType.SHORT` |
| No signal | `return SignalType.HOLD` |

---

## Calculating MIN_CANDLES_REQUIRED

Formula:
```
MIN_CANDLES_REQUIRED = (longest_indicator_period * resample_factor) + buffer
```

Examples:

| Strategy | Calculation | Result |
|----------|-------------|--------|
| ATR(14) on 3h, base 1h | `14 * 3 + 10` | 52 |
| SMA(50) on 1h, base 1h | `50 * 1 + 10` | 60 |
| EMA(200) on 4h, base 1h | `200 * 4 + 10` | 810 |

---

## Calculating lookback_hours

Formula:
```
lookback_hours = (longest_indicator_period * resample_hours) + 24h_buffer
```

Examples:

| Strategy | Calculation | Result |
|----------|-------------|--------|
| ATR(14) on 3h | `14 * 3 + 24` | 66 |
| SMA(50) on 1h | `50 * 1 + 24` | 74 |

---

## Critical: Shift Operations

Original Freqtrade:
```python
dataframe['atr'].shift(1)
```

Our adaptation:
```python
last_row = df.iloc[-1]
prev_row = df.iloc[-2]
atr_prev = prev_row["atr"]  # Equivalent to shift(1)
```

**Both are equivalent. Never skip the shift - it prevents look-ahead bias.**

---

## Validation Checklist

Before submitting adapted strategy:

- [ ] All indicator calculations identical to original
- [ ] All signal conditions identical to original
- [ ] All periods/multipliers unchanged
- [ ] Shift operations preserved
- [ ] MIN_CANDLES_REQUIRED correctly calculated
- [ ] lookback_hours correctly calculated
- [ ] Freqtrade-specific code removed
- [ ] Returns `StrategyRecommendation` with correct `SignalType`
- [ ] Handles NaN values gracefully (return `HOLD`)
- [ ] Handles insufficient data (return `HOLD`)

---

## Registry Entry

After adaptation, add to `strategies_registry.json`:

```json
{
  "StrategyName": {
    "enabled": true,
    "module": "strategies.strategy_module",
    "class_name": "StrategyClassName",
    "timeframe": "1h",
    "lookback_hours": 66
  }
}
```

---

## Example: Side-by-Side Comparison

### Original (Freqtrade)
```python
def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    dataframe.loc[
        (dataframe['close_change'] > dataframe['atr'].shift(1)),
        'enter_long'] = 1
    return dataframe
```

### Adapted (Our System)
```python
def _generate_signal(self, df: DataFrame) -> SignalType:
    if len(df) < 2:
        return SignalType.HOLD
    
    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]
    
    close_change = last_row["close_change"]
    atr_prev = prev_row["atr"]
    
    if pd.isna(close_change) or pd.isna(atr_prev):
        return SignalType.HOLD
    
    if close_change > atr_prev:
        return SignalType.LONG
    
    return SignalType.HOLD
```

**Logic is identical. Only the wrapper changed.**