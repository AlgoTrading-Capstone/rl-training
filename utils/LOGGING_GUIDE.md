# Logging Guide for Developers

## Quick Start

**Use logger for backend operations, use print() for user interaction.**

## When to Use What

### Use Logger (Backend Operations)
- Data processing steps
- Model training progress
- File I/O operations
- Error reporting
- Performance metrics
- Debug information

### Use print() (User Interaction)
- Interactive prompts asking for input
- User choice menus
- Confirmation dialogs
- **Note**: print() is ONLY allowed in `utils/user_input.py`

## Logger Levels

1. **DEBUG**: Detailed diagnostic info for troubleshooting
   - Use for: verbose variable values, intermediate calculations
   - Example: `logger.debug(f"Loaded {len(df)} rows from cache")`

2. **INFO**: General progress updates
   - Use for: workflow steps, milestones, status messages
   - Example: `logger.info("Starting data download")`

3. **WARNING**: Non-critical issues that don't stop execution
   - Use for: fallback behavior, deprecated features, potential problems
   - Example: `logger.warning("VIX data unavailable, using forward fill")`

4. **ERROR**: Failures requiring attention
   - Use for: caught exceptions, validation failures
   - Example: `logger.error("Failed to load model checkpoint")`

5. **SUCCESS**: Successful completion of operations
   - Use for: major milestones, completed phases
   - Example: `logger.success("Training completed successfully")`

6. **EXCEPTION**: Errors with full traceback
   - Use for: unexpected errors you want to log before re-raising
   - Example: `logger.exception("Unexpected error in data processing")`

## Usage Patterns

### Basic Logging

```python
from utils.logger import RLLogger, LogComponent

# Create logger (only done in main.py)
logger = RLLogger(run_path=run_path, component=LogComponent.MAIN)

# Log messages
logger.info("Processing started")
logger.debug(f"Processing {item_count} items")
logger.success("Processing completed")
```

### Phase Tracking (Auto-timing)

The `phase()` context manager automatically logs start time, duration, and completion status:

```python
with logger.phase("Data Processing", 1, 3):
    # Do work here
    df = load_data()
    df = clean_data(df)
    # Automatically logs:
    # [Phase 1/3] Data Processing
    # [Phase 1/3] Completed in 2.35s
```

### Component-Based Routing

Components route logs to specific log files for better organization:

- **MAIN** → `training.log` (main pipeline orchestration)
- **TRAINING** → `training.log` (training loops, metrics)
- **DATA** → `data_pipeline.log` (data download, preprocessing)
- **STRATEGY** → `strategy_execution.log` (strategy signals)
- **BACKTEST** → `backtest.log` (backtest execution, results)

### Getting Component-Specific Logger

```python
# In DataManager class
def __init__(self, logger: RLLogger):
    # Create component-specific logger
    self.logger = logger.for_component(LogComponent.DATA)
    self.logger.info("DataManager initialized")  # → data_pipeline.log
```

### Error Context Formatting

Use `Formatter.error_context()` for consistent error messages:

```python
from utils.formatting import Formatter

try:
    result = risky_operation()
except Exception as e:
    error_msg = Formatter.error_context(
        f"ERROR DURING OPERATION: {e}",
        "Check config.py for valid settings."  # Additional context
    )
    logger.error(error_msg)
    raise
```

## Architecture Rules

### Logger Propagation Flow

```
main.py creates temp_logger (run_path=None)
    ↓
temp_logger → DataManager.__init__(logger=temp_logger)
    ↓
DataManager creates self.logger = logger.for_component(LogComponent.DATA)
    ↓
After run_path created: main.py creates file_logger (run_path=path)
    ↓
file_logger → run_training_pipeline(logger=file_logger)
```

### CRITICAL Rules

1. **Only create ONE RLLogger instance** (in main.py)
2. **Pass logger through constructors** - never create new instances
3. **Use `.for_component()`** to get component-specific loggers
4. **NEVER mix print() and logger** in the same file (except user_input.py)
5. **All backend files MUST use logger** - no print() allowed

## Best Practices

### ✅ DO:

- Use logger for all backend operations
- Use descriptive messages that help debugging
- Include relevant context in error messages
- Use `phase()` for multi-step operations with timing
- Pass logger instance through function parameters
- Use appropriate log levels (DEBUG for verbose, INFO for normal)
- Use `Formatter.error_context()` for structured error messages

### ❌ DON'T:

- Create multiple RLLogger instances
- Use print() for debug/progress info
- Log passwords or sensitive data
- Mix print() and logger in same file (except user_input.py)
- Catch exceptions without logging them
- Use INFO level for verbose debug details
- Create logger in DataManager or other components

## Migration Checklist

When converting print() statements to logger:

1. **Import logger types**:
   ```python
   from utils.logger import RLLogger, LogComponent
   from utils.formatting import Formatter
   ```

2. **Receive logger as parameter** (don't create new instance):
   ```python
   def __init__(self, logger: RLLogger):
       self.logger = logger.for_component(LogComponent.DATA)
   ```

3. **Replace print statements**:
   - `print(msg)` → `logger.info(msg)`
   - `print(f"ERROR: {e}")` → `logger.error(Formatter.error_context(...))`
   - `print(f"DEBUG: {val}")` → `logger.debug(f"DEBUG: {val}")`

4. **Use appropriate levels**:
   - Progress updates → `logger.info()`
   - Verbose details → `logger.debug()`
   - Warnings → `logger.warning()`
   - Errors → `logger.error()` + `Formatter.error_context()`
   - Completion → `logger.success()`

5. **Wrap multi-step operations**:
   ```python
   # Before:
   print("Starting phase 1...")
   do_work()
   print("Phase 1 complete")

   # After:
   with logger.phase("Phase 1", 1, 3):
       do_work()
   ```

## Common Patterns

### Pattern 1: Class initialization with logger

```python
class DataProcessor:
    def __init__(
        self,
        exchange: str,
        logger: RLLogger,  # Required parameter
    ):
        self.exchange = exchange
        self.logger = logger.for_component(LogComponent.DATA)
        self.logger.info(f"{exchange} processor initialized")
```

### Pattern 2: Error handling with context

```python
try:
    data = load_data(path)
except FileNotFoundError as e:
    error_msg = Formatter.error_context(
        f"ERROR LOADING DATA: {e}",
        f"Expected file at: {path}"
    )
    logger.error(error_msg)
    raise
```

### Pattern 3: Multi-phase operation

```python
def process_data(self, df):
    with self.logger.phase("Data Validation", 1, 3):
        df = self.validate(df)

    with self.logger.phase("Feature Engineering", 2, 3):
        df = self.add_features(df)

    with self.logger.phase("Normalization", 3, 3):
        df = self.normalize(df)

    return df
```

### Pattern 4: Conditional debug logging

```python
# Only logged if log_level = DEBUG
self.logger.debug(f"Cache hit: {cache_key}")
self.logger.debug(f"Processing {len(items)} items")

# Always logged (INFO level)
self.logger.info(f"Processed {total} items successfully")
```

## Troubleshooting

### Problem: Logs not appearing in file

**Solution**: Ensure you're using a logger with `run_path` set:
```python
# Won't write to file (console only)
temp_logger = RLLogger(run_path=None)

# Will write to files
logger = RLLogger(run_path=run_path, component=LogComponent.MAIN)
```

### Problem: Duplicate log entries

**Solution**: You're creating multiple RLLogger instances. Only create ONE in main.py, pass it everywhere else.

### Problem: Wrong component routing

**Solution**: Use `.for_component()` to create component-specific loggers:
```python
# Wrong: uses parent component
self.logger = logger

# Right: creates DATA component logger
self.logger = logger.for_component(LogComponent.DATA)
```

### Problem: Phase timing not showing

**Solution**: Make sure you're using the context manager syntax:
```python
# Wrong:
logger.phase("Processing", 1, 3)
do_work()

# Right:
with logger.phase("Processing", 1, 3):
    do_work()
```

## Examples from Codebase

### Good Example: data_manager.py

```python
class DataManager:
    def __init__(
        self,
        exchange: str,
        trading_pair: str,
        base_timeframe: str,
        logger: RLLogger,  # Required parameter
    ):
        self.logger = logger.for_component(LogComponent.DATA)
        self.logger.info("DataManager initialized")

    def get_arrays(self, start_date, end_date, strategy_list):
        with self.logger.phase("Data Loading", 1, 5):
            df = self._load_data(start_date, end_date)

        with self.logger.phase("Feature Engineering", 2, 5):
            df = self._add_features(df)

        # ... more phases

        return arrays
```

### Bad Example: What NOT to do

```python
class BadDataManager:
    def __init__(self, exchange: str):
        # BAD: Creating new logger instance
        self.logger = RLLogger(run_path=None)

        # BAD: Using print() in backend code
        print("DataManager initialized")

    def process(self, df):
        # BAD: No phase tracking
        print("Processing data...")
        result = do_work(df)

        # BAD: Mixing print and logger
        self.logger.info("Processing done")
        print(f"Result: {result}")

        return result
```

## Summary

- **Backend code**: Always use logger with appropriate component
- **User interaction**: Only in `utils/user_input.py`, use print() or rich
- **One logger**: Created in main.py, passed everywhere else
- **Component routing**: Use `.for_component()` for clean file separation
- **Error handling**: Use `Formatter.error_context()` for consistency
- **Multi-step ops**: Use `logger.phase()` for automatic timing

For questions or issues, refer to `utils/logger.py` implementation.