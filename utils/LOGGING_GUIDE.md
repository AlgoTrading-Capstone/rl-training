# 📖 Logging Guide for Developers

## ⚡ Quick Rules (TL;DR)
1. **Backend Logic** → Use `self.logger` ONLY. **Never** use `print()`.
2. **User Interaction** → Use `print()` / `rich` (Only allowed in `user_input.py`).
3. **Instantiation** → **NEVER** do `RLLogger()` in your classes. Pass `logger` via `__init__`.

---

## 🛠 1. How to Init (The Right Way)

Always accept the logger as an argument and bind it to your **Component**.

```python
from utils.logger import RLLogger, LogComponent

class DataManager:
    def __init__(self, logger: RLLogger):
        # ✅ CORRECT: Create a view for this specific component
        self.logger = logger.for_component(LogComponent.DATA)
        self.logger.info("DataManager initialized")
```

### Available Components:
- `LogComponent.MAIN` → `training.log`
- `LogComponent.TRAINING` → `training.log`
- `LogComponent.DATA` → `data_pipeline.log`
- `LogComponent.STRATEGY` → `strategy_execution.log`
- `LogComponent.BACKTEST` → `backtest.log`

---

## 📝 2. Log Levels Cheat Sheet

| Level | Method | Usage Scenario | Output |
|-------|--------|----------------|--------|
| **DEBUG** | `self.logger.debug()` | Verbose variables, shapes, loops. | File Only |
| **INFO** | `self.logger.info()` | Standard progress updates. | Console + File |
| **WARNING** | `self.logger.warning()` | Non-critical issues / defaults used. | Console (🟡) + File |
| **ERROR** | `self.logger.error()` | Operation failed but app continues. | Console (🔴) + File |
| **SUCCESS** | `self.logger.success()` | Major step completed. | Console (🟢) + File |
| **EXCEPTION** | `self.logger.exception()` | Critical crash (prints traceback). | Console + File |

---

## ⏱ 3. Automatic Phase Tracking

Use the `phase()` context manager to automatically log **Start**, **End**, **Duration**, and **Success/Failure**.

```python
# Output: [Phase 1/3] Loading Data... [Completed in 2.4s]
with self.logger.phase("Loading Data", phase_num=1, total_phases=3):
    df = self.load_csv()
    self.process(df)
```

---

## 🚨 4. Error Handling Pattern

Use `Formatter` to add context to errors before logging them.

```python
from utils.formatting import Formatter

try:
    data = download_from_api()
except ConnectionError as e:
    # ✅ Add context to the error
    msg = Formatter.error_context(f"API Download Failed: {e}", "Check VPN connection")
    self.logger.error(msg)
    raise  # 👈 Always re-raise if you can't fix it
```

---

## ❌ 5. Anti-Patterns (DO NOT DO THIS)

- ❌ `logger = RLLogger(...)` inside a class (Creates duplicate logs!).
- ❌ `print("Starting...")` in backend files (Won't be saved to file).
- ❌ `self.logger = logger` (Without `.for_component(...)` → Logs will go to wrong files).

---

## 📋 Copy-Paste Template

```python
from utils.logger import RLLogger, LogComponent

class NewModule:
    def __init__(self, logger: RLLogger):
        # 1. Setup Logger
        self.logger = logger.for_component(LogComponent.TRAINING)

    def run(self):
        # 2. Use Phase
        with self.logger.phase("Executing Logic", 1, 1):
            self.logger.debug("Starting calculation...")

            # ... Your Code ...

            self.logger.success("Logic finished successfully")
```
