# Fast Data Transfer Guide

**Goal:** Move cached data to a client machine for an instant training start (0 wait time).

### Step 1: On Main Computer (Source)
Go to your project folder and copy this specific directory:
`data/download_data/training_data/`

*(Note: No need to copy the 'archived' folder)*

### Step 2: On Client Computer (Target)
1. Paste the folder into the exact same path in your project:
   `Project_Root/data/download_data/training_data/`

2. Open `config.py` and ensure this flag is set:
   USE_PREPROCESSED_DATA = True

### Step 3: Run
Execute the main script:
`python main.py`

**Result:** The system detects the cache, skips all calculations, and starts training instantly.