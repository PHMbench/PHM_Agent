import numpy as np
import h5py
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

for i in range(2):
    path = DATA_DIR / f"{i}.h5"
    with h5py.File(path, "w") as f:
        data = np.random.randn(1024, 4)
        f.create_dataset("data", data=data)
print(f"Created sample files under {DATA_DIR}")

