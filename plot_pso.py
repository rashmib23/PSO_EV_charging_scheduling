import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Load CSV file
df = pd.read_csv("results.csv")

print(df)

# Convert swarm sizes to numpy array
swarm = df["SwarmSize"].to_numpy()
serial = df["SerialTime"].to_numpy()
parallel = df["ParallelTime"].to_numpy()

# Create X positions for bars
x = np.arange(len(swarm))
bar_width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(x - bar_width/2, serial, width=bar_width, label="Serial")
plt.bar(x + bar_width/2, parallel, width=bar_width, label="Parallel")

plt.xticks(x, swarm)   # Ensure labels show 100, 1000, 5000

plt.xlabel("Swarm Size")
plt.ylabel("Execution Time (seconds)")
plt.title("Serial vs Parallel PSO Runtime Comparison")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("runtime_comparison_fixed.png", dpi=300)
plt.close()

print("Saved runtime_comparison_fixed.png")

# 2. Speedup Curve (Line Plot)
# ==============================

plt.figure(figsize=(8, 5))

plt.plot(df["SwarmSize"], df["Speedup"], marker='o', linewidth=2)

plt.xlabel("Swarm Size")
plt.ylabel("Speedup (Serial / Parallel)")
plt.title("PSO Speedup vs Swarm Size")
plt.grid(True, linestyle="--", alpha=0.5)

plt.savefig("speedup_curve.png", dpi=300)
plt.close()

print("[+] Saved speedup_curve.png")
