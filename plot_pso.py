import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load CSV file
df = pd.read_csv("results.csv")
print(df)

# Extract data
swarm = df["SwarmSize"].to_numpy()
serial = df["SerialTime"].to_numpy()
parallel = df["ParallelTime"].to_numpy()

# Line plot for runtime comparison
plt.figure(figsize=(10, 6))

plt.plot(swarm, serial, marker='o', linestyle='-', linewidth=2, label="Serial")
plt.plot(swarm, parallel, marker='s', linestyle='-', linewidth=2, label="Parallel")

plt.xticks(swarm)  # Ensure swarm sizes are labeled correctly
plt.xlabel("Swarm Size")
plt.ylabel("Execution Time (seconds)")
plt.title("Serial vs Parallel PSO Runtime Comparison")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("runtime_comparison_line.png", dpi=300)
plt.close()

print("[+] Saved runtime_comparison_line.png")

# Speedup curve (line plot)
plt.figure(figsize=(8, 5))
plt.plot(df["SwarmSize"], df["Speedup"], marker='o', linewidth=2)

plt.xlabel("Swarm Size")
plt.ylabel("Speedup (Serial / Parallel)")
plt.title("PSO Speedup vs Swarm Size")
plt.grid(True, linestyle="--", alpha=0.5)

plt.savefig("speedup_curve.png", dpi=300)
plt.close()

print("[+] Saved speedup_curve.png")
