import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("results_threads.csv")

# Make sure columns are correct
# print(df.head())

# Unique swarm sizes and thread counts
swarms = sorted(df["SwarmSize"].unique())
threads = sorted(df["Threads"].unique())

# 1) Runtime vs Threads for each swarm size
plt.figure()
for s in swarms:
    sub = df[df["SwarmSize"] == s].sort_values("Threads")
    plt.plot(sub["Threads"], sub["ParallelTime"],
             marker="o", label=f"Swarm {s}")
plt.xlabel("Number of Threads")
plt.ylabel("Parallel Runtime (s)")
plt.title("Parallel Runtime vs Threads for Different Swarm Sizes")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("runtime_vs_threads.png", dpi=200)

# 2) Speedup vs Threads for each swarm size
plt.figure()
for s in swarms:
    sub = df[df["SwarmSize"] == s].sort_values("Threads")
    plt.plot(sub["Threads"], sub["Speedup"],
             marker="o", label=f"Swarm {s}")
plt.xlabel("Number of Threads")
plt.ylabel("Speedup (SerialTime / ParallelTime)")
plt.title("Speedup vs Threads for Different Swarm Sizes")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("speedup_vs_threads.png", dpi=200)

print("Saved plots: runtime_vs_threads.png, speedup_vs_threads.png")
