import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
# Machine specs (Calculated in Q1)
BW_PEAK_GB_S = 51.2       
PEAK_PER_CORE_GFLOPS = 128.0
PEAK_ALL_CORES_GFLOPS = 512.0

# Ridge points (GFLOP/s / GB/s = FLOP/Byte)
RIDGE_PER_CORE = PEAK_PER_CORE_GFLOPS / BW_PEAK_GB_S
RIDGE_ALL_CORES = PEAK_ALL_CORES_GFLOPS / BW_PEAK_GB_S

# Experimental Data (N=4096, d=64)
# FLOPs = 4 * N * N * d
N = 4096
d = 64
total_flops = 4.0 * N * N * d  # ~4.295e9

# Variants: (Name, GFLOPs, Bytes_Traffic_Estimate)
# Note: Traffic estimates based on analysis in the report
variants = [
    ("fused",      1.57,  8.6e9),   # High traffic: streams K,V N times
    ("naive",      1.56,  268e6),   # Low traffic but poor locality
    ("loops",      1.82,  268e6),
    ("tiled",      4.99,  268e6),
    ("best (1T)",  12.86, 268e6),
    ("omp (8T)",   41.37, 268e6)
]

# Calculate Arithmetic Intensity (FLOP/Byte)
data_points = []
for name, gflops, bytes_traffic in variants:
    ai = total_flops / bytes_traffic
    data_points.append({"name": name, "ai": ai, "gflops": gflops})

# --- Plotting ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 7))

# 1. Draw Roofline Segments
# Bandwidth line: y = BW * x
x_bandwidth = np.logspace(-2, 2, 100)
y_bandwidth = BW_PEAK_GB_S * x_bandwidth

ax.plot(x_bandwidth, y_bandwidth, color='gray', linestyle='--', linewidth=2, label=f'Bandwidth Limit ({BW_PEAK_GB_S} GB/s)')

# Compute Plateaus
# Per-core plateau starts at RIDGE_PER_CORE
x_per_core = np.array([RIDGE_PER_CORE, 100])
y_per_core = np.array([PEAK_PER_CORE_GFLOPS, PEAK_PER_CORE_GFLOPS])
ax.plot(x_per_core, y_per_core, color='blue', linestyle='-', linewidth=2, label=f'Per-Core Peak ({PEAK_PER_CORE_GFLOPS:.0f} GFLOP/s)')

# All-core plateau starts at RIDGE_ALL_CORES
x_all_core = np.array([RIDGE_ALL_CORES, 100])
y_all_core = np.array([PEAK_ALL_CORES_GFLOPS, PEAK_ALL_CORES_GFLOPS])
ax.plot(x_all_core, y_all_core, color='red', linestyle='-.', linewidth=2, label=f'All-Core Peak ({PEAK_ALL_CORES_GFLOPS:.0f} GFLOP/s)')

# 2. Plot Data Points
colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33']
for i, pt in enumerate(data_points):
    ax.scatter(pt['ai'], pt['gflops'], s=150, c=[colors[i]], zorder=5, edgecolors='black', linewidth=1.5)
    ax.annotate(pt['name'], 
                (pt['ai'], pt['gflops']), 
                textcoords="offset points", 
                xytext=(5, 5), 
                fontsize=10, 
                fontweight='bold')

# 3. Formatting
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Arithmetic Intensity (FLOP/Byte)', fontsize=12)
ax.set_ylabel('Performance (GFLOP/s)', fontsize=12)
ax.set_title(f'Roofline Model: Attention Variants (N={N}, d={d})', fontsize=14, pad=15)

# Set limits to show relevant range
ax.set_xlim(0.01, 50)
ax.set_ylim(0.5, 600)

# Add grid and legend
ax.grid(True, which="both", ls="-", alpha=0.5)
ax.legend(loc='lower right', framealpha=0.9)

# Save
plt.tight_layout()
plt.savefig('plots/roofline.png', dpi=300, bbox_inches='tight')
print("Saved: plots/roofline.png")