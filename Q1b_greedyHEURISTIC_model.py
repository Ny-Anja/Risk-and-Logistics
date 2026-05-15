"""
Assignment 2 - Q1b: Construction Heuristic
k-means seeding with greedy assignment.

Phase 1: place n candidate station locations at k-means cluster centroids.
         n is driven by the expected charging demand (p_i) and gamma.
Phase 2: assign every robot to its nearest station with remaining capacity.
         drone if d_ij <= r_i, else human vehicle (no distance restriction).
Phase 3: set charger counts and open indicators.
Phase 4: evaluate objective.

Runs for gamma in {2, 5, 10} and multiple subset sizes, then compares
against the MINLP lower bounds from Q1a.
"""

import csv
import math
import time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# ==============================================================================
# 1. Parameters
# ==============================================================================

cb, cm, cc, ch = 5000, 500, 0.42, 1000
m, q   = 8, 2
lam    = 0.012      # lambda
r_min  = 10.0
r_max  = 175.0
GAMMAS = [2, 5, 10]

# ==============================================================================
# 2. Load and project data
# ==============================================================================

locs   = pd.read_csv('robot_locations.csv')
ranges = pd.read_csv('range.csv')
data   = pd.merge(locs, ranges, on='index').reset_index(drop=True)

def project(lon, lat):
    """Convert decimal degrees to flat km coordinates."""
    y = lat * 111.0
    x = lon * 111.0 * math.cos(math.radians(lat))
    return x, y

coords = np.array([project(row.longitude, row.latitude)
                   for _, row in data.iterrows()])   # shape (N, 2)
robot_x = coords[:, 0]
robot_y = coords[:, 1]
robot_r = data['range'].values                        # battery range (km)

# Charging probability for each robot (used only in Phase 1 to size n)
p_i = np.exp(-lam**2 * (robot_r - r_min)**2)

# ==============================================================================
# 3. Construction heuristic
# ==============================================================================

def construction_heuristic(subset_idx, gamma):
    """
    Run the greedy construction heuristic on a subset of robots.

    Parameters
    ----------
    subset_idx : array-like of int
        Indices into the full dataset selecting the robots to use.
    gamma : float
        Scaling factor for the number of candidate stations.

    Returns
    -------
    dict with keys: cost, build, maintenance, charging, human,
                    n_stations, n_chargers_total, n_drone, n_human,
                    n_unassigned, time_s
    """
    t0 = time.perf_counter()

    idx  = np.array(subset_idx)
    n_r  = len(idx)
    xi   = robot_x[idx]
    yi   = robot_y[idx]
    ri   = robot_r[idx]
    pi   = p_i[idx]

    # ------------------------------------------------------------------
    # Phase 1: k-means candidate station locations
    # p_i used ONLY here to estimate expected demand and size n
    # ------------------------------------------------------------------
    p_bar = pi.mean()
    n_min = math.ceil(n_r * p_bar / (m * q))
    n_stations = max(1, math.ceil(gamma * n_min))

    km = KMeans(n_clusters=n_stations, random_state=42, n_init=10)
    km.fit(np.column_stack([xi, yi]))
    cx = km.cluster_centers_[:, 0]   # station X coordinates (km)
    cy = km.cluster_centers_[:, 1]   # station Y coordinates (km)

    # Pre-compute all distances robot -> station  (n_r x n_stations)
    dx  = xi[:, None] - cx[None, :]   # broadcast
    dy  = yi[:, None] - cy[None, :]
    dij = np.sqrt(dx**2 + dy**2)      # (n_r, n_stations)

    # ------------------------------------------------------------------
    # Phase 2: sorted greedy assignment
    # Every robot must be assigned (deterministic: no p_i check here).
    # Drone if d <= r_i, else human vehicle (no upper distance limit).
    # ------------------------------------------------------------------
    a_ij      = np.zeros((n_r, n_stations), dtype=int)   # drone
    h_ij      = np.zeros((n_r, n_stations), dtype=int)   # human
    assigned  = np.zeros(n_stations, dtype=int)           # robots per station

    for i in range(n_r):
        # Sort stations by distance to robot i
        order = np.argsort(dij[i])

        j_star = -1
        for j in order:
            if assigned[j] < m * q:
                j_star = j
                break

        if j_star == -1:
            # All stations full -- robot unassigned (rare with good gamma)
            continue

        if dij[i, j_star] <= ri[i]:
            a_ij[i, j_star] = 1          # drone
        else:
            h_ij[i, j_star] = 1          # human vehicle (no range check)

        assigned[j_star] += 1

    # ------------------------------------------------------------------
    # Phase 3: charger counts and open indicators
    # ------------------------------------------------------------------
    n_j = np.minimum(m, np.ceil(assigned / q).astype(int))
    z_j = (n_j >= 1).astype(int)

    # ------------------------------------------------------------------
    # Phase 4: objective value
    # ------------------------------------------------------------------
    build_cost  = cb * z_j.sum()
    maint_cost  = cm * n_j.sum()

    # Charging cost: cc * (d_ij + r_max - r_i) for drone-assigned robots
    charge_cost = 0.0
    for i in range(n_r):
        for j in range(n_stations):
            if a_ij[i, j]:
                charge_cost += cc * (dij[i, j] + r_max - ri[i])

    human_cost  = ch * h_ij.sum()
    total       = build_cost + maint_cost + charge_cost + human_cost

    t1 = time.perf_counter()

    return {
        "n_robots":        n_r,
        "gamma":           gamma,
        "n_candidates":    n_stations,
        "n_open":          int(z_j.sum()),
        "n_chargers":      int(n_j.sum()),
        "n_drone":         int(a_ij.sum()),
        "n_human":         int(h_ij.sum()),
        "n_unassigned":    n_r - int(a_ij.sum()) - int(h_ij.sum()),
        "build":           round(build_cost, 2),
        "maintenance":     round(maint_cost, 2),
        "charging":        round(charge_cost, 2),
        "human":           round(human_cost, 2),
        "total":           round(total, 2),
        "time_s":          round(t1 - t0, 4),
    }

# ==============================================================================
# 4. Run experiments
# ==============================================================================

# Subset sizes to test (matching Q1a for comparison)
# Use a fixed shuffle so subsets are nested
rng      = np.random.default_rng(42)
shuffled = rng.permutation(len(data))

SUBSET_SIZES = [20, 30, 40, 60, 100, 268, 536, 1072]

# MINLP lower bounds from Q1a (fill in from your results CSV)
# Keys are subset sizes, values are LB values
MINLP_LB = {
    20:  None,   # replace with actual LB from Q1a
    30:  None,
    40:  None,
    60:  None,
    100: None,
}

results = []
print(f"{'n':>6}  {'gamma':>5}  {'n_cand':>6}  {'n_open':>6}  "
      f"{'Heuristic (£)':>14}  {'Time(s)':>8}  {'Gap_LB(%)':>10}")
print("-" * 70)

for size in SUBSET_SIZES:
    idx = shuffled[:size].tolist()
    for gamma in GAMMAS:
        res = construction_heuristic(idx, gamma)

        # Compute gap against MINLP LB if available
        lb  = MINLP_LB.get(size)
        if lb is not None and res["total"] > 0:
            gap = (res["total"] - lb) / res["total"] * 100
            gap_str = f"{gap:>10.2f}"
        else:
            gap_str = f"{'N/A':>10}"

        print(f"{size:>6}  {gamma:>5}  {res['n_candidates']:>6}  "
              f"{res['n_open']:>6}  {res['total']:>14,.2f}  "
              f"{res['time_s']:>8.4f}  {gap_str}")

        results.append({**res, "gap_lb_pct": gap_str.strip()})

# ==============================================================================
# 5. Save results
# ==============================================================================

df = pd.DataFrame(results)
df.to_csv("Q1b_heuristic_results.csv", index=False)
print("\nSaved to Q1b_heuristic_results.csv")

# ==============================================================================
# 6. Full instance detailed output (gamma=5, all 1072 robots)
# ==============================================================================

print("\n" + "=" * 60)
print("  FULL INSTANCE DETAIL  (n=1072, gamma=5)")
print("=" * 60)
res = construction_heuristic(list(range(len(data))), gamma=5)
for k, v in res.items():
    print(f"  {k:<20}: {v}")
