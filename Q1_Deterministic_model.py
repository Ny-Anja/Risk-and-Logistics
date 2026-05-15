
import pandas as pd
import xpress as xp
import numpy as np
from Plot import *

# 1. Load and Project Data
locs = pd.read_csv('robot_locations.csv')
ranges = pd.read_csv('range.csv')
data = pd.merge(locs, ranges, on='index')

#------------Data Sampling ---------------------
# Select random subset of Robots for 1a
subset_sizes = [20, 30, 40, 60,100]

# Sample the largest set once (no replacement = no duplicates)
sample_data = data.sample(n=max(subset_sizes), random_state=42, replace=False).reset_index(drop=True)

# Slice to get nested subsets
samples = {size: sample_data.iloc[:size].reset_index(drop=True) for size in subset_sizes}
#%%
#-----------------------------------------------
# Parameters from PDF
cb, cm, cc, ch, rmax = 5000, 500, 0.42, 1000,  175
m, q = 8, 2
subset_size=subset_sizes[4]
J_count = int(np.ceil(subset_size / (m * q)))

# Project robot coordinates to km
def get_coords(lon, lat):
    y = lat * 111
    x = lon * 111 * np.cos(np.radians(lat))
    return x, y

robot_km = [get_coords(sample_data.loc[i, 'longitude'], sample_data.loc[i, 'latitude']) 
            for i in range(subset_size)]

# Determine dynamic bounds for stations based on robot locations to avoid infeasibility
min_x = min(c[0] for c in robot_km)
max_x = max(c[0] for c in robot_km)
min_y = min(c[1] for c in robot_km)
max_y = max(c[1] for c in robot_km)

prob = xp.problem("Antarctica_Robots_MINLP")

# 2. Decision Variables (EXACT PDF notation)
Xj = [prob.addVariable(lb=min_x, ub=max_x, name=f'X_{j}') for j in range(J_count)]
Yj = [prob.addVariable(lb=min_y, ub=max_y, name=f'Y_{j}') for j in range(J_count)]
aij = {(i, j): prob.addVariable(vartype=xp.binary, name=f'a_{i}_{j}') for i in range(subset_size) for j in range(J_count)}
hij = {(i, j): prob.addVariable(vartype=xp.binary, name=f'h_{i}_{j}') for i in range(subset_size) for j in range(J_count)}
zj = [prob.addVariable(vartype=xp.binary, name=f'z_{j}') for j in range(J_count)]
nj = [prob.addVariable(vartype=xp.integer, lb=0, ub=m, name=f'n_{j}') for j in range(J_count)]
dij = {(i, j): prob.addVariable(lb=0, name=f'd_{i}_{j}') for i in range(subset_size) for j in range(J_count)}

# 3. Constraints
for i in range(subset_size):
    prob.addConstraint(xp.Sum(aij[i, j] + hij[i, j] for j in range(J_count)) == 1)

for j in range(J_count):
    prob.addConstraint(xp.Sum(aij[i, j] + hij[i, j] for i in range(subset_size)) <= q * nj[j])
    prob.addConstraint(nj[j] <= m * zj[j])
    prob.addConstraint(nj[j]>=zj[j])
for j in range(J_count):
    for i in range(subset_size):
        prob.addConstraint((aij[i, j] + hij[i, j] ) <= zj[j])

for i in range(subset_size):
    xi, yi = robot_km[i]
    ri = sample_data.loc[i, 'range']
    for j in range(J_count):
        # Nonlinear Distance (C5)
        prob.addConstraint(dij[i, j] == xp.sqrt((xi - Xj[j])**2 + (yi - Yj[j])**2))
        # Range Feasibility (Big-M)
        prob.addConstraint(dij[i, j] <= ri + 10000 * (1 - aij[i, j]))

# Symmetry Breaking
#for j in range(J_count - 1):
   # prob.addConstraint(zj[j] >= zj[j+1])

# 4. Objective (Bilinear: aij * dij)
objective = xp.Sum(cb * zj[j] + cm * nj[j] for j in range(J_count)) + \
            xp.Sum(cc * aij[i, j] * (rmax + dij[i, j]-sample_data.loc[i, 'range']) for i in range(subset_size) for j in range(J_count)) + \
            xp.Sum(ch * hij[i, j] for i in range(subset_size) for j in range(J_count))

prob.setObjective(objective, sense=xp.minimize)

prob.controls.maxtime = 500
# 5. Solve and Output with Updated API
prob.solve()

# Check Status using the 9.5+ attributes
sol_status = prob.attributes.solstatus
if sol_status in [xp.SolStatus.OPTIMAL, xp.SolStatus.FEASIBLE]:
    print(f"\nOptimization Successful!")
    print(f"Total Cost: £{prob.getObjVal():.2f}")
    for j in range(J_count):
        if prob.getSolution(zj[j]) > 0.5:
            print(f"Station {j}: ({prob.getSolution(Xj[j]):.2f}, {prob.getSolution(Yj[j]):.2f}) with {int(prob.getSolution(nj[j]))} chargers")
else:
    print(f"Solve failed with status: {sol_status}")


# ============================================================
# Results reporting for Q1a MINLP
# Paste this block AFTER prob.solve() in your script
# ============================================================

sol_status = prob.attributes.solstatus

if sol_status not in [xp.SolStatus.OPTIMAL, xp.SolStatus.FEASIBLE]:
    print(f"Solve failed with status: {sol_status}")
else:
    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------
    is_optimal = (sol_status == xp.SolStatus.OPTIMAL)
    try:
        gap = prob.attributes.miprelgap * 100
    except Exception:
        gap = 0.0

    print("=" * 60)
    print("  Q1a MINLP RESULTS")
    print("=" * 60)
    print(f"  Status        : {'Optimal' if is_optimal else 'Feasible (not proven optimal)'}")
    print(f"  MIP gap       : {gap:.4f} %")
    print(f"  Objective     : £{prob.getObjVal():,.2f}")
    print(f"  Robots        : {subset_size}")
    print(f"  Max stations  : {J_count}")

    # --------------------------------------------------------
    # Collect solution values
    # --------------------------------------------------------
    sol_z   = [prob.getSolution(zj[j])   for j in range(J_count)]
    sol_n   = [prob.getSolution(nj[j])   for j in range(J_count)]
    sol_X   = [prob.getSolution(Xj[j])   for j in range(J_count)]
    sol_Y   = [prob.getSolution(Yj[j])   for j in range(J_count)]
    sol_a   = {(i,j): prob.getSolution(aij[i,j])
               for i in range(subset_size) for j in range(J_count)}
    sol_h   = {(i,j): prob.getSolution(hij[i,j])
               for i in range(subset_size) for j in range(J_count)}
    sol_d   = {(i,j): prob.getSolution(dij[i,j])
               for i in range(subset_size) for j in range(J_count)}

    # --------------------------------------------------------
    # Station details
    # --------------------------------------------------------
    build_cost  = 0.0
    maint_cost  = 0.0
    charge_cost = 0.0
    human_cost  = 0.0
    n_open      = 0

    print("\n" + "-" * 60)
    print("  OPEN STATIONS")
    print("-" * 60)

    for j in range(J_count):
        if sol_z[j] > 0.5:
            n_open     += 1
            n_chargers  = round(sol_n[j])

            # Recover lat/lon from km
            lat_j = sol_Y[j] / 111.0
            cos_j = np.cos(np.radians(lat_j)) if abs(lat_j) < 89.9 else 1e-9
            lon_j = sol_X[j] / (111.0 * cos_j)

            # Which robots use this station
            drone_robots = [sample_data.loc[i, 'index'] for i in range(subset_size)
                            if sol_a[i,j] > 0.5]
            human_robots = [sample_data.loc[i, 'index'] for i in range(subset_size)
                            if sol_h[i,j] > 0.5]
            n_drone = len(drone_robots)
            n_human = len(human_robots)

            # Average drone distance for this station
            avg_drone_dist = (
                np.mean([sol_d[i,j] for i in range(subset_size) if sol_a[i,j] > 0.5])
                if n_drone > 0 else 0.0
            )

            print(f"\n  Station {j}")
            print(f"    Location   : lat={lat_j:.4f}, lon={lon_j:.4f}")
            print(f"    Chargers   : {n_chargers}")
            print(f"    Robots     : {n_drone} by drone, {n_human} by human")
            print(f"    Avg drone dist : {avg_drone_dist:.2f} km")
            print(f"    Drone robots   : {drone_robots}")
            print(f"    Human robots   : {human_robots}")

            build_cost += cb
            maint_cost += cm * n_chargers

    # --------------------------------------------------------
    # Robot-level summary
    # --------------------------------------------------------
    print("\n" + "-" * 60)
    print("  ROBOT ASSIGNMENTS")
    print("-" * 60)
    print(f"  {'Robot':>6}  {'Mode':>6}  {'Station':>8}  {'Dist (km)':>10}  {'Range (km)':>11}  {'In range?':>9}")
    print(f"  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*10}  {'-'*11}  {'-'*9}")

    for i in range(subset_size):
        ri = sample_data.loc[i, 'range']
        for j in range(J_count):
            if sol_a[i,j] > 0.5:
                dist    = sol_d[i,j]
                in_rng  = "Yes" if dist <= ri + 1e-4 else "No"
                rob_idx = sample_data.loc[i, 'index']
                print(f"  {rob_idx:>6}  {'drone':>6}  {j:>8}  {dist:>10.2f}  {ri:>11.2f}  {in_rng:>9}")
                charge_cost += cc * dist
            elif sol_h[i,j] > 0.5:
                dist    = sol_d[i,j]
                rob_idx = sample_data.loc[i, 'index']
                print(f"  {rob_idx:>6}  {'human':>6}  {j:>8}  {dist:>10.2f}  {ri:>11.2f}  {'N/A':>9}")
                human_cost += ch

    # --------------------------------------------------------
    # Cost breakdown
    # --------------------------------------------------------
    total = build_cost + maint_cost + charge_cost + human_cost

    print("\n" + "-" * 60)
    print("  COST BREAKDOWN")
    print("-" * 60)
    print(f"  Build cost    : £{build_cost:>10,.2f}  ({build_cost/total*100:.1f}%)")
    print(f"  Maintenance   : £{maint_cost:>10,.2f}  ({maint_cost/total*100:.1f}%)")
    print(f"  Charging      : £{charge_cost:>10,.2f}  ({charge_cost/total*100:.1f}%)")
    print(f"  Human transp  : £{human_cost:>10,.2f}  ({human_cost/total*100:.1f}%)")
    print(f"  {'─'*30}")
    print(f"  Total         : £{total:>10,.2f}")
    print(f"\n  Open stations : {n_open}")
    print("=" * 60)

#%%
plot_robots_and_stations_states(
    robot_points = robot_km,
    station_X    = sol_X,
    station_Y    = sol_Y,
    sol_n        = sol_n,
    sol_a        = sol_a,
    sol_h        = sol_h,
    robot_ranges = sample_data['range'].tolist()[:subset_size],
)

#%%
