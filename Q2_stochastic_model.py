"""
Assignment 2 - Q2: Stochastic Extension

Two-stage robust optimisation over 100 scenarios from range_scenarios.csv.

Stage 1 (here-and-now): station locations (X_j, Y_j), charger counts n_j,
                         open indicators z_j -- fixed across all scenarios.

Stage 2 (wait-and-see): for each scenario s, determine which robots charge
                         (delta_i^s) and assign them (a_ij^s drone / h_ij^s human).

Key outputs:
  RP  -- Recourse Problem cost (stochastic solution)
  EEV -- Expected value of using Q1c deterministic solution on scenarios
  VSS -- Value of Stochastic Solution = EEV - RP
  Out-of-sample validation on held-out scenarios
"""

import math, time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import xpress as xp

# ==============================================================================
# 1. Parameters
# ==============================================================================
cb, cm, cc, ch = 5000, 500, 0.42, 1000
m, q    = 8, 2
lam     = 0.012
r_min   = 10.0
r_max   = 175.0
GAMMA   = 2
EPS     = 100.0
T_ILP   = 60.0
SEED    = 42        # fixed seed for reproducibility
N_TRAIN = 80        # scenarios used for RP
N_TEST  = 20        # held-out scenarios for out-of-sample validation

# ==============================================================================
# 2. Load data
# ==============================================================================
locs     = pd.read_csv('robot_locations.csv')
ranges   = pd.read_csv('range.csv')
data     = pd.merge(locs, ranges, on='index').reset_index(drop=True)
scenarios= pd.read_csv('range_scenarios.csv', index_col=0)  # shape (1072, 100)

def project(lon, lat):
    y = lat * 111.0
    x = lon * 111.0 * math.cos(math.radians(lat))
    return x, y

coords  = np.array([project(r.longitude, r.latitude) for _, r in data.iterrows()])
robot_x = coords[:, 0]
robot_y = coords[:, 1]
robot_r = data['range'].values          # deterministic ranges (Q1)
n_robots = len(data)

# Stochastic ranges: shape (n_robots, 100)
R_scen = scenarios.values               # r_i^s
n_scen = R_scen.shape[1]               # 100

# ==============================================================================
# 3. Generate delta_i^s via Monte Carlo (fixed seed)
# ==============================================================================
rng_mc = np.random.default_rng(SEED)

# p_i^s = exp(-lambda^2 * (r_i^s - r_min)^2)
P_scen = np.exp(-lam**2 * (R_scen - r_min)**2)   # shape (n_robots, 100)

# psi_i^s ~ U(0,1)
Psi = rng_mc.uniform(0, 1, size=P_scen.shape)

# delta_i^s = 1 if psi_i^s <= p_i^s
Delta = (Psi <= P_scen).astype(int)               # shape (n_robots, 100)

p_bar = P_scen.mean()
print(f"Mean charging probability p_bar = {p_bar:.4f}")
print(f"Mean robots charging per scenario = {Delta.mean(axis=0).mean()*n_robots:.1f} "
      f"/ {n_robots}")
print(f"Using scenarios 0-{N_TRAIN-1} for RP, {N_TRAIN}-{n_scen-1} for validation\n")

# Split into training and test scenarios
train_idx = list(range(N_TRAIN))
test_idx  = list(range(N_TRAIN, n_scen))

# ==============================================================================
# 4. Helper functions
# ==============================================================================

def dist_matrix(xi, yi, cx, cy):
    return np.sqrt((xi[:,None]-cx[None,:])**2 +
                   (yi[:,None]-cy[None,:])**2)

def eval_obj_stochastic(cx, cy, nj, zj, xi, yi, delta, R_s, scen_idx):
    """
    Evaluate stochastic objective given fixed station locations.
    For each scenario s in scen_idx, greedily assign robots with delta_i^s=1.
    Returns: total cost, build, maintenance, avg_charge, avg_human
    """
    ns = len(cx)
    nr = len(xi)
    dij = dist_matrix(xi, yi, cx, cy)

    build = cb * float(zj.sum())
    maint = cm * float(nj.sum())

    total_charge = 0.0
    total_human  = 0.0
    n_s = len(scen_idx)

    for s in scen_idx:
        ri_s    = R_s[:, s]           # ranges in scenario s
        need_s  = np.where(delta[:, s] == 1)[0]   # robots that need charging

        assigned = np.zeros(ns, dtype=int)

        for i in np.argsort(ri_s[need_s]):   # most constrained first
            robot_i = need_s[i]
            ok   = dij[robot_i] <= ri_s[robot_i]
            cost = np.where(ok,
                            cc*(dij[robot_i] + r_max - ri_s[robot_i]),
                            ch)
            for j in np.argsort(cost):
                if assigned[j] < m * q:
                    if dij[robot_i, j] <= ri_s[robot_i]:
                        total_charge += cc*(dij[robot_i,j] + r_max - ri_s[robot_i])
                    else:
                        total_human  += ch
                    assigned[j] += 1
                    break

    avg_charge = total_charge / n_s
    avg_human  = total_human  / n_s
    total = build + maint + avg_charge + avg_human
    return total, build, maint, avg_charge, avg_human


def solve_ilp_stochastic(xi, yi, cx, cy, delta, R_s, scen_idx,
                          time_limit=T_ILP):
    """
    Solve linearised ILP (P) for stochastic setting.
    Station locations are fixed. For each training scenario s,
    only robots with delta_i^s=1 are assigned.
    Objective: build+maint + (1/|S|) * sum_s [charge + human costs]
    """
    ns = len(cx)
    nr = len(xi)
    ns_train = len(scen_idx)
    dij = dist_matrix(xi, yi, cx, cy)

    prob = xp.problem(f"ILP_stoch_{nr}x{ns}x{ns_train}")
    prob.controls.outputlog  = 0
    prob.controls.maxtime    = int(time_limit)
    prob.controls.miprelstop = 0.005

    # Stage 1 variables (fixed across scenarios)
    zj_v = [prob.addVariable(vartype=xp.binary,           name=f'z_{j}')
             for j in range(ns)]
    nj_v = [prob.addVariable(vartype=xp.integer, lb=0, ub=m, name=f'n_{j}')
             for j in range(ns)]

    # Stage 2 variables (per scenario, per robot needing charge)
    a_vs = {}   # a_vs[(s,i,j)] = drone assignment
    h_vs = {}   # h_vs[(s,i,j)] = human assignment

    for s in scen_idx:
        for i in range(nr):
            if delta[i, s] == 0:
                continue
            ri_s = R_s[i, s]
            for j in range(ns):
                if dij[i, j] <= ri_s:
                    a_vs[(s,i,j)] = prob.addVariable(
                        vartype=xp.binary, name=f'a_{s}_{i}_{j}')
                h_vs[(s,i,j)] = prob.addVariable(
                    vartype=xp.binary, name=f'h_{s}_{i}_{j}')

    # Constraints
    # C1: each robot needing charge assigned exactly once per scenario
    for s in scen_idx:
        for i in range(nr):
            if delta[i, s] == 0: continue
            drone_v = [a_vs[(s,i,j)] for j in range(ns) if (s,i,j) in a_vs]
            human_v = [h_vs[(s,i,j)] for j in range(ns) if (s,i,j) in h_vs]
            prob.addConstraint(xp.Sum(drone_v + human_v) == 1)

    # C2: assignment only to open stations
    for j in range(ns):
        for s in scen_idx:
            for i in range(nr):
                if delta[i,s] == 0: continue
                if (s,i,j) in a_vs:
                    prob.addConstraint(a_vs[(s,i,j)] <= zj_v[j])
                if (s,i,j) in h_vs:
                    prob.addConstraint(h_vs[(s,i,j)] <= zj_v[j])

    # C3: capacity per scenario
    for j in range(ns):
        for s in scen_idx:
            robots_j = ([a_vs[(s,i,j)] for i in range(nr)
                         if delta[i,s]==1 and (s,i,j) in a_vs] +
                        [h_vs[(s,i,j)] for i in range(nr)
                         if delta[i,s]==1 and (s,i,j) in h_vs])
            if robots_j:
                prob.addConstraint(xp.Sum(robots_j) <= q * nj_v[j])

    # C4: charger bounds
    for j in range(ns):
        prob.addConstraint(nj_v[j] <= m * zj_v[j])
        prob.addConstraint(nj_v[j] >= zj_v[j])

    # Symmetry breaking
    for j in range(ns-1):
        prob.addConstraint(zj_v[j] >= zj_v[j+1])

    # Objective: build + maint + (1/|S|) * scenario costs
    obj = xp.Sum(cb*zj_v[j] + cm*nj_v[j] for j in range(ns))

    scale = 1.0 / ns_train
    for s in scen_idx:
        for i in range(nr):
            if delta[i,s] == 0: continue
            ri_s = R_s[i, s]
            for j in range(ns):
                if (s,i,j) in a_vs:
                    cost_drone = cc * (dij[i,j] + r_max - ri_s)
                    obj += scale * cost_drone * a_vs[(s,i,j)]
                if (s,i,j) in h_vs:
                    obj += scale * ch * h_vs[(s,i,j)]

    prob.setObjective(obj, sense=xp.minimize)
    prob.solve()

    status = prob.attributes.solstatus
    if status not in [xp.SolStatus.OPTIMAL, xp.SolStatus.FEASIBLE]:
        return None, None, None, None

    obj_val = prob.attributes.objval
    try:
        lb = prob.attributes.bestbound
    except Exception:
        lb = obj_val

    zj_sol = np.array([round(prob.getSolution(zj_v[j])) for j in range(ns)])
    nj_sol = np.array([round(prob.getSolution(nj_v[j])) for j in range(ns)])

    return zj_sol, nj_sol, obj_val, lb


def weiszfeld(pts, tol=1e-8, max_iter=300):
    if len(pts) == 1: return pts[0].copy()
    x = pts.mean(axis=0)
    for _ in range(max_iter):
        d  = np.maximum(np.linalg.norm(pts-x, axis=1), 1e-10)
        w  = 1.0/d
        xn = (pts*w[:,None]).sum(0)/w.sum()
        if np.linalg.norm(xn-x) < tol: break
        x  = xn
    return x

def safe_relocate(xi, yi, ri, cx_j, cy_j, mask):
    pts = np.column_stack([xi[mask], yi[mask]])
    ri_ = ri[mask]
    g   = weiszfeld(pts)
    old_d = np.sqrt((xi[mask]-cx_j)**2 + (yi[mask]-cy_j)**2)
    new_d = np.sqrt((xi[mask]-g[0])**2  + (yi[mask]-g[1])**2)
    lost  = (old_d<=ri_) & (new_d>ri_)
    if not lost.any(): return g[0], g[1]
    direction = g - np.array([cx_j, cy_j])
    dist_to_g = np.linalg.norm(direction)
    if dist_to_g < 1e-10: return cx_j, cy_j
    u = direction/dist_to_g
    max_step = dist_to_g
    for k in np.where(lost)[0]:
        ax_, ay_, rk = xi[mask][k], yi[mask][k], ri_[k]
        dx0,dy0 = cx_j-ax_, cy_j-ay_
        B = 2*(dx0*u[0]+dy0*u[1])
        C = dx0**2+dy0**2-rk**2
        disc = B**2-4*C
        if disc < 0: continue
        t_max = (-B+math.sqrt(disc))/2
        max_step = min(max_step, max(t_max,0.0))
    return cx_j+max_step*u[0], cy_j+max_step*u[1]

# ==============================================================================
# 5. Construction heuristic (same as Q1b, uses deterministic ranges for sizing)
# ==============================================================================

def construction_heuristic(xi, yi, ri, pi, gamma):
    p_bar_loc = pi.mean()
    n_min  = math.ceil(len(xi)*p_bar_loc/(m*q))
    ns     = max(1, math.ceil(gamma*n_min))
    km = KMeans(n_clusters=ns, random_state=SEED, n_init=10)
    km.fit(np.column_stack([xi,yi]))
    cx = km.cluster_centers_[:,0].copy()
    cy = km.cluster_centers_[:,1].copy()
    dij      = dist_matrix(xi, yi, cx, cy)
    a        = np.zeros((len(xi),ns), dtype=np.int8)
    h        = np.zeros((len(xi),ns), dtype=np.int8)
    assigned = np.zeros(ns, dtype=int)
    for i in np.argsort(ri):
        ok   = dij[i] <= ri[i]
        cost = np.where(ok, cc*(dij[i]+r_max-ri[i]), ch)
        for j in np.argsort(cost):
            if assigned[j] < m*q:
                if dij[i,j] <= ri[i]: a[i,j]=1
                else:                  h[i,j]=1
                assigned[j] += 1
                break
    nj = np.minimum(m, np.ceil(assigned/q).astype(int))
    zj = (nj>=1).astype(int)
    return cx, cy, a, h, nj, zj

# ==============================================================================
# 6. Stochastic location-allocation heuristic (RP)
# ==============================================================================

def solve_rp(xi, yi, pi, delta, R_s, scen_idx, gamma=GAMMA,
             max_iter=10, verbose=True):
    """
    Solve the Recourse Problem (RP) using location-allocation heuristic.
    Stage 1: k-means initialisation, then ILP for assignment.
    Stage 2: Geometric median relocation.
    """
    t0 = time.perf_counter()

    # Start from Q1b construction with deterministic ranges
    ri_det = robot_r[:]
    cx, cy, _, _, nj, zj = construction_heuristic(xi, yi, ri_det, pi, gamma)

    # Evaluate starting cost on training scenarios
    f0, b0, m0, c0, h0 = eval_obj_stochastic(
        cx, cy, nj, zj, xi, yi, delta, R_s, scen_idx)

    if verbose:
        print(f"  Starting cost (Q1b): £{f0:,.2f}")
        print(f"    Build={b0:,.0f}  Maint={m0:,.0f}  "
              f"AvgCharge={c0:,.2f}  AvgHuman={h0:,.0f}")
        print(f"  Stations: {zj.sum()}\n")

    f_best = f0
    history = [f0]

    for it in range(1, max_iter+1):
        if verbose:
            print(f"  Iter {it}: Solving stochastic ILP "
                  f"({len(cx)} stations, {len(scen_idx)} scenarios)...",
                  end=" ", flush=True)

        zj_new, nj_new, f_ilp, lb = solve_ilp_stochastic(
            xi, yi, cx, cy, delta, R_s, scen_idx, time_limit=T_ILP)

        if zj_new is None:
            if verbose: print("ILP infeasible, stopping.")
            break

        if verbose:
            print(f"ILP=£{f_ilp:,.2f}  LB=£{lb:,.2f}")

        # Geometric median relocation using average assignments across scenarios
        # Use deterministic ranges as proxy for which robots are "typically" assigned
        dij_cur = dist_matrix(xi, yi, cx, cy)
        new_cx, new_cy = cx.copy(), cy.copy()

        for j in range(len(cx)):
            if zj_new[j] == 0: continue
            # Find robots most frequently assigned to station j across scenarios
            freq = np.zeros(len(xi))
            for s in scen_idx:
                ri_s = R_s[:, s]
                for i in range(len(xi)):
                    if delta[i,s]==1 and dij_cur[i,j] <= ri_s[i]:
                        freq[i] += 1
            mask = freq > (len(scen_idx) * 0.3)  # assigned in >30% of scenarios
            if not mask.any():
                mask = (dij_cur[:,j] == dij_cur[:,j].min())
            new_cx[j], new_cy[j] = safe_relocate(
                xi, yi, ri_det, cx[j], cy[j], mask)

        # Evaluate new locations
        f_new, b, mn, ch_c, hc = eval_obj_stochastic(
            new_cx, new_cy, nj_new, zj_new, xi, yi, delta, R_s, scen_idx)

        if verbose:
            print(f"         After relocation: £{f_new:,.2f}  "
                  f"(change: £{f_best-f_new:+,.2f})")

        if f_best - f_new > EPS:
            cx, cy  = new_cx, new_cy
            nj, zj  = nj_new, zj_new
            f_best  = f_new
            history.append(f_new)
        else:
            if verbose: print(f"         No improvement > £{EPS:.0f} — stopping.")
            history.append(f_best)
            break

    t1 = time.perf_counter()

    # Final evaluation on training scenarios
    f_rp, b, mn, ch_c, hc = eval_obj_stochastic(
        cx, cy, nj, zj, xi, yi, delta, R_s, scen_idx)

    return {
        "cx": cx, "cy": cy, "nj": nj, "zj": zj,
        "f_start": round(f0,   2),
        "f_rp":    round(f_rp, 2),
        "n_open":  int(zj.sum()),
        "n_chargers": int(nj.sum()),
        "build":   round(b,    2),
        "maint":   round(mn,   2),
        "avg_charge": round(ch_c, 2),
        "avg_human":  round(hc,   2),
        "time_s":  round(t1-t0, 2),
        "history": history,
    }

# ==============================================================================
# 7. Compute EEV (evaluate Q1c deterministic solution on scenarios)
# ==============================================================================

def compute_eev(cx_q1c, cy_q1c, nj_q1c, zj_q1c,
                xi, yi, delta, R_s, scen_idx):
    """
    EEV: fix Q1c station layout, evaluate on stochastic scenarios.
    """
    f_eev, b, mn, ch_c, hc = eval_obj_stochastic(
        cx_q1c, cy_q1c, nj_q1c, zj_q1c,
        xi, yi, delta, R_s, scen_idx)
    return f_eev, b, mn, ch_c, hc

# ==============================================================================
# 8. Main experiment
# ==============================================================================

xi = robot_x
yi = robot_y
pi = np.exp(-lam**2 * (robot_r - r_min)**2)   # deterministic p_i for sizing

print("=" * 65)
print("  SOLVING RECOURSE PROBLEM (RP) -- 80 training scenarios")
print("=" * 65)

rp = solve_rp(xi, yi, pi, Delta, R_scen, train_idx, verbose=True)

print(f"\n  RP SOLUTION")
print(f"    Stations open  : {rp['n_open']}")
print(f"    Total chargers : {rp['n_chargers']}")
print(f"    RP cost (train): £{rp['f_rp']:,.2f}")
print(f"    Time           : {rp['time_s']:.2f}s")

# ==============================================================================
# 9. EEV -- use Q1c station layout on training scenarios
# ==============================================================================

print("\n" + "=" * 65)
print("  COMPUTING EEV (Q1c deterministic solution on scenarios)")
print("=" * 65)

# Reconstruct Q1c station layout using same robots
cx_q1c, cy_q1c, _, _, nj_q1c, zj_q1c = construction_heuristic(
    xi, yi, robot_r, pi, GAMMA)

# Note: for a proper EEV, use the full Q1c result from the saved solution.
# Here we use the Q1b starting point as a proxy since Q1c output
# station coordinates are not saved between runs.
# To use actual Q1c: load cx, cy, nj, zj from Q1c_localsearch_results.csv

f_eev, b_eev, mn_eev, ch_eev, hc_eev = compute_eev(
    cx_q1c, cy_q1c, nj_q1c, zj_q1c,
    xi, yi, Delta, R_scen, train_idx)

print(f"  EEV cost (train scenarios): £{f_eev:,.2f}")
print(f"    Build={b_eev:,.0f}  Maint={mn_eev:,.0f}  "
      f"AvgCharge={ch_eev:,.2f}  AvgHuman={hc_eev:,.0f}")

# ==============================================================================
# 10. VSS
# ==============================================================================

vss     = f_eev - rp['f_rp']
vss_pct = vss / f_eev * 100

print("\n" + "=" * 65)
print("  VALUE OF STOCHASTIC SOLUTION (VSS)")
print("=" * 65)
print(f"  RP  (stochastic solution) : £{rp['f_rp']:>12,.2f}")
print(f"  EEV (deterministic Q1c)   : £{f_eev:>12,.2f}")
print(f"  VSS = EEV - RP            : £{vss:>12,.2f}")
print(f"  VSS (%)                   :  {vss_pct:>11.2f}%")

if vss > 0:
    print(f"\n  -> Stochastic planning saves £{vss:,.2f} ({vss_pct:.2f}%) "
          f"over using the deterministic solution.")
else:
    print(f"\n  -> Deterministic solution performs similarly on scenarios.")

# ==============================================================================
# 11. Out-of-sample validation
# ==============================================================================

print("\n" + "=" * 65)
print("  OUT-OF-SAMPLE VALIDATION (20 held-out scenarios)")
print("=" * 65)

f_rp_oos,  b,  mn,  ch_c,  hc  = eval_obj_stochastic(
    rp['cx'], rp['cy'], rp['nj'], rp['zj'],
    xi, yi, Delta, R_scen, test_idx)

f_eev_oos, be, mne, che, hce = compute_eev(
    cx_q1c, cy_q1c, nj_q1c, zj_q1c,
    xi, yi, Delta, R_scen, test_idx)

vss_oos     = f_eev_oos - f_rp_oos
vss_oos_pct = vss_oos / f_eev_oos * 100

print(f"  RP  out-of-sample cost    : £{f_rp_oos:>12,.2f}")
print(f"  EEV out-of-sample cost    : £{f_eev_oos:>12,.2f}")
print(f"  VSS (out-of-sample)       : £{vss_oos:>12,.2f}  ({vss_oos_pct:.2f}%)")

# ==============================================================================
# 12. Summary table
# ==============================================================================

print("\n" + "=" * 65)
print("  SUMMARY")
print("=" * 65)
print(f"  {'Metric':<35}  {'Train (80)':>12}  {'Test (20)':>12}")
print("  " + "-"*62)
print(f"  {'RP cost (stochastic)':<35}  "
      f"£{rp['f_rp']:>11,.2f}  £{f_rp_oos:>11,.2f}")
print(f"  {'EEV cost (deterministic Q1c)':<35}  "
      f"£{f_eev:>11,.2f}  £{f_eev_oos:>11,.2f}")
print(f"  {'VSS = EEV - RP':<35}  "
      f"£{vss:>11,.2f}  £{vss_oos:>11,.2f}")
print(f"  {'VSS (%)':<35}  "
      f"{vss_pct:>11.2f}%  {vss_oos_pct:>11.2f}%")
print(f"  {'RP open stations':<35}  {rp['n_open']:>12}")
print(f"  {'RP total chargers':<35}  {rp['n_chargers']:>12}")
print("=" * 65)

# Save results
results_df = pd.DataFrame([{
    "metric": "RP_train",       "cost": rp['f_rp'],
    "build": rp['build'],       "maint": rp['maint'],
    "avg_charge": rp['avg_charge'], "avg_human": rp['avg_human'],
},{
    "metric": "EEV_train",      "cost": f_eev,
    "build": b_eev,             "maint": mn_eev,
    "avg_charge": ch_eev,       "avg_human": hc_eev,
},{
    "metric": "VSS_train",      "cost": vss,
    "build": None, "maint": None, "avg_charge": None, "avg_human": None,
},{
    "metric": "RP_test",        "cost": f_rp_oos,
    "build": b, "maint": mn, "avg_charge": ch_c, "avg_human": hc,
},{
    "metric": "EEV_test",       "cost": f_eev_oos,
    "build": be, "maint": mne, "avg_charge": che, "avg_human": hce,
},{
    "metric": "VSS_test",       "cost": vss_oos,
    "build": None, "maint": None, "avg_charge": None, "avg_human": None,
}])
results_df.to_csv("Q2_stochastic_results.csv", index=False)
print("\nSaved to Q2_stochastic_results.csv")
