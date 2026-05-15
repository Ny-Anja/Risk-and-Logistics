"""
Assignment 2 - Q1c: Location-Allocation Local Search (Paper approach)

Matches Q1b exactly (same data, same shuffle, same gamma=2).

Algorithm (from Ljosheim et al. 2026, Algorithm 1):
  1. Start from Q1b construction heuristic solution (gamma=2)
  2. Loop:
     a. Solve ILP (P) with fixed PCL locations using Xpress
     b. For each open station, compute geometric median of assigned robots
        -> new improved PCL (range-safe constrained relocation)
     c. If warm-start cost improvement > eps -> continue; else stop
  3. Return best solution found

ILP (P): same as Q1a but d_ij are PARAMETERS (fixed by PCL locations),
         not variables. This removes all nonlinearity -> pure MILP.
"""

import math, time
import numpy as np
import pandas as pd
import xpress as xp
from sklearn.cluster import KMeans

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
MAX_ITER = 20

# ==============================================================================
# 2. Load and project data -- IDENTICAL to Q1b
# ==============================================================================
locs   = pd.read_csv('robot_locations.csv')
ranges = pd.read_csv('range.csv')
data   = pd.merge(locs, ranges, on='index').reset_index(drop=True)

def project(lon, lat):
    y = lat * 111.0
    x = lon * 111.0 * math.cos(math.radians(lat))
    return x, y

coords  = np.array([project(r.longitude, r.latitude) for _, r in data.iterrows()])
robot_x = coords[:, 0]
robot_y = coords[:, 1]
robot_r = data['range'].values
p_i     = np.exp(-lam**2 * (robot_r - r_min)**2)

# SAME shuffle as Q1b
rng      = np.random.default_rng(42)
shuffled = rng.permutation(len(data))

# ==============================================================================
# 3. Helper functions
# ==============================================================================

def dist_matrix(xi, yi, cx, cy):
    return np.sqrt((xi[:,None]-cx[None,:])**2 +
                   (yi[:,None]-cy[None,:])**2)

def evaluate_obj(a, h, nj, zj, dij, ri):
    build  = cb * float(zj.sum())
    maint  = cm * float(nj.sum())
    charge = cc * float(np.sum(a * (dij + r_max - ri[:,None])))
    human  = ch * float(h.sum())
    return build+maint+charge+human, build, maint, charge, human

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

    if not lost.any():
        return g[0], g[1]

    direction = g - np.array([cx_j, cy_j])
    dist_to_g = np.linalg.norm(direction)
    if dist_to_g < 1e-10: return cx_j, cy_j

    u        = direction / dist_to_g
    max_step = dist_to_g

    for k in np.where(lost)[0]:
        ax_, ay_, rk = xi[mask][k], yi[mask][k], ri_[k]
        dx0, dy0   = cx_j-ax_, cy_j-ay_
        B    = 2*(dx0*u[0]+dy0*u[1])
        C    = dx0**2+dy0**2-rk**2
        disc = B**2-4*C
        if disc < 0: continue
        t_max    = (-B+math.sqrt(disc))/2
        max_step = min(max_step, max(t_max, 0.0))

    return cx_j+max_step*u[0], cy_j+max_step*u[1]

# ==============================================================================
# 4. Solve ILP (P) with Xpress -- fixed PCL locations
# ==============================================================================

def solve_ilp(xi, yi, ri, cx, cy, time_limit=T_ILP,
              warm_a=None, warm_h=None, warm_nj=None, warm_zj=None):
    nr = len(xi)
    ns = len(cx)
    dij = dist_matrix(xi, yi, cx, cy)

    prob = xp.problem(f"ILP_P_{nr}x{ns}")
    prob.controls.outputlog  = 0
    prob.controls.maxtime    = int(time_limit)
    prob.controls.miprelstop = 0.001

    zj_v  = [prob.addVariable(vartype=xp.binary,           name=f'z_{j}')
              for j in range(ns)]
    nj_v  = [prob.addVariable(vartype=xp.integer, lb=0, ub=m, name=f'n_{j}')
              for j in range(ns)]
    aij_v = {(i,j): prob.addVariable(vartype=xp.binary,   name=f'a_{i}_{j}')
              for i in range(nr) for j in range(ns)
              if dij[i,j] <= ri[i]}
    hij_v = {(i,j): prob.addVariable(vartype=xp.binary,   name=f'h_{i}_{j}')
              for i in range(nr) for j in range(ns)}

    # C1: every robot assigned exactly once
    for i in range(nr):
        drone_v = [aij_v[i,j] for j in range(ns) if (i,j) in aij_v]
        human_v = [hij_v[i,j] for j in range(ns)]
        prob.addConstraint(xp.Sum(drone_v + human_v) == 1)

    # C2: assignment only to open stations
    for j in range(ns):
        for i in range(nr):
            if (i,j) in aij_v:
                prob.addConstraint(aij_v[i,j] <= zj_v[j])
            prob.addConstraint(hij_v[i,j] <= zj_v[j])

    # C3: capacity
    for j in range(ns):
        robots_j = ([aij_v[i,j] for i in range(nr) if (i,j) in aij_v] +
                    [hij_v[i,j] for i in range(nr)])
        prob.addConstraint(xp.Sum(robots_j) <= q * nj_v[j])

    # C4: charger bounds
    for j in range(ns):
        prob.addConstraint(nj_v[j] <= m * zj_v[j])
        prob.addConstraint(nj_v[j] >= zj_v[j])

    # Symmetry breaking
    for j in range(ns-1):
        prob.addConstraint(zj_v[j] >= zj_v[j+1])

    # Objective
    obj = xp.Sum(cb*zj_v[j] + cm*nj_v[j] for j in range(ns))
    for i in range(nr):
        for j in range(ns):
            if (i,j) in aij_v:
                obj += cc * dij[i,j] * aij_v[i,j]
                obj += cc * (r_max - ri[i]) * aij_v[i,j]
            obj += ch * hij_v[i,j]
    prob.setObjective(obj, sense=xp.minimize)

    # Warm start
    if warm_a is not None and warm_zj is not None:
        sol_vals = {}
        for j in range(ns):
            sol_vals[zj_v[j]] = float(warm_zj[j])
            sol_vals[nj_v[j]] = float(warm_nj[j])
            for i in range(nr):
                if (i,j) in aij_v:
                    sol_vals[aij_v[i,j]] = float(warm_a.get((i,j), 0))
                sol_vals[hij_v[i,j]] = float(warm_h.get((i,j), 0))
        try:
            prob.loadMipSol([sol_vals[v] for v in
                             list(zj_v) + list(nj_v) +
                             [aij_v[k] for k in sorted(aij_v)] +
                             [hij_v[k] for k in sorted(hij_v)]])
        except Exception:
            pass

    prob.solve()

    sol_status = prob.attributes.solstatus
    if sol_status not in [xp.SolStatus.OPTIMAL, xp.SolStatus.FEASIBLE]:
        return None, None, None, None, None, None

    obj_val = prob.attributes.objval
    try:
        lb      = prob.attributes.bestbound
    except Exception:
        lb = obj_val

    zj_sol = np.array([round(prob.getSolution(zj_v[j]))  for j in range(ns)])
    nj_sol = np.array([round(prob.getSolution(nj_v[j]))  for j in range(ns)])

    a_arr = np.zeros((nr, ns), dtype=np.int8)
    h_arr = np.zeros((nr, ns), dtype=np.int8)
    for i in range(nr):
        for j in range(ns):
            if (i,j) in aij_v:
                a_arr[i,j] = round(prob.getSolution(aij_v[i,j]))
            h_arr[i,j] = round(prob.getSolution(hij_v[i,j]))

    return a_arr, h_arr, nj_sol, zj_sol, obj_val, lb

# ==============================================================================
# 5. Construction heuristic -- identical to Q1b
# ==============================================================================

def construction_heuristic(xi, yi, ri, pi, gamma):
    n_r   = len(xi)
    p_bar = pi.mean()
    n_min = math.ceil(n_r * p_bar / (m*q))
    ns    = max(1, math.ceil(gamma * n_min))

    km = KMeans(n_clusters=ns, random_state=42, n_init=10)
    km.fit(np.column_stack([xi, yi]))
    cx = km.cluster_centers_[:,0].copy()
    cy = km.cluster_centers_[:,1].copy()

    dij      = dist_matrix(xi, yi, cx, cy)
    a        = np.zeros((n_r, ns), dtype=np.int8)
    h        = np.zeros((n_r, ns), dtype=np.int8)
    assigned = np.zeros(ns, dtype=int)

    for i in np.argsort(ri):
        ok   = dij[i] <= ri[i]
        cost = np.where(ok, cc*(dij[i]+r_max-ri[i]), ch)
        for j in np.argsort(cost):
            if assigned[j] < m*q:
                if dij[i,j] <= ri[i]: a[i,j] = 1
                else:                  h[i,j] = 1
                assigned[j] += 1
                break

    nj = np.minimum(m, np.ceil(assigned/q).astype(int))
    zj = (nj >= 1).astype(int)
    return cx, cy, a, h, nj, zj

# ==============================================================================
# 6. Find improved locations
# ==============================================================================

def find_improved_locations(xi, yi, ri, cx, cy, a, h, zj):
    new_cx = cx.copy()
    new_cy = cy.copy()
    for j in range(len(cx)):
        if zj[j] == 0: continue
        mask = (a[:,j]==1) | (h[:,j]==1)
        if not mask.any(): continue
        new_cx[j], new_cy[j] = safe_relocate(xi, yi, ri, cx[j], cy[j], mask)
    return new_cx, new_cy

# ==============================================================================
# 7. Construct warm start
# ==============================================================================

def construct_warmstart(xi, yi, ri, a, h, nj, zj, new_cx, new_cy):
    nr = len(xi)
    ns = len(new_cx)
    dij_new = dist_matrix(xi, yi, new_cx, new_cy)

    a_ws = np.zeros((nr, ns), dtype=np.int8)
    h_ws = np.zeros((nr, ns), dtype=np.int8)
    assigned = np.zeros(ns, dtype=int)

    for i in range(nr):
        for j in range(ns):
            if a[i,j] == 1 or h[i,j] == 1:
                if dij_new[i,j] <= ri[i]:
                    a_ws[i,j] = 1
                else:
                    h_ws[i,j] = 1
                assigned[j] += 1
                break

    nj_ws = np.minimum(m, np.ceil(assigned/q).astype(int))
    zj_ws = (nj_ws >= 1).astype(int)
    f_ws, *_ = evaluate_obj(a_ws, h_ws, nj_ws, zj_ws, dij_new, ri)
    return a_ws, h_ws, nj_ws, zj_ws, f_ws

# ==============================================================================
# 8. Main local search
# ==============================================================================

def local_search(xi, yi, ri, pi, gamma=GAMMA, eps=EPS,
                 t_ilp=T_ILP, max_iter=MAX_ITER, verbose=True):

    t0 = time.perf_counter()

    cx, cy, a, h, nj, zj = construction_heuristic(xi, yi, ri, pi, gamma)
    dij = dist_matrix(xi, yi, cx, cy)
    f0, b0, m0, c0, h0   = evaluate_obj(a, h, nj, zj, dij, ri)

    if verbose:
        print(f"  Starting cost (Q1b, gamma={gamma}): £{f0:,.2f}")
        print(f"    Build={b0:,.0f}  Maint={m0:,.0f}  "
              f"Charge={c0:,.2f}  Human={h0:,.0f}")
        print(f"  Stations: {zj.sum()}  Drone: {a.sum()}  Human: {h.sum()}\n")

    f_best   = f0
    history  = [f0]
    n_iter   = 0
    improved = True

    while improved and n_iter < max_iter:
        n_iter  += 1
        improved = False

        if verbose:
            print(f"  Iter {n_iter}: Solving ILP ({len(cx)} stations)...",
                  end=" ", flush=True)

        a_new, h_new, nj_new, zj_new, f_ilp, lb = solve_ilp(
            xi, yi, ri, cx, cy,
            time_limit=t_ilp,
            warm_a={k: v for k,v in np.ndenumerate(a) if v},
            warm_h={k: v for k,v in np.ndenumerate(h) if v},
            warm_nj=nj, warm_zj=zj)

        if a_new is None:
            if verbose: print("ILP infeasible, stopping.")
            break

        if verbose:
            print(f"ILP obj=£{f_ilp:,.2f}  LB=£{lb:,.2f}")

        new_cx, new_cy = find_improved_locations(
            xi, yi, ri, cx, cy, a_new, h_new, zj_new)

        a_ws, h_ws, nj_ws, zj_ws, f_ws = construct_warmstart(
            xi, yi, ri, a_new, h_new, nj_new, zj_new, new_cx, new_cy)

        if verbose:
            print(f"         Warm start: £{f_ws:,.2f}  "
                  f"(improvement vs current: £{f_best - f_ws:+,.2f})")

        if f_best - f_ws > eps:
            cx, cy  = new_cx, new_cy
            a, h    = a_ws, h_ws
            nj, zj  = nj_ws, zj_ws
            f_best  = f_ws
            improved = True
            history.append(f_ws)
        else:
            if f_best - f_ilp > eps:
                dij_cur = dist_matrix(xi, yi, cx, cy)
                f_check, *_ = evaluate_obj(a_new, h_new, nj_new, zj_new,
                                            dij_cur, ri)
                if f_best - f_check > eps:
                    a, h    = a_new, h_new
                    nj, zj  = nj_new, zj_new
                    f_best  = f_check
                    improved = True
                    history.append(f_check)
            if not improved:
                if verbose:
                    print(f"         No improvement > £{eps:.0f} — stopping.")
                history.append(f_best)

    t1 = time.perf_counter()

    dij_fin = dist_matrix(xi, yi, cx, cy)
    f_fin, b, mn, ch_c, hc = evaluate_obj(a, h, nj, zj, dij_fin, ri)

    return {
        "n_robots":        len(xi),
        "gamma":           gamma,
        "f_start":         round(f0,    2),
        "f_final":         round(f_fin, 2),
        "improvement_pct": round((f0-f_fin)/f0*100, 2),
        "n_iter":          n_iter,
        "n_open":          int(zj.sum()),
        "n_chargers":      int(nj.sum()),
        "n_drone":         int(a.sum()),
        "n_human":         int(h.sum()),
        "build":           round(b,    2),
        "maintenance":     round(mn,   2),
        "charging":        round(ch_c, 2),
        "human":           round(hc,   2),
        "time_s":          round(t1-t0, 3),
        "history":         history,
    }

# ==============================================================================
# 9. Run experiments
# ==============================================================================

SUBSET_SIZES = [20, 30, 40, 60, 100, 268, 536, 1072]

MINLP_UB = {
    20:  32111.30,
    30:  43664.17,
    40:  62116.34,
    60:  89400.07,
}

summary = []

for size in SUBSET_SIZES:
    idx = shuffled[:size].tolist()
    xi  = robot_x[idx]
    yi  = robot_y[idx]
    ri  = robot_r[idx]
    pi  = p_i[idx]

    print("=" * 65)
    print(f"  n = {size} robots")
    print("=" * 65)

    res = local_search(xi, yi, ri, pi, verbose=True)

    ub      = MINLP_UB.get(size)
    gap_str = (f"{(res['f_final']-ub)/res['f_final']*100:.2f}%"
               if ub and res['f_final'] > 0 else "N/A")

    print(f"\n  SUMMARY  n={size}")
    print(f"    Q1c final  : £{res['f_final']:>14,.2f}")
    print(f"    Improvement: {res['improvement_pct']:.2f}%  "
          f"({res['n_iter']} iters, {res['time_s']:.3f}s)")
    print(f"    Gap vs MINLP-UB: {gap_str}")
    print(f"    Stations: {res['n_open']}  Chargers: {res['n_chargers']}  "
          f"Drone: {res['n_drone']}  Human: {res['n_human']}\n")

    summary.append({**{k: v for k, v in res.items() if k != 'history'},
                    "gap_ub": gap_str})

# ==============================================================================
# 10. Summary table
# ==============================================================================

print("=" * 70)
print("  RESULTS SUMMARY")
print("=" * 70)
print(f"  {'n':>5}  {'f_LS (£)':>14}  {'Impr%':>6}  "
      f"{'Iters':>5}  {'Gap_UB':>8}  {'Time(s)':>7}")
print("  " + "-" * 65)

for r in summary:
    print(f"  {r['n_robots']:>5}  {r['f_final']:>14,.2f}  "
          f"{r['improvement_pct']:>6.2f}  "
          f"{r['n_iter']:>5}  "
          f"{str(r['gap_ub']):>8}  "
          f"{r['time_s']:>7.3f}")

print("=" * 70)

pd.DataFrame([{k: v for k, v in r.items() if k != 'history'}
              for r in summary]).to_csv(
    "Q1c_localsearch_results.csv", index=False)
print("\nSaved to Q1c_localsearch_results.csv")