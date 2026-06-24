"""
Fixed-charge extension and a tailored multi-objective ALNS.

Adding binary link-activation variables $z_{ij}\\in\\{0,1\\}$ with fixed costs
$\\phi_{ij}$ and fixed emissions $\\psi_{ij}$ turns the otherwise-convex base model
into a mixed-integer program: a MILP under the $\\ell_1$ ground metric (solved
exactly here) and a MISOCP under $\\ell_2$. This is the regime in which a
metaheuristic earns its keep, since the exact method becomes NP-hard.

This module provides
  * ``solve_fc_exact`` / ``fc_exact_front`` -- the exact MILP reformulation (the
    worst-case recourse is the same compact separable LP as in the convex model,
    now sitting on top of the binary first stage);
  * ``alns_fc_front`` -- an Adaptive Large Neighborhood Search that searches over
    the open-link set with destroy/repair operators, an adaptive operator-weight
    scheme, a simulated-annealing acceptance on random scalarisations, and a
    Pareto archive.
"""
from __future__ import annotations

import numpy as np
import cvxpy as cp

from .config import DROConfig
from .data_generation import Instance
from .dro_model import DROSolver
from .recourse import worst_case_l1_box, worst_case_full_support


# --------------------------------------------------------------------------- #
#  Exact MILP reformulation (l1 ground metric)
# --------------------------------------------------------------------------- #
def _fc_objectives(inst: Instance, dro: DROConfig):
    """Return cvxpy variables/expressions (x, z, f_cost, f_emis, cons)."""
    s = DROSolver(inst, dro)
    I, J, D = s.I, s.J, s.D
    x = cp.Variable((I, D), nonneg=True)
    z = cp.Variable((I, J), boolean=True)
    xbar = cp.sum(x, axis=0)
    cons = []
    for i in range(I):
        cons.append(s.Mk @ x[i, :] <= inst.supply[i, :])
        cons.append(s.Mj @ x[i, :] <= cp.multiply(inst.link_cap[i, :], z[i, :]))
    c_expr, c_cons, _ = s._wc_terms_compact_l1(xbar, s.cs, s.co)
    e_expr, e_cons, _ = s._wc_terms_compact_l1(xbar, s.es, s.eo)
    cons += c_cons + e_cons
    f_cost = cp.sum(cp.multiply(s.cost_id, x)) + cp.sum(cp.multiply(inst.fixed_cost, z)) + c_expr
    f_emis = cp.sum(cp.multiply(s.emis_id, x)) + cp.sum(cp.multiply(inst.fixed_emis, z)) + e_expr
    return s, x, z, f_cost, f_emis, cons


def _mi_solver(dro: DROConfig) -> str:
    if dro.solver:
        return dro.solver
    inst = set(cp.installed_solvers())
    for s in ("GUROBI", "MOSEK", "SCIP", "SCIPY", "GLPK_MI"):
        if s in inst:
            return s
    return None


def solve_fc_exact(inst: Instance, dro: DROConfig, primary: str = "cost",
                   secondary: str = "emission", kappa: float | None = None) -> dict:
    """Exact fixed-charge solve (MILP for l1). Minimise ``primary``; if ``kappa``
    is given, subject to the ``secondary`` robust objective <= kappa."""
    if dro.norm != "l1":
        raise NotImplementedError("exact fixed-charge is implemented for l1 (MILP); "
                                  "use alns_fc_front for l2.")
    s, x, z, f_cost, f_emis, cons = _fc_objectives(inst, dro)
    fmap = {"cost": f_cost, "emission": f_emis}
    if kappa is None:
        prob = cp.Problem(cp.Minimize(fmap[primary]), cons)
    else:
        cons = cons + [fmap[secondary] <= kappa]
        prob = cp.Problem(cp.Minimize(fmap[primary] + 1e-4 * fmap[secondary]), cons)
    prob.solve(solver=_mi_solver(dro))
    if prob.status not in ("optimal", "optimal_inaccurate"):
        return {"status": prob.status, "x": None}
    x_ijk = np.asarray(x.value).reshape(s.I, s.J, s.K)
    return {"status": prob.status, "x": x_ijk, "z": np.round(np.asarray(z.value)),
            "n_links": int(np.round(z.value).sum()),
            "cost_total": float(f_cost.value), "emis_total": float(f_emis.value)}


def _nondom(front: list[dict]) -> list[dict]:
    keep = []
    for p in front:
        c, e = p["cost_total"], p["emis_total"]
        if not any((q["cost_total"] <= c + 1e-9) and (q["emis_total"] <= e + 1e-9)
                   and (q["cost_total"] < c - 1e-9 or q["emis_total"] < e - 1e-9) for q in front):
            keep.append(p)
    keep.sort(key=lambda f: f["cost_total"])
    return keep


def fc_exact_front(inst: Instance, dro: DROConfig, n_points: int = 7) -> list[dict]:
    """Exact Pareto front of the fixed-charge model via epsilon-constraint."""
    c = solve_fc_exact(inst, dro, "cost")
    e = solve_fc_exact(inst, dro, "emission")
    front = [p for p in (c, e) if p.get("x") is not None]
    if c.get("x") is not None and e.get("x") is not None and c["emis_total"] > e["emis_total"] + 1e-6:
        for kappa in np.linspace(e["emis_total"], c["emis_total"], n_points)[1:-1]:
            sol = solve_fc_exact(inst, dro, "cost", "emission", float(kappa))
            if sol.get("x") is not None:
                front.append(sol)
    return _nondom(front)


# --------------------------------------------------------------------------- #
#  Multi-objective ALNS
# --------------------------------------------------------------------------- #
def _construct_flows(inst: Instance, zmat: np.ndarray, w: float) -> np.ndarray:
    """Greedy flows over the open links toward the newsvendor target quantile,
    with source preference w in [0,1] (w=1 cheapest, w=0 greenest); repaired to
    supply and link capacity."""
    I, J, K = inst.n_sources, inst.n_destinations, inst.n_products
    x = np.zeros((I, J, K))
    remaining = inst.supply.copy()
    # newsvendor critical fractile per (j,k), blended by the preference w
    qc = inst.short_cost / (inst.short_cost + inst.surp_cost)
    qe = inst.short_emis / (inst.short_emis + inst.surp_emis)
    level = np.clip(w * qc + (1.0 - w) * qe, 0.01, 0.99)     # (J,K)
    samp = inst.train_samples.reshape(-1, J, K)
    target = np.empty((J, K))
    for j in range(J):
        for k in range(K):
            target[j, k] = min(np.quantile(samp[:, j, k], level[j, k]),
                               inst.support_ub[j, k])
    score = w * inst.cost + (1.0 - w) * inst.emission        # (I,J,K) lower=better
    for k in range(K):
        for j in range(J):
            T = target[j, k]
            order = np.argsort(score[:, j, k])
            for i in order:
                if T <= 1e-9:
                    break
                if zmat[i, j] <= 0:                          # link closed
                    continue
                give = min(T, remaining[i, k])
                x[i, j, k] += give
                remaining[i, k] -= give
                T -= give
    # repair link capacity (scale down per (i,j)); supply already respected
    rho = inst.rho
    ton = (x * rho[None, None, :]).sum(axis=2)               # (I,J)
    scale = np.minimum(1.0, inst.link_cap / np.maximum(ton, 1e-12))
    return x * scale[:, :, None]


def _evaluate(inst: Instance, dro: DROConfig, zmat: np.ndarray, w: float) -> tuple[float, float]:
    x = _construct_flows(inst, zmat, w)
    xbar = (x.sum(axis=0)).reshape(-1)
    cs, co = inst.short_cost.reshape(-1), inst.surp_cost.reshape(-1)
    es, eo = inst.short_emis.reshape(-1), inst.surp_emis.reshape(-1)
    lb, ub = inst.support_lb.reshape(-1), inst.support_ub.reshape(-1)
    if dro.norm == "l1":
        rc = worst_case_l1_box(xbar, inst.train_samples, cs, co, lb, ub, dro.epsilon)
        re = worst_case_l1_box(xbar, inst.train_samples, es, eo, lb, ub, dro.epsilon)
    else:
        rc = worst_case_full_support(xbar, inst.train_samples, cs, co, dro.norm, dro.epsilon)
        re = worst_case_full_support(xbar, inst.train_samples, es, eo, dro.norm, dro.epsilon)
    f1 = float((inst.cost * x).sum() + (inst.fixed_cost * zmat).sum() + rc)
    f2 = float((inst.emission * x).sum() + (inst.fixed_emis * zmat).sum() + re)
    return f1, f2


class _Archive:
    """Pareto archive of (f1, f2, z, w)."""
    def __init__(self):
        self.pts = []

    def add(self, f1, f2, z, w) -> bool:
        for (g1, g2, _, _) in self.pts:
            if g1 <= f1 + 1e-9 and g2 <= f2 + 1e-9 and (g1 < f1 - 1e-9 or g2 < f2 - 1e-9):
                return False                                  # dominated
        self.pts = [p for p in self.pts
                    if not (f1 <= p[0] + 1e-9 and f2 <= p[1] + 1e-9
                            and (f1 < p[0] - 1e-9 or f2 < p[1] - 1e-9))]
        self.pts.append((f1, f2, z.copy(), w))
        return True

    def front(self):
        F = np.array([[p[0], p[1]] for p in self.pts])
        return F[np.argsort(F[:, 0])] if len(F) else F


def alns_fc_front(inst: Instance, dro: DROConfig, iters: int = 1500,
                  seed: int = 0, T0: float = 0.05, cooling: float = 0.999):
    """Adaptive Large Neighborhood Search for the multi-objective fixed-charge
    problem. Returns the Pareto front (array of [cost, emission])."""
    rng = np.random.default_rng(seed)
    I, J = inst.n_sources, inst.n_destinations

    # destroy / repair operators on the open-link matrix z (I, J)
    def d_random(z, q):
        z = z.copy(); opn = np.argwhere(z > 0)
        if len(opn):
            for idx in rng.choice(len(opn), min(q, len(opn)), replace=False):
                z[tuple(opn[idx])] = 0
        return z

    def d_expensive(z, q):
        z = z.copy(); opn = np.argwhere(z > 0)
        if len(opn):
            costs = np.array([inst.fixed_cost[i, j] + inst.fixed_emis[i, j] for i, j in opn])
            for idx in np.argsort(-costs)[:q]:
                z[tuple(opn[idx])] = 0
        return z

    def r_random(z, q):
        z = z.copy(); cl = np.argwhere(z == 0)
        if len(cl):
            for idx in rng.choice(len(cl), min(q, len(cl)), replace=False):
                z[tuple(cl[idx])] = 1
        return z

    def r_cheap(z, q):
        z = z.copy(); cl = np.argwhere(z == 0)
        if len(cl):
            costs = np.array([inst.fixed_cost[i, j] + inst.fixed_emis[i, j] for i, j in cl])
            for idx in np.argsort(costs)[:q]:
                z[tuple(cl[idx])] = 1
        return z

    destroys, repairs = [d_random, d_expensive], [r_random, r_cheap]
    wd, wr = np.ones(len(destroys)), np.ones(len(repairs))

    # ensure every destination is reachable by at least one open link
    def make_feasible(z):
        z = z.copy()
        for j in range(J):
            if z[:, j].sum() == 0:
                z[rng.integers(I), j] = 1
        return z

    z = make_feasible(np.zeros((I, J), dtype=int))            # start: minimal sparse
    w = 0.5
    arch = _Archive()
    f1, f2 = _evaluate(inst, dro, z, w)
    arch.add(f1, f2, z, w)
    cur = (z, w, f1, f2)
    lo = np.array([f1, f2]); hi = np.array([f1, f2]); T = T0
    maxq = max(2, (I * J) // 4)

    for it in range(iters):
        di = rng.choice(len(destroys), p=wd / wd.sum())
        ri = rng.choice(len(repairs), p=wr / wr.sum())
        # independent close/open magnitudes let the network shrink AND grow
        q1, q2 = rng.integers(0, maxq + 1), rng.integers(0, maxq + 1)
        znew = make_feasible(repairs[ri](destroys[di](cur[0], q1), q2))
        wnew = float(np.clip(cur[1] + rng.normal(0, 0.2), 0.0, 1.0))
        n1, n2 = _evaluate(inst, dro, znew, wnew)
        added = arch.add(n1, n2, znew, wnew)
 