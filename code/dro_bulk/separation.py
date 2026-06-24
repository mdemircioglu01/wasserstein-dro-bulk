"""
Separation oracle for the **type-2 (l2), box-support** Wasserstein DRO model.

Row generation needs, for every historical sample ``xi_n`` and the current plan
``xbar`` and multiplier ``lambda``, the worst-case demand realisation

    v_n  =  max_{xi in [lb, ub]}  Q(xbar, xi)  -  lambda * || xi - xi_n ||_2 ,

where  Q(xbar, xi) = sum_d max( b_d (xi_d - xbar_d),  -h_d (xi_d - xbar_d) )
is the (separable, convex, piecewise-linear) recourse value.  The maximiser
``xi*`` yields the violated recourse piece via its sign pattern
``sign(xi* - xbar)``, which is added to the master.

Two backends are provided:

* ``"misocp"`` -- exact.  The piecewise-linear recourse is linearised with one
  binary per coordinate (shortage vs. surplus) and the l2 penalty is a
  second-order cone, giving a mixed-integer SOCP.  Requires a MI-conic solver
  (GUROBI / MOSEK / SCIP / ECOS_BB).  Certifies global optimality of the cut.

* ``"fixedpoint"`` -- solver-agnostic.  For a fixed sign pattern the objective
  is concave (linear minus norm), so the inner problem is a small continuous
  SOCP; we alternate between solving it and updating the sign pattern, from
  several starts, and keep the best.  Strong in practice but not certified.

``mode="auto"`` uses ``"misocp"`` when a MI-conic solver is installed and falls
back to ``"fixedpoint"`` otherwise.
"""
from __future__ import annotations

import numpy as np
import cvxpy as cp

_MI_SOCP_SOLVERS = ["GUROBI", "MOSEK", "SCIP", "ECOS_BB"]


def available_mi_socp_solver() -> str | None:
    installed = set(cp.installed_solvers())
    for s in _MI_SOCP_SOLVERS:
        if s in installed:
            return s
    return None


def resolve_mode(mode: str) -> str:
    if mode == "auto":
        return "misocp" if available_mi_socp_solver() else "fixedpoint"
    return mode


def recourse_value(xbar, xi, cs, co) -> float:
    imb = np.asarray(xi) - np.asarray(xbar)
    return float(np.maximum(cs * imb, -co * imb).sum())


# --------------------------------------------------------------------------- #
#  exact MISOCP backend
# --------------------------------------------------------------------------- #
def worst_case_misocp(xbar, xi_n, lb, ub, cs, co, lam, solver=None):
    D = xbar.size
    xi = cp.Variable(D)
    ss = cp.Variable(D, nonneg=True)          # shortage (xi - xbar)_+
    so = cp.Variable(D, nonneg=True)          # surplus  (xbar - xi)_+
    z = cp.Variable(D, boolean=True)          # 1 -> shortage branch active
    t = cp.Variable(nonneg=True)
    # big-M large enough to remain feasible for any first-stage plan: shortage is
    # bounded by ub-xbar and surplus by xbar-lb, so M must cover xbar-lb too.
    M = np.maximum(ub - lb, xbar - lb) + 1.0
    cons = [
        ss - so == xi - xbar,
        ss <= cp.multiply(M, z),
        so <= cp.multiply(M, 1 - z),
        xi >= lb, xi <= ub,
        t >= cp.norm(xi - xi_n, 2),
    ]
    obj = cp.Maximize(cs @ ss + co @ so - lam * t)
    prob = cp.Problem(obj, cons)
    try:
        prob.solve(solver=solver or available_mi_socp_solver())
    except Exception:
        pass
    if xi.value is None:                       # solver returned no point
        return worst_case_fixedpoint(xbar, xi_n, lb, ub, cs, co, lam, solver=None)
    xi_star = np.asarray(xi.value)
    val = recourse_value(xbar, xi_star, cs, co) - lam * float(np.linalg.norm(xi_star - xi_n))
    return val, xi_star


# --------------------------------------------------------------------------- #
#  solver-agnostic fixed-point backend
# --------------------------------------------------------------------------- #
def _concave_step(xbar, xi_n, lb, ub, a, lam, solver=None):
    """max_{xi in [lb,ub]} a^T (xi - xbar) - lam ||xi - xi_n||_2  (concave)."""
    D = xbar.size
    xi = cp.Variable(D)
    t = cp.Variable(nonneg=True)
    cons = [xi >= lb, xi <= ub, t >= cp.norm(xi - xi_n, 2)]
    cp.Problem(cp.Maximize(a @ (xi - xbar) - lam * t), cons).solve(solver=solver)
    return np.asarray(xi.value)


def worst_case_fixedpoint(xbar, xi_n, lb, ub, cs, co, lam,
                          n_starts=5, seed=0, solver=None):
    rng = np.random.default_rng(seed)
    D = xbar.size
    seeds = [np.where(xi_n - xbar >= 0.0, 1, -1),
             np.where((ub - xbar) >= (xbar - lb), 1, -1)]
    while len(seeds) < n_starts:
        seeds.append(rng.choice([-1, 1], size=D))

    best_val, best_xi = -np.inf, xi_n.astype(float).copy()
    for sigma in seeds:
        sig = np.asarray(sigma)
        xi_star = best_xi
        for _ in range(12):                       # alternate until the sign pattern is stable
            a = np.where(sig > 0, cs, -co)
            xi_star = _concave_step(xbar, xi_n, lb, ub, a, lam, solver)
            new_sig = np.where(xi_star - xbar >= 0.0, 1, -1)
            if np.array_equal(new_sig, sig):
                break
            sig = new_sig
        val = recourse_value(xbar, xi_star, cs, co) - lam * float(np.linalg.norm(xi_star - xi_n))
        if val > best_val:
            best_val, best_xi = val, xi_star
    return best_val, best_xi


# --------------------------------------------------------------------------- #
#  unified entry point
# --------------------------------------------------------------------------- #
def worst_case_xi(xbar, xi_n, lb, ub, cs, co, lam,
                  mode="auto", solver=None, n_starts=5, seed=0):
    """Return (worst_value, xi_star) for one sample."""
    mode = resolve_mode(mode)
    if mode == "misocp":
        return worst_case_misocp(xbar, xi_n, lb, ub, cs, co, lam, solver)
    if mode == "fixedpoint":
        return worst_case_fixedpoint(xbar, xi_n, lb, ub, cs, co, lam,
                                     n_starts=n_starts, seed=seed, solver=solver)
    raise ValueError(f"unknown separation mode: {mode}")
