"""
Recourse evaluation and fast worst-case-expectation evaluators.

These routines are *evaluation* helpers (given a fixed first-stage plan they
return objective values).  They are used by NSGA-II, for out-of-sample testing,
and to cross-check the optimisation models in :mod:`dro_bulk.dro_model`.

All vectors here are flat of length ``D = J*K`` (see :mod:`dro_bulk.config`).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


# --------------------------------------------------------------------------- #
#  Second-stage recourse, equations (recourse-cost) / (recourse-emis)
# --------------------------------------------------------------------------- #
def recourse_value(xbar_d: np.ndarray, xi_d: np.ndarray,
                   coef_short: np.ndarray, coef_surp: np.ndarray) -> np.ndarray:
    """Recourse value per sample.

    ``xbar_d`` is length D (delivered quantities); ``xi_d`` is (n, D) or (D,).
    ``coef_short`` / ``coef_surp`` are the unit shortage / surplus coefficients
    (b, h) for cost or (beta, eta) for emission.  Returns an array of length n.
    """
    xi = np.atleast_2d(xi_d)
    imbalance = xi - xbar_d[None, :]
    shortage = np.maximum(imbalance, 0.0)
    surplus = np.maximum(-imbalance, 0.0)
    return shortage @ coef_short + surplus @ coef_surp


def lipschitz_modulus(coef_short: np.ndarray, coef_surp: np.ndarray,
                      norm: str) -> float:
    """Lipschitz constant of the recourse function w.r.t. the demand vector,
    measured in the *dual* of the ground norm.

    The recourse subgradient lives in the box [-coef_surp, coef_short], so the
    largest dual-norm vertex is the modulus:
        l1 ground  -> l_inf dual -> max_d max(b_d, h_d)
        l2 ground  -> l_2   dual -> sqrt( sum_d max(b_d, h_d)^2 )
    """
    vertex = np.maximum(coef_short, coef_surp)
    if norm == "l1":
        return float(np.max(vertex))
    if norm == "l2":
        return float(np.sqrt(np.sum(vertex ** 2)))
    raise ValueError(norm)


# --------------------------------------------------------------------------- #
#  Worst-case expected recourse  --  fast evaluators (fixed plan)
# --------------------------------------------------------------------------- #
def worst_case_full_support(xbar_d, samples, coef_short, coef_surp,
                            norm: str, epsilon: float) -> float:
    """Xi = R^D :  sup_P E_P[Q] = E_Phat[Q] + epsilon * Lipschitz modulus.

    Exact for the full-support ambiguity set; used as a fast surrogate.
    """
    saa = recourse_value(xbar_d, samples, coef_short, coef_surp).mean()
    return float(saa + epsilon * lipschitz_modulus(coef_short, coef_surp, norm))


def worst_case_l1_box(xbar_d, samples, coef_short, coef_surp,
                      lb_d, ub_d, epsilon: float) -> float:
    """Exact worst-case expected recourse for the **type-1 (l1), box-support**
    ambiguity set, evaluated for a *fixed* plan ``xbar_d``.

    Uses
        sup_P E_P[Q] = min_{lambda >= 0} lambda*eps
                       + (1/N) sum_n sup_{xi in Xi} ( Q(xbar, xi) - lambda||xi-xi_n||_1 )
    Because both Q and the l1 norm are separable across coordinates, the inner
    sup decomposes per coordinate and is attained at one of the breakpoints
    {lb, ub, xbar, xi_n}.  The remaining 1-D minimisation over lambda is convex.
    """
    xi = np.atleast_2d(samples)                  # (N, D)
    N, D = xi.shape
    b, h = coef_short, coef_surp

    def q(xi_val):                               # recourse value per coord, vectorised
        imb = xi_val - xbar_d
        return np.maximum(b * imb, -h * imb)

    # candidate breakpoints per (coord): lb, ub, xbar, and each sample xi_n
    # shape (N, D) for the per-sample candidate xi_n; lb/ub/xbar broadcast.
    def inner_sum(lam: float) -> float:
        # The inner sup over xi is piecewise linear with kinks at xbar and xi_n,
        # so it is attained at a breakpoint in {lb, ub, xbar, xi_n}.  We take a
        # running max over these candidates, per (sample n, coordinate d).
        best = np.full((N, D), -np.inf)
        for cand in (lb_d, ub_d, xbar_d):                  # n-independent kinks
            cand = np.clip(cand, lb_d, ub_d)
            val = q(cand)[None, :] - lam * np.abs(cand[None, :] - xi)
            best = np.maximum(best, val)
        best = np.maximum(best, q(xi))                     # the sample point (zero penalty)
        return best.sum() / N

    lip = lipschitz_modulus(b, h, "l1")
    res = minimize_scalar(lambda lam: lam * epsilon + inner_sum(lam),
                          bounds=(0.0, max(lip, 1e-9)), method="bounded")
    return float(res.fun)


def evaluate_plan(inst, x_ijk: np.ndarray, dro_cfg,
                  out_of_sample: bool = False) -> dict:
    """Evaluate both objectives for a first-stage plan ``x_ijk`` (I, J, K).

    Returns transportation, (worst-case or out-of-sample expected) recourse,
    and total for cost and emission.
    """
    xbar = inst.flat(x_ijk.sum(axis=0))          # (D,)
    transport_cost = float((inst.cost * x_ijk).sum())
    transport_emis = float((inst.emission * x_ijk).sum())

    cs, co = inst.flat(inst.short_cost), inst.flat(inst.surp_cost)
    es, eo = inst.flat(inst.short_emis), inst.flat(inst.surp_emis)
    lb, ub = inst.flat(inst.support_lb), inst.flat(inst.support_ub)

    if out_of_sample:
        rc = recourse_value(xbar, inst.test_samples, cs, co).mean()
        re = recourse_value(xbar, inst.test_samples, es, eo).mean()
    elif dro_cfg.support == "full":
        rc = worst_case_full_support(xbar, inst.train_samples, cs, co,
                                     dro_cfg.norm, dro_cfg.epsilon)
        re = worst_case_full_support(xbar, inst.train_samples, es, eo,
                                     dro_cfg.norm, dro_cfg.epsilon)
    elif dro_cfg.norm == "l1":
        rc = worst_case_l1_box(xbar, inst.train_samples, cs, co, lb, ub,
                               dro_cfg.epsilon)
        re = worst_case_l1_box(xbar, inst.train_samples, es, eo, lb, ub,
                               dro_cfg.epsilon)
    else:
        # exact l2-box evaluation has no separable closed form; fall back to the
        # (slightly conservative) full-support bound for fast evaluation.
        rc = worst_case_full_support(xbar, inst.train_samples, cs, co,
                                     dro_cfg.norm, dro_cfg.epsilon)
        re = worst_case_full_support(xbar, inst.train_samples, es, eo,
                                     dro_cfg.norm, dro_cfg.epsilon)

    return {
        "cost_transport": transport_cost,
        "cost_recourse": float(rc),
        "cost_total": transport_cost + float(rc),
        "emis_transport": transport_emis,
        "emis_recourse": float(re),
        "emis_total": transport_emis + float(re),
    }
