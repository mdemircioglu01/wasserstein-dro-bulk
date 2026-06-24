"""
Distributionally robust two-stage model -- tractable reformulation in cvxpy.

This module turns the semi-infinite DRO model of the paper into a finite convex
program and solves it.  Following Proposition 1 (Mohajerin Esfahani & Kuhn,
2018) the worst-case expected recourse is written with

  * a Wasserstein multiplier ``lambda``,
  * a per-sample epigraph variable ``s_n``,
  * per-(sample, piece) support-dual variables ``gamma`` for the box support,
  * one dual-norm constraint ``|| H^T gamma - a ||_*  <=  lambda`` per piece.

The recourse pieces ``a`` are the vertices of the box dual feasible set
Pi = [-h, b].  Enumerating all 2^D vertices is intractable, so we generate
them on the fly (row generation): the active piece for sample ``n`` at the
current plan is the subgradient with sign pattern ``sign(xi_n - xbar)``.  The
sign pattern is identical for the cost and the emission objective, which lets
the epsilon-constraint model share one pattern set across both objectives.

Choosing the ground norm selects the cone:
    norm == "l1"  ->  || . ||_inf <= lambda   (linear)            -> LP
    norm == "l2"  ->  || . ||_2   <= lambda   (second-order cone) -> SOCP
"""
from __future__ import annotations

import numpy as np
import cvxpy as cp

from .config import DROConfig
from .data_generation import Instance
from .recourse import evaluate_plan
from .separation import worst_case_xi, resolve_mode


class DROSolver:
    def __init__(self, inst: Instance, dro_cfg: DROConfig):
        self.inst = inst
        self.cfg = dro_cfg
        I, J, K, D = inst.n_sources, inst.n_destinations, inst.n_products, inst.D
        self.I, self.J, self.K, self.D = I, J, K, D

        # flat coefficient arrays
        self.cost_id = inst.cost.reshape(I, D)          # (I, D)
        self.emis_id = inst.emission.reshape(I, D)
        self.cs = inst.flat(inst.short_cost)            # b_{jk}
        self.co = inst.flat(inst.surp_cost)             # h_{jk}
        self.es = inst.flat(inst.short_emis)            # beta_{jk}
        self.eo = inst.flat(inst.surp_emis)             # eta_{jk}
        self.lb = inst.flat(inst.support_lb)
        self.ub = inst.flat(inst.support_ub)
        self.xi = inst.train_samples                    # (N, D)
        self.N = self.xi.shape[0]

        # selector matrices for the first-stage constraints
        self.Mk = np.zeros((K, D))                      # sum over j of x[i,j,k]
        self.Mj = np.zeros((J, D))                      # sum_k rho_k x[i,j,k]
        for j in range(J):
            for k in range(K):
                d = j * K + k
                self.Mk[k, d] = 1.0
                self.Mj[j, d] = inst.rho[k]

        self._pnorm = "inf" if dro_cfg.norm == "l1" else 2

        # genuine row generation for the l2 box-support case
        self._l2_sep = (dro_cfg.norm == "l2" and dro_cfg.support == "box"
                        and dro_cfg.l2_separation != "active")
        self._sep_mode = (resolve_mode(dro_cfg.l2_separation)
                          if self._l2_sep else None)

        # exact compact separable LP for the l1 box-support case
        self._compact_l1 = (dro_cfg.norm == "l1" and dro_cfg.support == "box"
                            and getattr(dro_cfg, "l1_compact", True))

    # ------------------------------------------------------------------ #
    #  first-stage variables and feasible region X
    # ------------------------------------------------------------------ #
    def _base(self):
        x = cp.Variable((self.I, self.D), nonneg=True)
        xbar = cp.sum(x, axis=0)                        # (D,)
        cons = []
        for i in range(self.I):
            cons.append(self.Mk @ x[i, :] <= self.inst.supply[i, :])
            cons.append(self.Mj @ x[i, :] <= self.inst.link_cap[i, :])
        return x, xbar, cons

    # ------------------------------------------------------------------ #
    #  worst-case expected recourse term for one objective
    # ------------------------------------------------------------------ #
    def _slope(self, pattern, coef_short, coef_surp):
        """Recourse subgradient for a +/-1 sign pattern (length D)."""
        return np.where(pattern > 0, coef_short, -coef_surp)

    def _wc_terms(self, xbar, coef_short, coef_surp, patterns):
        """Return (expr, constraints, lam, s) for sup_P E_P[Q] given the pieces."""
        s = cp.Variable(self.N)
        lam = cp.Variable(nonneg=True)
        cons = []
        for n in range(self.N):
            xi_n = self.xi[n]
            for pattern in patterns[n]:
                a = self._slope(pattern, coef_short, coef_surp)     # constant (D,)
                if self.cfg.support == "box":
                    gu = cp.Variable(self.D, nonneg=True)
                    gl = cp.Variable(self.D, nonneg=True)
                    epi = (a @ (xi_n - xbar)
                           + gu @ (self.ub - xi_n) + gl @ (xi_n - self.lb))
                    cons.append(s[n] >= epi)
                    cons.append(cp.norm(gu - gl - a, self._pnorm) <= lam)
                else:  # full support  ->  || a ||_* <= lambda  (constant bound)
                    cons.append(s[n] >= a @ (xi_n - xbar))
                    cons.append(cp.norm(-a, self._pnorm) <= lam)
        expr = lam * self.cfg.epsilon + cp.sum(s) / self.N
        return expr, cons, lam, s

    # ------------------------------------------------------------------ #
    #  exact compact separable LP for the l1 box-support worst case
    # ------------------------------------------------------------------ #
    def _wc_terms_compact_l1(self, xbar, coef_short, coef_surp):
        """Return (expr, constraints, lam) for sup_P E_P[Q] under the type-1
        (l1) ground norm and box support, as a single LP.

        Because the recourse, the l1 metric and the box support all decompose
        across coordinates, the worst-case expectation separates: for a shared
        multiplier lambda, each (sample n, coordinate d) contributes an epigraph
        variable s_{n,d} dominating its two recourse pieces (shortage slope b_d,
        surplus slope -h_d), with per-coordinate box-support duals and the
        l_inf dual-norm (Lipschitz) constraint. This is exactly equivalent to
        the general reformulation but needs no vertex enumeration or row
        generation -- there are only two pieces per coordinate.
        """
        N, D = self.N, self.D
        b = coef_short.reshape(1, D)
        h = coef_surp.reshape(1, D)
        ub = self.ub.reshape(1, D)
        lb = self.lb.reshape(1, D)
        xi = self.xi                                   # (N, D) constant
        ubgap = ub - xi                                # (N, D) >= 0
        lbgap = xi - lb                                # (N, D) >= 0

        xbar_mat = np.ones((N, 1)) @ cp.reshape(xbar, (1, D), order="C")   # (N, D)
        dev = xi - xbar_mat                            # xi - xbar, affine in x

        s = cp.Variable((N, D))
        lam = cp.Variable(nonneg=True)
        gpu = cp.Variable((N, D), nonneg=True)         # shortage upper dual
        gpl = cp.Variable((N, D), nonneg=True)         # shortage lower dual
        gnu = cp.Variable((N, D), nonneg=True)         # surplus upper dual
        gnl = cp.Variable((N, D), nonneg=True)         # surplus lower dual

        cons = [
            # epigraph: s_{n,d} >= each recourse piece + support-dual slack
            cp.multiply(b, dev) + cp.multiply(gpu, ubgap) + cp.multiply(gpl, lbgap) <= s,
            cp.multiply(-h, dev) + cp.multiply(gnu, ubgap) + cp.multiply(gnl, lbgap) <= s,
            # l_inf dual-norm (Lipschitz) constraints, per coordinate and piece
            (gpu - gpl) - b <= lam,  b - (gpu - gpl) <= lam,
            (gnu - gnl) + h <= lam,  -h - (gnu - gnl) <= lam,
        ]
        expr = lam * self.cfg.epsilon + cp.sum(s) / N
        return expr, cons, lam

    # ------------------------------------------------------------------ #
    #  piece (sign-pattern) generation helpers
    # ------------------------------------------------------------------ #
    def _active_pattern(self, xbar_val):
        """sign(xi_n - xbar) for every sample; +1 = shortage piece."""
        return np.where(self.xi - xbar_val[None, :] >= 0.0, 1, -1)

    def _saa_xbar(self, coef_short, coef_surp):
        """Solve the sample-average problem to seed the piece set."""
        x, xbar, cons = self._base()
        ss = cp.Variable((self.N, self.D), nonneg=True)
        so = cp.Variable((self.N, self.D), nonneg=True)
        for n in range(self.N):
            cons.append(ss[n] - so[n] == self.xi[n] - xbar)
        rec = cp.sum(ss @ coef_short + so @ coef_surp) / self.N
        transport = cp.sum(cp.multiply(self.cost_id, x))
        cp.Problem(cp.Minimize(transport + rec), cons).solve(
            solver=self.cfg.solver, verbose=False)
        return xbar.value

    def _init_patterns(self, coef_short, coef_surp):
        xbar0 = self._saa_xbar(coef_short, coef_surp)
        active = self._active_pattern(xbar0)
        return {n: [active[n]] for n in range(self.N)}

    @staticmethod
    def _add_patterns(patterns, new_active) -> int:
        """Add any unseen sign patterns; return how many were added."""
        added = 0
        for n, pat in enumerate(new_active):
            if not any(np.array_equal(pat, p) for p in patterns[n]):
                patterns[n].append(pat)
                added += 1
        return added

    def _grow_pieces(self, patterns, xbar_val, objectives) -> tuple[int, float]:
        """Add violated recourse pieces and return (n_added, max_violation).

        ``objectives`` is a list of (lam_value, coef_short, coef_surp, s_var),
        one entry per robust objective active in the current master.

        * l2 box-support: run the separation oracle (worst-case xi*) for every
          objective and sample; add the subgradient at xi* when its value
          exceeds the current epigraph variable s_n.  No violation -> certified.
        * otherwise: add the subgradient at the sample point sign(xi_n - xbar).
        """
        if not self._l2_sep:
            added = self._add_patterns(patterns, self._active_pattern(xbar_val))
            return added, float("inf")

        added, max_viol = 0, 0.0
        for (lam_val, cS, cO, s_var) in objectives:
            s_val = s_var.value
            for n in range(self.N):
                val, xi_star = worst_case_xi(
                    xbar_val, self.xi[n], self.lb, self.ub, cS, cO, lam_val,
                    mode=self._sep_mode, n_starts=self.cfg.fixedpoint_starts,
                    seed=n, solver=self.cfg.solver)
                viol = val - float(s_val[n])
                max_viol = max(max_viol, viol)
                if viol > self.cfg.tol:
                    pat = np.where(xi_star - xbar_val >= 0.0, 1, -1)
                    if not any(np.array_equal(pat, p) for p in patterns[n]):
                        patterns[n].append(pat)
                        added += 1
        return added, max_viol

    # ------------------------------------------------------------------ #
    #  public API
    # ------------------------------------------------------------------ #
    def solve_single(self, objective: str = "cost") -> dict:
        """Minimise one robust objective (cost or emission) over X."""
        cS, cO, T = self._obj_coeffs(objective)

        if self._compact_l1:                       # exact single-LP path
            x, xbar, cons = self._base()
            expr, wc, lam = self._wc_terms_compact_l1(xbar, cS, cO)
            transport = cp.sum(cp.multiply(T, x))
            prob = cp.Problem(cp.Minimize(transport + expr), cons + wc)
            prob.solve(solver=self.cfg.solver, verbose=self.cfg.verbose)
            if prob.status not in ("optimal", "optimal_inaccurate"):
                return {"status": prob.status, "x": None}
            x_ijk = self._reshape(x.value)
            out = evaluate_plan(self.inst, x_ijk, self.cfg)
            tr, rc = float(transport.value), float(expr.value)
            prefix = "cost" if objective == "cost" else "emis"
            out[f"{prefix}_transport"], out[f"{prefix}_recourse"] = tr, rc
            out[f"{prefix}_total"] = tr + rc
            out.update(status=prob.status, iters=1, max_violation=0.0,
                       certified=True, n_pieces=2 * self.N * self.D, x=x_ijk)
            return out

        patterns = self._init_patterns(cS, cO)

        max_viol = float("inf")
        for it in range(self.cfg.max_outer_iter):
            x, xbar, cons = self._base()
            expr, wc_cons, lam, s = self._wc_terms(xbar, cS, cO, patterns)
            transport = cp.sum(cp.multiply(T, x))
            prob = cp.Problem(cp.Minimize(transport + expr), cons + wc_cons)
            prob.solve(solver=self.cfg.solver, verbose=self.cfg.verbose)
            if prob.status not in ("optimal", "optimal_inaccurate"):
                return {"status": prob.status, "x": None}
            added, max_viol = self._grow_pieces(
                patterns, xbar.value, [(lam.value, cS, cO, s)])
            if added == 0 or max_viol <= self.cfg.tol:
                break

        x_ijk = self._reshape(x.value)
        out = evaluate_plan(self.inst, x_ijk, self.cfg)
        # report the optimised objective with the exact value from the master
        tr, rc = float(transport.value), float(expr.value)
        prefix = "cost" if objective == "cost" else "emis"
        out[f"{prefix}_transport"], out[f"{prefix}_recourse"] = tr, rc
        out[f"{prefix}_total"] = tr + rc
        out.update(status=prob.status, iters=it + 1, max_violation=max_viol,
                   certified=bool(added == 0 or max_viol <= self.cfg.tol),
                   n_pieces=sum(len(p) for p in patterns.values()), x=x_ijk)
        return out

    def solve_epsilon_constraint(self, primary: str, secondary: str,
                                 kappa: float) -> dict:
        """Minimise the ``primary`` robust objective subject to the ``secondary``
        robust objective being <= kappa (augmented epsilon-constraint)."""
        pS, pO, pT = self._obj_coeffs(primary)
        sS, sO, sT = self._obj_coeffs(secondary)
        delta = 1e-4  # augmentation weight to avoid weakly-dominated points

        if self._compact_l1:                       # exact single-LP path
            x, xbar, cons = self._base()
            p_expr, p_cons, p_lam = self._wc_terms_compact_l1(xbar, pS, pO)
            s_expr, s_cons, s_lam = self._wc_terms_compact_l1(xbar, sS, sO)
            p_transport = cp.sum(cp.multiply(pT, x))
            s_transport = cp.sum(cp.multiply(sT, x))
            cons += p_cons + s_cons
            cons.append(s_transport + s_expr <= kappa)
            obj = (p_transport + p_expr) + delta * (s_transport + s_expr)
            prob = cp.Problem(cp.Minimize(obj), cons)
            prob.solve(solver=self.cfg.solver, verbose=self.cfg.verbose)
            if prob.status not in ("optimal", "optimal_inaccurate"):
                return {"status": prob.status, "x": None}
            x_ijk = self._reshape(x.value)
            out = evaluate_plan(self.inst, x_ijk, self.cfg)
            out["cost_transport"] = float(p_transport.value)
            out["cost_recourse"] = float(p_expr.value)
            out["cost_total"] = float((p_transport + p_expr).value)
            out["emis_transport"] = float(s_transport.value)
            out["emis_recourse"] = float(s_expr.value)
            out["emis_total"] = float((s_transport + s_expr).value)
            out.update(status=prob.status, iters=1, max_violation=0.0,
                       certified=True, x=x_ijk)
            return out

        patterns = self._init_patterns(pS, pO)

        max_viol = float("inf")
        for it in range(self.cfg.max_outer_iter):
            x, xbar, cons = self._base()
            p_expr, p_cons, p_lam, p_s = self._wc_terms(xbar, pS, pO, patterns)
            s_expr, s_cons, s_lam, s_s = self._wc_terms(xbar, sS, sO, patterns)
            p_transport = cp.sum(cp.multiply(pT, x))
            s_transport = cp.sum(cp.multiply(sT, x))
            cons += p_cons + s_cons
            cons.append(s_transport + s_expr <= kappa)
            obj = (p_transport + p_expr) + delta * (s_transport + s_expr)
            prob = cp.Problem(cp.Minimize(obj), cons)
            prob.solve(solver=self.cfg.solver, verbose=self.cfg.verbose)
            if prob.status not in ("optimal", "optimal_inaccurate"):
                return {"status": prob.status, "x": None}
            # grow pieces against BOTH objectives (shared piece set)
            added, max_viol = self._grow_pieces(
                patterns, xbar.value,
                [(p_lam.value, pS, pO, p_s), (s_lam.value, sS, sO, s_s)])
            if added == 0 or max_viol <= self.cfg.tol:
                break

        x_ijk = self._reshape(x.value)
        out = evaluate_plan(self.inst, x_ijk, self.cfg)
        # both objectives are in the master -> report their exact values
        out["cost_transport"] = float(p_transport.value)
        out["cost_recourse"] = float(p_expr.value)
        out["cost_total"] = float((p_transport + p_expr).value)
        out["emis_transport"] = float(s_transport.value)
        out["emis_recourse"] = float(s_expr.value)
        out["emis_total"] = float((s_transport + s_expr).value)
        out.update(status=prob.status, iters=it + 1, max_violation=max_viol,
                   certified=bool(added == 0 or max_viol <= self.cfg.tol), x=x_ijk)
        return out

    # ------------------------------------------------------------------ #
    def _obj_coeffs(self, objective: str):
        if objective == "cost":
            return self.cs, self.co, self.cost_id
        if objective == "emission":
            return self.es, self.eo, self.emis_id
        raise ValueError(objective)

    def _reshape(self, x_flat):
        return np.asarray(x_flat).reshape(self.I, self.J, self.K)
