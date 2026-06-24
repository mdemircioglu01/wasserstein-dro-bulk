"""
Run the Turkey cement-export case study end-to-end on the REAL-demand instance
built by case_loader.py (demand from UN Comtrade; network calibrated).

Produces, in results/case/ :
  * pareto_case.csv / .png            -- exact cost-emission Pareto front
  * price_of_robustness_case.csv/.png -- fronts for several Wasserstein radii
  * fixed_charge_case.csv             -- exact MILP vs ALNS on the FC variant

Run locally:  python run_case.py
(Use a Gurobi/HiGHS-capable cvxpy; SCIPY (HiGHS) ships with scipy and handles the
LP/MILP. Set DROConfig.solver if you have GUROBI.)
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from case_loader import load_case
from dro_bulk.config import DROConfig
from dro_bulk.multiobjective import epsilon_constraint_front
from dro_bulk.fixed_charge import fc_exact_front, alns_fc_front

OUT = Path(__file__).parent / "results" / "case"
OUT.mkdir(parents=True, exist_ok=True)
SOLVER = None            # set to "GUROBI" if available; else None -> cvxpy default


def _demand_scale(inst):
    return float(np.sqrt((inst.demand_std.reshape(-1) ** 2).sum()))


def _save(name, F, header=("cost_total", "emis_total")):
    with open(OUT / f"{name}.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(header); w.writerows(np.atleast_2d(F).tolist())


def _plot(name, fronts, labels, title):
    # fully guarded: a broken matplotlib (e.g. NumPy 1.x/2.x ABI mismatch) must
    # never abort the run -- the CSV results are already saved.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4.2))
        for F, lab in zip(fronts, labels):
            F = np.atleast_2d(F)
            ax.plot(F[:, 0] / 1e6, F[:, 1] / 1e6, "o-", ms=5, label=lab)
        ax.set_xlabel("Total cost (million USD)")
        ax.set_ylabel(r"Total CO$_2$ (kt)")
        ax.set_title(title); ax.legend(frameon=False); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(OUT / f"{name}.png", dpi=140); plt.close(fig)
    except Exception as exc:
        print(f"  [plot skipped: {exc}]")


def main(n_destinations=10):
    inst = load_case(n_destinations=n_destinations)
    print(f"Real instance: I={inst.n_sources} J={inst.n_destinations} "
          f"K={inst.n_products} D={inst.D} N={inst.train_samples.shape[0]} years")
    s = _demand_scale(inst)

    # 1) exact cost-emission Pareto front (modest robustness)
    dro = DROConfig(norm="l1", epsilon=0.3 * s, support="box", solver=SOLVER)
    exact = epsilon_constraint_front(inst, dro, n_points=11)
    Fe = np.array([[p["cost_total"], p["emis_total"]] for p in exact])
    _save("pareto_case", Fe)
    print(f"[pareto] {len(Fe)} points, cost "
          f"${Fe[:,0].min()/1e6:.0f}-{Fe[:,0].max()/1e6:.0f}M")

    # 2) price of robustness: fronts at several radii
    eps_grid = [0.0, 0.3 * s, 0.8 * s]
    por = []
    for eps in eps_grid:
        f = epsilon_constraint_front(
            inst, DROConfig(norm="l1", epsilon=eps, support="box", solver=SOLVER),
            n_points=9)
        F = np.array([[p["cost_total"], p["emis_total"]] for p in f])
        por.append(F)
        _save(f"por_eps{int(eps)}", F)
        print(f"[robustness] eps={eps:,.0f}: {len(F)} pts, "
              f"cost ${F[:,0].min()/1e6:.0f}-{F[:,0].max()/1e6:.0f}M")
    _plot("price_of_robustness_case", por,
          [f"eps={int(e):,}" for e in eps_grid], "Price of robustness (real instance)")
    _plot("pareto_case", [Fe], ["exact e-constraint"],
          "Cost-emission Pareto front (real instance)")

    # 3) fixed-charge variant: exact MILP vs ALNS
    print("[fixed-charge] solving (exact MILP may be slow on the real network)...")
    drofc = DROConfig(norm="l1", epsilon=0.3 * s, support="box",
                      fixed_charge=True, solver=SOLVER or "SCIPY")
    rows = [("method", "cost_total", "emis_total")]
    try:
        ex = fc_exact_front(inst, drofc, n_points=5)
        for p in ex:
            rows.append(("exact_MILP", p["cost_total"], p["emis_total"]))
        print(f"  exact MILP: {len(ex)} points")
    except Exception as exc:
        print(f"  exact MILP skipped/failed: {exc}")
    Fa = alns_fc_front(inst, drofc, iters=1500, seed=0)
    for c, e in np.atleast_2d(Fa).tolist():
        rows.append(("ALNS", c, e))
    print(f"  ALNS: {len(Fa)} points")
    with open(OUT / "fixed_charge_case.csv", "w", newline="") as fh:
        csv.writer(fh).writerows(rows)

    print(f"\nresults -> {OUT}")


if __name__ == "__main__":
    main()
