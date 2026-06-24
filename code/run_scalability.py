"""
Scalability study (paper Section 5) -- journal-quality, Gurobi-ready.

For each instance class (S / M / L) and several random replications we trace the
exact augmented epsilon-constraint front and an NSGA-II approximation of the same
robust objectives, then report, as mean +/- std over replications:

  * exact solve time and NSGA-II solve time,
  * number of exact Pareto points,
  * hypervolume of each front (objectives normalised to a common ideal/nadir box,
    reference point [1.1, 1.1]; larger = better coverage),
  * IGD of the NSGA-II front w.r.t. the exact front (smaller = closer),
  * fraction of subproblems that terminated with an optimality certificate.

Outputs (in ./results):
  * scalability_runs.jsonl   -- one line per (scale, seed) replication
  * scalability_table.csv    -- aggregated mean/std table
  * scalability_table.tex    -- booktabs LaTeX table, ready for \\input

--------------------------------------------------------------------------------
QUICK START (on a machine with Gurobi)

  pip install -r requirements.txt          # plus gurobipy + a Gurobi license
  python run_scalability.py --check        # verify the solver is visible
  python run_scalability.py --quality journal --norm both --solver GUROBI
  # ... go get a coffee; then the CSV/TeX tables are in ./results

For a fast sanity run with the open-source solver:
  python run_scalability.py --quality smoke
--------------------------------------------------------------------------------
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import cvxpy as cp

from dro_bulk.config import InstanceConfig, DROConfig
from dro_bulk.data_generation import generate_instance
from dro_bulk.multiobjective import epsilon_constraint_front, nsga2_front

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)
LOG = OUT / "scalability_runs.jsonl"

# --------------------------------------------------------------------------- #
#  Experiment budgets.  Each scale maps to:
#     (n_train, n_front_points, max_outer_iter, nsga_pop, nsga_gen)
#  "journal" uses the full sample size and publication-grade budgets;
#  "smoke" is a seconds-long sanity configuration for the open-source solver.
# --------------------------------------------------------------------------- #
PRESETS = {
    "journal": {
        "seeds": 10,
        "scales": {
            "S": (100, 21, 30, 120, 400),
            "M": (100, 21, 30, 150, 500),
            "L": (100, 17, 25, 200, 600),
        },
    },
    "smoke": {
        "seeds": 2,
        "scales": {
            "S": (20, 5, 3, 30, 20),
            "M": (15, 5, 3, 30, 15),
            "L": (10, 4, 2, 24, 12),
        },
    },
}


# --------------------------------------------------------------------------- #
#  metrics
# --------------------------------------------------------------------------- #
def _hv(points, ideal, nadir, ref=1.3):
    """Normalised hypervolume in [0,1].  Objectives are scaled to the exact
    front's ideal/nadir box; the reference point sits ``ref`` beyond the nadir
    (so an approximate front within (ref-1)*range of the true front still scores
    positively), and the result is divided by ref^2 to land in [0,1]."""
    from pymoo.indicators.hv import HV
    if points is None or len(points) == 0:
        return 0.0
    span = np.where(nadir - ideal > 0, nadir - ideal, 1.0)
    norm = (np.atleast_2d(points) - ideal) / span
    norm = norm[(norm <= ref).all(axis=1)]
    if len(norm) == 0:
        return 0.0
    return float(HV(ref_point=np.array([ref, ref])).do(norm)) / (ref * ref)


def _igd(points, reference, ideal, nadir):
    """IGD of `points` w.r.t. the `reference` (exact) front, in normalised space."""
    from pymoo.indicators.igd import IGD
    if points is None or len(points) == 0 or reference is None or len(reference) == 0:
        return float("nan")
    span = np.where(nadir - ideal > 0, nadir - ideal, 1.0)
    ref = (np.atleast_2d(reference) - ideal) / span
    pts = (np.atleast_2d(points) - ideal) / span
    return float(IGD(ref).do(pts))


# --------------------------------------------------------------------------- #
#  one replication
# --------------------------------------------------------------------------- #
def run_once(scale, budget, norm, epsilon, solver, seed):
    n_train, n_pts, moi, pop, gen = budget
    inst = generate_instance(InstanceConfig(
        scale=scale, seed=seed, n_train=n_train,
        mu_shortage_cost=6.0, supply_tightness=1.05))
    dro = DROConfig(norm=norm, epsilon=epsilon, support="box",
                    max_outer_iter=moi, solver=solver, l2_separation="auto")

    t0 = time.time()
    exact = epsilon_constraint_front(inst, dro, n_points=n_pts)
    t_exact = time.time() - t0
    Fe = np.array([[p["cost_total"], p["emis_total"]] for p in exact])
    certified = [bool(p.get("certified", True)) for p in exact]
    cert_frac = float(np.mean(certified)) if certified else 1.0

    t0 = time.time()
    try:
        Fn = nsga2_front(inst, dro, pop_size=pop, n_gen=gen, seed=seed)
    except Exception as exc:
        print(f"  [seed {seed}] NSGA-II failed: {exc}")
        Fn = np.empty((0, 2))
    t_nsga = time.time() - t0

    # Normalise against the EXACT front (the reference Pareto front), not the
    # union: this keeps HV/IGD stable and comparable across instances and
    # prevents a poor NSGA front from distorting the box (and producing HV>1).
    ideal, nadir = Fe.min(axis=0), Fe.max(axis=0)

    row = dict(scale=scale, norm=norm, seed=seed, D=int(inst.D), N=int(n_train),
               n_exact=int(len(Fe)), n_nsga=int(len(Fn)),
               t_exact=round(t_exact, 3), t_nsga=round(t_nsga, 3),
               hv_exact=round(_hv(Fe, ideal, nadir), 5),
               hv_nsga=round(_hv(Fn, ideal, nadir), 5),
               igd_nsga=round(_igd(Fn, Fe, ideal, nadir), 5),
               cert_frac=round(cert_frac, 3), solver=str(solver))
    with open(LOG, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(f"  [seed {seed}] t_exact={row['t_exact']}s t_nsga={row['t_nsga']}s "
          f"pts={row['n_exact']} HVe={row['hv_exact']} HVn={row['hv_nsga']} "
          f"IGD={row['igd_nsga']} cert={row['cert_frac']}")
    return row


# --------------------------------------------------------------------------- #
#  aggregation + reporting
# --------------------------------------------------------------------------- #
def _ms(values):
    vals = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not vals:
        return (float("nan"), 0.0)
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return (m, s)


def _fmt(m, s, dec=2):
    if np.isnan(m):
        return "--"
    return f"{m:.{dec}f}$\\pm${s:.{dec}f}"


def summarize():
    if not LOG.exists():
        print("no results yet -- run a scale first")
        return
    rows = [json.loads(l) for l in open(LOG)]
    groups = {}
    for r in rows:
        groups.setdefault((r["norm"], r["scale"]), []).append(r)

    order = {"S": 0, "M": 1, "L": 2}
    keys = sorted(groups, key=lambda k: (k[0], order[k[1]]))

    # console table
    hdr = (f"{'scale':<6}{'norm':<5}{'D':>5}{'N':>5}{'reps':>5}{'pts':>6}"
           f"{'t_exact(s)':>16}{'t_nsga(s)':>16}{'HV_exact':>14}{'HV_nsga':>14}"
           f"{'IGD':>12}{'cert':>7}")
    print(hdr); print("-" * len(hdr))
    agg = []
    for k in keys:
        g = groups[k]
        D = g[0]["D"]; N = g[0]["N"]; reps = len(g)
        te = _ms([r["t_exact"] for r in g]); tn = _ms([r["t_nsga"] for r in g])
        he = _ms([r["hv_exact"] for r in g]); hn = _ms([r["hv_nsga"] for r in g])
        ig = _ms([r["igd_nsga"] for r in g]); pt = _ms([r["n_exact"] for r in g])
        cf = _ms([r["cert_frac"] for r in g])
        print(f"{k[1]:<6}{k[0]:<5}{D:>5}{N:>5}{reps:>5}{pt[0]:>6.0f}"
              f"{te[0]:>10.2f}+/-{te[1]:<3.2f}{tn[0]:>10.2f}+/-{tn[1]:<3.2f}"
              f"{he[0]:>9.3f}+/-{he[1]:<.3f}{hn[0]:>9.3f}+/-{hn[1]:<.3f}"
              f"{ig[0]:>9.3f}{cf[0]:>7.2f}")
        agg.append(dict(scale=k[1], norm=k[0], D=D, N=N, reps=reps,
                        pts=pt[0], t_exact=te, t_nsga=tn, hv_exact=he,
                        hv_nsga=hn, igd=ig, cert=cf[0]))

    _write_csv(agg)
    _write_latex(agg)
    print(f"\nCSV  -> {OUT / 'scalability_table.csv'}")
    print(f"LaTeX-> {OUT / 'scalability_table.tex'}")


def _write_csv(agg):
    import csv
    keys = ["scale", "norm", "D", "N", "reps", "pts",
            "t_exact_mean", "t_exact_std", "t_nsga_mean", "t_nsga_std",
            "hv_exact_mean", "hv_exact_std", "hv_nsga_mean", "hv_nsga_std",
            "igd_mean", "cert_frac"]
    with open(OUT / "scalability_table.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(keys)
        for a in agg:
            w.writerow([a["scale"], a["norm"], a["D"], a["N"], a["reps"],
                        f"{a['pts']:.1f}",
                        f"{a['t_exact'][0]:.3f}", f"{a['t_exact'][1]:.3f}",
                        f"{a['t_nsga'][0]:.3f}", f"{a['t_nsga'][1]:.3f}",
                        f"{a['hv_exact'][0]:.4f}", f"{a['hv_exact'][1]:.4f}",
                        f"{a['hv_nsga'][0]:.4f}", f"{a['hv_nsga'][1]:.4f}",
                        f"{a['igd'][0]:.4f}", f"{a['cert']:.2f}"])


def _write_latex(agg):
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Scalability of the exact $\varepsilon$-constraint method and "
        r"NSGA-II across instance classes (mean$\pm$std over replications). "
        r"$D=|J||K|$ is the demand dimension; HV is the normalised hypervolume "
        r"(reference point $[1.1,1.1]$); IGD is computed against the exact front.}",
        r"\label{tab:scalability}",
        r"\begin{tabular}{llrrrrrrrr}", r"\toprule",
        r"Class & Norm & $D$ & $N$ & $|\mathcal{P}|$ & $t_{\text{exact}}$ (s) & "
        r"$t_{\text{NSGA}}$ (s) & HV$_{\text{exact}}$ & HV$_{\text{NSGA}}$ & IGD \\",
        r"\midrule",
    ]
    for a in agg:
        lines.append(
            f"{a['scale']} & {a['norm']} & {a['D']} & {a['N']} & {a['pts']:.0f} & "
            f"{_fmt(*a['t_exact'])} & {_fmt(*a['t_nsga'])} & "
            f"{_fmt(*a['hv_exact'], dec=3)} & {_fmt(*a['hv_nsga'], dec=3)} & "
            f"{a['igd'][0]:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (OUT / "scalability_table.tex").write_text("\n".join(lines))


# --------------------------------------------------------------------------- #
def preflight(requested_solver):
    print("Python    :", platform.python_version())
    print("cvxpy     :", cp.__version__)
    installed = cp.installed_solvers()
    print("solvers   :", installed)
    for s in ("GUROBI", "MOSEK", "CPLEX", "SCIP", "ECOS_BB"):
        print(f"  {s:<8}: {'available' if s in installed else 'not found'}")
    mi = [s for s in ("GUROBI", "MOSEK", "SCIP", "ECOS_BB") if s in installed]
    print("MI-conic  :", mi or "none (l2 separation will use 'fixedpoint')")
    chosen = pick_solver(requested_solver)
    print("will use  :", chosen or "cvxpy default")
    try:
        import pymoo
        print("pymoo     :", pymoo.__version__)
    except Exception:
        print("pymoo     : NOT INSTALLED (NSGA-II will be skipped)")


def pick_solver(requested):
    installed = cp.installed_solvers()
    if requested:
        if requested not in installed:
            print(f"WARNING: requested solver {requested} not installed; "
                  f"falling back to cvxpy default")
            return None
        return requested
    return "GUROBI" if "GUROBI" in installed else None     # prefer Gurobi if present


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality", default="journal", choices=["journal", "smoke"])
    ap.add_argument("--norm", default="l1", choices=["l1", "l2", "both"])
    ap.add_argument("--scales", nargs="+", default=["S", "M", "L"],
                    choices=["S", "M", "L"])
    ap.add_argument("--seeds", type=int, default=None,
                    help="override the number of replications")
    ap.add_argument("--epsilon", type=float, default=0.05)
    ap.add_argument("--solver", default=None,
                    help="cvxpy solver name, e.g. GUROBI / MOSEK")
    # optional budget overrides (override the per-scale PRESETS components)
    ap.add_argument("--n-train", type=int, default=None, dest="n_train")
    ap.add_argument("--points", type=int, default=None)
    ap.add_argument("--max-iter", type=int, default=None, dest="max_iter")
    ap.add_argument("--pop", type=int, default=None)
    ap.add_argument("--gen", type=int, default=None)
    ap.add_argument("--check", action="store_true", help="preflight and exit")
    ap.add_argument("--summarize", action="store_true",
                    help="aggregate existing runs and exit")
    args = ap.parse_args()

    if args.check:
        preflight(args.solver); raise SystemExit
    if args.summarize:
        summarize(); raise SystemExit

    solver = pick_solver(args.solver)
    preset = PRESETS[args.quality]
    n_seeds = args.seeds or preset["seeds"]
    norms = ["l1", "l2"] if args.norm == "both" else [args.norm]

    overrides = (args.n_train, args.points, args.max_iter, args.pop, args.gen)
    for norm in norms:
        for scale in args.scales:
            b = list(preset["scales"][scale])
            for idx, val in enumerate(overrides):
                if val is not None:
                    b[idx] = val
            budget = tuple(b)
            print(f"\n=== scale {scale} | norm {norm} | N={budget[0]} | "
                  f"pts={budget[1]} moi={budget[2]} nsga={budget[3]}x{budget[4]} | "
                  f"{n_seeds} reps | solver={solver or 'default'} ===")
            for seed in range(n_seeds):
                run_once(scale, budget, norm, args.epsilon, solver, seed)
    summarize()
