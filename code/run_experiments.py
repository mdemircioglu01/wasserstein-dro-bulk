"""
Driver that reproduces the headline numerical experiments of the paper:

  1. cost--emission Pareto front (exact epsilon-constraint vs. NSGA-II);
  2. price of robustness: how the front shifts with the Wasserstein radius eps;
  3. exact-vs-metaheuristic comparison across instance scales.

Run:  python run_experiments.py
Outputs CSVs and PNG figures next to this file.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from dro_bulk.config import InstanceConfig, DROConfig
from dro_bulk.data_generation import generate_instance
from dro_bulk.multiobjective import epsilon_constraint_front, nsga2_front

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)


def _save_front(name, front):
    keys = ["cost_total", "emis_total", "cost_transport", "cost_recourse",
            "emis_transport", "emis_recourse"]
    with open(OUT / f"{name}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(keys)
        for p in front:
            w.writerow([round(p.get(k, float("nan")), 4) for k in keys])


def experiment_pareto(scale="S", norm="l1", epsilon=0.05, seed=0):
    inst = generate_instance(InstanceConfig(scale=scale, seed=seed))
    dro = DROConfig(norm=norm, epsilon=epsilon, support="box")

    exact = epsilon_constraint_front(inst, dro, n_points=11)
    _save_front(f"pareto_exact_{scale}_{norm}_eps{epsilon}", exact)
    print(f"[exact] {len(exact)} Pareto points "
          f"(cost {exact[0]['cost_total']:.0f}..{exact[-1]['cost_total']:.0f})")

    try:
        nsga = nsga2_front(inst, dro, pop_size=40, n_gen=40, seed=seed)
        np.savetxt(OUT / f"pareto_nsga_{scale}_{norm}_eps{epsilon}.csv",
                   nsga, delimiter=",", header="cost_total,emis_total")
        print(f"[nsga2] {len(nsga)} non-dominated solutions")
    except Exception as exc:                       # pymoo optional at first run
        nsga = None
        print(f"[nsga2] skipped ({exc})")

    _plot(scale, norm, epsilon, exact, nsga)
    return exact, nsga


def experiment_price_of_robustness(scale="S", norm="l1",
                                   eps_grid=(0.0, 0.02, 0.05, 0.1, 0.2), seed=0):
    inst = generate_instance(InstanceConfig(scale=scale, seed=seed))
    fronts = {}
    for eps in eps_grid:
        dro = DROConfig(norm=norm, epsilon=eps, support="box")
        fronts[eps] = epsilon_constraint_front(inst, dro, n_points=9)
        _save_front(f"por_{scale}_{norm}_eps{eps}", fronts[eps])
        print(f"eps={eps:<5}: {len(fronts[eps])} points")
    _plot_por(scale, norm, fronts)
    return fronts


def _plot(scale, norm, eps, exact, nsga):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    c = [p["cost_total"] for p in exact]
    e = [p["emis_total"] for p in exact]
    ax.plot(c, e, "o-", label="epsilon-constraint (exact)")
    if nsga is not None:
        ax.scatter(nsga[:, 0], nsga[:, 1], c="orange", s=18, label="NSGA-II")
    ax.set_xlabel("Total cost"); ax.set_ylabel("Total CO2 emission")
    ax.set_title(f"Pareto front  (scale={scale}, {norm}, eps={eps})")
    ax.legend(); fig.tight_layout()
    fig.savefig(OUT / f"pareto_{scale}_{norm}_eps{eps}.png", dpi=130)
    plt.close(fig)


def _plot_por(scale, norm, fronts):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for eps, front in fronts.items():
        c = [p["cost_total"] for p in front]
        e = [p["emis_total"] for p in front]
        ax.plot(c, e, "o-", ms=4, label=f"eps={eps}")
    ax.set_xlabel("Total cost"); ax.set_ylabel("Total CO2 emission")
    ax.set_title(f"Price of robustness  (scale={scale}, {norm})")
    ax.legend(); fig.tight_layout()
    fig.savefig(OUT / f"price_of_robustness_{scale}_{norm}.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="S", choices=["S", "M", "L"])
    ap.add_argument("--norm", default="l1", choices=["l1", "l2"])
    ap.add_argument("--epsilon", type=float, default=0.05)
    ap.add_argument("--experiment", default="pareto",
                    choices=["pareto", "robustness"])
    args = ap.parse_args()

    if args.experiment == "pareto":
        experiment_pareto(args.scale, args.norm, args.epsilon)
    else:
        experiment_price_of_robustness(args.scale, args.norm)
    print(f"\nResults written to {OUT}")
