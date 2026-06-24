# Wasserstein DRO — Multi-Product Bulk Transportation (code skeleton)

Reference implementation for the paper *A Wasserstein-based Distributionally
Robust Two-Stage Multi-Objective Optimization Model for Multi-Product Bulk
Transportation under Demand Uncertainty* (`../formulation.tex`).

## Layout

| File | Purpose |
|------|---------|
| `dro_bulk/config.py` | instance presets (S/M/L) and DRO settings |
| `dro_bulk/data_generation.py` | synthetic instances: geography, costs, emissions, supply/capacity, correlated demand samples |
| `dro_bulk/recourse.py` | recourse evaluation, Lipschitz modulus, fast worst-case evaluators |
| `dro_bulk/dro_model.py` | tractable reformulation in cvxpy: exact compact LP (`l1`), SOCP (`l2`) with row generation |
| `dro_bulk/separation.py` | exact l2-box separation oracle (MISOCP / fixed-point) for row generation |
| `dro_bulk/multiobjective.py` | exact epsilon-constraint front + NSGA-II |
| `dro_bulk/fixed_charge.py` | NP-hard fixed-charge variant: exact MILP/MISOCP + tailored multi-objective ALNS |
| `run_experiments.py` | driver: synthetic Pareto fronts, price of robustness |
| `run_scalability.py` | scalability study (S/M/L, time + hypervolume + IGD), CSV/LaTeX tables |
| `comtrade_fetch.py` | pull the real Turkey cement-export demand panel from UN Comtrade |
| `case_loader.py` | build the real-data `Instance` (real demand + calibrated network) |
| `run_case.py` | run the Turkey case study (Pareto, price of robustness, fixed-charge) |
| `make_paper_figures.py` | regenerate the paper figures (`figures/pareto_fig.pdf`, `figures/case_pareto.pdf`) |

## Install & run

```bash
pip install -r requirements.txt
python -m dro_bulk.data_generation            # sanity check the generator
python run_experiments.py --scale S --norm l1 --epsilon 0.05
python run_experiments.py --experiment robustness --scale S --norm l1
```

## Reproducing the paper figures

```bash
pip install -U matplotlib          # needs a NumPy-2-compatible build
python make_paper_figures.py       # writes figures/pareto_fig.{pdf,png} and figures/case_pareto.{pdf,png}
```

`make_paper_figures.py` regenerates both figures from scratch: the synthetic
exact-vs-NSGA-II / price-of-robustness panel and the real Turkey case-study panel.

## Real-data case study (Turkey cement exports)

```bash
python comtrade_fetch.py           # pull the real demand panel -> case_data/demand_panel_long.csv
                                   # (key-free UN Comtrade preview; set API_KEY for the full endpoint)
python run_case.py                 # Pareto, price of robustness, fixed-charge MILP vs ALNS -> results/case/
```

The demand panel is real (UN Comtrade, 2005–2023 cement-family exports by partner);
the network parameters are calibrated from public sources in `case_loader.CaseCalib`
(Baltic freight, IMO/GLEC emission factors, port tariffs) and can be refined.

## Reproducing the journal-quality scalability table (Gurobi)

`run_scalability.py` produces Table `tab:scalability` (solve time + hypervolume
+ IGD across S/M/L, mean±std over replications) and writes a ready-to-`\input`
LaTeX file.

```bash
pip install -r requirements.txt          # + gurobipy and a Gurobi license
python run_scalability.py --check        # confirm GUROBI is visible to cvxpy
python run_scalability.py --quality journal --norm both --solver GUROBI
```

What the journal preset does:

* full sample size `N = 100`, 10 random replications per class;
* 17–21 exact Pareto points per front, up to 25–30 row-generation iterations;
* NSGA-II with population 120–200 and 400–600 generations;
* `l2` uses the **exact, certified MISOCP separation** automatically once a
  MI-conic solver (Gurobi) is present (`l2_separation="auto"`).

Outputs land in `results/`:

* `scalability_runs.jsonl` — every replication (resumable; safe to re-run);
* `scalability_table.csv` — aggregated mean/std;
* `scalability_table.tex` — booktabs table for the paper.

Notes:

* The run is long (hours for `L` with `--norm both`); use `--seeds 3` or
  `--scales S M` to shorten, or `--quality smoke` for a seconds-long check on
  the open-source solver.
* Without `--solver`, the script auto-selects GUROBI if installed, else the
  cvxpy default; `--check` reports exactly what will be used.
* Set Gurobi up via `pip install gurobipy` and a license
  (`grbgetkey` for academic/WLS licenses).

## Modelling notes

* **Index map.** A (destination *j*, product *k*) pair maps to the flat
  coordinate `d = j*K + k`; the demand vector has dimension `D = J*K`.
* **Two objectives.** Cost `f1` and CO2 `f2`; each is robustified with its own
  worst-case expectation over the same type-1 Wasserstein ball.
* **Norm = cone.** `norm="l1"` gives the dual `l_inf` constraint → **LP**;
  `norm="l2"` gives the dual `l2` constraint → **SOCP**. cvxpy picks the cone.
* **Row generation.** The recourse pieces are box-dual vertices (2^D of them),
  generated on the fly per sample. The sign pattern is shared by both
  objectives, so the epsilon-constraint model reuses one piece set.
  * `l1` (LP): solved as an **exact compact separable LP** (`l1_compact=True`,
    default) — one solve, no row generation, two pieces per coordinate. Verified
    against an independent closed-form worst-case evaluator and ~1-2 orders of
    magnitude faster than row generation. Set `l1_compact=False` only to exercise
    the (slower, and for large radii optimistic) active-piece row-generation path.
  * `l2` (SOCP): the active-at-sample piece is *optimistic*, so a genuine
    **separation oracle** (`dro_bulk/separation.py`) finds the worst-case
    realisation `xi*` and adds the subgradient there. Two backends:
    `misocp` (exact & certified, needs a MI-conic solver such as Gurobi),
    and `fixedpoint` (solver-agnostic multi-start, used when none is
    installed). The loop terminates with `certified=True` once no sample
    has a violation above `tol`. Set via `DROConfig.l2_separation`
    (`"auto"` by default); `"active"` restores the fast optimistic heuristic.
* **Support.** `support="box"` is the meaningful case (robustness changes the
  plan). `support="full"` reduces the worst case to `SAA + eps * Lipschitz`, a
  constant shift, and is kept only as a fast baseline.

## Where to extend (research TODOs)

* fixed-charge link activation (binary `z_ij` → mixed-integer conic);
* cross-validated / `N^{-1/2}` schedule for the radius `epsilon`;
* mean–CVaR objective variants; alternative ground metrics;
* scalability study sweeping S/M/L and reporting solve times + hypervolume.
