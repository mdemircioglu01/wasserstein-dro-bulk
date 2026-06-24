"""
Build a real-data Instance for the Turkey cement/clinker export case study.

Demand side  : REAL -- read from case_data/demand_panel_long.csv (UN Comtrade,
               produced by comtrade_fetch.py); the historical years are the
               empirical demand samples that the Wasserstein ball is built on.
Network side : CALIBRATED from public sources -- Turkish export-port clusters,
               great-circle sea distances, a voyage-cost freight model anchored
               to Baltic time-charter + bunker prices, IMO/GLEC emission factors,
               and recourse penalties from Comtrade unit values. Every calibrated
               number is collected in CALIB below; refine it with your sources.

The returned object is the same `Instance` used by the synthetic experiments, so
dro_model / multiobjective / fixed_charge all run unchanged on the real instance.

Usage:
    from case_loader import load_case
    inst = load_case(n_destinations=10, products=("clinker","portland_grey","portland_white"))
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from dro_bulk.config import InstanceConfig
from dro_bulk.data_generation import Instance

PANEL = Path(__file__).parent / "case_data" / "demand_panel_long.csv"

# --------------------------------------------------------------------------- #
#  Geography (representative ports; lat, lon).  Public/known coordinates.
# --------------------------------------------------------------------------- #
TR_PORTS = {                                   # Turkish export-port clusters
    "Marmara":      (40.97, 28.70),
    "Aegean":       (38.43, 27.00),
    "Mediterranean":(36.80, 34.60),
    "BlackSea":     (41.30, 36.30),
}
DEST_PORTS = {                                 # destination representative ports
    "USA": (29.75, -95.0), "Israel": (32.80, 35.00), "Romania": (44.17, 28.65),
    "Ghana": (5.63, 0.00), "CotedIvoire": (5.30, -4.00), "Cameroon": (4.05, 9.70),
    "Senegal": (14.67, -17.43), "Guinea": (9.50, -13.70), "Libya": (32.90, 13.20),
    "Syria": (35.50, 35.80), "Spain": (41.40, 2.20), "Egypt": (31.20, 29.90),
    "Haiti": (18.55, -72.34), "DominicanRep": (18.46, -69.90), "Peru": (-12.05, -77.15),
    "Colombia": (10.40, -75.50), "Gambia": (13.45, -16.58), "GuineaBissau": (11.86, -15.60),
    "Mauritania": (18.09, -15.98), "Togo": (6.13, 1.29), "Benin": (6.35, 2.43),
    "Bangladesh": (22.30, 91.80), "SriLanka": (6.95, 79.85), "UK": (51.50, 0.00),
    "Japan": (35.60, 139.70), "Morocco": (33.60, -7.60), "Oman": (23.60, 58.50),
    "Albania": (41.30, 19.45), "Algeria": (36.80, 3.06),
}

# --------------------------------------------------------------------------- #
#  Calibration knobs (public-source anchored; document/refine before submission)
# --------------------------------------------------------------------------- #
@dataclass
class CaseCalib:
    # voyage-cost freight model  (Baltic Handysize TC + bunker; representative)
    tc_daily_usd: float = 12000.0      # time-charter $/day (Baltic Handysize avg)
    speed_kn: float = 12.0             # service speed (knots)
    port_days: float = 4.0            # load+discharge days per voyage
    vessel_dwt: float = 35000.0       # representative handysize cargo (t)
    bunker_usd_per_t: float = 550.0    # VLSFO $/ton (Ship & Bunker, representative)
    bunker_t_per_day: float = 22.0     # main-engine consumption at sea (t/day)
    handling_usd_per_t: float = 6.0    # port loading/handling $/ton (port tariff)
    prod_handling: dict = field(default_factory=lambda: {       # per-product factor
        "clinker": 1.0, "portland_grey": 1.05, "portland_white": 1.20,
        "gypsum": 0.8})
    # emissions (IMO/GLEC handysize bulk carrier)
    ef_g_co2_per_t_nm: float = 8.0     # g CO2 per ton-nautical-mile
    # supply / capacity
    supply_tightness: float = 1.15     # total supply = tightness x peak demand
    link_cap_factor: float = 0.6       # per port-lane cap as share of total tonnage
    # recourse penalties as multiples of the product's FOB unit value ($/ton)
    mu_shortage: float = 3.0
    mu_surplus: float = 0.5
    mu_shortage_emis: float = 2.5      # emergency-supply emission vs regular
    mu_surplus_emis: float = 0.8
    # demand model
    recent_window: int = 12            # years used as the empirical sample set
    support_sigma: float = 4.0
    fixed_cost_factor: float = 0.25
    fixed_emis_factor: float = 0.25


def _haversine_nm(a, b) -> float:
    R = 3440.065                                   # earth radius in nautical miles
    la1, lo1, la2, lo2 = map(math.radians, [a[0], a[1], b[0], b[1]])
    h = (math.sin((la2 - la1) / 2) ** 2
         + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(h))


def _freight_per_ton(dist_nm: float, c: CaseCalib) -> float:
    sea_days = dist_nm / (24.0 * c.speed_kn)
    voyage_days = sea_days + c.port_days
    charter = c.tc_daily_usd * voyage_days / c.vessel_dwt
    bunker = c.bunker_usd_per_t * c.bunker_t_per_day * sea_days / c.vessel_dwt
    return charter + bunker                          # $/ton ocean freight


# --------------------------------------------------------------------------- #
def _read_panel():
    rows = list(csv.DictReader(open(PANEL)))
    data = {}        # (year, country, product) -> (tons, value)
    for r in rows:
        try:
            tons = float(r["tons"]); val = float(r["value_usd"])
        except (ValueError, TypeError):
            continue
        data[(int(r["year"]), r["partner_name"], r["product"])] = (tons, val)
    years = sorted({k[0] for k in data})
    return data, years


def load_case(n_destinations: int = 10,
              products=("clinker", "portland_grey", "portland_white"),
              calib: CaseCalib | None = None, seed: int = 0) -> Instance:
    c = calib or CaseCalib()
    rng = np.random.default_rng(seed)
    data, years = _read_panel()
    years = years[-c.recent_window:]               # recent window = sample set
    K = len(products)

    # pick top destinations by mean total tonnage, restricted to known coords
    tot = {}
    for ctry in DEST_PORTS:
        s = [sum(data.get((y, ctry, p), (0, 0))[0] for p in products) for y in years]
        if sum(s) > 0:
            tot[ctry] = float(np.mean(s))
    dests = sorted(tot, key=lambda d: -tot[d])[:n_destinations]
    J, I = len(dests), len(TR_PORTS)
    ports = list(TR_PORTS)

    # demand panel: samples (N, J, K) in tonnes
    N = len(years)
    samp = np.zeros((N, J, K))
    unit_val = np.zeros((J, K))                    # FOB $/ton per (dest, product)
    for jj, d in enumerate(dests):
        for kk, p in enumerate(products):
            ts, vs = [], []
            for n, y in enumerate(years):
                t, v = data.get((y, d, p), (0.0, 0.0))
                samp[n, jj, kk] = t
                if t > 0:
                    ts.append(t); vs.append(v)
            unit_val[jj, kk] = (sum(vs) / sum(ts)) if ts else 50.0
    demand_mean = samp.mean(axis=0)                # (J, K)
    demand_std = samp.std(axis=0)
    support_lb = np.zeros((J, K))
    support_ub = demand_mean + c.support_sigma * np.maximum(demand_std, 1.0)

    # distances (nm), costs ($/ton), emissions (kg CO2/ton)
    dist = np.array([[_haversine_nm(TR_PORTS[ports[i]], DEST_PORTS[dests[j]])
                      for j in range(J)] for i in range(I)])
    rho = np.ones(K)                               # demand already in tonnes
    cost = np.zeros((I, J, K)); emission = np.zeros((I, J, K))
    for i in range(I):
        for j in range(J):
            base = c.handling_usd_per_t + _freight_per_ton(dist[i, j], c)
            em = c.ef_g_co2_per_t_nm * dist[i, j] / 1000.0      # kg CO2 per ton
            for k, p in enumerate(products):
                cost[i, j, k] = base * c.prod_handling.get(p, 1.0)
                emission[i, j, k] = em
    cost_rate_like = cost.mean(axis=(0, 1))        # per-product avg (for emis recourse)

    # supply & capacity
    peak_k = samp.sum(axis=1).max(axis=0)          # (K,) peak total demand per product
    total_k = c.supply_tightness * peak_k
    split = rng.dirichlet(np.ones(I), size=K).T    # (I, K)
    supply = split * total_k[None, :]
    exp_tonnage = float(demand_mean.sum())
    link_cap = c.link_cap_factor * exp_tonnage * rng.uniform(0.7, 1.3, size=(I, J))

    # fixed-charge coefficients
    cost_rate_ij = cost.mean(axis=2)               # (I,J) avg $/ton
    emis_rate_ij = emission.mean(axis=2)
    fixed_cost = c.fixed_cost_factor * cost_rate_ij * link_cap
    fixed_emis = c.fixed_emis_factor * emis_rate_ij * link_cap

    # recourse penalties from FOB unit values
    short_cost = c.mu_shortage * unit_val
    surp_cost = c.mu_surplus * unit_val
    short_emis = c.mu_shortage_emis * cost_rate_like[None, :] * 0 + \
        c.mu_shortage_emis * emission.mean(axis=0)     # (J,K) ~ emission scale
    surp_emis = c.mu_surplus_emis * emission.mean(axis=0)

    # out-of-sample test set: bootstrap resample of the historical years
    idx = rng.integers(0, N, size=2000)
    test = samp[idx].reshape(2000, J * K)

    cfg = InstanceConfig(scale="M", seed=seed, n_train=N)
    return Instance(
        n_sources=I, n_destinations=J, n_products=K,
        src_xy=np.array([TR_PORTS[p] for p in ports]),
        dst_xy=np.array([DEST_PORTS[d] for d in dests]),
        distance=dist, rho=rho, cost=cost, emission=emission,
        supply=supply, link_cap=link_cap, fixed_cost=fixed_cost, fixed_emis=fixed_emis,
        short_cost=short_cost, surp_cost=surp_cost,
        short_emis=short_emis, surp_emis=surp_emis,
        demand_mean=demand_mean, demand_std=demand_std,
        support_lb=support_lb, support_ub=support_ub,
        train_samples=samp.reshape(N, J * K), test_samples=test, cfg=cfg)


if __name__ == "__main__":
    inst = load_case(n_destinations=10)
    print(f"Real-data instance: I={inst.n_sources} ports, J={inst.n_destinations} "
          f"destinations, K={inst.n_products} products, D={inst.D}, N={inst.train_samples.shape[0]} years")
    print("destinations:", list(inst.dst_xy[:, 0].round(1)))
    print("mean demand (kt) per dest (summed over products):",
          (inst.demand_mean.sum(axis=1) / 1000).round(0))
    print("distance range (nm):", int(inst.distance.min()), "-", int(inst.distance.max()))
    print("unit transport cost range ($/t):", round(float(inst.cost.min()), 1),
          "-", round(float