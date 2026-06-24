"""
Synthetic instance generator for the multi-product bulk-transportation problem.

Everything is reproducible from ``InstanceConfig.seed``.  The generator returns
an :class:`Instance` dataclass holding all sets, parameters and demand samples
needed by the model.  The (destination, product) pair (j, k) is stored on the
flat coordinate ``d = j * K + k`` of the demand vector of dimension ``D = J*K``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .config import InstanceConfig, SCALE_PRESETS


@dataclass
class Instance:
    # sizes
    n_sources: int          # I
    n_destinations: int     # J
    n_products: int         # K

    # geography
    src_xy: np.ndarray      # (I, 2)
    dst_xy: np.ndarray      # (J, 2)
    distance: np.ndarray    # (I, J)

    # per-unit product mass
    rho: np.ndarray         # (K,)

    # transportation cost / emission  (I, J, K)
    cost: np.ndarray
    emission: np.ndarray

    # supply / capacity
    supply: np.ndarray      # (I, K)
    link_cap: np.ndarray    # (I, J)
    fixed_cost: np.ndarray  # (I, J) fixed cost of activating a link
    fixed_emis: np.ndarray  # (I, J) fixed emission of activating a link

    # recourse coefficients per (J, K)
    short_cost: np.ndarray  # b_{jk}
    surp_cost: np.ndarray   # h_{jk}
    short_emis: np.ndarray  # beta_{jk}
    surp_emis: np.ndarray   # eta_{jk}

    # demand description (J, K)
    demand_mean: np.ndarray
    demand_std: np.ndarray
    support_lb: np.ndarray
    support_ub: np.ndarray

    # samples, stored flat as (n, D)
    train_samples: np.ndarray
    test_samples: np.ndarray

    cfg: InstanceConfig

    # ------------------------------------------------------------------ #
    @property
    def D(self) -> int:
        return self.n_destinations * self.n_products

    def flat(self, a_jk: np.ndarray) -> np.ndarray:
        """Flatten a (J, K) array to length D = J*K (row-major: d = j*K + k)."""
        return np.asarray(a_jk).reshape(-1)

    def unflat(self, a_d: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`flat`."""
        return np.asarray(a_d).reshape(self.n_destinations, self.n_products)


# --------------------------------------------------------------------------- #
def _correlated_link_emission(rng, cost_rate, ef_low, ef_high, target_corr):
    """Draw an emission factor per link negatively correlated with the cost rate.

    We build a latent variable that is a convex combination of the (negated)
    standardised cost rate and fresh noise, then map it to [ef_low, ef_high]
    through its empirical rank (a simple Gaussian-copula-style construction).
    """
    cr = (cost_rate - cost_rate.mean()) / (cost_rate.std() + 1e-12)
    rho = np.clip(-target_corr, -0.99, 0.99)        # negate: high cost -> low ef
    noise = rng.standard_normal(cost_rate.shape)
    latent = rho * cr + np.sqrt(1.0 - rho ** 2) * noise
    ranks = stats.rankdata(latent).reshape(cost_rate.shape) / (cost_rate.size + 1)
    return ef_low + (ef_high - ef_low) * ranks


def _demand_covariance(std_jk, product_corr, J, K):
    """Block covariance: products correlated within a destination, independent
    across destinations."""
    D = J * K
    corr = np.eye(D)
    for j in range(J):
        for k1 in range(K):
            for k2 in range(K):
                if k1 != k2:
                    corr[j * K + k1, j * K + k2] = product_corr
    std = std_jk.reshape(-1)
    return np.outer(std, std) * corr


def _sample_demand(rng, mean_d, cov, lb_d, ub_d, n, dist):
    """Draw ``n`` correlated, non-negative, bounded demand vectors (n, D)."""
    D = mean_d.size
    # correlated standard normals via Cholesky of the correlation matrix
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    L = np.linalg.cholesky(corr + 1e-9 * np.eye(D))
    z = rng.standard_normal((n, D)) @ L.T

    if dist == "normal":
        x = mean_d + std * z
    elif dist == "lognormal":
        # match mean / std of the target marginal with a log-normal law
        sigma2 = np.log1p((std / np.maximum(mean_d, 1e-9)) ** 2)
        mu = np.log(np.maximum(mean_d, 1e-9)) - 0.5 * sigma2
        x = np.exp(mu + np.sqrt(sigma2) * z)
    else:
        raise ValueError(f"unknown demand_dist: {dist}")

    return np.clip(x, lb_d, ub_d)


# --------------------------------------------------------------------------- #
def generate_instance(cfg: InstanceConfig) -> Instance:
    """Build a fully specified, reproducible problem instance."""
    preset = SCALE_PRESETS[cfg.scale]
    I, J, K = preset.n_sources, preset.n_destinations, preset.n_products
    rng = np.random.default_rng(cfg.seed)

    # --- geography & distances ---
    src_xy = rng.uniform(0, cfg.grid_km, size=(I, 2))
    dst_xy = rng.uniform(0, cfg.grid_km, size=(J, 2))
    distance = np.linalg.norm(src_xy[:, None, :] - dst_xy[None, :, :], axis=2)  # (I,J)

    # --- product mass ---
    rho = rng.uniform(cfg.rho_min, cfg.rho_max, size=K)

    # --- emission factor per link, negatively correlated with the cost rate ---
    cost_rate = cfg.theta0 + cfg.theta1 * distance            # (I, J), per ton
    ef = _correlated_link_emission(rng, cost_rate, cfg.ef_low,
                                   cfg.ef_high, cfg.ef_cost_correlation)

    # --- unit cost and unit emission (I, J, K) ---
    u = rng.uniform(0, cfg.cost_noise, size=(I, J, K))
    cost = cost_rate[:, :, None] * rho[None, None, :] * (1.0 + u)
    emission = ef[:, :, None] * distance[:, :, None] * rho[None, None, :]

    # --- demand description (J, K) ---
    demand_mean = rng.uniform(cfg.demand_mean_min, cfg.demand_mean_max, size=(J, K))
    demand_std = cfg.coeff_variation * demand_mean
    support_lb = np.zeros((J, K))
    support_ub = demand_mean + cfg.support_sigma * demand_std

    # --- recourse coefficients (J, K) ---
    avg_cost = cost.mean(axis=0)        # (J, K) average over sources
    avg_emis = emission.mean(axis=0)
    short_cost = cfg.mu_shortage_cost * avg_cost
    surp_cost = cfg.mu_surplus_cost * avg_cost
    short_emis = cfg.mu_shortage_emis * avg_emis
    surp_emis = cfg.mu_surplus_emis * avg_emis

    # --- supply & capacity ---
    exp_demand_k = demand_mean.sum(axis=0)              # (K,) total expected per product
    total_supply_k = cfg.supply_tightness * exp_demand_k
    split = rng.dirichlet(np.ones(I), size=K).T         # (I, K) random shares
    supply = split * total_supply_k[None, :]            # (I, K)

    exp_tonnage = float((demand_mean * rho[None, :]).sum())
    link_cap = cfg.link_capacity_factor * exp_tonnage * \
        rng.uniform(0.5, 1.5, size=(I, J))

    # fixed-charge coefficients: a fraction of the cost/emission of filling the
    # link to capacity once (longer/larger links are more expensive to open)
    fixed_cost = cfg.fixed_cost_factor * cost_rate * link_cap          # (I, J)
    fixed_emis = cfg.fixed_emis_factor * (ef * distance) * link_cap    # (I, J)

    # --- samples ---
    mean_d = demand_mean.reshape(-1)
    lb_d, ub_d = support_lb.reshape(-1), support_ub.reshape(-1)
    cov = _demand_covariance(demand_std, cfg.product_correlation, J, K)
    train = _sample_demand(rng, mean_d, cov, lb_d, ub_d, cfg.n_train, cfg.demand_dist)
    test = _sample_demand(rng, mean_d, cov, lb_d, ub_d, cfg.n_test, cfg.demand_dist)

    return Instance(
        n_sources=I, n_destinations=J, n_products=K,
        src_xy=src_xy, dst_xy=dst_xy, distance=distance,
        rho=rho, cost=cost, emission=emission,
        supply=supply, link_cap=link_cap,
        fixed_cost=fixed_cost, fixed_emis=fixed_emis,
        short_cost=short_cost, surp_cost=surp_cost,
        short_emis=short_emis, surp_emis=surp_emis,
        demand_mean=demand_mean, demand_std=demand_std,
        support_lb=support_lb, support_ub=support_ub,
        train_samples=train, test_samples=test,
        cfg=cfg,
    )


if __name__ == "__main__":
    inst = generate_instance(InstanceConfig(scale="S", seed=1))
    print(f"Instance S: I={inst.n_sources} J={inst.n_destinations} "
          f"K={inst.n_products} D={inst.D}")
    print("train samples:", inst.train_samples.shape,
          "test samples:", inst.test_samples.shape)
    print("mean demand (J,K):\n", np.round(inst.demand_mean, 1))
