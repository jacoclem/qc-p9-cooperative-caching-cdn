"""Binary Quadratic Model (BQM) builder for Cooperative Caching in CDN."""

from typing import Dict, Tuple

from dimod import BinaryQuadraticModel

from src.parser import CachingInstance


def _add_decision_variables(bqm: BinaryQuadraticModel, m: int, n: int) -> None:
    """Add all binary decision variables x_{h}_{j} to the BQM."""
    for h in range(m):
        for j in range(n):
            bqm.add_variable(f"x_{h}_{j}")


def build_bqm(
    instance: CachingInstance,
    gamma_cap: float = 10.0,
    gamma_rep: float = 10.0,
) -> Tuple[BinaryQuadraticModel, Dict[str, int]]:
    """Build the BQM and return both the model and variable/interaction statistics."""
    m = instance.m
    n = instance.n

    bqm = BinaryQuadraticModel(vartype="BINARY")
    _add_decision_variables(bqm, m, n)

    # Objective function: linear benefit terms (negated for minimization)
    for h in range(m):
        for j in range(n):
            bqm.set_linear(f"x_{h}_{j}", -instance.b[h][j])

    # Objective function: quadratic redundancy penalties for neighboring servers
    for h, k in instance.N:
        for j in range(n):
            d_val = instance.get_delta(h, k, j)
            if d_val != 0:
                bqm.add_quadratic(f"x_{h}_{j}", f"x_{k}_{j}", d_val)

    # Constraint 1: edge server memory capacity (sum_j sigma_j * x_{h,j} <= K_h)
    for h in range(m):
        inequality = [(f"x_{h}_{j}", instance.sigma[j]) for j in range(n)]
        bqm.add_linear_inequality_constraint(
            inequality,
            lagrange_multiplier=gamma_cap,
            lb=0,
            ub=instance.K[h],
            label=f"capacity_{h}",
        )

    # Constraint 2: minimum replication per content (r_j <= sum_h x_{h,j} <= m)
    for j in range(n):
        inequality = [(f"x_{h}_{j}", 1) for h in range(m)]
        bqm.add_linear_inequality_constraint(
            inequality,
            lagrange_multiplier=gamma_rep,
            lb=instance.r[j],
            ub=m,
            label=f"replication_{j}",
        )

    num_logical = instance.num_variables
    num_total = len(bqm.variables)
    bqm_stats = {
        "num_logical_vars": num_logical,
        "num_slack_vars": num_total - num_logical,
        "num_total_vars": num_total,
        "num_quadratic_interactions": len(bqm.quadratic),
    }

    return bqm, bqm_stats