"""Solution validator and objective evaluator for Cooperative Caching in CDN."""

from collections.abc import Mapping
from typing import Any, Dict, Union

import numpy as np

from src.parser import CachingInstance


def extract_decision_matrix(
    solution: Union[Mapping[str, Any], np.ndarray, list],
    m: int,
    n: int,
) -> np.ndarray:
    """Extract the (m x n) binary decision matrix from a mapping or array."""
    if isinstance(solution, Mapping):
        x = np.zeros((m, n), dtype=int)
        for h in range(m):
            for j in range(n):
                val = solution.get(f"x_{h}_{j}", 0)
                x[h, j] = int(round(val))
        return x

    x = np.array(solution, dtype=int)
    if x.shape != (m, n):
        raise ValueError(f"Expected matrix shape {(m, n)}, got {x.shape}")
    return x


def validate_solution(
    instance: CachingInstance,
    solution: Union[Mapping[str, Any], np.ndarray, list],
) -> Dict[str, Any]:
    """Check constraint feasibility and compute the true objective value."""
    m = instance.m
    n = instance.n
    sigma = np.array(instance.sigma, dtype=int)
    capacities = np.array(instance.K, dtype=int)
    benefits = np.array(instance.b, dtype=float)
    min_replicas = np.array(instance.r, dtype=int)

    x = extract_decision_matrix(solution, m, n)

    if not np.all(np.isin(x, [0, 1])):
        raise ValueError("Solution contains non-binary values.")

    # Capacity constraint check: sum_j sigma_j * x_{h,j} <= K_h
    used_capacity = x @ sigma
    cap_excess = np.maximum(0, used_capacity - capacities)
    cap_violations = int(np.sum(cap_excess > 0))

    # Minimum replication constraint check: sum_h x_{h,j} >= r_j
    copies_per_content = np.sum(x, axis=0)
    rep_deficit = np.maximum(0, min_replicas - copies_per_content)
    rep_violations = int(np.sum(rep_deficit > 0))

    is_valid = (cap_violations == 0) and (rep_violations == 0)

    # Objective computation: -sum(b_{h,j} * x_{h,j}) + sum(delta_{h,k,j} * x_{h,j} * x_{k,j})
    benefit_total = float(np.sum(benefits * x))
    redundancy_total = 0.0
    for h, k in instance.N:
        for j in range(n):
            if x[h, j] == 1 and x[k, j] == 1:
                redundancy_total += float(instance.get_delta(h, k, j))

    objective_value = -benefit_total + redundancy_total

    return {
        "is_valid": is_valid,
        "objective_value": objective_value,
        "benefit_total": benefit_total,
        "redundancy_total": redundancy_total,
        "capacity_violations": cap_violations,
        "replication_violations": rep_violations,
        "details": {
            "max_capacity_excess": int(np.max(cap_excess)) if m > 0 else 0,
            "max_replication_deficit": int(np.max(rep_deficit)) if n > 0 else 0,
        },
    }