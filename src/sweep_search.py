"""Analisi delle performance del Simulated Annealing al variare di num_sweeps."""

import json
import logging
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dwave.samplers import SimulatedAnnealingSampler

from src.bqm_builder import build_bqm
from src.parser import CachingInstance
from src.tuning import evaluate_with_optimality, solve_exact_ilp

DEFAULT_SWEEP_VALUES = [1000, 5000, 10000, 20000]


def plot_sweep_curve(
    results: list[dict],
    output_img: Path,
    gamma_cap: float,
    gamma_rep: float,
) -> None:
    """Genera la curva delle performance (Score, Ottimalità, Ammissibilità) al variare di num_sweeps."""
    sweeps = [r["num_sweeps"] for r in results]
    scores = [r["combined_score"] * 100.0 for r in results]
    opts = [r["optimality_quality"] * 100.0 for r in results]
    feasibles = [r["feasible_ratio"] * 100.0 for r in results]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    ax.plot(sweeps, scores, marker="o", linewidth=2.5, label="Combined Score (%)", color="#1f77b4")
    ax.plot(sweeps, opts, marker="s", linestyle="--", linewidth=1.8, label="Optimality Quality (%)", color="#2ca02c")
    ax.plot(sweeps, feasibles, marker="^", linestyle=":", linewidth=1.8, label="Feasible Ratio (%)", color="#ff7f0e")

    for x, y in zip(sweeps, scores):
        ax.annotate(
            f"{y:.1f}%",
            (x, y),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=9,
            fontweight="bold",
            color="#1f77b4",
        )

    ax.set_title(
        f"Performance vs num_sweeps (gamma_cap={gamma_cap:g}, gamma_rep={gamma_rep:g})",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("num_sweeps", fontsize=11)
    ax.set_ylabel("Performance (%)", fontsize=11)
    ax.set_xticks(sweeps)
    ax.set_ylim(-5, 108)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right")

    output_img.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_img, dpi=300)
    plt.close(fig)


def run_sweep_search(
    instances: list[CachingInstance],
    sweep_values: list[int] | None = None,
    gamma_cap: float = 10.0,
    gamma_rep: float = 10.0,
    num_reads: int = 100,
    output_path: Path = Path("sweep-result/sweep_search_metrics.json"),
    exp_name: str | None = None,
) -> list[dict]:
    """Esegue il benchmark al variare di num_sweeps, salva i risultati in sweep-result/ e traccia la curva."""
    sweeps_list = (
        sorted(set(DEFAULT_SWEEP_VALUES))
        if not sweep_values
        else sorted(set(sweep_values))
    )

    sampler = SimulatedAnnealingSampler()
    exact_opts = {inst.id: solve_exact_ilp(inst) for inst in instances}
    bqms = {
        inst.id: build_bqm(inst, gamma_cap=gamma_cap, gamma_rep=gamma_rep)[0]
        for inst in instances
    }
    logging.info(f"Ottimi esatti e BQM pronti per {len(instances)} istanze.")

    results: list[dict] = []

    for sweeps in sweeps_list:
        scores, opt_qualities, feasibles, infeasibles, best_objs, times = (
            [], [], [], [], [], []
        )

        for inst in instances:
            t0 = time.perf_counter()
            sampleset = sampler.sample(
                bqms[inst.id], num_reads=num_reads, num_sweeps=sweeps
            )
            times.append(time.perf_counter() - t0)

            stats = evaluate_with_optimality(inst, sampleset, exact_opts[inst.id])
            scores.append(stats["combined_score"])
            opt_qualities.append(stats["optimality_quality"])
            feasibles.append(stats["feasible_ratio"])
            infeasibles.append(stats["infeasible_ratio"])
            if stats["best_feasible_obj"] is not None:
                best_objs.append(stats["best_feasible_obj"])

        avg_score = float(np.mean(scores))
        avg_opt = float(np.mean(opt_qualities))
        avg_feasible = float(np.mean(feasibles))
        avg_infeasible = float(np.mean(infeasibles))
        avg_time = float(np.mean(times))
        avg_best_obj = float(np.mean(best_objs)) if best_objs else None

        results.append(
            {
                "num_sweeps": sweeps,
                "gamma_cap": gamma_cap,
                "gamma_rep": gamma_rep,
                "combined_score": avg_score,
                "optimality_quality": avg_opt,
                "feasible_ratio": avg_feasible,
                "infeasible_ratio": avg_infeasible,
                "avg_sa_time_s": avg_time,
                "avg_best_feasible_obj": avg_best_obj,
            }
        )

        logging.info(
            f"num_sweeps={sweeps:6d} | Score: {avg_score:5.1%} | "
            f"Opt: {avg_opt:5.1%} | Feasible: {avg_feasible:5.1%} | Time: {avg_time:.3f}s"
        )

    if exp_name:
        exp_dir = Path("sweep-result") / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        json_path = exp_dir / "sweep_metrics.json"
        img_path = exp_dir / "sweep_curve.png"
    else:
        sweep_dir = Path("sweep-result")
        json_path = (
            output_path
            if output_path.parts and output_path.parts[0] == "sweep-result"
            else sweep_dir / output_path.name
        )
        img_path = json_path.with_suffix(".png")

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    plot_sweep_curve(results, img_path, gamma_cap=gamma_cap, gamma_rep=gamma_rep)
    logging.info(f"Risultati e curva sweep salvati in {json_path} e {img_path}")

    return results
