from typing import Any
from src.parser import CachingInstance
import json
import logging
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dwave.samplers import SimulatedAnnealingSampler
from ortools.linear_solver import pywraplp

from src.bqm_builder import build_bqm
from src.parser import CachingInstance
from src.validator import validate_solution

def solve_exact_ilp(instance: CachingInstance) -> float:
    """Calcola il valore ottimo esatto dell'istanza tramite OR-Tools MILP."""
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if solver is None:
        solver = pywraplp.Solver.CreateSolver("CBC")

    m, n = instance.m, instance.n
    x = {(h, j): solver.BoolVar(f"x_{h}_{j}") for h in range(m) for j in range(n)}

    # Vincolo capacità del server
    for h in range(m):
        solver.Add(
            sum(instance.sigma[j] * x[h, j] for j in range(n)) <= instance.K[h]
        )

    # Vincolo ridondanza del contenuto
    for j in range(n):
        solver.Add(sum(x[h, j] for h in range(m)) >= instance.r[j])

    objective = solver.Objective()

    # Termini lineari
    for h in range(m):
        for j in range(n):
            objective.SetCoefficient(x[h, j], float(-instance.b[h][j]))

    # Termini quadratici linearizzati
    for (h, k, j), pen in instance.delta.items():
        if pen != 0:
            y = solver.BoolVar(f"y_{h}_{k}_{j}")
            solver.Add(y >= x[h, j] + x[k, j] - 1)
            objective.SetCoefficient(y, float(pen))

    objective.SetMinimization()

    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise RuntimeError(f"OR-Tools non ha trovato l'ottimo per {instance.id}")

    return float(objective.Value())

def evaluate_with_optimality(
    instance: CachingInstance,
    sampleset: Any,
    exact_opt: float,
) -> dict:
    total_reads = 0
    feasible_reads = 0
    best_feasible_obj = None

    for datum in sampleset.data(["sample", "num_occurrences"]):
        occ = int(datum.num_occurrences)
        total_reads += occ
        val = validate_solution(instance, datum.sample)

        if val["is_valid"]:
            feasible_reads += occ
            obj = float(val["objective_value"])
            if best_feasible_obj is None or obj < best_feasible_obj:
                best_feasible_obj = obj

    feasible_ratio = feasible_reads / total_reads if total_reads > 0 else 0.0
    infeasible_ratio = 1.0 - feasible_ratio

    if best_feasible_obj is not None and abs(exact_opt) > 1e-9:
        gap = max(0.0, (best_feasible_obj - exact_opt) / abs(exact_opt))
        optimality_quality = max(0.0, 1.0 - gap)
    elif best_feasible_obj is not None:
        optimality_quality = 1.0
    else:
        optimality_quality = 0.0

    combined_score = feasible_ratio * optimality_quality

    return {
        "combined_score": combined_score,
        "feasible_ratio": feasible_ratio,
        "infeasible_ratio": infeasible_ratio,
        "best_feasible_obj": best_feasible_obj,
    }

def plot_tuning_heatmap(
    results: list[dict], gamma_values: list[float], output_img: Path
) -> None:
    """Genera la heatmap singola basata sullo score: 100% Verde = 0% infeasible e 100% soluzione ottima."""
    n = len(gamma_values)
    idx_map = {val: i for i, val in enumerate(gamma_values)}
    score_grid = np.full((n, n), np.nan)
    infeasible_grid = np.full((n, n), np.nan)

    for r in results:
        i = idx_map[r["gamma_rep"]]
        j = idx_map[r["gamma_cap"]]
        score_grid[i, j] = r["combined_score"] * 100.0
        infeasible_grid[i, j] = r["infeasible_ratio"] * 100.0

    fig, ax = plt.subplots(figsize=(8, 6.5))
    labels = [f"{g:g}" for g in gamma_values]

    im = ax.imshow(score_grid, cmap="RdYlGn", vmin=0, vmax=100, origin="lower")
    ax.set_title(
        "Tuning Moltiplicatori di Lagrange (100% = 0% Infeasible & Ottimo)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel(r"$\gamma_{\mathrm{cap}}$ (Moltiplicatore Capacità)", fontsize=11)
    ax.set_ylabel(r"$\gamma_{\mathrm{rep}}$ (Moltiplicatore Replicazione)", fontsize=11)
    ax.set_xticks(range(n), labels=labels)
    ax.set_yticks(range(n), labels=labels)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Score Combinato (%)", fontsize=10)

    for i in range(n):
        for j in range(n):
            score = score_grid[i, j]
            infeasible = infeasible_grid[i, j]
            if not np.isnan(score):
                color = "white" if score < 25 or score > 80 else "black"
                ax.text(
                    j,
                    i,
                    f"{score:.0f}%\n({infeasible:.0f}% infeas)",
                    ha="center",
                    va="center",
                    color=color,
                    fontweight="bold",
                    fontsize=9,
                )

    output_img.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_img, dpi=300)
    plt.close(fig)


def run_tuning(
    instances: list[CachingInstance],
    num_reads: int,
    output_path: Path,
    exp_name: str | None = None,
    gamma_values: list[float] | None = None,
    num_sweeps: int = 10000,
) -> None:
    """Orchestra la grid search, salva i risultati in cartella dedicata e genera la heatmap."""

    #utilizzo i gamma_values dati come parametro altrimenti uso dati di default
    if gamma_values is None:
        gamma_values=[5.0, 10.0, 15.0, 20.0, 30.0, 50.0]
    else:
        gamma_values = sorted(set(gamma_values))
    
    #carico il simulated annealing come sampler
    sampler = SimulatedAnnealingSampler()

    #estraggo le soluzioni ottime
    exact_opts = {inst.id: solve_exact_ilp(inst) for inst in instances}
    logging.info(f"Ottimi esatti calcolati per {len(instances)} istanze")

    tuning_results=[]

    #ciclo su tutti e due i valori per provare tutte le combinazioni possibili
    for g_cap in gamma_values:
        for g_rep in gamma_values:
            scores, infeasibles, best_objs = [], [], []

            #ciclo su tutte le istanze a disposizione
            for inst in instances:
                #costruzione del modello matematico
                bqm, _ = build_bqm(inst, gamma_cap=g_cap, gamma_rep=g_rep)
                sampleset = sampler.sample(
                    bqm, num_reads=num_reads, num_sweeps=num_sweeps
                )
                stats = evaluate_with_optimality(inst, sampleset, exact_opts[inst.id])

                #inserisco i risultati ottenuti nella risoluzione in scores e in infeasibles
                scores.append(stats["combined_score"])
                infeasibles.append(stats["infeasible_ratio"])
                if stats["best_feasible_obj"] is not None:
                    best_objs.append(stats["best_feasible_obj"])

            avg_score = float(np.mean(scores))
            avg_infeasible = float(np.mean(infeasibles))
            avg_best_obj = float(np.mean(best_objs)) if best_objs else None

            tuning_results.append({
                "gamma_cap": g_cap,
                "gamma_rep": g_rep,
                "combined_score": avg_score,
                "infeasible_ratio": avg_infeasible,
                "feasible_ratio": 1.0 - avg_infeasible,
                "avg_best_feasible_obj": avg_best_obj,
            })

            logging.info(
                f"gamma_cap={g_cap:4.1f}, gamma_rep={g_rep:4.1f} | "
                f"Score: {avg_score:5.1%} | Infeasible: {avg_infeasible:5.1%}"
            )

    if exp_name:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
        exp_dir = Path("tuning") / f"{timestamp}_{exp_name}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        json_path = exp_dir / "metrics.json"
        img_path = exp_dir / "tuning_heatmap.png"
        readme_path = exp_dir / "README.md"
    else:
        tuning_dir = Path("tuning")
        json_path = (
            output_path
            if output_path.parts and output_path.parts[0] == "tuning"
            else tuning_dir / output_path.name
        )
        img_path = json_path.with_suffix(".png")
        readme_path = None

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(tuning_results, f, indent=4)

    plot_tuning_heatmap(tuning_results, gamma_values, img_path)

    # Miglior coppia: score più alto, poi feasible_ratio, poi gamma più piccoli
    best = max(
        tuning_results,
        key=lambda r: (
            round(r["combined_score"], 2),
            round(r["feasible_ratio"], 2),
            -(r["gamma_cap"] + r["gamma_rep"]),
        ),
    )

    if readme_path:
        readme_text = (
            f"# Tuning Experiment: {exp_name}\n\n"
            f"- **Data:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            f"- **Istanze ({len(instances)}):** {[i.id for i in instances]}\n"
            f"- **Num reads:** {num_reads}\n"
            f"- **Num sweeps:** {num_sweeps}\n"
            f"- **Griglia gamma:** {gamma_values}\n\n"
            f"## Miglior Coppia Individuata\n"
            f"- **gamma_cap:** `{best['gamma_cap']}`\n"
            f"- **gamma_rep:** `{best['gamma_rep']}`\n"
            f"- **Score Combinato:** {best['combined_score']:.1%}\n"
            f"- **Soluzioni Infeasible:** {best['infeasible_ratio']:.1%}\n"
        )
        readme_path.write_text(readme_text, encoding="utf-8")

    print("\n=== COPPIA OTTIMALE ===")
    print(f"gamma_cap={best['gamma_cap']}, gamma_rep={best['gamma_rep']} -> Score: {best['combined_score']:.1%} (Infeasible: {best['infeasible_ratio']:.1%})")