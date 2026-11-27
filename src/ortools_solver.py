"""Risolutore classico esatto (Google OR-Tools CP-SAT) e benchmark per blocchi per il problema P09."""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from dwave.samplers import SimulatedAnnealingSampler
from ortools.sat.python import cp_model

from src.bqm_builder import build_bqm
from src.parser import CachingInstance
from src.validator import validate_solution

DEFAULT_ORTOOLS_RESULTS_DIR = Path("ortools-result")


def solve_instance_ortools(
    instance: CachingInstance,
    time_limit_s: float = 60.0,
) -> Dict[str, Any]:
    """Risolve un'istanza P09 all'ottimo globale certificato tramite OR-Tools CP-SAT."""
    t_build_start = time.perf_counter()
    model = cp_model.CpModel()

    m, n = instance.m, instance.n

    # Variabili decisionali binarie: x_{h,j} = 1 se il server h memorizza il contenuto j
    x = {
        (h, j): model.NewBoolVar(f"x_{h}_{j}")
        for h in range(m)
        for j in range(n)
    }

    # Vincolo 1: Capacità massima di ciascun server edge
    for h in range(m):
        model.Add(
            sum(int(instance.sigma[j]) * x[h, j] for j in range(n))
            <= int(instance.K[h])
        )

    # Vincolo 2: Numero minimo di repliche per ciascun contenuto
    for j in range(n):
        model.Add(
            sum(x[h, j] for h in range(m)) >= int(instance.r[j])
        )

    # Funzione obiettivo lineare: beneficio (-b_{h,j} * x_{h,j})
    linear_terms = [
        -int(instance.b[h][j]) * x[h, j]
        for h in range(m)
        for j in range(n)
    ]

    # Funzione obiettivo quadratica linearizzata: penalità di ridondanza (+delta_{h,k,j} * y_{h,k,j})
    # dove y_{h,k,j} = x_{h,j} AND x_{k,j}
    quadratic_terms = []
    num_aux_vars = 0
    for h, k in instance.N:
        for j in range(n):
            d_val = int(instance.get_delta(h, k, j))
            if d_val != 0:
                y = model.NewBoolVar(f"y_{h}_{k}_{j}")
                model.Add(y <= x[h, j])
                model.Add(y <= x[k, j])
                model.Add(y >= x[h, j] + x[k, j] - 1)
                quadratic_terms.append(d_val * y)
                num_aux_vars += 1

    model.Minimize(sum(linear_terms) + sum(quadratic_terms))
    model_build_time_s = time.perf_counter() - t_build_start

    # Risoluzione esatta con CP-SAT
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s

    t_solve_start = time.perf_counter()
    status = solver.Solve(model)
    solve_time_s = time.perf_counter() - t_solve_start

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(
            f"CP-SAT non ha trovato alcuna soluzione ammissibile per {instance.id} "
            f"(status={solver.StatusName(status)})"
        )

    solution_dict = {
        f"x_{h}_{j}": int(solver.Value(x[h, j]))
        for h in range(m)
        for j in range(n)
    }
    solution_matrix = [
        [int(solver.Value(x[h, j])) for j in range(n)]
        for h in range(m)
    ]

    # Verifica incrociata con il validatore ufficiale del progetto
    validation = validate_solution(instance, solution_dict)
    solver_obj = float(solver.ObjectiveValue())

    if not validation["is_valid"]:
        raise RuntimeError(
            f"Errore interno: la soluzione CP-SAT per {instance.id} non supera validate_solution!"
        )
    if abs(validation["objective_value"] - solver_obj) > 1e-6:
        raise RuntimeError(
            f"Discrepanza obiettivo per {instance.id}: "
            f"CP-SAT={solver_obj} vs validator={validation['objective_value']}"
        )

    return {
        "instance_id": instance.id,
        "block": instance.block,
        "m": m,
        "n": n,
        "num_logical_vars": instance.num_variables,
        "num_linearization_vars": num_aux_vars,
        "status": solver.StatusName(status),
        "is_valid": validation["is_valid"],
        "exact_objective": solver_obj,
        "benefit_total": validation["benefit_total"],
        "redundancy_total": validation["redundancy_total"],
        "model_build_time_s": model_build_time_s,
        "solve_time_s": solve_time_s,
        "solver_wall_time_s": float(solver.WallTime()),
        "solution_matrix": solution_matrix,
        "solution": solution_dict,
    }


def evaluate_sampleset_with_exact(
    instance: CachingInstance,
    sampleset: Any,
    exact_objective: float,
    tol: float = 1e-6,
) -> Dict[str, Any]:
    """Valuta un SampleSet (SA o QPU) rispetto all'ottimo esatto secondo le metriche della traccia."""
    total_reads = 0
    feasible_reads = 0
    optimal_reads = 0
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
            if abs(obj - exact_objective) <= tol:
                optimal_reads += occ

    feasible_ratio = feasible_reads / total_reads if total_reads > 0 else 0.0
    optimal_reads_ratio = optimal_reads / total_reads if total_reads > 0 else 0.0

    if best_feasible_obj is not None:
        abs_gap = max(0.0, best_feasible_obj - exact_objective)
        rel_gap_pct = (
            (abs_gap / abs(exact_objective)) * 100.0
            if abs(exact_objective) > 1e-9
            else 0.0
        )
    else:
        abs_gap = None
        rel_gap_pct = None

    return {
        "total_reads": total_reads,
        "feasible_reads": feasible_reads,
        "feasible_ratio": feasible_ratio,
        "optimal_reads": optimal_reads,
        "optimal_reads_ratio": optimal_reads_ratio,
        "best_feasible_obj": best_feasible_obj,
        "exact_objective": exact_objective,
        "abs_gap": abs_gap,
        "rel_gap_pct": rel_gap_pct,
    }


def summarize_by_block(
    records: List[Dict[str, Any]],
    include_sa: bool = False,
) -> List[Dict[str, Any]]:
    """Aggrega i risultati per blocco (media e deviazione standard, come richiesto dalla traccia)."""
    blocks = sorted({r["block"] for r in records})
    summary: List[Dict[str, Any]] = []

    for blk in blocks:
        blk_records = [r for r in records if r["block"] == blk]
        objs = [r["exact_objective"] for r in blk_records]
        times_ms = [r["solve_time_s"] * 1000.0 for r in blk_records]

        entry: Dict[str, Any] = {
            "block": blk,
            "num_instances": len(blk_records),
            "m": blk_records[0]["m"],
            "n": blk_records[0]["n"],
            "num_logical_vars": blk_records[0]["num_logical_vars"],
            "avg_exact_objective": round(float(np.mean(objs)), 4),
            "std_exact_objective": round(float(np.std(objs)), 4),
            "avg_exact_time_ms": round(float(np.mean(times_ms)), 4),
            "std_exact_time_ms": round(float(np.std(times_ms)), 4),
        }

        if include_sa:
            sa_times_ms = [r["sa_time_s"] * 1000.0 for r in blk_records]
            sa_feas = [r["sa_feasible_ratio"] * 100.0 for r in blk_records]
            sa_opt_reads = [r["sa_optimal_reads_ratio"] * 100.0 for r in blk_records]
            sa_gaps = [
                r["sa_rel_gap_pct"]
                for r in blk_records
                if r["sa_rel_gap_pct"] is not None
            ]
            entry.update(
                {
                    "avg_sa_time_ms": round(float(np.mean(sa_times_ms)), 4),
                    "std_sa_time_ms": round(float(np.std(sa_times_ms)), 4),
                    "avg_sa_feasible_pct": round(float(np.mean(sa_feas)), 4),
                    "std_sa_feasible_pct": round(float(np.std(sa_feas)), 4),
                    "avg_sa_optimal_reads_pct": round(float(np.mean(sa_opt_reads)), 4),
                    "std_sa_optimal_reads_pct": round(float(np.std(sa_opt_reads)), 4),
                    "avg_sa_rel_gap_pct": round(float(np.mean(sa_gaps)), 4) if sa_gaps else None,
                    "std_sa_rel_gap_pct": round(float(np.std(sa_gaps)), 4) if sa_gaps else None,
                }
            )

        summary.append(entry)

    return summary


def export_csv_tables(
    records: List[Dict[str, Any]],
    block_summary: List[Dict[str, Any]],
    instances_csv_path: Path,
    blocks_csv_path: Path,
    include_sa: bool = False,
) -> None:
    """Esporta su file CSV sia i risultati per singola istanza sia il riepilogo per blocco."""
    instance_fields = [
        "instance_id",
        "block",
        "m",
        "n",
        "num_logical_vars",
        "num_linearization_vars",
        "status",
        "is_valid",
        "exact_objective",
        "benefit_total",
        "redundancy_total",
        "solve_time_ms",
    ]
    if include_sa:
        instance_fields.extend(
            [
                "gamma_cap",
                "gamma_rep",
                "num_bqm_vars",
                "num_slack_vars",
                "sa_time_ms",
                "sa_best_feasible_obj",
                "sa_feasible_pct",
                "sa_optimal_reads_pct",
                "sa_abs_gap",
                "sa_rel_gap_pct",
            ]
        )

    instances_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with instances_csv_path.open("w", encoding="utf-8", newline="") as f_inst:
        writer = csv.DictWriter(f_inst, fieldnames=instance_fields)
        writer.writeheader()
        for r in records:
            row: Dict[str, Any] = {
                "instance_id": r["instance_id"],
                "block": r["block"],
                "m": r["m"],
                "n": r["n"],
                "num_logical_vars": r["num_logical_vars"],
                "num_linearization_vars": r["num_linearization_vars"],
                "status": r["status"],
                "is_valid": r["is_valid"],
                "exact_objective": round(r["exact_objective"], 4),
                "benefit_total": round(r["benefit_total"], 4),
                "redundancy_total": round(r["redundancy_total"], 4),
                "solve_time_ms": round(r["solve_time_s"] * 1000.0, 4),
            }
            if include_sa:
                row.update(
                    {
                        "gamma_cap": r["gamma_cap"],
                        "gamma_rep": r["gamma_rep"],
                        "num_bqm_vars": r["num_bqm_vars"],
                        "num_slack_vars": r["num_slack_vars"],
                        "sa_time_ms": round(r["sa_time_s"] * 1000.0, 4),
                        "sa_best_feasible_obj": r["sa_best_feasible_obj"],
                        "sa_feasible_pct": round(r["sa_feasible_ratio"] * 100.0, 2),
                        "sa_optimal_reads_pct": round(
                            r["sa_optimal_reads_ratio"] * 100.0, 2
                        ),
                        "sa_abs_gap": (
                            round(r["sa_abs_gap"], 4)
                            if r["sa_abs_gap"] is not None
                            else None
                        ),
                        "sa_rel_gap_pct": (
                            round(r["sa_rel_gap_pct"], 4)
                            if r["sa_rel_gap_pct"] is not None
                            else None
                        ),
                    }
                )
            writer.writerow(row)

    if block_summary:
        block_fields = list(block_summary[0].keys())
        with blocks_csv_path.open("w", encoding="utf-8", newline="") as f_blk:
            writer = csv.DictWriter(f_blk, fieldnames=block_fields)
            writer.writeheader()
            writer.writerows(block_summary)


def plot_ortools_benchmark(
    block_summary: List[Dict[str, Any]],
    output_img: Path,
    include_sa: bool = False,
) -> None:
    """Genera i grafici di sintesi per blocco (tempi e qualità, media ± deviazione standard)."""
    x_labels = [
        f"B{b['block']}\n({b['num_logical_vars']} var)" for b in block_summary
    ]
    x_pos = np.arange(len(block_summary))

    exact_times = np.array([b["avg_exact_time_ms"] for b in block_summary], dtype=float)
    exact_times_std = np.array([b["std_exact_time_ms"] for b in block_summary], dtype=float)

    if not include_sa:
        exact_objs = [b["avg_exact_objective"] for b in block_summary]
        exact_objs_std = [b["std_exact_objective"] for b in block_summary]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8))

        ax1.errorbar(
            x_pos,
            exact_objs,
            yerr=exact_objs_std,
            fmt="-o",
            capsize=5,
            linewidth=2.2,
            color="#1f77b4",
            label="Ottimo Esatto (media ± std)",
        )
        ax1.set_title("Valore Ottimo per Blocco (CP-SAT)", fontweight="bold")
        ax1.set_xticks(x_pos, x_labels)
        ax1.set_ylabel("Funzione Obiettivo")
        ax1.grid(True, linestyle="--", alpha=0.4)
        ax1.legend()

        exact_yerr = [np.minimum(exact_times, exact_times_std), exact_times_std]
        bars = ax2.bar(
            x_pos,
            exact_times,
            yerr=exact_yerr,
            capsize=5,
            color="#2ca02c",
            alpha=0.85,
            width=0.55,
            label="Tempo CP-SAT (media ± std)",
        )
        max_top = float(np.max(exact_times + exact_times_std)) if len(exact_times) else 1.0
        for bar, val, std in zip(bars, exact_times, exact_times_std):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                val + std + max_top * 0.02,
                f"{val:.1f} ms",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
            )
        ax2.set_title("Tempo di Risoluzione Esatta per Blocco", fontweight="bold")
        ax2.set_xticks(x_pos, x_labels)
        ax2.set_ylabel("Tempo (ms)")
        ax2.set_ylim(0, max_top * 1.18)
        ax2.grid(True, axis="y", linestyle="--", alpha=0.4)
        ax2.legend()
    else:
        sa_times = np.array([b["avg_sa_time_ms"] for b in block_summary], dtype=float)
        sa_times_std = np.array([b["std_sa_time_ms"] for b in block_summary], dtype=float)
        sa_feas = [b["avg_sa_feasible_pct"] for b in block_summary]
        sa_feas_std = [b["std_sa_feasible_pct"] for b in block_summary]
        sa_opt = [b["avg_sa_optimal_reads_pct"] for b in block_summary]
        sa_opt_std = [b["std_sa_optimal_reads_pct"] for b in block_summary]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.0))

        ax1.errorbar(
            x_pos,
            sa_feas,
            yerr=sa_feas_std,
            fmt="-^",
            capsize=5,
            linewidth=2.0,
            color="#ff7f0e",
            label="Letture Ammissibili SA (%)",
        )
        ax1.errorbar(
            x_pos,
            sa_opt,
            yerr=sa_opt_std,
            fmt="-o",
            capsize=5,
            linewidth=2.2,
            color="#1f77b4",
            label="Letture all'Ottimo Esatto SA (%)",
        )
        ax1.set_title("Qualità SA rispetto all'Ottimo Esatto", fontweight="bold")
        ax1.set_xticks(x_pos, x_labels)
        ax1.set_ylabel("Percentuale Letture (%)")
        ax1.set_ylim(-5, 108)
        ax1.grid(True, linestyle="--", alpha=0.4)
        ax1.legend(loc="best")

        width = 0.35
        exact_yerr = [np.minimum(exact_times, exact_times_std), exact_times_std]
        sa_yerr = [np.minimum(sa_times, sa_times_std), sa_times_std]
        ax2.bar(
            x_pos - width / 2,
            exact_times,
            width,
            yerr=exact_yerr,
            capsize=4,
            label="CP-SAT Esatto (ms)",
            color="#2ca02c",
        )
        ax2.bar(
            x_pos + width / 2,
            sa_times,
            width,
            yerr=sa_yerr,
            capsize=4,
            label="Simulated Annealing (ms)",
            color="#9467bd",
        )
        ax2.set_title("Confronto Tempi Classici per Blocco", fontweight="bold")
        ax2.set_xticks(x_pos, x_labels)
        ax2.set_ylabel("Tempo (ms)")
        ax2.set_ylim(bottom=0)
        ax2.grid(True, axis="y", linestyle="--", alpha=0.4)
        ax2.legend(loc="best")

    output_img.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_img, dpi=300)
    plt.close(fig)


def run_ortools_baseline(
    instances: List[CachingInstance],
    compare_sa: bool = False,
    gamma_cap: float = 10.0,
    gamma_rep: float = 30.0,
    num_reads: int = 100,
    num_sweeps: int = 1000,
    output_dir: Path = DEFAULT_ORTOOLS_RESULTS_DIR,
    exp_name: str | None = None,
) -> Dict[str, Any]:
    """Esegue il solver esatto CP-SAT su tutte le istanze e salva CSV, JSON e grafico in ortools-result/."""
    target_dir = output_dir / exp_name if exp_name else output_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    sa_sampler = SimulatedAnnealingSampler() if compare_sa else None
    records: List[Dict[str, Any]] = []

    for inst in instances:
        exact_res = solve_instance_ortools(inst)

        log_msg = (
            f"[{inst.id} - Block {inst.block}] CP-SAT {exact_res['status']} | "
            f"opt={exact_res['exact_objective']:.1f} "
            f"(benefit={exact_res['benefit_total']:.1f}, pen={exact_res['redundancy_total']:.1f}) | "
            f"time={exact_res['solve_time_s'] * 1000.0:.2f}ms"
        )

        if compare_sa and sa_sampler is not None:
            bqm, bqm_stats = build_bqm(inst, gamma_cap=gamma_cap, gamma_rep=gamma_rep)
            t_sa = time.perf_counter()
            sampleset = sa_sampler.sample(
                bqm, num_reads=num_reads, num_sweeps=num_sweeps
            )
            sa_time_s = time.perf_counter() - t_sa

            sa_eval = evaluate_sampleset_with_exact(
                inst, sampleset, exact_res["exact_objective"]
            )
            exact_res.update(
                {
                    "gamma_cap": gamma_cap,
                    "gamma_rep": gamma_rep,
                    "num_bqm_vars": bqm_stats["num_total_vars"],
                    "num_slack_vars": bqm_stats["num_slack_vars"],
                    "sa_time_s": sa_time_s,
                    "sa_best_feasible_obj": sa_eval["best_feasible_obj"],
                    "sa_feasible_ratio": sa_eval["feasible_ratio"],
                    "sa_optimal_reads_ratio": sa_eval["optimal_reads_ratio"],
                    "sa_abs_gap": sa_eval["abs_gap"],
                    "sa_rel_gap_pct": sa_eval["rel_gap_pct"],
                }
            )
            log_msg += (
                f" || SA: feas={sa_eval['feasible_ratio']:.1%}, "
                f"opt_reads={sa_eval['optimal_reads_ratio']:.1%}, "
                f"best={sa_eval['best_feasible_obj']}, "
                f"time={sa_time_s * 1000.0:.1f}ms"
            )

        logging.info(log_msg)
        records.append(exact_res)

    block_summary = summarize_by_block(records, include_sa=compare_sa)

    payload = {
        "solver": "OR-Tools CP-SAT",
        "compare_sa": compare_sa,
        "block_summary": block_summary,
        "instances": records,
    }

    instances_csv_path = target_dir / "instances.csv"
    blocks_csv_path = target_dir / "blocks_summary.csv"
    json_path = target_dir / "metrics.json"
    img_path = target_dir / "benchmark.png"

    export_csv_tables(
        records=records,
        block_summary=block_summary,
        instances_csv_path=instances_csv_path,
        blocks_csv_path=blocks_csv_path,
        include_sa=compare_sa,
    )

    with json_path.open("w", encoding="utf-8") as f_out:
        json.dump(payload, f_out, indent=4)

    plot_ortools_benchmark(block_summary, img_path, include_sa=compare_sa)

    logging.info(
        f"Risultati salvati in {target_dir}: "
        f"{instances_csv_path.name}, {blocks_csv_path.name}, {json_path.name}, {img_path.name}"
    )
    return payload
