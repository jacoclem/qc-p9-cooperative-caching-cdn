"""Main entry point for Cooperative Caching in CDN experiments."""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict

from dwave.samplers import SimulatedAnnealingSampler

from src.bqm_builder import build_bqm
from src.parser import (
    CachingInstance,
    load_instance_by_id,
    load_instances,
    load_instances_by_block,
)
from src.validator import validate_solution

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the solver pipeline."""
    parser = argparse.ArgumentParser(
        description="Solver for the Cooperative Caching in CDN problem (P09)."
    )

    group_loader = parser.add_mutually_exclusive_group()
    group_loader.add_argument(
        "--load-by-id",
        type=str,
        help="Load a single instance by its ID (e.g., P09-00).",
    )
    group_loader.add_argument(
        "--load-by-block",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Load all instances belonging to a specific block (1..5).",
    )

    parser.add_argument(
        "--gamma-cap",
        type=float,
        default=10.0,
        help="Lagrange multiplier for server capacity constraints.",
    )
    parser.add_argument(
        "--gamma-rep",
        type=float,
        default=10.0,
        help="Lagrange multiplier for content replication constraints.",
    )
    parser.add_argument(
        "--num-reads",
        type=int,
        default=100,
        help="Number of reads/samples per instance.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/sa_tuning_metrics.json",
        help="Output path for the JSON metrics report.",
    )
    return parser.parse_args()


def evaluate_sampleset(
    instance: CachingInstance,
    sampleset: Any,
) -> Dict[str, Any]:
    """Validate all samples in a SampleSet, accounting for num_occurrences."""
    total_reads = 0
    feasible_reads = 0
    best_feasible_obj = None
    best_feasible_energy = None
    best_overall_validation = None

    for idx, datum in enumerate(
        sampleset.data(["sample", "energy", "num_occurrences"])
    ):
        occurrences = int(datum.num_occurrences)
        total_reads += occurrences
        validation = validate_solution(instance, datum.sample)

        if idx == 0:
            best_overall_validation = {
                "energy": float(datum.energy),
                **validation,
            }

        if validation["is_valid"]:
            feasible_reads += occurrences
            obj_val = validation["objective_value"]
            if best_feasible_obj is None or obj_val < best_feasible_obj:
                best_feasible_obj = obj_val
                best_feasible_energy = float(datum.energy)

    feasible_ratio = feasible_reads / total_reads if total_reads > 0 else 0.0
    return {
        "total_reads": total_reads,
        "feasible_reads": feasible_reads,
        "feasible_ratio": feasible_ratio,
        "best_feasible_obj": best_feasible_obj,
        "best_feasible_energy": best_feasible_energy,
        "lowest_energy_sample": best_overall_validation,
    }


def main() -> None:
    args = parse_args()

    if args.load_by_id:
        instances = [load_instance_by_id(args.load_by_id)]
    elif args.load_by_block:
        instances = load_instances_by_block(args.load_by_block)
    else:
        instances = load_instances()

    logging.info(f"Loaded {len(instances)} instances.")
    sampler = SimulatedAnnealingSampler()
    results_metrics = []

    for instance in instances:
        logging.info(
            f"--- Processing {instance.id} (Block {instance.block}) "
            f"| gamma_cap={args.gamma_cap}, gamma_rep={args.gamma_rep} ---"
        )

        # Phase 1: BQM construction
        t0 = time.perf_counter()
        bqm, bqm_stats = build_bqm(
            instance,
            gamma_cap=args.gamma_cap,
            gamma_rep=args.gamma_rep,
        )
        bqm_build_time = time.perf_counter() - t0

        logging.info(
            f"BQM built in {bqm_build_time:.4f}s | "
            f"logical={bqm_stats['num_logical_vars']}, "
            f"slack={bqm_stats['num_slack_vars']}, "
            f"total={bqm_stats['num_total_vars']}, "
            f"interactions={bqm_stats['num_quadratic_interactions']}"
        )

        # Phase 2: Local Simulated Annealing execution (multiplier tuning)
        t1 = time.perf_counter()
        sampleset = sampler.sample(bqm, num_reads=args.num_reads)
        sa_time = time.perf_counter() - t1

        # Phase 3: Full SampleSet validation
        eval_stats = evaluate_sampleset(instance, sampleset)

        logging.info(
            f"SA done in {sa_time:.4f}s | "
            f"feasible={eval_stats['feasible_reads']}/{eval_stats['total_reads']} "
            f"({eval_stats['feasible_ratio']:.1%}) | "
            f"best_feasible_obj={eval_stats['best_feasible_obj']}"
        )

        results_metrics.append(
            {
                "instance_id": instance.id,
                "block": instance.block,
                "gamma_cap": args.gamma_cap,
                "gamma_rep": args.gamma_rep,
                "bqm_build_time_s": bqm_build_time,
                "sa_time_s": sa_time,
                **bqm_stats,
                **eval_stats,
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_out:
        json.dump(results_metrics, file_out, indent=4)

    logging.info(f"Metrics saved to {output_path}")


if __name__ == "__main__":
    main()