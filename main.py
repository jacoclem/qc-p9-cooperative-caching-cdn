import argparse
import json
import logging
import time
from pathlib import Path

from src.parser import (
    CachingInstance,
    load_instance_by_id,
    load_instances,
    load_instances_by_block,
)
# from src.bqm_builder import build_bqm
# from src.classical_solver import solve_classically
# from src.validator import is_feasible

# Configure logging for clean output
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

def parse_args() -> argparse.Namespace:
    """
    Parses command line arguments for the CDN cooperative caching solver.
    """
    parser = argparse.ArgumentParser(
        description="Solver for the Cooperative Caching in CDN problem."
    )
    
    # Mutually exclusive group: load by ID or by block
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
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    
    # 1. Load instances based on CLI arguments
    if args.load_by_id:
        instances = [load_instance_by_id(args.load_by_id)]
    elif args.load_by_block:
        instances = load_instances_by_block(args.load_by_block)
    else:
        instances = load_instances()

    logging.info(f"Successfully loaded {len(instances)} instances.")

    results_metrics = []

    # 2. Execution loop for each loaded instance
    for instance in instances:
        # I am assuming instance object has attributes like 'id' and 'block'
        instance_id = getattr(instance, "id", "unknown")
        block_id = getattr(instance, "block", "unknown")
        
        logging.info(f"--- Processing Instance: {instance_id} (Block {block_id}) ---")
        
        metrics = {
            "instance_id": instance_id,
            "block": block_id
        }

        # ==========================================
        # PHASE A: Classical Solver (Ground Truth)
        # ==========================================
        # start_time = time.perf_counter()
        # best_classical_solution, classical_obj_val = solve_classically(instance)
        # metrics["classical_time"] = time.perf_counter() - start_time
        # metrics["classical_obj"] = classical_obj_val

        # ==========================================
        # PHASE B: BQM Construction
        # ==========================================
        # lagrange_capacity = 10.0 
        # lagrange_copies = 15.0
        # start_time = time.perf_counter()
        # bqm = build_bqm(instance, lagrange_capacity, lagrange_copies)
        # metrics["bqm_build_time"] = time.perf_counter() - start_time
        # logging.info(f"BQM created with {len(bqm.variables)} variables.")

        # ==========================================
        # PHASE C: Quantum Execution (D-Wave)
        # ==========================================
        # (Insert DWaveSampler / EmbeddingComposite logic here)

        # ==========================================
        # PHASE D: Physical Constraints Validation
        # ==========================================
        # is_valid = is_feasible(instance, best_quantum_sample)
        # metrics["is_valid"] = is_valid

        results_metrics.append(metrics)
        logging.info(f"Instance {instance_id} processed.\n")

    # 3. Export global metrics for the final report
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    metrics_file = output_dir / "results_metrics.json"
    
    with open(metrics_file, "w") as file_out:
        json.dump(results_metrics, file_out, indent=4)
        
    logging.info(f"All metrics saved to {metrics_file}")

if __name__ == "__main__":
    main()