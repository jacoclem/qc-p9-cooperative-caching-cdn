import ast
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import dwave.inspector
import minorminer
from dwave.embedding.chain_strength import uniform_torque_compensation
from dwave.system import DWaveSampler, FixedEmbeddingComposite

from src.bqm_builder import build_bqm
from src.parser import CachingInstance
from src.validator import validate_solution


def _parse_var_key(key_str: str) -> Any:
    """Riconverte la chiave JSON nel tipo originale usato nel BQM."""
    try:
        return ast.literal_eval(key_str)
    except (ValueError, SyntaxError):
        return key_str


def compute_and_save_embeddings(
    instances: List[CachingInstance],
    solver_name: str = "Advantage2_system1",
    output_dir: Path = Path("embeddings"),
    gamma_cap: float = 10.0,
    gamma_rep: float = 10.0,
    timeout: int = 120,
) -> None:
    """Calcola e salva su JSON gli embedding per le istanze fornite."""
    output_dir.mkdir(parents=True, exist_ok=True)

    sampler = DWaveSampler(solver=solver_name)
    target_edges = list(sampler.edgelist)
    actual_solver = sampler.solver.name
    sampler.client.close()

    logging.info(f"Target QPU: {actual_solver} ({len(target_edges)} couplers)")
    summary = []

    for inst in instances:
        # I valori numerici di gamma (> 0) non alterano il grafo di interazione del BQM
        bqm, stats = build_bqm(inst, gamma_cap=gamma_cap, gamma_rep=gamma_rep)
        source_edges = list(bqm.quadratic.keys())

        logging.info(
            f"[{inst.id} - Block {inst.block}] Finding embedding "
            f"(vars={stats['num_total_vars']}, edges={len(source_edges)})..."
        )

        t0 = time.perf_counter()
        embedding = minorminer.find_embedding(
            source_edges,
            target_edges,
            timeout=timeout,
            random_seed=42,
        )
        elapsed = time.perf_counter() - t0

        if not embedding:
            logging.error(f"[{inst.id}] Embedding FALLITO entro {timeout}s!")
            continue

        chain_lengths = [len(chain) for chain in embedding.values()]
        total_qubits = sum(chain_lengths)
        max_chain = max(chain_lengths)
        avg_chain = total_qubits / len(chain_lengths)

        logging.info(
            f"  -> OK in {elapsed:.2f}s | physical_qubits={total_qubits}, "
            f"max_chain={max_chain}, avg_chain={avg_chain:.2f}"
        )

        emb_file = output_dir / f"{inst.id}_{actual_solver}.json"
        with emb_file.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "instance_id": inst.id,
                    "block": inst.block,
                    "solver": actual_solver,
                    "embedding_time_s": round(elapsed, 4),
                    "num_bqm_vars": len(embedding),
                    "num_physical_qubits": total_qubits,
                    "max_chain_length": max_chain,
                    "avg_chain_length": round(avg_chain, 3),
                    "embedding": {str(k): v for k, v in embedding.items()},
                },
                f,
                indent=4,
            )

        summary.append(
            {
                "instance_id": inst.id,
                "block": inst.block,
                "bqm_vars": len(embedding),
                "physical_qubits": total_qubits,
                "max_chain": max_chain,
                "avg_chain": round(avg_chain, 2),
                "time_s": round(elapsed, 3),
            }
        )

    with (output_dir / f"summary_{actual_solver}.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)


def load_embedding(
    instance_id: str,
    solver_name: str = "Advantage2_system1",
    embeddings_dir: Path = Path("embeddings"),
) -> Dict[Any, List[int]]:
    """Carica da disco l'embedding salvato per una specifica istanza e QPU."""
    emb_file = embeddings_dir / f"{instance_id}_{solver_name}.json"
    if not emb_file.exists():
        raise FileNotFoundError(f"Embedding non trovato: {emb_file}")

    with emb_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return {_parse_var_key(k): v for k, v in data["embedding"].items()}


def run_qpu_probe(
    instance: CachingInstance,
    gamma_cap: float = 10.0,
    gamma_rep: float = 30.0,
    chain_strength: Optional[float] = None,
    num_reads: int = 100,
    solver_name: str = "Advantage2_system1",
    open_inspector: bool = False,
) -> Dict[str, Any]:
    """Esegue una run di prova su QPU usando l'embedding salvato e analizza le broken chains."""
    bqm, _ = build_bqm(instance, gamma_cap=gamma_cap, gamma_rep=gamma_rep)
    embedding = load_embedding(instance.id, solver_name=solver_name)

    qpu = DWaveSampler(solver=solver_name)
    sampler = FixedEmbeddingComposite(qpu, embedding)

    # Se chain_strength non è specificato, usa la compensazione automatica D-Wave
    cs = (
        chain_strength
        if chain_strength is not None
        else float(uniform_torque_compensation(bqm, sampler))
    )

    logging.info(
        f"Running QPU probe on {instance.id} ({solver_name}) | "
        f"chain_strength={cs:.2f}, num_reads={num_reads}"
    )

    sampleset = sampler.sample(
        bqm,
        num_reads=num_reads,
        chain_strength=cs,
        label=f"P09_{instance.id}_cs{cs:.1f}",
    )

    # Analisi broken chains e feasibility
    total_reads = 0
    feasible_reads = 0
    weighted_break_fraction = 0.0
    best_obj = None

    for datum in sampleset.data(["sample", "num_occurrences", "chain_break_fraction"]):
        occ = int(datum.num_occurrences)
        total_reads += occ
        weighted_break_fraction += float(datum.chain_break_fraction) * occ

        val = validate_solution(instance, datum.sample)
        if val["is_valid"]:
            feasible_reads += occ
            obj = float(val["objective_value"])
            if best_obj is None or obj < best_obj:
                best_obj = obj

    avg_chain_break_pct = (weighted_break_fraction / total_reads) * 100.0
    feasible_pct = (feasible_reads / total_reads) * 100.0

    logging.info(
        f"QPU Result [{instance.id}] | chain_strength={cs:.2f} | "
        f"broken_chains={avg_chain_break_pct:.2f}% | "
        f"feasible={feasible_reads}/{total_reads} ({feasible_pct:.1f}%) | "
        f"best_obj={best_obj}"
    )

    if open_inspector:
        # Avvia il server locale di dwave-inspector (porta 18000 di default)
        os.environ["DWAVE_INSPECTOR_HOST"] = "0.0.0.0"
        os.environ["DWAVE_INSPECTOR_PORT"] = "18000"
        os.environ["BROWSER"] = "echo"
        dwave.inspector.show(sampleset)
        input("Inspector attivo su http://localhost:18000 — Premi INVIO per chiudere...")

    qpu.client.close()

    return {
        "instance_id": instance.id,
        "chain_strength": cs,
        "avg_chain_break_pct": avg_chain_break_pct,
        "feasible_pct": feasible_pct,
        "best_feasible_obj": best_obj,
    }