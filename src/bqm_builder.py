from dimod import BinaryQuadraticModel
from typing import Dict, Any, Tuple


def _add_variables(bqm: BinaryQuadraticModel, m: int, n: int) -> None:
    """Aggiunge al BQM le m*n variabili decisionali binarie x_{h}_{j}."""
    for h in range(m):
        for j in range(n):
            bqm.add_variable(f"x_{h}_{j}")


def _extract_instance_params(instance: Any) -> Tuple[int, int, Any, Any, Any, Any, Any, Any]:
    """Estrae i parametri dell'istanza supportando sia dict sia CachingInstance."""
    getter = instance.__getitem__ if isinstance(instance, dict) else lambda k: getattr(instance, k)
    return (
        getter("m"),
        getter("n"),
        getter("sigma"),
        getter("K"),
        getter("b"),
        getter("r"),
        getter("N"),
        getter("delta"),
    )


def build_bqm(
    instance: Any,
    gamma_cap: float = 10.0,
    gamma_rep: float = 10.0,
) -> Tuple[BinaryQuadraticModel, Dict[str, int]]:
    """Costruisce il BinaryQuadraticModel per il problema P09 e restituisce le statistiche."""
    # Parametri
    m, n, sigma, K, b, r, neighbors, delta = _extract_instance_params(instance)

    # Inizializzazione del BQM con variabili binarie
    bqm = BinaryQuadraticModel(vartype="BINARY")
    _add_variables(bqm, m, n)

    # Funzione obiettivo termini lineari (beneficio)
    for h in range(m):
        for j in range(n):
            bqm.set_linear(f"x_{h}_{j}", -b[h][j])  # incentivo del beneficio nel salvare il contenuto

    # Funzione obiettivo termine quadratico (penalità di ridondanza)
    for edge in neighbors:
        h, k = edge[0], edge[1]
        for j in range(n):
            # se salvato come un dizionario
            if isinstance(delta, dict):
                d_val = delta.get((h, k, j), delta.get((k, h, j), 0))
            else:
                d_val = delta[h][k][j]

            if d_val != 0:
                bqm.add_quadratic(f"x_{h}_{j}", f"x_{k}_{j}", d_val)

    # Vincolo capacità del server
    for h in range(m):
        inequality = [(f"x_{h}_{j}", sigma[j]) for j in range(n)]
        bqm.add_linear_inequality_constraint(
            inequality,
            lagrange_multiplier=gamma_cap,
            lb=0,
            ub=K[h],
            label=f"capacity_{h}",
        )

    # Vincolo numero minimo di copie
    for j in range(n):
        inequality = [(f"x_{h}_{j}", 1) for h in range(m)]
        bqm.add_linear_inequality_constraint(
            inequality,
            lagrange_multiplier=gamma_rep,
            lb=r[j],
            ub=m,
            label=f"replication_{j}",
        )

    num_logical_vars = m * n
    num_total_vars = bqm.num_variables
    bqm_stats = {
        "num_logical_vars": num_logical_vars,
        "num_slack_vars": num_total_vars - num_logical_vars,
        "num_total_vars": num_total_vars,
        "num_quadratic_interactions": bqm.num_interactions,
    }

    return bqm, bqm_stats
