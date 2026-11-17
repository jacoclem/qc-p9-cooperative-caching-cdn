from dimod import BinaryQuadraticModel
from typing import Dict, Any, Tuple

def _add_variable(bqm, m, n):
    for h in range(m):
        for j in range(n):
            bqm.add_variable(f"x_{h}_{j}")


def build_bqm( instance: Dict[str, Any], gamma_cap: float = 10.0, gamma_rep: float = 10.0)-> Tuple[BinaryQuadraticModel, Dict[str, int]]:

    #Parametri
    m = instance["m"]
    n = instance["n"]
    sigma = instance["sigma"]
    K = instance["K"]
    b = instance["b"]
    r = instance["r"]
    neighbors = instance["N"]
    delta = instance["delta"]

    #Inizializzazione del BQM con variabili binarie
    bqm = BinaryQuadraticModel(vartype="BINARY")
    _add_variable(bqm, m, n)

    #Funzione obiettivo termini lineari (beneficio)
    for h in range(m):
        for j in range(n):
            bqm.set_linear(f"x_{h}_{j}", -b[h][j]) #incentivo del beneficio nel salvare il contenuto
        
    #Funzione obiettivo termine quadratico (penalità di ridondanza)
    for edge in neighbors:
        h, k = edge[0], edge[1]
        for j in range(n):
            #se salvato come un dizionario
            if isinstance(delta, dict):
                d_val=delta.get((h, k, j), delta.get((k, h, j), 0))
            else:
                d_val = delta[h][k][j]

            if d_val != 0:
                bqm.add_quadratic(f"x_{h}_{j}", f"x_{k}_{j}", d_val)

    #Vincolo capacità del server
    for h in range(m):
        inequality=[]
        for j in range(n):
            inequality.append((f"x_{h}_{j}", sigma[j]))

        bqm.add_linear_inequality_constraint(
            inequality, 
            lagrange_multiplier=gamma_cap,
            lb=0,
            ub=K[h],
            label=f"capacity_{h}"
        )

    #Vincolo numero minimo di copie
    for j in range(n):
        inequality=[]
        for h in range(m):
            inequality.append((f"x_{h}_{j}", 1))

        bqm.add_linear_inequality_constraint(
            inequality,
            lagrange_multiplier=gamma_rep,
            lb=r[j],
            ub=m,
            label=f"replication_{j}"
        )
    
    return bqm