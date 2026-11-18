import numpy as np
from typing import Dict, Union, Any

# Estrae la matrice decisionale (m x n) dalla soluzione dove x è un array bidimensionale
def extract_decision_matrix(solution: Union[Dict[str, Any], np.ndarray, list], m: int, n: int) -> np.ndarray:

    # Se è un dizionario
    if isinstance(solution, dict):
        # Creo una matrice x e sostituisco i suoi valori con i valori della matrice soluzione
        x = np.zeros((m, n), dtype=int)
        for h in range(m):
            for j in range(n):
                val = solution.get(f"x_{h}_{j}", 0)
                x[h, j] = int(round(val))
        return x
    else:
        # se non è un dizionario (Solutori classici)
        x = np.array(solution, dtype=int)
        # controllo delle dimensioni della matrice
        if x.shape != (m, n):
            raise ValueError(f"Dimensione matrice attesa {(m, n)}, ricevuta {x.shape}")
        return x


# Funzione che verifica se la soluzione rispetta i vincoli del problema e calcola il suo punteggio
def validate_solution(
    instance: Any,
    solution: Union[Dict[str, Any], np.ndarray, list]
    ):

    # Estraggo i parametri da instance (supportando sia dict sia CachingInstance)
    getter = instance.__getitem__ if isinstance(instance, dict) else lambda k: getattr(instance, k)
    m = getter("m") # numero server edge
    n = getter("n") # numero contenuti
    sigma = np.array(getter("sigma"), dtype=float) # dimensione del contenuto
    K = np.array(getter("K"), dtype=float) # capacità del server
    b = np.array(getter("b"), dtype=float) # beneficio di salvare un contenuto in un server
    r = np.array(getter("r"), dtype=int) # numero minimo di server che possiedono un contenuto
    neighbors = getter("N")
    delta = getter("delta")

    # ottengo la matrice binaria
    x = extract_decision_matrix(solution, m, n)

    # controllo che la matrice abbia solo valori binari
    if not np.all(np.isin(x, [0, 1])):
        raise ValueError("La soluzione contiene valori non binari.")

    # controllo capacità dei server
    used_capacity = x @ sigma # prodotto matrice vettore -> spazio utilizzato in ogni server
    cap_excess = np.maximum(0.0, used_capacity - K) # salva gli spazi in eccesso nei vari server altrimenti salva 0
    cap_violations = int(np.sum(cap_excess > 1e-9)) # Numero di server la cui memoria è stata superata

    # controllo sul numero minimo
    copies_per_content = np.sum(x, axis=0) # somma (per colonna) il numero di contenuti salvati
    rep_deficit = np.maximum(0, r - copies_per_content) # calcolo dei contenuti non abbastanza ridondanti
    rep_violations = int(np.sum(rep_deficit > 0)) # restituisce il numero di violazioni del vincolo

    is_valid = (cap_violations == 0) and (rep_violations == 0)

    # Calcolo della funzione obiettivo
    benefit_term = np.sum(b * x) # somma(moltiplicazione del fattore del beneficio per la matrice soluzione)

    # calcolo della penalità per server vicini che hanno salvato lo stesso contenuto
    redundancy_penalty = 0.0
    for edge in neighbors:
        h, k = edge[0], edge[1]
        for j in range(n):
            if x[h, j] == 1 and x[k, j] == 1:
                # se è caricato come dizionario
                if isinstance(delta, dict):
                    d_val = delta.get((h, k, j), delta.get((k, h, j), 0.0))
                else:
                    d_val = delta[h][k][j]
                redundancy_penalty += d_val

    # calcolo del valore della funzione obiettivo finale
    objective_value = float(-benefit_term + redundancy_penalty)

    return {
        "is_valid": is_valid,
        "objective_value": objective_value,
        "benefit_total": float(benefit_term),
        "redundancy_total": float(redundancy_penalty),
        "capacity_violations": cap_violations,
        "replication_violations": rep_violations,
        "details": {
            "max_capacity_excess": float(np.max(cap_excess)) if m > 0 else 0.0,
            "max_replication_deficit": int(np.max(rep_deficit)) if n > 0 else 0,
        }
    }
