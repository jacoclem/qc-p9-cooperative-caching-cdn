"""Parser per il caricamento delle istanze del problema P09 (Cooperative Caching CDN)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#Configurazione del default path dei dati
DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "data.json"


@dataclass(frozen=True)
class CachingInstance:
    """Singola istanza del problema P09 di caching cooperativo in una CDN."""

    id: str         # identificativo dell'istanza
    blocco: int     # dimensione del problema
    m: int          # numero di server edge (|S|)
    n: int          # numero di contenuti (|C|)
    sigma: list[int]        # sigma[i]: dimensione del i-esimo contenuto in C
    K: list[int]            # K[h]:     spazio di memoria dell'h-esimo server in S
    b: list[list[int]]      # b[h][j]:  tabella che rappreenta il guadagno che si ottiene se
                            #           il server h salva il contenuto j
    r: list[int]            # r[j]:     indica il numero di copie di un contenuto j da distribuire
    N: list[tuple[int, int]]    # N[(h, k)]:  vicini nella rete
    delta: dict[tuple[int, int, int], int]      # (h, k, j)-> valore: Perdita di guadagno che si ottiene se il server
                                                # h e il server k salvano il contenuto j

    # Verifica della validità dei dati
    def __post_init__(self) -> None:
        # Controlla che le dimensioni di sigma, K, b ed r siano coerenti con m ed n
        if len(self.sigma) != self.n:
            raise ValueError(f"{self.id}: len(sigma)={len(self.sigma)} != n={self.n}")
        if len(self.K) != self.m:
            raise ValueError(f"{self.id}: len(K)={len(self.K)} != m={self.m}")
        if len(self.b) != self.m or any(len(row) != self.n for row in self.b):
            raise ValueError(f"{self.id}: dimensioni di b non coerenti con m={self.m}, n={self.n}")
        if len(self.r) != self.n:
            raise ValueError(f"{self.id}: len(r)={len(self.r)} != n={self.n}")

    # Numero totale di variabili decisionali binarie
    @property
    def num_variables(self) -> int:
        """Numero di variabili decisionali binarie x_{h,j} (m * n)."""
        return self.m * self.n

    # Bonus di cooperazione tra i server h e k per i file j
    def get_delta(self, h: int, k: int, j: int) -> int:
        """Restituisce delta_{hkj} (0 se la combinazione è assente). Normalizza h < k."""
        if h > k:
            h, k = k, h
        return self.delta.get((h, k, j), 0)

    # Costruzione dell'istanza a partire dal dizionario letto dal JSON
    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> CachingInstance:
        """Costruisce una CachingInstance a partire da un dizionario JSON."""
        neighbors = [(int(pair[0]), int(pair[1])) for pair in raw["N"]]
        delta_map = {
            (int(h), int(k), int(j)): int(val)
            for h, k, j, val in raw["delta"]
        }
        return cls(
            id=str(raw["id"]),
            blocco=int(raw["blocco"]),
            m=int(raw["m"]),
            n=int(raw["n"]),
            sigma=[int(x) for x in raw["sigma"]],
            K=[int(x) for x in raw["K"]],
            b=[[int(val) for val in row] for row in raw["b"]],
            r=[int(x) for x in raw["r"]],
            N=neighbors,
            delta=delta_map,
        )


# Lettura del file JSON completo
def load_raw_data(path: str | Path = DEFAULT_DATA_PATH) -> dict[str, Any]:
    """Carica il contenuto JSON completo del dataset."""
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


# Caricamento di tutte le istanze del dataset
def load_instances(path: str | Path = DEFAULT_DATA_PATH) -> list[CachingInstance]:
    """Carica tutte le istanze dal file JSON e le restituisce come lista di CachingInstance."""
    data = load_raw_data(path)
    return [CachingInstance.from_dict(item) for item in data["istanze"]]


# Ricerca di una singola istanza tramite il suo identificativo
def load_instance_by_id(
    instance_id: str, path: str | Path = DEFAULT_DATA_PATH
) -> CachingInstance:
    """Carica una singola istanza identificata dal suo id (es. 'P09-00')."""
    for instance in load_instances(path):
        if instance.id == instance_id:
            return instance
    raise KeyError(f"Istanza '{instance_id}' non trovata in {path}")


# Filtro delle istanze appartenenti a uno specifico blocco dimensionale
def load_instances_by_block(
    blocco: int, path: str | Path = DEFAULT_DATA_PATH
) -> list[CachingInstance]:
    """Carica tutte le istanze appartenenti a un determinato blocco (1..5)."""
    return [inst for inst in load_instances(path) if inst.blocco == blocco]


if __name__ == "__main__":
    instances = load_instances()
    print(f"Caricate {len(instances)} istanze da {DEFAULT_DATA_PATH}")
    for inst in instances:
        print(
            f"  {inst.id} | blocco={inst.blocco} | m={inst.m}, n={inst.n} "
            f"| vars={inst.num_variables} | |N|={len(inst.N)} | |delta|={len(inst.delta)}"
        )

