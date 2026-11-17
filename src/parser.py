"""Parser for loading Problem 09 (Cooperative Caching in CDN) instances."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "data.json"


@dataclass(frozen=True)
class CachingInstance:
    """Single instance of the Cooperative Caching in CDN problem (P09)."""

    id: str
    block: int
    m: int  # Number of edge servers (|S|)
    n: int  # Number of contents (|C|)
    sigma: list[int]  # Content sizes
    K: list[int]  # Edge server memory capacities
    b: list[list[int]]  # Benefit matrix b[h][j]
    r: list[int]  # Minimum required replicas per content
    N: list[tuple[int, int]]  # Neighboring server pairs
    delta: dict[tuple[int, int, int], int]  # Redundancy penalties (h, k, j) -> value

    def __post_init__(self) -> None:
        if len(self.sigma) != self.n:
            raise ValueError(f"{self.id}: len(sigma)={len(self.sigma)} != n={self.n}")
        if len(self.K) != self.m:
            raise ValueError(f"{self.id}: len(K)={len(self.K)} != m={self.m}")
        if len(self.b) != self.m or any(len(row) != self.n for row in self.b):
            raise ValueError(
                f"{self.id}: dimensions of b do not match m={self.m}, n={self.n}"
            )
        if len(self.r) != self.n:
            raise ValueError(f"{self.id}: len(r)={len(self.r)} != n={self.n}")

    @property
    def num_variables(self) -> int:
        """Number of binary decision variables x_{h,j} (m * n)."""
        return self.m * self.n

    def get_delta(self, h: int, k: int, j: int) -> int:
        """Return delta_{hkj} (0 if absent), normalizing h < k."""
        if h > k:
            h, k = k, h
        return self.delta.get((h, k, j), 0)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> CachingInstance:
        """Build a CachingInstance from a raw JSON dictionary."""
        neighbors = [(int(pair[0]), int(pair[1])) for pair in raw["N"]]
        delta_map = {}
        for h, k, j, val in raw["delta"]:
            u, v = int(h), int(k)
            if u > v:
                u, v = v, u
            delta_map[(u, v, int(j))] = int(val)

        return cls(
            id=str(raw["id"]),
            block=int(raw["blocco"]),  # Original JSON schema key
            m=int(raw["m"]),
            n=int(raw["n"]),
            sigma=[int(x) for x in raw["sigma"]],
            K=[int(x) for x in raw["K"]],
            b=[[int(val) for val in row] for row in raw["b"]],
            r=[int(x) for x in raw["r"]],
            N=neighbors,
            delta=delta_map,
        )


def load_raw_data(path: str | Path = DEFAULT_DATA_PATH) -> dict[str, Any]:
    """Load the full JSON dataset."""
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as file_in:
        return json.load(file_in)


def load_instances(path: str | Path = DEFAULT_DATA_PATH) -> list[CachingInstance]:
    """Load all instances from the JSON file as a list of CachingInstance objects."""
    data = load_raw_data(path)
    return [CachingInstance.from_dict(item) for item in data["istanze"]]


def load_instance_by_id(
    instance_id: str, path: str | Path = DEFAULT_DATA_PATH
) -> CachingInstance:
    """Load a single instance by its identifier (e.g., 'P09-00')."""
    for instance in load_instances(path):
        if instance.id == instance_id:
            return instance
    raise KeyError(f"Instance '{instance_id}' not found in {path}")


def load_instances_by_block(
    block: int, path: str | Path = DEFAULT_DATA_PATH
) -> list[CachingInstance]:
    """Load all instances belonging to a specific size block (1..5)."""
    return [inst for inst in load_instances(path) if inst.block == block]


if __name__ == "__main__":
    loaded_instances = load_instances()
    print(f"Loaded {len(loaded_instances)} instances from {DEFAULT_DATA_PATH}")
    for inst in loaded_instances:
        print(
            f"  {inst.id} | block={inst.block} | m={inst.m}, n={inst.n} "
            f"| vars={inst.num_variables} | |N|={len(inst.N)} | |delta|={len(inst.delta)}"
        )