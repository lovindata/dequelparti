from dataclasses import dataclass
from typing import Sequence


@dataclass
class SlindingAvgLemmasVo:
    value: Sequence[Sequence[float]]  # List of 1D vectors

    def vector_size(self):
        return len(self.value[0])

    def __post_init__(self):
        sizes = set([len(x) for x in self.value])
        if len(sizes) != 1:
            raise RuntimeError(f"Different sizes found: {sizes}.")
