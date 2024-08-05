from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass
class LemmaEmbeddingsVo:
    value: Mapping[str, Sequence[float]]

    def size(self):
        return len(self.value)

    def vector_size(self):
        return len(list(self.value.values())[0])

    def __post_init__(self):
        sizes = set([len(x) for x in self.value.values()])
        if len(sizes) != 1:
            raise RuntimeError(f"Different sizes found: {sizes}.")
