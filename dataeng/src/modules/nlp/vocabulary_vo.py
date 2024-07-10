from dataclasses import dataclass
from typing import Mapping


@dataclass
class VocabularyVo:
    value: Mapping[str, str]

    def size(self) -> int:
        return len(self.value)

    def nb_unique_lemmas(self) -> int:
        return len(set(self.value.values()))

    def __post_init__(self):
        keys = self.value.keys()
        values = self.value.values()
        circular_lemmas = [value for value in values if value in keys]
        if len(circular_lemmas) > 0:
            raise RuntimeError(f"Circular lemmas found: {circular_lemmas}.")
