from dataclasses import dataclass
from typing import List, Mapping


@dataclass
class VocabularyVo:
    root: Mapping[str, List[float]]

    def __post_init__(self) -> None:
        self.root = dict(sorted(self.root.items()))
