from dataclasses import dataclass
from typing import List, Mapping


@dataclass
class VocabularyVo:
    root: Mapping[str, List[float]]
