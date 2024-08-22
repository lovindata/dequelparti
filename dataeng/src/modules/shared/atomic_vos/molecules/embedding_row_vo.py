from dataclasses import dataclass

from src.modules.shared.atomic_vos.atoms.decoded_embedding_vo import DecodedEmbeddingVo
from src.modules.shared.atomic_vos.atoms.id_vo import IdVo
from src.modules.shared.atomic_vos.atoms.label_vo import LabelVo
from src.modules.shared.atomic_vos.atoms.vector_embedding_vo import VectorEmbeddingVo


@dataclass
class EmbeddingRowVo:
    id: IdVo
    decoded_embedding: DecodedEmbeddingVo
    vector_embedding: VectorEmbeddingVo
    label: LabelVo
