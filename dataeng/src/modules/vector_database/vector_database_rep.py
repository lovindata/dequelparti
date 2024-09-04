from __future__ import annotations

import json
import os
from dataclasses import dataclass

from loguru import logger

from src.confs import envs_conf
from src.modules.shared.atomic_vos.organisms.embedding_table_vo import EmbeddingTableVo


@dataclass(frozen=True)
class VectorDatabaseRep:
    envs_conf: envs_conf.EnvsConf = envs_conf.impl

    def save(self, embedding_table: EmbeddingTableVo) -> None:
        logger.info(f"Saving the vector database: {embedding_table.__len__()} rows.")
        embedding_table_as_dicts = [
            embedding_row.__dict__ for embedding_row in embedding_table
        ]
        dirpath = os.path.dirname(self.envs_conf.vector_database_filepath)
        os.makedirs(dirpath, exist_ok=True)
        with open(self.envs_conf.vector_database_filepath, "w") as file:
            json.dump(embedding_table_as_dicts, file)


impl = VectorDatabaseRep()
