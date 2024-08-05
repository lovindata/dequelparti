from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from hashlib import sha3_512
from typing import Any, List, Sequence

import torch

from src.confs import envs_conf
from src.modules.file_system import file_system_svc
from src.modules.file_system.pdf_vo.pdf_vo import PdfVo
from src.modules.llm_prep.llm_row_vo import LLMRowVo
from src.modules.vocabulary_prep.vocabulary_vo import VocabularyVo


@dataclass(frozen=True)
class CacheSvc:
    envs_conf: envs_conf.EnvsConf = envs_conf.impl
    file_system_svc: file_system_svc.FileSystemSvc = file_system_svc.impl

    def load_feature_tensor_cache_from_disk(
        self, llm_rows: Sequence[LLMRowVo], vocabulary: VocabularyVo
    ) -> torch.Tensor | None:
        ...
        """
        filename = self._build_filename(pdfs)
        filepath = os.path.join(
            self.envs_conf.llm_prep_svc_compute_llm_rows_cache_dirpath, filename
        )
        if os.path.exists(filepath) is False:
            return None
        with open(filepath, "r") as f:
            llm_rows_as_python = json.load(f)
        llm_rows = [
            LLMRowVo.model_validate(llm_row_as_python)
            for llm_row_as_python in llm_rows_as_python
        ]
        return llm_rows
        """

    def cache_feature_tensor_to_disk(self, feature_tensor: torch.Tensor) -> None:
        ...
        """
        filename = self._build_filename(pdfs)
        filepath = os.path.join(
            self.envs_conf.llm_prep_svc_compute_llm_rows_cache_dirpath, filename
        )
        llm_rows_serialized = [rows.model_dump() for rows in llm_rows]
        self.file_system_svc.write_as_json(llm_rows_serialized, filepath)
        """

    def load_label_tensor_cache_from_disk(
        self, llm_rows: Sequence[LLMRowVo]
    ) -> torch.Tensor | None:
        ...
        """
        filename = self._build_filename(pdfs)
        filepath = os.path.join(
            self.envs_conf.llm_prep_svc_compute_llm_rows_cache_dirpath, filename
        )
        if os.path.exists(filepath) is False:
            return None
        with open(filepath, "r") as f:
            llm_rows_as_python = json.load(f)
        llm_rows = [
            LLMRowVo.model_validate(llm_row_as_python)
            for llm_row_as_python in llm_rows_as_python
        ]
        return llm_rows
        """

    def cache_label_tensor_to_disk(self, label_tensor: torch.Tensor) -> None:
        ...
        """
        filename = self._build_filename(pdfs)
        filepath = os.path.join(
            self.envs_conf.llm_prep_svc_compute_llm_rows_cache_dirpath, filename
        )
        llm_rows_serialized = [rows.model_dump() for rows in llm_rows]
        self.file_system_svc.write_as_json(llm_rows_serialized, filepath)
        """

    def _build_feature_tensor_filename(
        self, llm_rows: Sequence[LLMRowVo], vocabulary: VocabularyVo
    ) -> str:
        ...
        """
        return f"{sha3_512(pickle.dumps(pdfs)).hexdigest()}.json"
        """

    def _build_label_tensor_filename(self, llm_rows: Sequence[LLMRowVo]) -> str:
        ...
        """
        return f"{sha3_512(pickle.dumps(pdfs)).hexdigest()}.json"
        """


impl = CacheSvc()
