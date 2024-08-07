from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from hashlib import sha3_512
from typing import Sequence, Tuple

import torch

from src.confs import envs_conf
from src.modules.file_system import file_system_svc
from src.modules.llm_prep.llm_row_vo import LLMRowVo
from src.modules.vocabulary_prep.vocabulary_vo import VocabularyVo


@dataclass(frozen=True)
class CacheSvc:
    envs_conf: envs_conf.EnvsConf = envs_conf.impl
    file_system_svc: file_system_svc.FileSystemSvc = file_system_svc.impl

    def load_feature_and_label_tensors_from_disk(
        self, llm_rows: Sequence[LLMRowVo], vocabulary: VocabularyVo
    ) -> Tuple[torch.Tensor, torch.Tensor] | None:
        feature_tensor_filepath, label_tensor_filepath = (
            self._build_feature_and_label_tensors_filepath(llm_rows, vocabulary)
        )
        if (
            os.path.exists(feature_tensor_filepath) is False
            or os.path.exists(label_tensor_filepath) is False
        ):
            return None
        feature_tensor: torch.Tensor = torch.load(
            feature_tensor_filepath, weights_only=True
        )
        label_tensor: torch.Tensor = torch.load(
            label_tensor_filepath, weights_only=True
        )
        return feature_tensor, label_tensor

    def save_feature_and_label_tensors_to_disk(
        self,
        llm_rows: Sequence[LLMRowVo],
        vocabulary: VocabularyVo,
        feature_tensor: torch.Tensor,
        label_tensor: torch.Tensor,
    ) -> None:
        feature_tensor_filepath, label_tensor_filepath = (
            self._build_feature_and_label_tensors_filepath(llm_rows, vocabulary)
        )
        torch.save(feature_tensor, feature_tensor_filepath)
        torch.save(label_tensor, label_tensor_filepath)

    def _build_feature_and_label_tensors_filepath(
        self, llm_rows: Sequence[LLMRowVo], vocabulary: VocabularyVo
    ) -> Tuple[str, str]:
        def build_feature_tensor_filename() -> str:
            llm_rows_bytes = pickle.dumps(llm_rows)
            vocabulary_bytes = pickle.dumps(vocabulary)
            args_identifier = llm_rows_bytes + vocabulary_bytes
            return f"feature_tensor_{sha3_512(pickle.dumps(args_identifier)).hexdigest()}.pt"

        def build_label_tensor_filename() -> str:
            llm_rows_bytes = pickle.dumps(llm_rows)
            return (
                f"label_tensor_{sha3_512(pickle.dumps(llm_rows_bytes)).hexdigest()}.pt"
            )

        os.makedirs(
            self.envs_conf.prepare_data_svc_prepare_data_cache_dirpath, exist_ok=True
        )
        feature_tensor_filename = build_feature_tensor_filename()
        feature_tensor_filepath = os.path.join(
            self.envs_conf.prepare_data_svc_prepare_data_cache_dirpath,
            feature_tensor_filename,
        )
        label_tensor_filename = build_label_tensor_filename()
        label_tensor_filepath = os.path.join(
            self.envs_conf.prepare_data_svc_prepare_data_cache_dirpath,
            label_tensor_filename,
        )
        return feature_tensor_filepath, label_tensor_filepath


impl = CacheSvc()
