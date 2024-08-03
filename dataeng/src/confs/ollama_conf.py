from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha3_512
from typing import Any, Generator, Mapping

from ollama import Client

from src.confs import envs_conf


@dataclass(frozen=True)
class OllamaConf:
    envs_conf: envs_conf.EnvsConf = envs_conf.impl

    _ollama = Client(host=f"http://{envs_conf.ollama_ip}:{envs_conf.ollama_port}")

    @contextmanager
    def get_prediction(self, command: str) -> Generator[str, None, None]:
        loaded_message_content = self._load_cache_from_disk(command)
        message_content = (
            loaded_message_content if loaded_message_content else self._predict(command)
        )
        try:
            yield message_content
        except:
            raise
        else:
            if not loaded_message_content:
                self._cache_to_disk(command, message_content)

    def _predict(self, command: str) -> str:
        assistant_response: Mapping[str, Any] = self._ollama.chat(
            model="gemma2:9b",
            messages=[
                {
                    "role": "user",
                    "content": command,
                },
            ],
        )  # type: ignore
        message_content: str = assistant_response["message"]["content"]
        return message_content

    def _load_cache_from_disk(self, command: str) -> str | None:
        filename = self._build_filename(command)
        filepath = os.path.join(
            self.envs_conf.ollama_conf_get_prediction_cache_dirpath, filename
        )
        message_content = None
        if os.path.exists(filepath):
            with open(filepath, "r") as file:
                message_content = file.read()
        return message_content

    def _cache_to_disk(self, command: str, message_content: str) -> None:
        filename = self._build_filename(command)
        filepath = os.path.join(
            self.envs_conf.ollama_conf_get_prediction_cache_dirpath, filename
        )
        os.makedirs(
            self.envs_conf.ollama_conf_get_prediction_cache_dirpath, exist_ok=True
        )
        with open(filepath, "w+") as f:
            f.write(message_content)

    def _build_filename(self, command: str) -> str:
        return f"{sha3_512(command.encode()).hexdigest()}.txt"


impl = OllamaConf()
