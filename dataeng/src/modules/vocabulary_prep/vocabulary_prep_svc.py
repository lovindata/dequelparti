import os
from dataclasses import dataclass
from typing import Dict, List

from loguru import logger
from tqdm import tqdm

from src.confs import envs_conf, spacy_conf
from src.modules.file_system import file_system_svc
from src.modules.vocabulary_prep.vocabulary_vo import VocabularyVo


@dataclass
class VocabularyPrepSvc:
    envs_conf = envs_conf.impl
    spacy_conf = spacy_conf.impl
    file_system_svc = file_system_svc.impl

    def compute_vocabulary(self) -> VocabularyVo:
        logger.info("Extracting tokens with embedding vectors from SpaCy.")
        tokens = [
            word
            for word in tqdm(self.spacy_conf.spacy.vocab.strings)
            if self.spacy_conf.spacy.vocab.has_vector(word)
        ]
        logger.info("Cleaning up tokens to only keep words.")
        tokens = [
            word
            for word in tqdm(tokens)
            if (lexeme := self.spacy_conf.spacy.vocab[word])
            and lexeme.is_stop is False
            and lexeme.is_punct is False
            and lexeme.is_space is False
        ]
        tokens = self.spacy_conf.spell_checker.known(tokens)
        logger.info("Building vocabulary.")
        vocabulary_data: Dict[str, List[float]] = {
            word: self.spacy_conf.spacy.vocab[word].vector.tolist() for word in tokens
        }
        vocabulary = VocabularyVo(vocabulary_data)
        logger.info(f"Saving vocabulary of {len(vocabulary.root)} words.")
        self._write_vocabulary_grouped_by_initial(vocabulary)
        return vocabulary

    def _write_vocabulary_grouped_by_initial(self, vocabulary: VocabularyVo) -> None:
        initials = set([word[0] for word in vocabulary.root.keys()])
        initials = sorted(list(initials))
        for initial in initials:
            vocabulary_only_with_initial = {
                word: vocabulary.root[word]
                for word in vocabulary.root
                if word[0] == initial
            }
            filename = f"{initial}.json"
            filepath = os.path.join(self.envs_conf.vocabulary_dirpath, filename)
            self.file_system_svc.write_as_json(vocabulary_only_with_initial, filepath)


impl = VocabularyPrepSvc()
