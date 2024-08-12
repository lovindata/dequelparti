import hashlib
import os
import pickle
import sys
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha3_512
from typing import Any, Generator, List, Mapping, Sequence, Tuple

import lightning as L
import numpy as np
import onnxruntime as ort
import plotly.express as px
import spacy
import torch
import torch.nn.functional as F
from loguru import logger
from numpy.typing import NDArray
from ollama import Client
from PyPDF2 import PdfReader
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from spacy.tokens import Token
from spellchecker import SpellChecker
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchmetrics import Accuracy, F1Score
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast

from src.confs import envs_conf, spacy_conf
from src.modules.file_system import file_system_svc
from src.modules.llm_prep import llm_prep_svc
from src.modules.llm_prep.llm_row_vo import LLMRowVo
from src.modules.vocabulary_prep import vocabulary_prep_svc
from src.modules.vocabulary_prep.vocabulary_vo import VocabularyVo


def testing_all_MiniLM_L6_v2_onnx_as_python():
    tokenizer: BertTokenizerFast = AutoTokenizer.from_pretrained(
        "../frontend/public/artifacts/all-MiniLM-L6-v2",
        clean_up_tokenization_spaces=True,
    )  # type: ignore
    ort_sess = ort.InferenceSession(
        "../frontend/public/artifacts/all-MiniLM-L6-v2/onnx/model.onnx",
        providers=["TensorrtExecutionProvider"],
    )
    for input in ort_sess.get_inputs():
        print("input", type(input), input)
    for output in ort_sess.get_outputs():
        print("output", type(output), output)

    input = "That is a happy person"
    input_tokenized = tokenizer(
        input, return_tensors="np", truncation=True, padding=True
    )
    input_ids: NDArray[np.int64] = input_tokenized["input_ids"]  # type: ignore
    print("input_ids", type(input_ids), type(input_ids[0][0]))
    attention_mask: NDArray[np.int64] = input_tokenized["attention_mask"]  # type: ignore
    print("attention_mask", type(attention_mask), type(attention_mask[0][0]))

    output: NDArray[np.float32] = ort_sess.run(
        ["sentence_embedding"],
        {"input_ids": input_ids, "attention_mask": attention_mask},
    )
    print("output", type(output), output)


testing_all_MiniLM_L6_v2_onnx_as_python()


def testing_all_MiniLM_L6_v2_installation_and_simple_usage():
    def get_embedding(
        tokenizer: BertTokenizerFast, model: BertModel, sentence: str
    ) -> torch.Tensor:
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
        model.eval()
        with torch.no_grad():
            outputs: BaseModelOutputWithPoolingAndCrossAttentions = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
        return embeddings

    def cosine_similarity_pytorch(
        tensor1: torch.Tensor, tensor2: torch.Tensor
    ) -> float:
        dot_product = torch.dot(tensor1, tensor2)
        norm1 = torch.norm(tensor1)
        norm2 = torch.norm(tensor2)
        return (dot_product / (norm1 * norm2)).item()

    tokenizer: BertTokenizerFast = AutoTokenizer.from_pretrained(
        # "sentence-transformers/all-MiniLM-L6-v2",
        "./draft/all_MiniLM_L6_v2/tokenizer_prod",
        # "draft/all_MiniLM_L6_v2/model/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/8b3219a92973c328a8e22fadcfa821b5dc75636a/config.json",
        # cache_dir="./draft/all_MiniLM_L6_v2/tokenizer",
        clean_up_tokenization_spaces=True,
    )  # type: ignore
    # tokenizer.save_pretrained("./draft/all_MiniLM_L6_v2/tokenizer_prod")
    model: BertModel = AutoModel.from_pretrained(
        # "sentence-transformers/all-MiniLM-L6-v2",
        "./draft/all_MiniLM_L6_v2/model_prod"
        # "./draft/all_MiniLM_L6_v2/model",
        # cache_dir="./draft/all_MiniLM_L6_v2/model",
    )  # type: ignore
    # model.save_pretrained("./draft/all_MiniLM_L6_v2/model_prod")

    input = "That is a happy person"
    input = get_embedding(tokenizer, model, input)

    docs = [
        "That is a happy dog",
        "That is a very happy person",
        "Today is a sunny day",
    ]
    docs = [get_embedding(tokenizer, model, doc) for doc in docs]

    scores = [cosine_similarity_pytorch(input, doc) for doc in docs]
    print(scores)


# testing_all_MiniLM_L6_v2_installation_and_simple_usage()


def testing_onnx_model_predict():
    def softmax(x):
        # Shift the input x by subtracting the maximum value for numerical stability
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    vocabulary = vocabulary_prep_svc.impl.compute_vocabulary()
    ort_sess = ort.InferenceSession(envs_conf.impl.model_filepath)
    input_name: str = ort_sess.get_inputs()[0].name
    output_name: str = ort_sess.get_outputs()[0].name
    while True:
        print("Enter text (type 'exit' to quit):")
        # Read a line from standard input
        line = sys.stdin.readline().strip()

        # Check for termination condition
        if line.lower() == "exit":
            print("Exiting...")
            break

        # Process the input line
        line = [word for word in line.split(" ") if word in vocabulary.root]
        if not line:
            continue
        print(f"You entered: {line}")
        input_data_vector: NDArray[np.float32] = np.array(
            [vocabulary.root[word] for word in line],
            dtype=np.float32,
        )
        input_data_vector = np.mean(input_data_vector, axis=0, keepdims=True)
        print(input_data_vector.shape)

        # Predict
        results = ort_sess.run([output_name], {input_name: input_data_vector})
        output_data = softmax(results[0])
        print(output_data)


# testing_onnx_model_predict()


def testing_build_torch_tensor():
    class DeQuelPartiLightningModule(L.LightningModule):
        def __init__(
            self,
            llm_rows: Sequence[LLMRowVo],
            vocabulary: VocabularyVo,
            val_split: float,
            batch_size: int,
            lr: float,
        ):
            super().__init__()
            self.llm_rows = llm_rows
            self.vocabulary = vocabulary
            self.val_split = val_split
            self.batch_size = batch_size
            self.lr = lr
            self._fc1 = nn.Linear(300, 256)
            self._fc2 = nn.Linear(256, 128)
            self._fc3 = nn.Linear(128, 3)
            self.dropout = nn.Dropout(0.5)
            self._loss_fn = nn.CrossEntropyLoss()
            self._accuracy = Accuracy(task="multiclass", num_classes=3)
            self._f1_score = F1Score(task="multiclass", num_classes=3, average="macro")

        def forward(self, x: Tensor):
            x = F.relu(self._fc1(x))
            x = self.dropout(x)
            x = F.relu(self._fc2(x))
            x = self.dropout(x)
            x = self._fc3(x)
            return x

        def prepare_data(self) -> None:
            feature_tensor = self._convert_llm_rows_to_feature_tensor(
                llm_rows, vocabulary
            )
            label_tensor = self._convert_llm_rows_to_label_tensor(llm_rows)
            self._tensor_dataset = TensorDataset(feature_tensor, label_tensor)

        def setup(self, stage: str) -> None:
            total_size = len(self._tensor_dataset)
            val_size = int(total_size * self.val_split)
            train_size = total_size - val_size
            train_dataset, val_dataset = random_split(
                self._tensor_dataset, [train_size, val_size]
            )
            self._train_loader = DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=11
            )
            self._val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=11
            )

        def train_dataloader(self) -> DataLoader[Tuple[Tensor, ...]]:
            return self._train_loader

        def val_dataloader(self) -> DataLoader[Tuple[Tensor, ...]]:
            return self._val_loader

        def training_step(self, batch: Tuple[Tensor, Tensor]) -> Tensor:
            x, y = batch
            y_pred = self.forward(x)
            loss: Tensor = self._loss_fn(y_pred, y)
            acc: Tensor = self._accuracy(y_pred, y)
            f1: Tensor = self._f1_score(y_pred, y)
            self.log_dict(
                {"train_loss": loss, "train_acc": acc, "train_f1": f1},
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            return loss

        def validation_step(self, batch: Tuple[Tensor, Tensor]) -> Tensor:
            x, y = batch
            y_pred = self.forward(x)
            loss: Tensor = self._loss_fn(y_pred, y)
            acc: Tensor = self._accuracy(y_pred, y)
            f1: Tensor = self._f1_score(y_pred, y)
            self.log_dict(
                {"val_loss": loss, "val_acc": acc, "val_f1": f1},
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            return loss

        def configure_optimizers(
            self,
        ) -> optim.Optimizer:  # type: ignore
            return optim.Adam(self.parameters(), lr=self.lr)  # type: ignore

        def _convert_llm_rows_to_feature_tensor(
            self, llm_rows: Sequence[LLMRowVo], vocabulary: VocabularyVo
        ) -> Tensor:
            logger.info("Converting feature from LLM rows to tensor.")
            tokens_per_feature = [
                list(spacy_conf.impl.spacy(llm_row.feature))
                for llm_row in tqdm(llm_rows)
            ]
            tensor_per_feature = [
                torch.vstack(
                    [
                        torch.FloatTensor(vocabulary.root[token.text])
                        for token in tokens
                        if token.text in vocabulary.root
                    ]
                ).mean(dim=0)
                for tokens in tqdm(tokens_per_feature)
            ]
            tensor = torch.vstack(tensor_per_feature)
            return tensor

        def _convert_llm_rows_to_label_tensor(
            self,
            llm_rows: Sequence[LLMRowVo],
        ) -> torch.Tensor:
            logger.info("Converting label from LLM rows to tensor.")
            unique_labels = list(set([llm_row.label for llm_row in llm_rows]))
            tensor = torch.zeros(len(llm_rows), dtype=torch.long)
            for i, llm_row in enumerate(tqdm(llm_rows)):
                label_enc_idx = unique_labels.index(llm_row.label)
                tensor[i] = label_enc_idx
            return tensor

    def build_and_train_dnn(
        llm_rows: Sequence[LLMRowVo],
        vocabulary: VocabularyVo,
        val_split=0.30,
        batch_size=64,
        lr=0.001,
    ) -> DeQuelPartiLightningModule:
        model = DeQuelPartiLightningModule(
            llm_rows, vocabulary, val_split, batch_size, lr
        )
        # early_stop_callback_f1 = EarlyStopping(
        #     monitor="val_f1",  # Monitor validation F1 score
        #     min_delta=0.00,
        #     patience=3,
        #     verbose=True,
        #     mode="max",  # 'max' because we want to maximize F1 score
        # )
        trainer = L.Trainer(max_epochs=-1, logger=False, enable_checkpointing=False)
        trainer.fit(model)
        trainer.validate(model)
        return model

    vocabulary = vocabulary_prep_svc.impl.compute_vocabulary()
    pdfs = file_system_svc.impl.read_pdfs(envs_conf.impl.input_dirpath)
    llm_rows = llm_prep_svc.impl.compute_llm_rows(pdfs, vocabulary)
    dequelparti_dnn = build_and_train_dnn(llm_rows, vocabulary)


# testing_build_torch_tensor()


def testing_pickle_sha3_512():
    pdfs = file_system_svc.impl.read_pdfs(envs_conf.impl.input_dirpath)
    byte_data = pickle.dumps(pdfs)
    print(len(byte_data))
    test = hashlib.sha3_512(byte_data).hexdigest()
    print(test)


# testing_pickle_sha3_512()


def testing_vocabulary_compute():
    vocabulary = vocabulary_prep_svc.impl.compute_vocabulary()


# testing_vocabulary_compute()


def testing_spacy_word_vector_to_list_float():
    nlp = spacy.load("fr_core_news_lg")
    word = "avion"
    vector_python: List[float] = nlp.vocab[word].vector.tolist()
    print(type(vector_python[0]))
    print(vector_python)


# testing_spacy_word_vector_to_list_float()


def testing_context_manager():
    @contextmanager
    def get_prediction() -> Generator[None, None, None]:
        print("get_prediction")
        try:
            print("try")
            yield
        except:
            print("except")
        else:
            print("else")
        finally:
            print("finally")

    with get_prediction():
        print("with")
        return


# testing_context_manager()


def testing_sha3_512_for_filename():
    print(sha3_512("".encode()).hexdigest())
    print(
        sha3_512(
            "697f2d856172cb8309d6b8b97dac4de344b549d4dee61edfb4962d8698b7fa803f4f93ff24393586e28b5b957ac3d1d369420ce53332712f997bd336d09ab02a697f2d856172cb8309d6b8b97dac4de344b549d4dee61edfb4962d8698b7fa803f4f93ff24393586e28b5b957ac3d1d369420ce53332712f997bd336d09ab02a697f2d856172cb8309d6b8b97dac4de344b549d4dee61edfb4962d8698b7fa803f4f93ff24393586e28b5b957ac3d1d369420ce53332712f997bd336d09ab02a697f2d856172cb8309d6b8b97dac4de344b549d4dee61edfb4962d8698b7fa803f4f93ff24393586e28b5b957ac3d1d369420ce53332712f997bd336d09ab02a697f2d856172cb8309d6b8b97dac4de344b549d4dee61edfb4962d8698b7fa803f4f93ff24393586e28b5b957ac3d1d369420ce53332712f997bd336d09ab02a697f2d856172cb8309d6b8b97dac4de344b549d4dee61edfb4962d8698b7fa803f4f93ff24393586e28b5b957ac3d1d369420ce53332712f997bd336d09ab02a".encode()
        ).hexdigest()
    )


# testing_sha3_512_for_filename()


def testing_ollama_same_model():
    def read_pdf_pages(path: str) -> List[str]:
        prgm = [page.extract_text() for page in PdfReader(path).pages]
        return prgm

    def gemma2_predict(page: str) -> str:
        ollama = Client(host="http://localhost:11434")
        prompt = f"""{page}

Pour le texte ci-dessus, dans le cadre de l'augmentation de données pour entraîner un réseau de neurones dense:
- Génère environ 25 conséquences positives de ces mesures
- Une réponse EXACTEMENT FORMATTER "- conséquence0\n- conséquence1\n ... - conséquence25\n"
- Pas d'entêtes ou autres artéfacts dans ta réponse, il faut VRAIMENT RESPECTER LE FROMAT "- conséquence0\n- conséquence1\n ... - conséquence25\n"
"""
        response: Mapping[str, Any] = ollama.chat(
            model="gemma2:9b",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )  # type: ignore
        assitent_content: str = response["message"]["content"]
        return assitent_content

    prgm_nfp = "../frontend/public/programs/program_nouveau_front_populaire.pdf"
    pages = read_pdf_pages(prgm_nfp)
    for page in tqdm(pages[7:9]):
        print("##### ORIGINAL DATA #####")
        print(page)
        out_page = gemma2_predict(page)
        print("##### TRANSFORMED DATA #####")
        print(out_page)


# testing_ollama_same_model()


def testing_ollama():
    def read_pdf_pages(path: str) -> List[str]:
        prgm = [page.extract_text() for page in PdfReader(path).pages]
        return prgm

    def gemma2b_predict(page: str) -> str:
        ollama = Client(host="http://localhost:11434")
        prompt = f"""{page}

Ce text provient d'une extraction brut via PyPDF2. Pour le text ci-dessus:

- Génère moi un paragraph de phrases qui résument
- Qui explique les causes de ce text
- Qui explique les conséquences de ce text
- Bref, un bon paragraph clean pour faire de la data augmentation
- Ne pas hésiter à renvoyer un string vide si y'a rien il y'a pas assez de contenu pour faire une bonne génération
- Pas d'artéfacts dans ta réponse, il faut que je puisse directement copier et coller ce UNIQUE PARAGRAPHE
"""
        response: Mapping[str, Any] = ollama.chat(
            model="gemma2:9b",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )  # type: ignore
        assitent_content: str = response["message"]["content"]
        return assitent_content

    prgm_nfp = "../frontend/public/programs/program_nouveau_front_populaire.pdf"
    pages = read_pdf_pages(prgm_nfp)
    for page in tqdm(pages):
        print("ORIGINAL DATA:")
        print(page)
        out_page = gemma2b_predict(page)
        print("TRANSFORMED DATA:")
        print(out_page)


# testing_ollama()


def testing_dataviz_on_words():
    nlp = spacy.load("fr_core_news_lg")
    spell_checker = SpellChecker(language="fr")
    manual_stop_words = {"l", "etc."}

    def read_pdf_pages(path: str) -> List[str]:
        prgm = [page.extract_text() for page in PdfReader(path).pages]
        return prgm

    def get_tokens(pages: List[str]) -> List[Token]:
        tokens = [
            token
            for page in pages
            for token in nlp(page)
            if token.is_stop is False
            and token.is_punct is False
            and token.is_space is False
            and token.has_vector
            and token.text not in manual_stop_words
            and token.lemma_ not in manual_stop_words
        ]
        unique_words = set([token.text for token in tokens])
        unique_words_correctly_spelled = spell_checker.known(unique_words)
        tokens = [
            token for token in tokens if token.text in unique_words_correctly_spelled
        ]
        return tokens

    def sliding_avg_windows(
        tokens: List[Token], window: int = 100, stride: int = 25
    ) -> List[NDArray[np.float64]]:
        windows: List[NDArray[np.float64]] = []
        for i in range(0, len(tokens) - window, stride):
            window_tokens = tokens[i : i + window]
            window_vectors = [np.array(token.vector) for token in window_tokens]
            window_avg_vector = np.vstack(window_vectors)
            window_avg_vector = np.mean(window_avg_vector, axis=0)
            windows.append(window_avg_vector)
        return windows

    def build_and_save_plot_using_pca(
        path: str, avg_windows: List[NDArray[np.float64]], colors: List[str]
    ) -> None:
        logger.info(f"{len(avg_windows)} data points with {len(set(colors))} classes.")
        avg_windows_ = np.vstack(avg_windows)
        scaler = StandardScaler()
        scaled_data: NDArray[np.float64] = scaler.fit_transform(avg_windows_)
        pca = PCA(n_components=3)
        pca_transformed_data: NDArray[np.float64] = pca.fit_transform(scaled_data)
        x = pca_transformed_data[:, 0]
        y = pca_transformed_data[:, 1]
        z = pca_transformed_data[:, 2]
        fig = px.scatter_3d(x=x, y=y, z=z, color=colors, title=path)
        fig.write_html(path)
        fig.show()

    prgms_path = [
        "../frontend/public/programs/program_nouveau_front_populaire.pdf",
        "../frontend/public/programs/program_ensemble.pdf",
        "../frontend/public/programs/program_rassemblement_national.pdf",
    ]

    prgms_pages = [read_pdf_pages(path) for path in prgms_path]
    prgms_tokens = [get_tokens(pages) for pages in prgms_pages]
    prgms_avg_windows = [sliding_avg_windows(tokens) for tokens in prgms_tokens]
    avg_windows = [window for windows in prgms_avg_windows for window in windows]
    colors = [
        os.path.basename(path)
        for path, windows in zip(prgms_path, prgms_avg_windows)
        for _ in windows
    ]
    build_and_save_plot_using_pca(
        "/mnt/d/lovin/Downloads/testing_dataviz_on_words.html", avg_windows, colors
    )


# testing_dataviz_on_words()


def testing_some_words_by_fixing_got_spellchecked():
    spell_checker = SpellChecker(language="fr")
    word = "interdirons"

    print(spell_checker.correction(word))


# testing_some_words_by_fixing_got_spellchecked()


def testing_vector_word_but_not_lemma():
    nlp = spacy.load("fr_core_news_lg")
    word = "interdirons"

    lexeme = nlp.vocab[word]
    print(lexeme.has_vector)
    print(lexeme.vector)

    token = nlp(word)[0]
    print(token.has_vector)
    print(token.vector)

    lemma = nlp(word)[0].lemma_
    print(lemma)
    print(nlp.vocab[lemma].has_vector)
    print(nlp.vocab[lemma].vector)


# testing_vector_word_but_not_lemma()


def testing_not_removing_stop_words():  # NO STOP WORDS BECAUSE IT CAUSES VECTOR ANGLES TO BE MORE CLOSE
    def angle_between_vectors(u: NDArray[np.float64], v: NDArray[np.float64]) -> float:
        # Compute the dot product
        dot_product = np.dot(u, v)

        # Compute the norms (magnitudes) of the vectors
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)

        # Compute the cosine of the angle
        cos_theta = dot_product / (norm_u * norm_v)

        # Handle potential floating point errors
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        # Compute the angle in radians
        angle_rad = np.arccos(cos_theta)

        # Convert radians to degrees
        angle_deg = np.degrees(angle_rad)

        return angle_deg

    def compute_avg_1darray_of_tokens(tokens: List[Token]) -> NDArray[np.float64]:
        avg_vector = np.vstack([np.array(token.vector) for token in tokens])
        return np.mean(avg_vector, axis=0)

    nlp = spacy.load("fr_core_news_lg")
    spell_checker = SpellChecker(language="fr")
    tokens = nlp("Nous interdirons l’accès aux réseaux sociaux avant 15 ans")
    random_tokens = nlp("Augmenter les salaires par le passage du SMIC à 1600€ net")

    token_without_stop_words = [
        token
        for token in tokens  # spell_checker.known([token.text]) and
        if token.is_stop is False
        # and token.is_punct is False
        and token.is_space is False and token.has_vector
    ]

    for token in token_without_stop_words:
        print(token)

    token_with_stop_words = [
        token
        for token in tokens  # spell_checker.known([token.text]) and
        # and token.is_punct is False
        if token.is_space is False and token.has_vector
    ]

    random_tokens = [
        token
        for token in random_tokens  # spell_checker.known([token.text]) and
        # and token.is_punct is False
        if token.is_space is False and token.has_vector
    ]

    for token in token_with_stop_words:
        print(token)

    avg_token_without_stop_words = compute_avg_1darray_of_tokens(
        token_without_stop_words
    )
    avg_token_with_stop_words = compute_avg_1darray_of_tokens(token_with_stop_words)
    print(
        angle_between_vectors(avg_token_without_stop_words, avg_token_with_stop_words)
    )

    avg_random_tokens = compute_avg_1darray_of_tokens(random_tokens)
    print(
        "without stop words:",
        angle_between_vectors(avg_random_tokens, avg_token_without_stop_words),
    )
    print(
        "with stop words:",
        angle_between_vectors(avg_random_tokens, avg_token_with_stop_words),
    )


# testing_not_removing_stop_words()


def testing_text_blob():
    ...
    """
    word = Word("dormira")
    print(word.spellcheck())
    """


# testing_text_blob()


def testing_has_vector_and_spellchecked():
    nlp = spacy.load("fr_core_news_lg")
    spell_checker = SpellChecker(language="fr")

    vocabulary = [
        word for word in tqdm(nlp.vocab.strings) if nlp.vocab.has_vector(word)
    ]
    print(len(vocabulary))

    vocabulary = [
        word
        for word in tqdm(vocabulary)
        if (lexeme := nlp.vocab[word])
        and lexeme.is_stop is False
        and lexeme.is_punct is False
        and lexeme.is_space is False
    ]
    print(len(vocabulary))

    vocabulary = spell_checker.known(vocabulary)
    print(len(vocabulary))

    first_letter_vocabulary = set([word[0] for word in vocabulary])
    print(len(first_letter_vocabulary))
    print(sorted(list(first_letter_vocabulary)))


# testing_has_vector_and_spellchecked()


def testing_get_all_keys_with_vectors():
    nlp = spacy.load("fr_core_news_lg")
    # nlp = spacy.load("fr_core_news_md")

    vocabulary = list(nlp.vocab.strings)
    print(len(vocabulary))

    word = vocabulary[434333]
    print(word)

    vector = nlp.vocab.get_vector(word)
    print(vector)

    vocabulary_has_vector = [word for word in vocabulary if nlp.vocab.has_vector(word)]
    print(len(vocabulary_has_vector))

    # for word in vocabulary:
    #     if nlp.vocab.has_vector(word) is True:
    #         print(word)


# testing_get_all_keys_with_vectors()


def testing_if_same_word_embeddings_lemmas_vs_word():
    nlp = spacy.load("fr_core_news_lg")
    doc0 = nlp("expulsé expulser")
    token0, token1 = doc0
    print(f"WORD: {token0.text} | LEMMA: {token0.lemma_}")
    print(f"WORD: {token1.text} | LEMMA: {token1.lemma_}")
    vec0 = [float(x) for x in token0.vector]
    vec1 = [float(x) for x in token1.vector]
    print(vec0 == vec1)
    print("#######")
    print(vec0)
    print("#######")
    print(vec1)


# testing_if_same_word_embeddings_lemmas_vs_word()


def testing_word_embedding():
    nlp = spacy.load("fr_core_news_lg")
    doc0 = nlp("expulser")
    for token in doc0:
        test = list(token.vector)
        print(test)


# testing_word_embedding()


def testing_cosine_sim():
    nlp = spacy.load("fr_core_news_lg")
    doc0 = nlp("protéger immigrés")
    doc1 = nlp("limiter migration")
    print(doc0.similarity(doc1))


# testing_cosine_sim()


def testing_no_word_embeddings():
    nlp = spacy.load("fr_core_news_lg")
    for token in nlp("vieaaaaaa abaissement l etc. mercure"):
        print(token.text, token.lemma_, token.has_vector)


# testing_no_word_embeddings()


def testing_spacy_sim_score():
    nlp = spacy.load("fr_core_news_lg")
    print(nlp("abaissement").similarity(nlp("vieaaaaaa")))


# testing_spacy_sim_score()


def testing_spacy_vocab():
    nlp = spacy.load("fr_core_news_lg")
    for lexeme in nlp.vocab:
        print(lexeme.text)


# testing_spacy_vocab()


def testing_spacy_oov():
    nlp = spacy.load("fr_core_news_lg")
    tokens = nlp("BARDELLA")
    for token in tokens:
        print(token.is_oov)


# testing_spacy_oov()


def testing_get_all_values_from_dict():
    my_dict = {"a": 1, "b": 2, "c": 2}
    values = my_dict.values()
    print(len(set(values)))


# testing_get_all_values_from_dict()


def testing_generics_and_abc():
    @dataclass
    class VizFormDto(ABC): ...

    @dataclass
    class VizDataDto(ABC): ...

    @dataclass
    class ChartSvc[I: VizFormDto, O: VizDataDto](ABC):
        @abstractmethod
        def compute_viz_data(self, viz_form: I) -> O: ...

    @dataclass
    class ChartCtrl[I: VizFormDto, O: VizDataDto](ABC):
        @property
        @abstractmethod
        def chart_svc(self) -> ChartSvc[I, O]: ...

        def compute_viz_data(self, viz_form: I) -> O:
            viz_data = self.chart_svc.compute_viz_data(viz_form)
            return viz_data

    @dataclass
    class VizFormVerticalBarsDto(VizFormDto): ...

    @dataclass
    class VizDataVerticalBarsDto(VizDataDto): ...

    @dataclass
    class ChartVerticalBarsSvc(
        ChartSvc[VizFormVerticalBarsDto, VizDataVerticalBarsDto]
    ):
        def compute_viz_data(
            self, viz_form: VizFormVerticalBarsDto
        ) -> VizDataVerticalBarsDto: ...

    impl0 = ChartVerticalBarsSvc()

    @dataclass
    class ChartVerticalBarsCtrl(
        ChartCtrl[VizFormVerticalBarsDto, VizDataVerticalBarsDto]
    ):
        @property
        def chart_svc(self) -> ChartVerticalBarsSvc:
            return impl0

    impl0_ctrl = ChartVerticalBarsCtrl()

    @dataclass
    class VizFormPieDto(VizFormDto): ...

    @dataclass
    class VizDataPieDto(VizDataDto): ...

    @dataclass
    class ChartPieSvc(ChartSvc[VizFormPieDto, VizDataPieDto]):
        def compute_viz_data(
            self, viz_form: VizFormVerticalBarsDto
        ) -> VizDataVerticalBarsDto: ...

    impl1 = ChartPieSvc()


# testing_generics_and_abc()
