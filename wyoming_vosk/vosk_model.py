import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional, List, Dict
from zipfile import ZipFile
from urllib.request import urlretrieve

import onnxruntime
import requests
from tqdm import tqdm

try:
    from tokenizers.implementations import BertWordPieceTokenizer
except ImportError:
    BertWordPieceTokenizer = None

log = logging.getLogger(__name__)

# Константы путей поиска моделей
MODEL_PRE_URL = "https://alphacephei.com/vosk/models/"
MODEL_LIST_URL = MODEL_PRE_URL + "model-list.json"
MODEL_DIRS = [
    os.getenv("VOSK_MODEL_PATH"),
    Path("/usr/share/vosk"),
    Path.home() / "AppData/Local/vosk",
    Path.home() / ".cache/vosk",
]


class VoskModel:
    """Класс для управления моделью Vosk, словарем и BERT-токенизатором."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_name: Optional[str] = None,
        lang: str = "ru",
        provider: Optional[str] = None,
    ):
        """
        Инициализация модели.

        :param model_path: Прямой путь к папке модели.
        :param model_name: Имя модели для поиска/скачивания.
        :param lang: Язык модели.
        :param provider: Явное указание провайдера ONNX (напр. 'CUDAExecutionProvider').
        """
        if model_path is None:
            self.model_path = self.get_model_path(model_name, lang)
        else:
            self.model_path = Path(model_path)

        # 1. Настройка ONNX провайдеров
        available_providers = onnxruntime.get_available_providers()
        if provider and provider in available_providers:
            providers = [provider]
        else:
            # Авто-выбор: CUDA если есть, иначе CPU
            providers = [
                p for p in available_providers
                if p in ["CUDAExecutionProvider", "CPUExecutionProvider"]
            ]

        # 2. Оптимизация сессии ONNX
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        sess_options.enable_mem_pattern = True

        log.info("Loading Vosk model from %s", self.model_path)
        log.info("Using ONNX providers: %s", providers)

        # Инициализация основной TTS модели
        self.onnx = onnxruntime.InferenceSession(
            str(self.model_path / "model.onnx"),
            sess_options=sess_options,
            providers=providers,
        )

        # 3. Загрузка текстового словаря
        self.dic = self._load_dictionary(self.model_path)

        # 4. Загрузка конфигурации модели
        with open(self.model_path / "config.json", encoding="utf-8") as f:
            self.config = json.load(f)

        # 5. Инициализация BERT (если предусмотрено моделью)
        self.tokenizer = None
        self.bert_onnx = None
        bert_vocab = self.model_path / "bert/vocab.txt"
        bert_model = self.model_path / "bert/model.onnx"

        if BertWordPieceTokenizer and bert_vocab.exists() and bert_model.exists():
            log.info("Loading BERT tokenizer and model...")
            self.tokenizer = BertWordPieceTokenizer(
                vocab=str(bert_vocab),
                unk_token="[UNK]",
                lowercase=False
            )
            self.bert_onnx = onnxruntime.InferenceSession(
                str(bert_model),
                sess_options=sess_options,
                providers=providers,
            )

    def _load_dictionary(self, model_path: Path) -> Dict[str, str]:
        """Читает текстовый файл словаря и выбирает варианты с лучшей вероятностью."""
        dict_file = model_path / "dictionary"
        
        log.info("Parsing dictionary...")
        new_dic = {}
        probs = {}

        # Читаем построчно для экономии памяти
        with open(dict_file, encoding="utf-8") as f:
            for line in f:
                parts = line.split(maxsplit=2)
                if len(parts) < 3:
                    continue
                
                word, prob_str, phonemes = parts
                prob = float(prob_str)
                
                # Если слово встретилось впервые или у него выше вероятность
                if word not in probs or prob > probs[word]:
                    new_dic[word] = phonemes.strip()
                    probs[word] = prob

        return new_dic

    def get_model_path(self, model_name: Optional[str], lang: str) -> Path:
        """Ищет путь к модели локально или скачивает ее при необходимости."""
        # 1. Поиск в локальных директориях
        for directory in [d for d in MODEL_DIRS if d and Path(d).exists()]:
            for model_file in os.listdir(directory):
                path = Path(directory, model_file)
                # Если указано имя — ищем по точному совпадению
                if model_name and model_file == model_name:
                    return path
                # Если имя не указано — ищем по префиксу и языку
                if not model_name and re.match(
                    rf"vosk-model(-small)?-{lang}", model_file
                ):
                    return path

        # 2. Если не нашли локально — ищем в репозитории Vosk
        log.info("Model not found locally. Searching remotely...")
        try:
            response = requests.get(MODEL_LIST_URL, timeout=10).json()
        except Exception as e:
            log.error("Failed to fetch model list: %s", e)
            sys.exit(1)
        
        target = None
        if model_name:
            target = next((m for m in response if m["name"] == model_name), None)
        else:
            target = next(
                (m for m in response 
                 if m["lang"] == lang and m["type"] == "small"), None
            )

        if not target:
            log.error("Model %s (lang: %s) not found in remote list.", model_name, lang)
            sys.exit(1)

        # Скачиваем в последнюю директорию из списка (обычно .cache)
        dest_dir = Path(MODEL_DIRS[-1])
        dest_dir.mkdir(parents=True, exist_ok=True)
        model_path = dest_dir / target["name"]
        
        self._download_and_extract(target["name"], dest_dir)
        return model_path

    def _download_and_extract(self, name: str, dest_dir: Path):
        """Скачивает архив модели и распаковывает его."""
        url = f"{MODEL_PRE_URL}{name}.zip"
        zip_path = dest_dir / f"{name}.zip"

        log.info("Downloading model from %s", url)
        with tqdm(unit="B", unit_scale=True, unit_divisor=1024, miniters=1) as t:
            def reporthook(b=1, bsize=1, tsize=None):
                if tsize:
                    t.total = tsize
                t.update((b - t.n / bsize) * bsize)
            urlretrieve(url, str(zip_path), reporthook=reporthook)

        log.info("Extracting model...")
        with ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(dest_dir)
        
        # Удаляем архив после распаковки
        zip_path.unlink()