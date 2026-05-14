import asyncio
import logging
import re
import time
from typing import Tuple, List, Optional, Any

import numpy as np
from .vosk_g2p import convert

log = logging.getLogger(__name__)


class VoskEngine:
    """Движок синтеза речи на базе Vosk и ONNX Runtime."""

    def __init__(self, model):
        self.model = model
        self._lock = asyncio.Lock()
        self.sample_rate = 22050
        self.sample_width = 2
        self.channels = 1
        self.num_speakers = self.model.config.get("num_speakers", 5)

    def audio_float_to_int16(self, audio: np.ndarray) -> np.ndarray:
        """Конвертация float32 аудио в PCM 16-бит."""
        return np.clip(audio * 32767.0, -32767.0, 32767.0).astype("int16")

    def get_word_bert(self, text: str, nopunc: bool = False) -> np.ndarray:
        """Извлечение эмбеддингов BERT для слов."""
        tokens = self.model.tokenizer.encode(text.replace("+", ""))
        
        # Инференс BERT
        bert = self.model.bert_onnx.run(None, {
            "input_ids": [tokens.ids],
            "attention_mask": [tokens.attention_mask],
            "token_type_ids": [tokens.type_ids],
        })[0]

        # Фильтрация токенов (исключаем подслова ## и пунктуацию)
        punct_pattern = r"[-,.?!;:\"]"
        selected = [
            i for i, t in enumerate(tokens.tokens)
            if t[0] != '#' and not (nopunc and re.match(punct_pattern, t))
        ]
        return bert[selected]

    def _sync_synthesize(
        self,
        text: str,
        speaker_id: int = 0,
        speech_rate: float = 1.0
    ) -> Optional[bytes]:
        """Синхронный запуск инференса модели."""
        inf_cfg = self.model.config.get("inference", {})
        noise_level = inf_cfg.get("noise_level", 0.8)
        duration_noise_level = inf_cfg.get("duration_noise_level", 0.8)
        scale = inf_cfg.get("scale", 1.0)

        # Очистка текста от разных видов тире
        text = text.strip()
        for dash in ["—", "–", "−"]:
            text = text.replace(dash, "-")

        if not text:
            return None

        model_type = self.model.config.get("model_type")
        has_tokenizer = self.model.tokenizer is not None

        # 1. G2P и подготовка тензоров в зависимости от типа модели
        if has_tokenizer and model_type == "multistream_v2":
            bert_embs_raw = self.get_word_bert(text, nopunc=True)
            phoneme_ids, bert_embs = self.g2p_multistream(
                text, bert_embs_raw, word_pos=True
            )
            text_tensor = np.expand_dims(
                np.transpose(np.array(phoneme_ids, dtype=np.int64)), 0
            )
        elif has_tokenizer and model_type == "multistream_v1":
            bert_embs_raw = self.get_word_bert(text, nopunc=True)
            phoneme_ids, bert_embs = self.g2p_multistream(text, bert_embs_raw)
            text_tensor = np.expand_dims(
                np.transpose(np.array(phoneme_ids, dtype=np.int64)), 0
            )
        elif has_tokenizer and self.model.config.get("no_blank", 0) != 0:
            phoneme_ids, bert_embs = self.g2p_noblank(
                text, self.get_word_bert(text)
            )
            text_tensor = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)
        elif has_tokenizer:
            phoneme_ids, bert_embs = self.g2p(text, self.get_word_bert(text))
            text_tensor = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)
        else:
            phoneme_ids = self.g2p_noembed(text)
            bert_embs = np.zeros((1, 768, len(phoneme_ids)), dtype=np.float32)
            text_tensor = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)

        # 2. Формирование аргументов для ONNX
        scales = np.array(
            [noise_level, 1.0 / speech_rate, duration_noise_level],
            dtype=np.float32
        )
        
        onnx_inputs = {
            "input": text_tensor,
            "input_lengths": np.array([text_tensor.shape[-1]], dtype=np.int64),
            "scales": scales,
            "sid": np.array([speaker_id], dtype=np.int64),
        }

        if has_tokenizer:
            onnx_inputs["bert"] = np.expand_dims(
                np.transpose(np.array(bert_embs, dtype=np.float32)), 0
            )

        # 3. Запуск инференса
        audio = self.model.onnx.run(None, onnx_inputs)[0].squeeze() * scale
        return self.audio_float_to_int16(audio).tobytes()

    async def synthesize(
        self,
        text: str,
        speaker_id: int,
        speech_rate: float
    ) -> bytes:
        """Асинхронная обертка над синтезом."""
        async with self._lock:
            return await asyncio.to_thread(
                self._sync_synthesize, text, speaker_id, speech_rate
            )

    def g2p(self, text: str, embeddings: np.ndarray):
        pattern = r"([,.?!;:\"() ])"
        phonemes = ["^"]
        phone_embeddings = [embeddings[0]]
        word_idx = 1
        
        for word in re.split(pattern, text.lower()):
            if not word:
                continue
            if re.match(pattern, word) or word == '-':
                phonemes.append(word)
                phone_embeddings.append(embeddings[word_idx])
            elif word in self.model.dic:
                for p in self.model.dic[word].split():
                    phonemes.append(p)
                    phone_embeddings.append(embeddings[word_idx])
            else:
                for p in convert(word).split():
                    phonemes.append(p)
                    phone_embeddings.append(embeddings[word_idx])
            if word != " ":
                word_idx += 1
                
        phonemes.append("$")
        phone_embeddings.append(embeddings[-1])

        # Добавление бланк-символов (0)
        phoneme_id_map = self.model.config["phoneme_id_map"]
        phoneme_ids = [phoneme_id_map[phonemes[0]]]
        phone_embs_is = [phone_embeddings[0]]
        
        for i in range(1, len(phonemes)):
            phoneme_ids.append(0)
            phoneme_ids.append(phoneme_id_map[phonemes[i]])
            phone_embs_is.append(phone_embeddings[i])
            phone_embs_is.append(phone_embeddings[i])

        return phoneme_ids, phone_embs_is

    def g2p_noblank(self, text: str, embeddings: np.ndarray):
        pattern = r"([,.?!;:\"() ])"
        phonemes = ["^"]
        phone_embeddings = [embeddings[0]]
        word_idx = 1
        
        for word in re.split(pattern, text.lower()):
            if not word:
                continue
            if re.match(pattern, word) or word == '-':
                phonemes.append(word)
                phone_embeddings.append(embeddings[word_idx])
            elif word in self.model.dic:
                for p in self.model.dic[word].split():
                    phonemes.append(p)
                    phone_embeddings.append(embeddings[word_idx])
            else:
                for p in convert(word).split():
                    phonemes.append(p)
                    phone_embeddings.append(embeddings[word_idx])
            if word != " ":
                word_idx += 1
                
        phonemes.append("$")
        phone_embeddings.append(embeddings[-1])

        phoneme_id_map = self.model.config["phoneme_id_map"]
        phoneme_ids = [phoneme_id_map[p] for p in phonemes]

        return phoneme_ids, phone_embeddings

    def g2p_noembed(self, text: str):
        pattern = r"([,.?!;:\"() ])"
        phonemes = ["^"]
        
        for word in re.split(pattern, text.lower()):
            if not word:
                continue
            if re.match(pattern, word) or word == '-':
                phonemes.append(word)
            elif word in self.model.dic:
                for p in self.model.dic[word].split():
                    phonemes.append(p)
            else:
                for p in convert(word).split():
                    phonemes.append(p)
        phonemes.append("$")

        phoneme_id_map = self.model.config["phoneme_id_map"]
        start_p = phoneme_id_map[phonemes[0]]
        
        if isinstance(start_p, list):
            phoneme_ids = []
            phoneme_ids.extend(start_p)
            for i in range(1, len(phonemes)):
                phoneme_ids.append(0)
                phoneme_ids.extend(phoneme_id_map[phonemes[i]])
        else:
            phoneme_ids = [start_p]
            for i in range(1, len(phonemes)):
                phoneme_ids.append(0)
                phoneme_ids.append(phoneme_id_map[phonemes[i]])

        return phoneme_ids

    def add_pos(self, x: List[str]) -> List[str]:
        """Добавление B/I/E/S тегов позиции в слове."""
        if len(x) == 1:
            return [x[0] + "_S"]

        res = []
        for i, p in enumerate(x):
            if i == 0:
                res.append(p + "_B")
            elif i == len(x) - 1:
                res.append(p + "_E")
            else:
                res.append(p + "_I")
        return res

    def g2p_multistream(self, text: str, bert_embs: np.ndarray, word_pos: bool = False):
        phonemes = [("^", [], 0, 0)]
        pattern = r"(\.\.\.|- |[ ,.?!;:\"()])"
        
        # Унификация тире
        text = text.replace(" -", "- ")

        in_quote = 0
        cur_punc = []
        bert_word_idx = 1

        for word in re.split(pattern, text.lower()):
            if not word:
                continue

            if word == "\"":
                in_quote = 1 if in_quote == 0 else 0
                continue

            if word in ["- ", "-"]:
                cur_punc.append('-')
                continue

            if re.match(pattern, word) and word != " ":
                cur_punc.append(word)
                continue

            if word == " ":
                phonemes.append((' ', cur_punc, in_quote, bert_word_idx))
                cur_punc = []
                continue

            # Фонетизация слова
            if word in self.model.dic:
                word_ph = self.model.dic[word].split()
            else:
                word_ph = convert(word).split()

            if word_pos:
                word_ph = self.add_pos(word_ph)

            for p in word_ph:
                phonemes.append((p, [], in_quote, bert_word_idx))

            cur_punc = []
            bert_word_idx += 1

        phonemes.append((" ", cur_punc, in_quote, bert_word_idx))
        phonemes.append(("$", [], 0, bert_word_idx))

        last_punc = " "
        last_sent_punc = " "
        lp_phonemes = []
        phone_bert_embs = []
        ph_id_map = self.model.config["phoneme_id_map"]

        # Обработка в обратном порядке для определения контекста пунктуации
        for p in reversed(phonemes):
            punc_list = p[1]
            if "..." in punc_list:
                last_sent_punc = "..."
            elif "." in punc_list:
                last_sent_punc = "."
            elif "!" in punc_list:
                last_sent_punc = "!"
            elif "?" in punc_list:
                last_sent_punc = "?"
            elif "-" in punc_list:
                last_sent_punc = "-"

            if punc_list:
                last_punc = punc_list[0]
                cur_punc_val = punc_list[0]
            else:
                cur_punc_val = "_"

            # Сборка фичей фонемы
            lp_phonemes.append((
                ph_id_map[p[0]],
                ph_id_map[cur_punc_val],
                p[2],
                ph_id_map[last_punc],
                ph_id_map[last_sent_punc]
            ))
            phone_bert_embs.append(bert_embs[p[3]])

        return list(reversed(lp_phonemes)), list(reversed(phone_bert_embs))