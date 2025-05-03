# --- START OF FILE speech_tts.py ---

import logging
import asyncio
import contextlib
import re
from num2words import num2words

from vosk_tts import Model, Synth

log = logging.getLogger(__name__)

# Параметры аудио по умолчанию для Vosk TTS моделей
# ВАЖНО: Убедитесь, что эти значения соответствуют вашей модели!
DEFAULT_SAMPLE_RATE = 22050
DEFAULT_SAMPLE_WIDTH = 2  # 16 бит = 2 байта
DEFAULT_CHANNELS = 1      # Моно

# Словарь для транслитерации английских букв в русские
ENGLISH_TO_RUSSIAN = {
    'a': 'а', 'b': 'б', 'c': 'к', 'd': 'д', 'e': 'е', 'f': 'ф', 'g': 'г',
    'h': 'х', 'i': 'и', 'j': 'ж', 'k': 'к', 'l': 'л', 'm': 'м', 'n': 'н',
    'o': 'о', 'p': 'п', 'q': 'к', 'r': 'р', 's': 'с', 't': 'т', 'u': 'у',
    'v': 'в', 'w': 'в', 'x': 'х', 'y': 'ай', 'z': 'з'
}

class SpeechTTS:
    """
    Класс для обработки TTS с использованием Vosk с предзагрузкой модели
    и возвратом аудио данных в памяти.
    """

    def __init__(self, vosk_model_name: str) -> None:
        """
        Инициализация с предзагрузкой модели Vosk TTS.
        Args:
            vosk_model_name: Имя или путь к модели Vosk TTS.
        Raises:
            RuntimeError: Если не удалось загрузить модель или создать синтезатор.
        """
        log.info(f"Initializing Vosk Speech TTS and preloading model: {vosk_model_name}")
        self.vosk_model_name = vosk_model_name
        self._lock = asyncio.Lock()  # Блокировка для потокобезопасного доступа к synth

        # Сохраняем предполагаемые параметры аудио
        # TODO: Уточнить, можно ли их получить из объекта model или synth
        self.sample_rate = DEFAULT_SAMPLE_RATE
        self.sample_width = DEFAULT_SAMPLE_WIDTH
        self.channels = DEFAULT_CHANNELS
        log.info(f"Assuming audio parameters: Rate={self.sample_rate}, Width={self.sample_width}, Channels={self.channels}")

        try:
            self.model = Model(model_name=self.vosk_model_name)
            log.info(f"Vosk model '{self.vosk_model_name}' loaded successfully.")
            self.synth = Synth(self.model)
            log.info("Vosk Synth initialized successfully.")
        except Exception as e:
            log.error(f"Failed to preload Vosk model '{self.vosk_model_name}' or initialize Synth: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize Vosk TTS: {e}") from e

    def _normalize_numbers(self, text: str) -> str:
        """
        Преобразует числа в тексте в их словесное представление на русском языке.
        Args:
            text: Входной текст.
        Returns:
            Текст с числами, преобразованными в слова.
        """
        def replace_number(match):
            num = match.group(0)
            try:
                return num2words(int(num), lang='ru')
            except (ValueError, OverflowError):
                log.warning(f"Could not normalize number: {num}")
                return num
        return re.sub(r'\b\d+\b', replace_number, text)

    def _normalize_english(self, text: str) -> str:
        """
        Транслитерирует английские слова и буквы в русские, сохраняя пробелы между словами.
        Args:
            text: Входной текст.
        Returns:
            Текст с транслитерированными английскими словами и буквами.
        """
        def replace_english(match):
            word = match.group(0).lower()  # Приводим к нижнему регистру
            # Транслитерируем каждую букву и соединяем без пробелов внутри слова
            return ''.join(ENGLISH_TO_RUSSIAN.get(c, c) for c in word)

        # Ищем английские слова или отдельные буквы (a-z, A-Z)
        return re.sub(r'\b[a-zA-Z]+\b', replace_english, text)

    async def synthesize(self, text: str, speaker_id: int) -> bytes | None:
        """
        Синтезирует текст в речь, используя предзагруженный синтезатор Vosk.
        Возвращает сырые байты аудио PCM 16-bit little-endian mono @ 22050 Hz (предположительно).
        Args:
            text: Текст для синтеза.
            speaker_id: ID спикера для использования в модели Vosk.
        Returns:
            Байтовый массив аудиоданных или None в случае ошибки синтеза.
        """
        log.debug(f"Requested in-memory Vosk TTS synthesis for speaker_id: {speaker_id}, text: [{text[:50]}...]")

        # Нормализуем числа и английские слова
        normalized_text = self._normalize_numbers(text)
        normalized_text = self._normalize_english(normalized_text)
        log.debug(f"Normalized text: [{normalized_text[:50]}...]")

        try:
            # Используем блокировку и запускаем в потоке
            async with self._lock:
                audio_data = await asyncio.to_thread(
                    self.synth.synth_audio,  # Считывает текст и возвращает аудиоданные (bytes или numpy.ndarray)
                    normalized_text,  # Используем нормализованный текст
                    speaker_id=speaker_id
                )

            # Преобразуем результат в bytes, если необходимо
            if hasattr(audio_data, 'tobytes'):
                audio_bytes = audio_data.tobytes()
            elif isinstance(audio_data, bytes):
                audio_bytes = audio_data
            else:
                log.error(f"Unexpected return type from synth_audio: {type(audio_data)}. Expected bytes or object with .tobytes()")
                return None

            log.info(f'Vosk in-memory synthesis complete for speaker {speaker_id}, data length: {len(audio_bytes)} bytes')
            return audio_bytes

        except AttributeError:
            log.error(f"Method 'synth_audio' not found in vosk_tts.Synth. Cannot perform in-memory synthesis.", exc_info=True)
            return None
        except Exception as e:
            log.error(f"Vosk TTS in-memory synthesis failed: {e}", exc_info=True)
            return None  # Возвращаем None при ошибке

# --- END OF FILE speech_tts.py ---