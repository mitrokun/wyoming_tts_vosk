# --- START OF FILE speech_tts.py ---

import logging
import asyncio
import contextlib
import re
from num2words import num2words

from vosk_tts import Model, Synth

log = logging.getLogger(__name__)

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

    _emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # Emoticons
        u"\U0001F300-\U0001F5FF"  # Symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # Transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # Flags (iOS / Regional Indicator Symbols)
        u"\U00002600-\U000026FF"  # Miscellaneous symbols
        u"\U00002700-\U000027BF"  # Dingbats
        u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        u"\u200D"                # Zero Width Joiner
        u"\uFE0F"                # Variation Selector-16
        "]+",
        flags=re.UNICODE
    )

    _chars_to_delete_for_translate = "=#$“”„«»<>*\"‘’‚‹›'/"
    _map_1_to_1_from_for_translate = "—–−\xa0"
    _map_1_to_1_to_for_translate   = "--- " 
    
    _translation_table = str.maketrans(
        _map_1_to_1_from_for_translate,
        _map_1_to_1_to_for_translate,
        _chars_to_delete_for_translate
    )

    # Паттерн для финальной очистки
    _FINAL_CLEANUP_PATTERN = re.compile(r'[^а-яА-ЯёЁ+?!., ]')


    def __init__(self, vosk_model_name: str) -> None:
        log.debug(f"Initializing Vosk Speech TTS and preloading model: {vosk_model_name}")
        self.vosk_model_name = vosk_model_name
        self._lock = asyncio.Lock()
        self.sample_rate = DEFAULT_SAMPLE_RATE
        self.sample_width = DEFAULT_SAMPLE_WIDTH
        self.channels = DEFAULT_CHANNELS
        log.debug(f"Assuming audio parameters: Rate={self.sample_rate}, Width={self.sample_width}, Channels={self.channels}")

        try:
            self.model = Model(model_name=self.vosk_model_name)
            log.debug(f"Vosk model '{self.vosk_model_name}' loaded successfully.")
            self.synth = Synth(self.model)
            log.debug("Vosk Synth initialized successfully.")
        except Exception as e:
            log.error(f"Failed to preload Vosk model '{self.vosk_model_name}' or initialize Synth: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize Vosk TTS: {e}") from e

    def _choose_percent_form(self, number: int) -> str:
        """Выбирает правильную форму слова 'процент' в зависимости от числа."""
        if 10 < number % 100 < 20:
            return "процентов"
        
        last_digit = number % 10
        if last_digit == 1:
            return "процент"
        if last_digit in [2, 3, 4]:
            return "процента"
        
        return "процентов"

    def _normalize_percentages(self, text: str) -> str:
        """Находит и заменяет конструкции вида '123%' и одиночные '%'."""
        
        def replace_match(match):
            number_str = match.group(1)
            try:
                number_int = int(number_str)
                number_words = num2words(number_int, lang='ru')
                percent_word = self._choose_percent_form(number_int)
                return f" {number_words} {percent_word} "
            except (ValueError, OverflowError):
                return f" {number_str} процентов "

        # Шаг 1: Обрабатываем случаи "число + %"
        processed_text = re.sub(r'(\d+)\s*\%', replace_match, text)

        # Шаг 2: Обрабатываем оставшиеся одиночные символы '%', заменяя их на "процентов".
        processed_text = processed_text.replace('%', ' процентов ')
        
        return processed_text


    def _normalize_special_chars(self, text: str) -> str:
        """Заменяет 'сложные' символы, удаляет эмодзи, разделяет буквы/цифры и нормализует пробелы."""
        
        text = self._emoji_pattern.sub(r'', text)
        text = text.translate(self._translation_table)
        text = text.replace('…', '.')
        text = re.sub(r':(?!\d)', ',', text)
        text = re.sub(r'([a-zA-Zа-яА-ЯёЁ])(\d)', r'\1 \2', text)
        text = re.sub(r'(\d)([a-zA-Zа-яА-ЯёЁ])', r'\1 \2', text)
        text = text.replace('\n', ' ').replace('\t', ' ')
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def _normalize_numbers(self, text: str) -> str:
        def replace_number(match):
            num = match.group(0)
            try:
                if num.isdigit():
                    return num2words(int(num), lang='ru')
                return num 
            except (ValueError, OverflowError):
                log.warning(f"Could not normalize number: {num}")
                return num
        return re.sub(r'\b\d+\b', replace_number, text)

    def _normalize_english(self, text: str) -> str:
        def replace_english(match):
            word = match.group(0).lower()
            return ''.join(ENGLISH_TO_RUSSIAN.get(c, c) for c in word)
        return re.sub(r'\b[a-zA-Z]+\b', replace_english, text)

    def _cleanup_final_text(self, text: str) -> str:
        """
        Удаляет из текста все символы, кроме разрешенных, заменяя их на пробелы.
        Разрешенные символы: русские буквы (включая ё/Ё), пробелы и знак '+'.
        """
        # Заменяем все неразрешенные символы на пробел
        cleaned_text = self._FINAL_CLEANUP_PATTERN.sub(' ', text)
        return cleaned_text

    async def synthesize(self, text: str, speaker_id: int, speech_rate: float = 1.0) -> bytes | None:
        log.debug(f"Requested TTS. Speaker: {speaker_id}, Rate: {speech_rate}, Original text: [{text[:100]}...]")

        if not (0.5 <= speech_rate <= 2.0):
            log.warning(f"Speech rate {speech_rate} out of range [0.5, 2.0]. Clamping.")
            speech_rate = max(0.5, min(2.0, speech_rate))

        # Этап 1: "Умная" обработка процентов
        normalized_text = self._normalize_percentages(text)
        
        # Этап 2: Базовая нормализация символов, пробелов, эмодзи
        normalized_text = self._normalize_special_chars(normalized_text)
        
        # Этап 3: Оставшиеся числа в слова
        normalized_text = self._normalize_numbers(normalized_text)
        
        # Этап 4: Английские слова в русскую транслитерацию
        normalized_text = self._normalize_english(normalized_text)
        
        # Этап 5: Финальная очистка от неразрешенных символов
        normalized_text = self._cleanup_final_text(normalized_text)

        # Финальная чистка пробелов, которые могли добавиться на предыдущих шагах
        normalized_text = re.sub(r'\s+', ' ', normalized_text).strip()

        if not normalized_text.strip():
            log.warning("Normalized text is empty or whitespace only. Skipping synthesis.")
            return None

        try:
            async with self._lock:
                audio_data = await asyncio.to_thread(
                    self.synth.synth_audio,
                    normalized_text,
                    speaker_id=speaker_id,
                    speech_rate=speech_rate
                )

            if hasattr(audio_data, 'tobytes'):
                audio_bytes = audio_data.tobytes()
            elif isinstance(audio_data, bytes):
                audio_bytes = audio_data
            else:
                log.error(f"Unexpected return type from synth_audio: {type(audio_data)}. Expected bytes or object with .tobytes()")
                return None

            log.debug(f'Vosk synthesis complete. Speaker {speaker_id}, Rate: {speech_rate}, Audio length: {len(audio_bytes)} bytes')
            return audio_bytes

        except AttributeError as e:
            if 'synth_audio' in str(e):
                 log.error(f"Method 'synth_audio' not found. vosk_tts library might be outdated/different.", exc_info=True)
            else:
                log.error(f"AttributeError during Vosk TTS synthesis: {e}", exc_info=True)
            return None
        except Exception as e:
            log.error(f"Vosk TTS synthesis failed: {e}", exc_info=True)
            return None

# --- END OF FILE speech_tts.py ---