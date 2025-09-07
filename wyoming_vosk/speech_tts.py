import logging
import asyncio
import contextlib
import re
from num2words import num2words

from vosk_tts import Model, Synth

try:
    from ruaccent import RUAccent
    RUACCENT_AVAILABLE = True
except ImportError:
    RUACCENT_AVAILABLE = False

log = logging.getLogger(__name__)

# Параметры аудио по умолчанию для Vosk TTS моделей
DEFAULT_SAMPLE_RATE = 22050
DEFAULT_SAMPLE_WIDTH = 2
DEFAULT_CHANNELS = 1

# Словарь для транслитерации
ENGLISH_TO_RUSSIAN = {
    'a': 'э', 'b': 'б', 'c': 'к', 'd': 'д', 'e': 'е', 'f': 'ф', 'g': 'г',
    'h': 'х', 'i': 'и', 'j': 'ж', 'k': 'к', 'l': 'л', 'm': 'м', 'n': 'н',
    'o': 'о', 'p': 'п', 'q': 'к', 'r': 'р', 's': 'с', 't': 'т', 'u': 'у',
    'v': 'в', 'w': 'в', 'x': 'х', 'y': 'ай', 'z': 'з'
}

class SpeechTTS:
    _emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F" u"\U0001F300-\U0001F5FF" u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF" u"\U00002600-\U000026FF" u"\U00002700-\U000027BF"
        u"\U0001F900-\U0001F9FF" u"\u200D" u"\uFE0F"
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
    _FINAL_CLEANUP_PATTERN = re.compile(r'[^а-яА-ЯёЁ+?!., ]')

    # <<< ИЗМЕНЕНО: Конструктор теперь принимает или имя, или путь >>>
    def __init__(self, vosk_model_name: str = None, vosk_model_path: str = None, use_accentizer: bool = False) -> None:
        if not vosk_model_name and not vosk_model_path:
            raise ValueError("Either 'vosk_model_name' or 'vosk_model_path' must be provided.")
        if vosk_model_name and vosk_model_path:
            log.warning(f"Both model name ('{vosk_model_name}') and path ('{vosk_model_path}') were provided. Using local path.")
            vosk_model_name = None

        log.debug(f"Initializing Vosk Speech TTS...")
        self.vosk_model_name = vosk_model_name
        self.vosk_model_path = vosk_model_path
        self._lock = asyncio.Lock()
        self.sample_rate = DEFAULT_SAMPLE_RATE
        self.sample_width = DEFAULT_SAMPLE_WIDTH
        self.channels = DEFAULT_CHANNELS
        log.debug(f"Assuming audio parameters: Rate={self.sample_rate}, Width={self.sample_width}, Channels={self.channels}")

        try:
            model_args = {}
            if self.vosk_model_path:
                log.info(f"Loading Vosk model from local path: {self.vosk_model_path}")
                model_args['model_path'] = self.vosk_model_path
            else:
                log.info(f"Loading Vosk model by name: {self.vosk_model_name}")
                model_args['model_name'] = self.vosk_model_name
            
            self.model = Model(**model_args)
            self.synth = Synth(self.model)
            log.debug("Vosk Synth initialized successfully.")

            # <<< НОВОЕ: Считываем количество спикеров из конфига модели >>>
            self.num_speakers = self.model.config.get("num_speakers")
            if self.num_speakers is None:
                log.warning("Key 'num_speakers' not found in model's config.json. Falling back to 5 speakers.")
                self.num_speakers = 5 # Запасной вариант для очень старых или некорректных моделей
            log.info(f"Model reports {self.num_speakers} available speakers.")

        except Exception as e:
            log.error(f"Failed to load Vosk model or initialize Synth: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize Vosk TTS: {e}") from e

        self.accentizer = None
        if use_accentizer:
            if RUACCENT_AVAILABLE:
                log.info("Loading RUAccent model for automatic stress marking...")
                try:
                    self.accentizer = RUAccent()
                    self.accentizer.load(omograph_model_size='turbo3.1', use_dictionary=True, tiny_mode=False)
                    log.info("RUAccent model loaded successfully.")
                except Exception as e:
                    log.error(f"Failed to load RUAccent model: {e}", exc_info=True)
                    self.accentizer = None
            else:
                log.warning("`use_accentizer` is True, but `ruaccent` library is not installed. Please run `pip install ruaccent`.")
    
    # ... (все остальные методы _normalize_*, _add_accents, synthesize и т.д. остаются БЕЗ ИЗМЕНЕНИЙ) ...
    def _choose_percent_form(self, number_str: str) -> str:
        if '.' in number_str or ',' in number_str: return "процента"
        try:
            number = int(number_str)
            if 10 < number % 100 < 20: return "процентов"
            last_digit = number % 10
            if last_digit == 1: return "процент"
            if last_digit in [2, 3, 4]: return "процента"
            return "процентов"
        except (ValueError, OverflowError): return "процентов"
    def _normalize_percentages(self, text: str) -> str:
        def replace_match(match):
            number_str_clean = match.group(1).replace(',', '.')
            try: return f" {number_str_clean} {self._choose_percent_form(number_str_clean)} "
            except (ValueError, OverflowError): return f" {number_str_clean} процентов "
        processed_text = re.sub(r'(\d+([.,]\d+)?)\s*\%', replace_match, text)
        processed_text = processed_text.replace('%', ' процентов ')
        return processed_text
    def _normalize_special_chars(self, text: str) -> str:
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
            num_str = match.group(0).replace(',', '.')
            try:
                if '.' in num_str:
                    parts = num_str.split('.')
                    integer_part_str, fractional_part_str = parts[0], parts[1]
                    if not integer_part_str or not fractional_part_str:
                        valid_num_str = num_str.replace('.', '')
                        return num2words(int(valid_num_str), lang='ru') if valid_num_str.isdigit() else num_str
                    integer_part_val, fractional_part_val = int(integer_part_str), int(fractional_part_str)
                    fractional_len = len(fractional_part_str)
                    integer_words, fractional_words = num2words(integer_part_val, lang='ru'), num2words(fractional_part_val, lang='ru')
                    if fractional_len == 1: return f"{integer_words} и {fractional_words}"
                    if fractional_part_val % 10 == 1 and fractional_part_val % 100 != 11:
                        if fractional_words.endswith("один"): fractional_words = fractional_words[:-4] + "одна"
                    if fractional_part_val % 10 == 2 and fractional_part_val % 100 != 12:
                        if fractional_words.endswith("два"): fractional_words = fractional_words[:-3] + "две"
                    if fractional_len == 2: return f"{integer_words} и {fractional_words} сотых"
                    if fractional_len == 3: return f"{integer_words} и {fractional_words} тысячных"
                    return f"{integer_words} точка {fractional_words}"
                else: return num2words(int(num_str), lang='ru')
            except (ValueError, OverflowError) as e:
                log.warning(f"Could not normalize number '{num_str}': {e}")
                return num_str
        return re.sub(r'\b\d+([.,]\d+)?\b', replace_number, text)
    def _normalize_english(self, text: str) -> str:
        def replace_english(match):
            word = match.group(0).lower()
            return ''.join(ENGLISH_TO_RUSSIAN.get(c, c) for c in word)
        return re.sub(r'\b[a-zA-Z]+\b', replace_english, text)
    def _add_accents(self, text: str) -> str:
        if self.accentizer is None or '+' in text or not text or not text.strip(): return text
        try:
            processed_text = self.accentizer.process_all(text)
            log.debug(f"Accentizer processed text. Before: '{text[:50]}...'. After: '{processed_text[:50]}...'")
            return processed_text
        except Exception as e:
            log.warning(f"RUAccent failed to process text: {e}")
            return text
    def _cleanup_final_text(self, text: str) -> str:
        return self._FINAL_CLEANUP_PATTERN.sub(' ', text)
    async def synthesize(self, text: str, speaker_id: int, speech_rate: float = 1.0) -> bytes | None:
        log.debug(f"Requested TTS. Speaker: {speaker_id}, Rate: {speech_rate}, Original text: [{text[:100]}...]")
        speech_rate = max(0.5, min(2.0, speech_rate))
        normalized_text = self._normalize_percentages(text)
        normalized_text = self._normalize_special_chars(normalized_text)
        normalized_text = self._normalize_numbers(normalized_text)
        normalized_text = self._normalize_english(normalized_text)
        normalized_text = self._add_accents(normalized_text)
        normalized_text = self._cleanup_final_text(normalized_text)
        normalized_text = re.sub(r'\s+', ' ', normalized_text).strip()
        if not normalized_text:
            log.warning("Normalized text is empty or whitespace only. Skipping synthesis.")
            return None
        try:
            async with self._lock:
                audio_data = await asyncio.to_thread(self.synth.synth_audio, normalized_text, speaker_id=speaker_id, speech_rate=speech_rate)
            if hasattr(audio_data, 'tobytes'): audio_bytes = audio_data.tobytes()
            elif isinstance(audio_data, bytes): audio_bytes = audio_data
            else:
                log.error(f"Unexpected return type from synth_audio: {type(audio_data)}.")
                return None
            log.debug(f'Vosk synthesis complete. Speaker {speaker_id}, Rate: {speech_rate}, Audio length: {len(audio_bytes)} bytes')
            return audio_bytes
        except Exception as e:
            log.error(f"Vosk TTS synthesis failed: {e}", exc_info=True)
            return None