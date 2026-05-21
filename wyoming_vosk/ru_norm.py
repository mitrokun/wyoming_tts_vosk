import logging
import re
from num2words import num2words
import eng_to_ipa as ipa

try:
    from silero_stress import load_accentor
    SILERO_STRESS_AVAILABLE = True
except ImportError:
    SILERO_STRESS_AVAILABLE = False

_LOGGER = logging.getLogger(__name__)

class _EnglishToRussianNormalizer:
    """Internal helper to transliterate English words to Russian phonetics."""

    SIMPLE_TRANSLIT = str.maketrans({
        'a': 'а', 'b': 'б', 'c': 'к', 'd': 'д', 'e': 'е', 'f': 'ф', 'g': 'г',
        'h': 'х', 'i': 'и', 'j': 'дж', 'k': 'к', 'l': 'л', 'm': 'м', 'n': 'н',
        'o': 'о', 'p': 'п', 'q': 'к', 'r': 'р', 's': 'с', 't': 'т', 'u': 'у',
        'v': 'в', 'w': 'в', 'x': 'кс', 'y': 'и', 'z': 'з'
    })

    ENGLISH_EXCEPTIONS = {
        "google": "гугл", "apple": "эпл", "microsoft": "майкрософт",
        "xiaomi": "сяом+и", "samsung": "самсунг", "toyota": "тойота",
        "volkswagen": "фольцваген", "coca": "кока", "cola": "кола",
        "pepsi": "пэпси", "whatsapp": "вотсап", "telegram": "телеграм",
        "youtube": "ютуб", "instagram": "инстаграм", "facebook": "фэйсбук",
        "twitter": "твиттер", "iphone": "айф+он", "tesla": "тесла",
        "spacex": "спэйс икс", "amazon": "амазон", "camera": "к+амера",
        "python": "пайтон", "AI": "эй+ай", "api": "эйпиай",
        "glados": "гл+адос", "IT": "+ай т+и", "wi-fi": "вай фай",
        "rtx": "эрте+икс", "nasa": "н+аса", "photoshop": "фотош+оп",
        "SOS": "сос", "pdf": "пэдэ+эф", "raw": "р+оу", "scp": "эссипи",
        "cuda": "к+уда", "ibm": "эйбиэм", "usb": "юэсб+и",
        "chatgpt": "чат джипит+и", "gpt": "джипит+и", "copilot": "копайлот",
        "intel": "интел", "android": "андроид", "linux": "линукс",
        "3d": "трид+э", "amd": "айэмд+и", "enter": "+энта",
        "setup": "сет+ап", "mode": "мод", "pc": "пис+и",
        "work": "ворк", "world": "ворлд", "bird": "бёрд", "girl": "гёрл",
        "burn": "бёрн", "her": "хёр", "early": "ёрли", "service": "сёрвис",
        "a": "э", "the": "зе", "of": "оф", "and": "энд", "for": "фо",
        "to": "ту", "in": "ин", "on": "он", "is": "из", "or": "ор",
        "knowledge": "ноуледж", "new": "нью", "just": "джаст", "error": "+эрор",
        "video": "видео", "ru": "ру", "com": "ком", "done": "дон",
        "media": "медиа", "hot": "хот", "https": "аштитипиэс",
        "http": "аштитипи", "upper": "аппер", "xxx": "иксыкс+ыкс",
    }

    IPA_TO_RUSSIAN_MAP = {
        "ˈ": "", "ˌ": "", "ː": "",
        "p": "п", "b": "б", "t": "т", "d": "д", "k": "к", "g": "г",
        "m": "м", "n": "н", "f": "ф", "v": "в", "s": "с", "z": "з",
        "h": "х", "l": "л", "r": "р", "w": "в", "j": "й",
        "ʃ": "ш", "ʒ": "ж", "tʃ": "ч", "ʧ": "ч", "dʒ": "дж", "ʤ": "дж",
        "ŋ": "нг", "θ": "с", "ð": "з",
        "i": "и", "ɪ": "и", "ɛ": "э", "æ": "э", "ɑ": "а", "ɔ": "о",
        "u": "у", "ʊ": "у", "ʌ": "а", "ə": "э", "ər": "эр", "ɚ": "эр",
        "eɪ": "эй", "aɪ": "ай", "ɔɪ": "ой", "aʊ": "ау", "oʊ": "оу",
        "ɪə": "иэ", "eə": "еэ", "ʊə": "уэ",
    }

    def __init__(self):
        self._max_ipa_len = max(len(k) for k in self.IPA_TO_RUSSIAN_MAP.keys())

    def _convert_ipa_to_russian(self, ipa_text: str) -> str:
        result, pos = "", 0
        while pos < len(ipa_text):
            found = False
            for length in range(self._max_ipa_len, 0, -1):
                chunk = ipa_text[pos:pos + length]
                if chunk in self.IPA_TO_RUSSIAN_MAP:
                    result += self.IPA_TO_RUSSIAN_MAP[chunk]
                    pos += length
                    found = True
                    break
            if not found:
                pos += 1
        return result

    def _transliterate_word(self, match: re.Match) -> str:
        word_original = match.group(0).replace("’", "'")
        word_lower = word_original.lower()

        if word_original in self.ENGLISH_EXCEPTIONS:
            return self.ENGLISH_EXCEPTIONS[word_original]
        if word_lower in self.ENGLISH_EXCEPTIONS:
            return self.ENGLISH_EXCEPTIONS[word_lower]

        try:
            ipa_trans = ipa.convert(word_lower)
            ipa_trans = re.sub(r'[/]', '', ipa_trans).strip()
            if '*' in ipa_trans:
                raise ValueError("IPA not found")

            phonetics = self._convert_ipa_to_russian(ipa_trans)
            phonetics = re.sub(r'йй', 'й', phonetics)
            return re.sub(r'([чшщждж])ь', r'\1', phonetics)
        except Exception:
            return word_lower.translate(self.SIMPLE_TRANSLIT)

    def normalize(self, text: str) -> str:
        pattern = r"\b[a-zA-Z]+(?:[-'’][a-zA-Z]+)*\b"
        return re.sub(pattern, self._transliterate_word, text)


class RussianTextNormalizer:
    """Main normalization class for Russian TTS."""

    # These words (clitics) usually don't have their own stress in speech
    SKIP_STRESS_WORDS = {
        "в", "во", "на", "за", "под", "подо", "из", "изо", "ко", "с", "со",
        "от", "ото", "по", "о", "об", "обо", "у", "при", "над", "надо", "и", 
        "пред", "предо", "без", "безо", "для", "про", "до", "а", "но", "да",
        "что", "кто", "то", "кого", "не", "ни", "чего", "да", "где", "ты", "мы",
        "какой", "какая", "тоже", "конечно", "бока",
    }

    _EMOJI_PATTERN = re.compile(
        "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF\u2600-\u26FF\u2700-\u27BF"
        "\U0001F900-\U0001F9FF\u200D\uFE0F]+",
        flags=re.UNICODE
    )

    def __init__(self, use_stress=False):
        self._eng_norm = _EnglishToRussianNormalizer()
        self.accentor = None
        if SILERO_STRESS_AVAILABLE and use_stress:
            try:
                self.accentor = load_accentor()
            except Exception as e:
                _LOGGER.error(f"Failed to load silero-stress: {e}")

        self._year_pattern = re.compile(
            r'\b(?P<num>\d{1,4})(?:-?[а-яё]{1,3})?\s+(?P<god>год[а-яё]{0,3})\b',
            re.IGNORECASE
        )

    # --- PUBLIC API ---

    def normalize(self, text: str) -> str:
        """Main entry point for text normalization."""
        # 1. Cleanup
        text = self._EMOJI_PATTERN.sub('', text)

        # 2. Math & Numbers
        text = self._handle_math_and_symbols(text)
        text = self._normalize_numbers(text)

        # 3. Linguistic processing
        text = self._eng_norm.normalize(text)
        text = self._add_accents(text)

        # 4. Final sanitation
        text = self._sanitize_output(text)

        return text.strip()

    # --- HIGH LEVEL STEPS ---

    def _handle_math_and_symbols(self, text: str) -> str:
        # Convert "+" to word when used with numbers
        text = re.sub(r'\s*\+\s*(?=\d)', ' плюс ', text)
        # Handle percentages
        text = re.sub(r'(\d+(?:[.,]\d+)?)\s*%', self._replace_percentages, text)
        # Handle years (e.g. 2024 год)
        text = self._year_pattern.sub(self._replace_years, text)
        return text

    def _normalize_numbers(self, text: str) -> str:
        def replacer(m):
            s = m.group(0)
            if '.' in s or ',' in s:
                return self._float_to_text(s)
            try:
                return num2words(int(s), lang='ru')
            except Exception:
                return s
        return re.sub(r'\b\d+([.,]\d+)?\b', replacer, text)

    def _add_accents(self, text: str) -> str:
        """Adds stress marks using Silero, skipping clitics and pre-stressed words."""
        if self.accentor is None or not text.strip():
            return text

        try:
            # Tokenize into word units (including existing + marks) and everything else
            tokens = re.findall(r'([а-яА-ЯёЁ+]+|[^а-яА-ЯёЁ+]+)', text)
            
            # Identify words that actually need accentuation
            words_to_acc = []
            for t in tokens:
                # Word must be pure Cyrillic, not in skip list, and not already stressed
                if re.fullmatch(r'[а-яА-ЯёЁ]+', t):
                    if t.lower() not in self.SKIP_STRESS_WORDS:
                        words_to_acc.append(t)

            if not words_to_acc:
                return text

            # Batch process words
            acc_res = self.accentor(' '.join(words_to_acc)).split()
            acc_iter = iter(acc_res)
            
            result = []
            for t in tokens:
                # Same check as above to replace original word with stressed version
                if re.fullmatch(r'[а-яА-ЯёЁ]+', t) and t.lower() not in self.SKIP_STRESS_WORDS:
                    try:
                        result.append(next(acc_iter))
                    except StopIteration:
                        result.append(t)
                else:
                    result.append(t)
                    
            return "".join(result)
        except Exception as e:
            _LOGGER.error(f"Accentuation failed: {e}")
            return text

    def _sanitize_output(self, text: str) -> str:
        # Keep only Cyrillic, numbers, basic punctuation and the stress mark (+)
        text = re.sub(r'[^а-яА-ЯёЁ0-9\s\.,!\?\-\+:\(\)\"\']', ' ', text)
        # Specific pronunciation fix
        text = re.sub(r'м\+э-\+я', 'М+э-йа', text, flags=re.IGNORECASE)
        # Clean up whitespaces
        text = re.sub(r'\s+', ' ', text)
        # Convert standalone plus signs (math leftovers)
        text = re.sub(r'\s\+\s', ' плюс ', text)
        return text

    # --- NUMERIC HELPERS ---

    def _get_noun_form(self, n: int, forms: list) -> str:
        if 10 < n % 100 < 20:
            return forms[2]
        last = n % 10
        if last == 1:
            return forms[0]
        if 2 <= last <= 4:
            return forms[1]
        return forms[2]

    def _replace_percentages(self, match: re.Match) -> str:
        num_str = match.group(1).replace(',', '.')
        if '.' in num_str:
            return f"{self._float_to_text(num_str)} процента"
        val = int(num_str)
        forms = ['процент', 'процента', 'процентов']
        return f"{num_str} {self._get_noun_form(val, forms)}"

    def _float_to_text(self, num_str: str) -> str:
        try:
            parts = num_str.replace(',', '.').split('.')
            if len(parts) != 2:
                return num_str
            int_p, frac_p = int(parts[0]), int(parts[1])
            frac_len = len(parts[1])

            int_t = num2words(int_p, lang='ru')
            frac_t = num2words(frac_p, lang='ru')

            # Gender adjustment for decimal parts
            last_digit, last_two = frac_p % 10, frac_p % 100
            if last_digit == 1 and last_two != 11:
                frac_t = re.sub(r'\bодин$', 'одна', frac_t)
            elif last_digit == 2 and last_two != 12:
                frac_t = re.sub(r'\bдва$', 'две', frac_t)

            suffixes = {1: " десятая", 2: " сотая", 3: " тысячная"}
            plurals = {1: " десятых", 2: " сотых", 3: " тысячных"}

            if frac_len in suffixes:
                suffix = suffixes[frac_len] if (last_digit == 1 and last_two != 11) else plurals[frac_len]
                return f"{int_t} и {frac_t}{suffix}"

            return f"{int_t} точка {frac_t}"
        except Exception:
            return num_str

    def _replace_years(self, match: re.Match) -> str:
        num_str, god_raw = match.group('num'), match.group('god')
        try:
            ord_t = num2words(int(num_str), to='ordinal', lang='ru')
            s_map = {
                'год': {'ый': 'ый', 'ой': 'ой', 'ий': 'ий'},
                'года': {'ый': 'ого', 'ой': 'ого', 'ий': 'ьего'},
                'году': {'ый': 'ом', 'ой': 'ом', 'ий': 'ьем'},
                'годом': {'ый': 'ым', 'ой': 'ым', 'ий': 'ьим'},
                'годы': {'ый': 'ые', 'ой': 'ые', 'ий': 'ьи'},
                'годов': {'ый': 'ых', 'ой': 'ых', 'ий': 'ьих'},
            }
            rules = s_map.get(god_raw.lower(), s_map['год'])
            words = ord_t.split()
            for base, new in rules.items():
                if words[-1].endswith(base):
                    words[-1] = words[-1][:-len(base)] + new
                    break
            return f"{' '.join(words)} {god_raw}"
        except Exception:
            return match.group(0)
