"""
Определяет границы предложений в потоке текста для русского и английского языков.
"""

from collections.abc import Iterable
import regex as re

# Концы предложений для русского и английского
SENTENCE_END = r"[.!?…]"

# Сокращения из 1-3 букв, чтобы не рвать на "Mr. Smith"
ABBREVIATION_RE = re.compile(r"\b[a-zA-Zа-яА-ЯёЁ]{1,3}\.$")

# Основное правило: конец предложения, если дальше идет пробел и заглавная буква
# ИЛИ если дальше идет нумерованный список (например, " 1. ").
SENTENCE_BOUNDARY_RE = re.compile(
    rf"(.*?(?:{SENTENCE_END}+))"  # Сам текст предложения
    # Позитивный просмотр вперед (условие, но не часть совпадения):
    r"(?="
    r"\s+[A-ZА-ЯЁ]"  # ...пробел и заглавная буква
    r"|"             # ИЛИ
    r"(?:\s+\d+\.\s+)" # ...пробел, число, точка, пробел (для списков)
    r")",
    re.DOTALL, # Позволяет '.' совпадать с символом новой строки
)


class SentenceBoundaryDetector:
    """Накапливает текст и отдает готовые предложения по мере их формирования."""
    def __init__(self) -> None:
        self.remaining_text = ""
        self.current_sentence = ""

    def add_chunk(self, chunk: str) -> Iterable[str]:
        """Добавляет чанк текста и возвращает (yields) найденные предложения."""
        self.remaining_text += chunk
        while self.remaining_text:
            match = SENTENCE_BOUNDARY_RE.search(self.remaining_text)
            if not match:
                # Больше нет явных границ, выходим из цикла
                break

            match_text = match.group(0)

            if not self.current_sentence:
                self.current_sentence = match_text
            elif ABBREVIATION_RE.search(self.current_sentence[-5:]):
                # Если похоже на аббревиатуру, присоединяем следующий кусок
                self.current_sentence += match_text
            else:
                yield self.current_sentence.strip()
                self.current_sentence = match_text

            if not ABBREVIATION_RE.search(self.current_sentence[-5:]):
                yield self.current_sentence.strip()
                self.current_sentence = ""

            # Убираем обработанную часть из буфера
            self.remaining_text = self.remaining_text[match.end() :]

    def finish(self) -> str:
        """Возвращает весь оставшийся в буфере текст как одно финальное предложение."""
        text = (self.current_sentence + self.remaining_text).strip()
        self.remaining_text = ""
        self.current_sentence = ""

        return text