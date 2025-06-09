import os
import io
import logging
import soundfile as sf
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response
from vosk_tts import Model, Synth
from dotenv import load_dotenv
import asyncio
from num2words import num2words
import re

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Фильтр для исключения сообщений с фонемами
class PhonemeFilter(logging.Filter):
    def filter(self, record):
        return "Phonemes" not in record.msg

# Применяем фильтр к корневому логгеру
logging.getLogger("").addFilter(PhonemeFilter())

# Загрузка переменных окружения
load_dotenv()

# --- Конфигурация ---
DEFAULT_MODEL_NAME = "vosk-model-tts-ru-0.7-multi"
MODEL_NAME = os.getenv("VOSK_MODEL_NAME", DEFAULT_MODEL_NAME)
DEFAULT_SPEAKER_ID = 4
SPEAKER_ID = int(os.getenv("VOSK_SPEAKER_ID", DEFAULT_SPEAKER_ID))
MAX_TEXT_LENGTH = 3250
ERROR_MESSAGE = "Превышен лимит ввода"
AVAILABLE_SPEAKERS = [0, 1, 2, 3, 4]  # Фиксированный список дикторов из документации

# --- Инициализация Vosk TTS ---
try:
    logger.info(f"Загрузка модели: {MODEL_NAME}")
    model = Model(model_name=MODEL_NAME)
    synth = Synth(model)
    logger.info(f"Модель '{MODEL_NAME}' загружена. Диктор ID: {SPEAKER_ID}")
except Exception as e:
    logger.critical(f"Ошибка инициализации модели {MODEL_NAME}: {str(e)}")
    raise RuntimeError(f"Не удалось загрузить модель: {str(e)}")

# --- Создание FastAPI приложения ---
app = FastAPI(
    title="Vosk TTS Server",
    description="API для синтеза речи с использованием vosk-tts",
    version="1.1.0"
)

# --- Функция нормализации специальных символов ---
def normalize_special_chars(text: str) -> str:
    """Заменяет 'сложные' или неподдерживаемые символы на более простые аналоги. Актуально для 0.3.58 и старше"""
    replacements = {
        '—': '-',  # Em dash -> Hyphen
        '–': '-',  # En dash -> Hyphen
        '−': '-',  # Minus sign -> Hyphen
            # Удаление всех видов кавычек
     	'“': '',  # Left double quotation mark -> Remove
   	    '”': '',  # Right double quotation mark -> Remove
	    '„': '',  # Double low-9 quotation mark -> Remove
        '«': '',  # Left-pointing double angle quotation mark -> Remove
        '»': '',  # Right-pointing double angle quotation mark -> Remove
 	    '"': '',  # Straight double quote -> Remove 

      	'‘': '',  # Left single quotation mark -> Remove
      	'’': '',  # Right single quotation mark -> Remove
    	'‚': '',  # Single low-9 quotation mark -> Remove
    	'‹': '',  # Single left-pointing angle quotation mark -> Remove
        '›': '',  # Single right-pointing angle quotation mark -> Remove
     	"'": "",  # Straight single quote -> Remove
        '…': '...', # Ellipsis -> Three periods
        '\xa0': ' ', # Non-breaking space -> Regular space
        # Добавьте другие замены по необходимости
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Удаляем символы, которые могут вызвать проблемы и не имеют простого аналога
    # Например, управляющие символы, кроме \n, \t (хотя их тоже лучше заменить)
    # Здесь можно использовать re.sub для удаления всего, что не является
    # буквами, цифрами, пробелами и основной пунктуацией.
    # text = re.sub(r'[^\w\s\.,!?-:"\'()%]', '', text) # Пример более агрессивной очистки

    # Нормализуем пробелы и переносы строк
    text = text.replace('\n', ' ') # Заменяем переносы строк на пробелы
    text = text.replace('\t', ' ') # Заменяем табы на пробелы
    text = re.sub(r'\s+', ' ', text).strip() # Заменяем множественные пробелы на один

    return text

# --- Функция нормализации чисел ---
def normalize_numbers(text: str) -> str:
    """Преобразует числа в текстовую форму (например, 123 -> сто двадцать три)."""

    def replace_number(match):
        num = match.group(0)
        try:
            # Преобразуем число в слова на русском языке
            return num2words(int(num), lang='ru')
        except (ValueError, TypeError):
            # Если не удалось преобразовать (например, очень большое число или не число),
            # попробуем произнести по цифрам или вернуть как есть
            # return ' '.join(num2words(int(digit), lang='ru') for digit in num) # Вариант: по цифрам
            return num # Возвращаем оригинальное значение

    # Ищем все числа в тексте (целые числа)
    # Добавляем \b (границы слова), чтобы не заменять части слов, похожие на числа
    normalized_text = re.sub(r'\b\d+\b', replace_number, text)
    return normalized_text

async def synthesize_audio(text: str, speaker_id: int, speech_rate: float, model: Model, synth: Synth) -> bytes:
    """Синтезирует речь и возвращает аудиоданные в формате WAV."""
    if not text: # Проверка на пустую строку после нормализации
        logger.warning("Получен пустой текст для синтеза после нормализации.")
        raise ValueError("Текст для синтеза не может быть пустым после нормализации.")
    try:
        # logger.info(f"Начало синтеза для текста (начало): '{text[:100]}...'") # <-- Убрано логирование текста
        # Выполняем синтез в отдельном потоке
        def sync_synthesize():
            buffer = io.BytesIO()
            synth.synth(text, buffer, speaker_id=speaker_id, speech_rate=speech_rate)
            return buffer.getvalue()

        loop = asyncio.get_event_loop()
        wav_bytes = await loop.run_in_executor(None, sync_synthesize)

        if not wav_bytes:
            logger.error("Метод synth.synth выполнился, но не вернул аудиоданные.")
            raise RuntimeError("Синтезатор не вернул аудиоданные.")

        logger.info(f"Синтез завершен. Размер аудио: {len(wav_bytes)} байт") # <-- Лог размера аудио оставлен
        return wav_bytes
    except ValueError as e:
        logger.error(f"Ошибка валидации во время синтеза: {str(e)}. Текст (начало): '{text[:100]}...'") # <-- Лог текста при ошибке оставлен для диагностики
        raise
    except Exception as e:
        logger.error(f"Неожиданная ошибка во время выполнения synth.synth: {str(e)}. Текст (начало): '{text[:100]}...'") # <-- Лог текста при ошибке оставлен для диагностики
        logger.exception("Полный traceback ошибки синтеза:")
        raise RuntimeError(f"Ошибка синтеза: {str(e)}")


# --- Эндпоинт синтеза ---
@app.get("/synthesize",
         summary="Синтезировать речь из текста",
         response_description="Аудиофайл в формате WAV",
         responses={
             200: {"content": {"audio/wav": {}}, "description": "Успешный синтез речи или аудио с сообщением об ошибке длины текста"},
             400: {"description": "Неверные параметры запроса (ID диктора, пустой текст, ошибка валидации синтезатора)"},
             500: {"description": "Внутренняя ошибка синтеза речи"}
         })
async def synthesize_speech(
    text: str = Query(..., description="Текст для синтеза речи", min_length=1),
    speaker: int = Query(SPEAKER_ID, description=f"ID диктора. Доступные: {AVAILABLE_SPEAKERS}"),
    speech_rate: float = Query(1.0, description="Скорость речи (0.2–2.0)", ge=0.2, le=2.0)
):
    """
    Синтезирует текст в аудио (WAV).
    Перед синтезом текст нормализуется: заменяются специальные символы (тире, кавычки и т.д.)
    и числа преобразуются в слова.
    Если итоговый текст длиннее MAX_TEXT_LENGTH, возвращает аудио с сообщением об ошибке.
    """
    try:
        # Проверка доступных дикторов
        if speaker not in AVAILABLE_SPEAKERS:
            raise HTTPException(status_code=400, detail=f"Неверный ID диктора: {speaker}. Доступные ID: {AVAILABLE_SPEAKERS}")

        # logger.info(f"Получен текст (начало): '{text[:100]}...'") # <-- Убрано

        # 1. Нормализация специальных символов
        cleaned_text = normalize_special_chars(text)
        # logger.info(f"Текст после очистки символов (начало): '{cleaned_text[:100]}...'") # <-- Убрано

        # 2. Нормализация чисел
        normalized_text = normalize_numbers(cleaned_text)
        # logger.info(f"Текст после нормализации чисел (начало): '{normalized_text[:100]}...'") # <-- Убрано
        final_text_len = len(normalized_text)
        logger.info(f"Длина итогового текста для синтеза: {final_text_len}") # <-- Этот лог оставлен

        # Проверка на пустую строку после всех нормализаций
        if not normalized_text:
             logger.warning("Итоговый текст для синтеза пуст.")
             raise HTTPException(status_code=400, detail="Текст для синтеза не может быть пустым после нормализации.")

        # Проверка длины текста
        if final_text_len > MAX_TEXT_LENGTH:
            logger.warning(f"Текст превышает {MAX_TEXT_LENGTH} символов (длина: {final_text_len}). Синтезируется сообщение об ошибке.")
            # Используем исходный speaker и speech_rate для синтеза сообщения об ошибке
            error_wav_bytes = await synthesize_audio(ERROR_MESSAGE, speaker, speech_rate, model, synth)
            return Response(content=error_wav_bytes, media_type="audio/wav")

        # Синтез основного текста
        wav_bytes = await synthesize_audio(normalized_text, speaker, speech_rate, model, synth)
        return Response(content=wav_bytes, media_type="audio/wav")

    except HTTPException as e:
        raise e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.critical(f"Неожиданная ошибка в эндпоинте /synthesize: {str(e)}")
        logger.exception("Полный traceback неожиданной ошибки:")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера при обработке запроса.")

# --- Эндпоинт проверки состояния ---
@app.get("/health", summary="Проверка состояния сервера")
async def health_check():
    """Проверяет, что сервер и модель готовы к работе."""
    # vosk-tts Model не имеет стандартного метода list_speakers в текущих версиях
    # Оставляем просто проверку факта загрузки
    return {
        "status": "ok",
        "model_name": MODEL_NAME,
        "model_loaded": model is not None and synth is not None, # Проверяем, что объекты созданы
        "default_speaker_id": SPEAKER_ID,
        "available_speaker_ids": AVAILABLE_SPEAKERS # Используем наш фиксированный список
    }

# --- Запуск сервера ---
if __name__ == "__main__":
    import uvicorn
    logger.info(f"Запуск сервера на http://0.0.0.0:5002 (Модель: {MODEL_NAME}, Диктор: {SPEAKER_ID})")
    uvicorn.run(app, host="0.0.0.0", port=5002)

