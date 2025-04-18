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
MAX_TEXT_LENGTH = 1200
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

# --- Функция нормализации чисел ---
def normalize_numbers(text: str) -> str:
    """Преобразует числа в текстовую форму (например, 123 -> сто двадцать три)."""
    def replace_number(match):
        num = match.group(0)
        try:
            # Преобразуем число в слова на русском языке
            return num2words(int(num), lang='ru')
        except (ValueError, TypeError):
            return num  # Если не удалось преобразовать, возвращаем оригинальное значение

    # Ищем все числа в тексте (целые числа)
    normalized_text = re.sub(r'\d+', replace_number, text)
    return normalized_text

# --- Логика синтеза ---
async def synthesize_audio(text: str, speaker_id: int, speech_rate: float, model: Model, synth: Synth) -> bytes:
    """Синтезирует речь и возвращает аудиоданные в формате WAV."""
    try:
        # Выполняем синтез в отдельном потоке
        def sync_synthesize():
            buffer = io.BytesIO()
            synth.synth(text, buffer, speaker_id=speaker_id, speech_rate=speech_rate)
            return buffer.getvalue()

        loop = asyncio.get_event_loop()
        wav_bytes = await loop.run_in_executor(None, sync_synthesize)  # Ожидаем результат асинхронно

        if not wav_bytes:
            raise RuntimeError("Метод synth не вернул аудиоданные.")
        
        logger.info(f"Синтез завершен. Размер аудио: {len(wav_bytes)} байт")
        return wav_bytes
    except ValueError as e:
        logger.warning(f"Ошибка валидации: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Ошибка синтеза: {str(e)}")
        raise RuntimeError(f"Ошибка синтеза: {str(e)}")

# --- Эндпоинт синтеза ---
@app.get("/synthesize",
         summary="Синтезировать речь из текста",
         response_description="Аудиофайл в формате WAV",
         responses={
             200: {"content": {"audio/wav": {}}, "description": "Успешный синтез речи или сообщение о превышении лимита"},
             400: {"description": "Неверные параметры запроса"},
             500: {"description": "Ошибка синтеза речи"}
         })
async def synthesize_speech(
    text: str = Query(..., description="Текст для синтеза речи", min_length=1),
    speaker: int = Query(SPEAKER_ID, description="ID диктора"),
    speech_rate: float = Query(1.0, description="Скорость речи (0.2–2.0)", ge=0.2, le=2.0)
):
    """Синтезирует текст в аудио (WAV). Если текст длиннее MAX_TEXT_LENGTH символов, возвращает аудио с сообщением ERROR_MESSAGE."""
    try:
        # Проверка доступных дикторов
        if speaker not in AVAILABLE_SPEAKERS:
            raise HTTPException(status_code=400, detail=f"Неверный ID диктора: {speaker}. Доступные ID: {AVAILABLE_SPEAKERS}")

        # Нормализация чисел в тексте
        normalized_text = normalize_numbers(text)
        logger.info(f"Нормализованный текст: {normalized_text}")

        # Проверка длины текста
        if len(normalized_text) > MAX_TEXT_LENGTH:
            logger.warning(f"Текст превышает {MAX_TEXT_LENGTH} символов (длина: {len(normalized_text)}). Синтезируется сообщение об ошибке")
            wav_bytes = await synthesize_audio(ERROR_MESSAGE, speaker, speech_rate, model, synth)
            return Response(content=wav_bytes, media_type="audio/wav")

        wav_bytes = await synthesize_audio(normalized_text, speaker, speech_rate, model, synth)
        return Response(content=wav_bytes, media_type="audio/wav")
    except HTTPException as e:
        raise e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.critical(f"Неожиданная ошибка: {str(e)}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")

# --- Эндпоинт проверки состояния ---
@app.get("/health", summary="Проверка состояния сервера")
async def health_check():
    """Проверяет, что сервер и модель готовы к работе."""
    return {
        "status": "ok",
        "model_name": MODEL_NAME,
        "model_loaded": True,
        "default_speaker_id": SPEAKER_ID,
        "available_speaker_ids": AVAILABLE_SPEAKERS
    }

# --- Запуск сервера ---
if __name__ == "__main__":
    import uvicorn
    logger.info(f"Запуск сервера на http://0.0.0.0:5002 (Модель: {MODEL_NAME}, Диктор: {SPEAKER_ID})")
    uvicorn.run(app, host="0.0.0.0", port=5002)
