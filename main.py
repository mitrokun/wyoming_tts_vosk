import os
import io
import soundfile as sf
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response
from vosk_tts import Model, Synth # Используем правильные классы
from dotenv import load_dotenv
import numpy as np # vosk-tts может возвращать numpy массив

# Загрузка переменных окружения (если есть .env файл)
load_dotenv()

# --- Конфигурация ---
# Имя модели Vosk TTS. Будет скачана автоматически при первом запуске, если нет.
# Можно задать через переменную окружения VOSK_MODEL_NAME.
DEFAULT_MODEL_NAME = "vosk-model-tts-ru-0.7-multi" # Используем имя модели
MODEL_NAME = os.getenv("VOSK_MODEL_NAME", DEFAULT_MODEL_NAME)

# ID диктора. Можно задать через VOSK_SPEAKER_ID.
DEFAULT_SPEAKER_ID = 4 # Измените на 2, если предпочитаете этот голос по умолчанию
SPEAKER_ID = int(os.getenv("VOSK_SPEAKER_ID", DEFAULT_SPEAKER_ID))

# --- Инициализация Vosk TTS ---
try:
    print(f"Загрузка/проверка модели: {MODEL_NAME}")
    # Модель скачается в ~/.cache/vosk-tts/ если ее там нет
    model = Model(model_name=MODEL_NAME)
    synth = Synth(model)
    print(f"Модель '{MODEL_NAME}' успешно загружена/найдена.")
    print(f"Используется диктор ID: {SPEAKER_ID}")
    # Проверка доступных дикторов (опционально)
    # print("Доступные ID дикторов:", model.list_speakers())
except Exception as e:
    print(f"Ошибка при инициализации Vosk TTS ({MODEL_NAME}): {e}")
    print("Убедитесь, что имя модели правильное и есть доступ в интернет для скачивания.")
    exit(1)

# --- Создание FastAPI приложения ---
app = FastAPI(
    title="Vosk TTS Server (vosk-tts library)",
    description="API для синтеза речи с использованием библиотеки vosk-tts",
    version="1.1.0"
)

@app.get("/synthesize",
         summary="Синтезировать речь из текста",
         response_description="Аудиофайл в формате WAV",
         responses={
             200: {
                 "content": {"audio/wav": {}},
                 "description": "Успешный синтез речи. Возвращает WAV аудио.",
             },
             400: {"description": "Параметр 'text' отсутствует или пуст"},
             500: {"description": "Ошибка во время синтеза речи"},
         })
async def synthesize_speech(
    text: str = Query(..., description="Текст для синтеза речи", min_length=1),
    speaker: int = Query(SPEAKER_ID, description="ID диктора (переопределяет стандартный)") # Добавим опциональный параметр диктора
):
    """
    Принимает текстовую строку в query параметре 'text' и возвращает
    аудиофайл в формате WAV. Можно указать 'speaker' для выбора диктора.
    """
    if not text:
        raise HTTPException(status_code=400, detail="Query параметр 'text' не может быть пустым.")

    current_speaker_id = speaker # Используем ID из запроса или стандартный

    try:
        print(f"Синтезирую текст: '{text}' (Диктор ID: {current_speaker_id})...")

        # Синтез речи. Ожидаем, что synth.synth без имени файла вернет данные
        # Проверяем документацию/исходники, если это не так.
        # Часто возвращает кортеж (sample_rate, audio_data_numpy)
        # Если synth.synth *требует* имя файла, нам придется генерировать
        # временный файл и читать из него. Но попробуем без этого.

        # *** Важное предположение: ***
        # Предполагаем, что synth.synth может работать так же, как в
        # оригинальной vosk-api/vosk-tts, возвращая данные, если имя файла None
        # Если это НЕ так, и он возвращает None или что-то другое,
        # код ниже нужно будет адаптировать!

        # Попытка получить данные напрямую (если API это поддерживает неявно)
        # Это может не сработать с текущей версией vosk-tts, которая фокусируется на файлах
        # audio_data = synth.synth(text, None, speaker_id=current_speaker_id) # Это может не сработать

        # --- Запасной вариант: Запись во временный файл в памяти ---
        # (Более вероятно сработает с API, ориентированным на файлы)
        buffer = io.BytesIO()
        synth.synth(text, buffer, speaker_id=current_speaker_id) # Передаем buffer как файловый объект
        # sample_rate нужно указать явно, т.к. берется из модели

        # Проверим, что данные записаны в буфер
        wav_bytes = buffer.getvalue()
        if not wav_bytes:
             # Если synth.synth не записал в buffer или вернул не то, что ожидалось
             raise RuntimeError("Метод synth не вернул или не записал ожидаемые аудио данные.")

        # --- Преобразование в нужный формат (если synth вернул numpy массив) ---
        # Этот блок нужен, если бы synth.synth вернул (rate, nparray)
        # sample_rate = model.sample_rate # или получаем из возврата synth
        # buffer = io.BytesIO()
        # sf.write(buffer, audio_data, sample_rate, format='WAV', subtype='PCM_16')
        # wav_bytes = buffer.getvalue()
        # --- Конец блока преобразования ---

        print(f"Синтез завершен. Размер аудио: {len(wav_bytes)} байт.")
        return Response(content=wav_bytes, media_type="audio/wav")

    except Exception as e:
        print(f"Ошибка синтеза: {e}")
        # Если ошибка связана с ID диктора, дать подсказку
        if "speaker_id" in str(e).lower():
             print(f"Возможно, ID диктора {current_speaker_id} не существует в модели {MODEL_NAME}.")
             # print("Доступные ID:", model.list_speakers()) # Раскомментируйте для отладки
             raise HTTPException(status_code=400, detail=f"Неверный ID диктора: {current_speaker_id}. Ошибка: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка синтеза речи: {str(e)}")


@app.get("/health", summary="Проверка состояния сервера")
async def health_check():
    """Простая проверка, что сервер запущен и модель загружена."""
    speakers = []
    try:
        # Попробуем получить список дикторов, если метод существует
        if hasattr(model, 'list_speakers'):
             speakers = model.list_speakers()
        elif hasattr(model, 'speakers'): # Или может быть атрибутом
             speakers = model.speakers
    except Exception:
        speakers = ["Не удалось получить список"] # Обработка ошибки

    return {
        "status": "ok",
        "model_name": MODEL_NAME,
        "model_loaded": True,
        "default_speaker_id": SPEAKER_ID,
        "available_speaker_ids": speakers # Показываем доступные ID, если можем
        }

# --- Запуск сервера (для локального тестирования) ---
if __name__ == "__main__":
    import uvicorn
    print("Запуск сервера на http://0.0.0.0:5002")
    print(f"Используемая модель: {MODEL_NAME}")
    print(f"Стандартный ID диктора: {SPEAKER_ID}")
    print("Для синтеза используйте: http://<ip>:5002/synthesize?text=ВАШ_ТЕКСТ")
    print("Для смены диктора: http://<ip>:5002/synthesize?text=ВАШ_ТЕКСТ&speaker=ID")
    uvicorn.run(app, host="0.0.0.0", port=5002)