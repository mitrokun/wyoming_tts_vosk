# Vosk TTS Home Assistant [Wyoming Protocol]
Сервер и интеграция для [vosk-tts](https://github.com/alphacep/vosk-tts).
Создано Grok и Gemini. Понятие не имею как оно работает, сколько неоптимальных решений и лишнего кода используется.

## Обновленный вариант сервера для Wyoming Protocol
Нативная поддержка протокола в HA (не требуется пользовательская интеграция). Возможность выбора спикера из интерфейса. Проверку длины ввода убрал.
Работа с числами и английскими словами максимально упрощенная (смотрите speech_tts.py), лишь бы воск не мычал.
Лучше сразу делайте правильный системный промт для LLM, а в шаблонах можно использовать [интеграцию](https://github.com/AlexxIT/MorphNumbers) AlexxIT
```
# Установите требуемые библиотеки (возможно что-то ещё)
pip install vosk-tts==0.3.56 wyoming num2words numpy
# Скопируйте папку wyoming_vosk и перейдите в неё
# Запустите сервер с кастомным портом
python __main__.py --uri tcp://0.0.0.0:10205
# в HA добавьте службу в интеграции Wyoming Protocol
```
---

## Предыдущая реализация сервера
```
git clone https://github.com/mitrokun/vosk_tts_hass.git
cd vosk_tts_hass
# Создайте и активируйте виртуальное окружение (опционально)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate  # Windows

# Установите необходимые библиотеки (для 0.8 ставьте актуальную версию vosk-tts)
pip install vosk-tts==0.3.56 fastapi uvicorn[standard] soundfile python-dotenv num2words numpy

# Для ипользования с esp32 спутниками HA скорректируйте значение MAX_TEXT_LENGTH в main.py,
# число символов подбирайте под производительность cpu, суммарно процесс должен быть короче 5с
# высокий лимит для стандартных прошивок некритичен, будет отсутствовать ответ и терминал вернется к обнаружению WW
# Используется 0.7 версия модели, она сильно быстрее, чем 0.8. Измените, если требуется.

# Запуск
uvicorn main:app --host 0.0.0.0 --port 5002 --reload

# Тест
http://127.0.0.1:5002/synthesize?text=Привет мир!
```
## cuda (12.x) 
Ставим пакет
`pip install onnxruntime-gpu`
Заменить провайдера на CUDAExecutionProvider в файле model.py (актульно на момент написания с vosk_tts-0.3.58).
Ищем в каталоге python `...\Lib\site-packages\vosk_tts\model.py`  
Если есть ошибки при запуске main.py, выполняйте предписания и устанавливайте требуемые версии cuda, cuDNN...
Когда всё завелось, MAX_TEXT_LENGTH можно прилично задрать. 

## Интеграция

Скопировать каталог vosk_tts в /homeassistant/custom_components
В конфигурационном файле добавить запись, не забудьте указать верный адрес сервера:
```
tts:
  - platform: vosk_tts
    url: "http://192.168.1.xxx:5002/synthesize"
    default_voice: "4"
    default_speech_rate: 1.0
    # ускорение речи уменьшает продолжительность обработки
```


Перезапустить HA.

Можно использовать действие с выбором одного из пяти голосов (0-4):
```
action: tts.vosk_tts_say
data:
  entity_id: media_player.lg
  message: Привет 
  options:
    voice: "2"
    speech_rate: 0.85
```

Дорабатывайте и улучшайте на свое усмотретние.
