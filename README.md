# Vosk TTS Home Assistant [Wyoming Protocol]
Сервер ~~и интеграция~~ для [vosk-tts](https://github.com/alphacep/vosk-tts).
При участии Grok и Gemini. 
## Обновленный вариант сервера для Wyoming Protocol
Нативная поддержка протокола в HA (не требуется пользовательская интеграция). Возможность выбора спикера из интерфейса. Проверку длины ввода убрал.
Работа с числами и английскими словами максимально упрощенная (смотрите speech_tts.py), лишь бы воск не мычал.
Лучше сразу делайте правильный системный промт для LLM, а в шаблонах можно использовать [интеграцию](https://github.com/AlexxIT/MorphNumbers) AlexxIT
```
git clone https://github.com/mitrokun/wyoming_tts_vosk.git
cd wyoming_tts_vosk

# Установите требуемые библиотеки вручную. Простой путь для Win
pip install vosk-tts wyoming num2words numpy
# Запустите сервер (здесь пример с кастомным портом)
python -m wyoming_vosk --uri tcp://0.0.0.0:10205 --speech-rate 1.0

# Либо воспользуйтесь скриптами (linux) для поднятия venv и запуска 
script/setup
script/run
# Или с ключами
script/run --uri tcp://0.0.0.0:10222 --speech-rate 1.5

# в HA добавьте службу в интеграции Wyoming Protocol [IP и 10200, если порт не назначен ключем]
# Используется 0.7 версия модели, она сильно быстрее, чем 0.8. Измените, если требуется.

```
#### Опционально добавляем поддержку стримминга через интеграцию, полезно для работы с LLM.
https://github.com/mitrokun/streaming_tts_proxy
#### CUDA (12.x) 
Ставим пакет
`pip install onnxruntime-gpu`
Если есть ошибки при запуске, выполняйте предписания и устанавливайте требуемые версии cuda, cuDNN...



---

#### Предыдущая реализация сервера. Можно не читать.
```
git clone https://github.com/mitrokun/vosk_tts_hass.git
cd wyoming_tts_vosk
# Создайте и активируйте виртуальное окружение (опционально)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate  # Windows

# Установите необходимые библиотеки
pip install vosk-tts fastapi uvicorn[standard] soundfile python-dotenv num2words numpy

# Для ипользования с esp32 спутниками HA скорректируйте значение MAX_TEXT_LENGTH в main.py,
# число символов подбирайте под производительность cpu, суммарно процесс должен быть короче 5с
# высокий лимит для стандартных прошивок некритичен, будет отсутствовать ответ и терминал вернется к обнаружению WW
# Используется 0.7 версия модели, она сильно быстрее, чем 0.8. Измените, если требуется.

# Запуск
uvicorn main:app --host 0.0.0.0 --port 5002 --reload

# Тест
http://127.0.0.1:5002/synthesize?text=Привет мир!
```

### Интеграция

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
