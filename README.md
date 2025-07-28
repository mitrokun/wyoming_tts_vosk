# Vosk TTS Home Assistant

## Сервер [vosk-tts](https://github.com/alphacep/vosk-tts) для HA, подключение через Wyoming Protocol
- Нативная поддержка протокола в HA.
- Пять русских голосов. 3f/2m
 
Работа с числами и английскими словами максимально упрощенная (смотрите speech_tts.py), лишь бы воск не мычал.
Синтезатор негативно реагирует на разные символы, поэтому всё вычищено. Знаком `+` перед гласной можно указать ударение.
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

# Используется 0.7 версия модели, она сильно быстрее, чем 0.8.
# Если требуется, используйте --vosk-model-name vosk-model-tts-ru-0.8-multi

```
В в HA добавьте службу в интеграции Wyoming Protocol [`IP хоста` и `10205`, если порт не назначен ключем]

[![Open your Home Assistant instance and start setting up a new integration.](https://my.home-assistant.io/badges/config_flow_start.svg)](https://my.home-assistant.io/redirect/config_flow_start/?domain=wyoming)

Дополнительно и/или альтернативно добавляем поддержку стримминга через [интеграцию](https://github.com/mitrokun/streaming_tts_proxy) альтернативного клиента. Полезно для работы с LLM.

upd

Есть тестовый [бранч](https://github.com/mitrokun/wyoming_tts_vosk/tree/streaming) с нативной поддержкой сервером стриминга, который представил в 2025.07. Вариант для тех, кому не требуется расширеный функционал предыдущей интеграции.

#### CUDA (12.x) 
Ставим пакет
`pip install onnxruntime-gpu`
Если есть ошибки при запуске, выполняйте предписания и устанавливайте требуемые версии cuda, cuDNN...
Возможность выбирать устройство обработки отстутвует в библиотеке vosk-tts, если обнаружен onnxruntime-gpu, то выполняться будет на gpu.

---
Создано при участии Grok и Gemini. 

Дорабатывайте и улучшайте на свое усмотретние.
