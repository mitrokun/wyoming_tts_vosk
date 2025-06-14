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
