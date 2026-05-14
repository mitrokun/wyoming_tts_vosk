# Vosk TTS Home Assistant

## Сервер [vosk-tts](https://github.com/alphacep/vosk-tts) для HA, подключение через Wyoming Protocol
- Нативная поддержка в HA.
- 5 русских голосов [3f/2m] в стандартных моделях. 57 голосов [24f/33m] в версии 0.10.
- Реализована поддержка стриминга. Выполняется синтез отдельных предложений, что положительно влияет на объем используемой памяти.

 
Выполняется базовая конвертация чисел и английских слов (смотрите ru_norm.py). Лучше сразу делайте правильный системный промт для LLM, чтобы получать корректные формы числительных, а в шаблонах можно использовать [интеграцию](https://github.com/AlexxIT/MorphNumbers) AlexxIT

TTS негативно реагирует на нестандартные символы, поэтому весь текст отчищается перед синтезом. Знаком `+` перед гласной можно указать ударение.
```
git clone https://github.com/mitrokun/wyoming_tts_vosk.git
cd wyoming_tts_vosk

# Установите требуемые библиотеки вручную. Простой путь для Win
pip install wyoming num2words numpy eng_to_ipa regex silero-stress requests tqdm onnxruntime
# Запустите сервер (здесь пример с кастомным портом)
python -m wyoming_vosk --uri tcp://0.0.0.0:10205 --speech-rate 1.0

# Либо воспользуйтесь скриптами (linux) для поднятия venv и запуска 
script/setup
script/run
# Или с ключами
script/run --uri tcp://0.0.0.0:10222 --speech-rate 1.5

# Изначально используется 0.7 версия модели, она проще, но сильно быстрее последующих версий
# Если требуется, укажите другую версию --vosk-model-name vosk-model-tts-ru-0.9-multi
```
Версия 0.10 содержит [57 спикеров](https://mitrokun.github.io/voskvoice/), но недоступна для автоматической загрузки.

Требуется самостоятельно скачать [модель](https://alphacephei.com/vosk/models/vosk-model-tts-ru-0.10-multi.zip), после чего указать путь к каталогу
```
--vosk-model-path "D:\vosk-model-tts-ru-0.10-multi"
or
--vosk-model-path "/home/username/vosk-model-tts-ru-0.10-multi"
```


В в HA добавьте службу в интеграции Wyoming Protocol [`IP хоста` и `10205`, если порт не назначен ключем]
[![Open your Home Assistant instance and start setting up a new integration.](https://my.home-assistant.io/badges/config_flow_start.svg)](https://my.home-assistant.io/redirect/config_flow_start/?domain=wyoming)

#### CUDA (12.x) и прочие onnxruntime
Работоспособность тестировалась только на nvidia в windows.

Для небольшой модели 0.7 использовать не целесообразно. Для 0.10 — возможно, смотрите на rtfx в логах, запуская с `--debug` 

Ставим пакет (глобально или в венв)
`pip install onnxruntime-gpu`
Если есть ошибки при запуске, выполняйте предписания и устанавливайте требуемые версии cuda, cuDNN... 
в моём случае были установлены onnxruntime-gpu	1.21.1 + cuda toolkit 12.8 + cuddn 9.8

Для активации используется ключ `--provider "CUDAExecutionProvider"`

Другие [варианты](https://onnxruntime.ai/docs/install/) для самостоятельной проверки на соответствующем оборудобании
```
"TensorRTExecutionProvider",    # NVIDIA RTX
"ROCMExecutionProvider",        # AMD Linux
"CoreMLExecutionProvider",      # Apple Mac
"OpenVINOExecutionProvider",    # Intel
"DmlExecutionProvider",         # Windows (AMD/Intel/NVIDIA)
"VulkanExecutionProvider",      # Универсальный GPU
```

