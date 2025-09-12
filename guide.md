### Краткая инструкция по запуску Vosk TTS в WSL2 (Ubuntu)

#### Шаг 1: Базовая настройка WSL (Ubuntu)

Откройте терминал Ubuntu и выполните:

```bash
# Обновление системы и установка базовых утилит
sudo apt update && sudo apt upgrade -y
sudo apt install -y git python3-pip python3-venv unzip
```

#### Шаг 2: Установка правильных версий CUDA и cuDNN

Это самый важный шаг. Мы установим **CUDA Toolkit 12.5** и **cuDNN 9.8.0**, так как эта комбинация доказала свою работоспособность.

```bash
# Настройка репозитория NVIDIA
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Установка CUDA Toolkit
sudo apt-get install -y cuda-toolkit-12-8

# Полное удаление любых других версий cuDNN для чистоты
sudo apt-get purge "libcudnn9*"

# Установка ТОЧНОЙ версии cuDNN 9.8.0
sudo apt-get install -y libcudnn9-dev-cuda-12=9.8.0.87-1 libcudnn9-cuda-12=9.8.0.87-1

# БЛОКИРОВКА версии cuDNN, чтобы она не обновлялась автоматически
sudo apt-mark hold libcudnn9-dev-cuda-12 libcudnn9-cuda-12
```

#### Шаг 3: Настройка переменных окружения

Это нужно, чтобы система знала, где искать установленный CUDA Toolkit.

```bash
# Добавляем пути в .bashrc для постоянной настройки
echo 'export PATH=/usr/local/cuda-12.5/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.5/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc

# Применяем изменения к текущей сессии терминала
source ~/.bashrc
```

#### Шаг 4: Установка проекта и модели Vosk

```bash
# Клонируем репозиторий сервера
git clone https://github.com/mitrokun/wyoming_tts_vosk.git
cd wyoming_tts_vosk

# Скачиваем и распаковываем модель (57 голосов)
wget https://alphacephei.com/vosk/models/vosk-model-tts-ru-0.10-multi.zip
unzip vosk-model-tts-ru-0.10-multi.zip

# Установка
script/setup
```

#### Шаг 5: Установка опциональной библиотеки

```bash
# Устанавливаем onnxruntime-gpu отдельно
source .venv/bin/activate
pip install onnxruntime-gpu
deactivate
```

#### Шаг 6: Запуск сервера

Теперь все готово к запуску.

```bash
# Запускаем сервер, указав путь к модели
# Замените USERNAME на ваше имя пользователя
script/run --uri tcp://0.0.0.0:10205 --streaming --vosk-model-path "/home/USERNAME/wyoming_tts_vosk/vosk-model-tts-ru-0.10-multi"
```
