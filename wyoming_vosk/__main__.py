# --- START OF FILE __main__.py ---

import os
from argparse import ArgumentParser
import asyncio
import contextlib
import logging
import sys
from functools import partial

from wyoming.info import Attribution, Info, TtsProgram, TtsVoice, Describe
from wyoming.server import AsyncServer
from wyoming.error import Error

from .handler import SpeechEventHandler
from .speech_tts import SpeechTTS

log = logging.getLogger(__name__)
logging.getLogger('vosk_tts').setLevel(logging.WARNING)

# --- Константы ---
DEFAULT_VOSK_MODEL_NAME = "vosk-model-tts-ru-0.7-multi"
DEFAULT_VOSK_SPEAKER_IDS = "0,1,2,3,4"
DEFAULT_SPEAKER_ID = 3
DEFAULT_SPEECH_RATE = 1.0
MODEL_LANGUAGE = "ru"
DEFAULT_VOICE_VERSION = "1.1"
VOSK_ATTRIBUTION_NAME = "Vosk"
VOSK_ATTRIBUTION_URL = "https://alphacephei.com/vosk/"
PROGRAM_NAME = "vosk-tts-wyoming"
PROGRAM_DESCRIPTION = "Wyoming server for Vosk TTS"
PROGRAM_VERSION = "1.1"

# Ключ: числовой ID спикера. Значение: (Описание для пользователя, Часть имени для Wyoming)
VOICE_MAP = {
    0: ("Female 01", "female_01"),
    1: ("Female 02", "female_02"),
    2: ("Female 03", "female_03"),
    3: ("Male 01", "male_01"),
    4: ("Male 02", "male_02"),
}

async def main() -> None:
    logging.basicConfig(level=os.getenv("LOGLEVEL", "INFO"))
    parser = ArgumentParser()
    parser.add_argument(
        "--uri", default="tcp://0.0.0.0:10200", help="unix:// or tcp://"
    )
    parser.add_argument("--samples-per-chunk", type=int, default=1024)
    parser.add_argument(
        "--vosk-model-name",
        default=DEFAULT_VOSK_MODEL_NAME,
        help=f"Name or path of the Vosk TTS model. Default: {DEFAULT_VOSK_MODEL_NAME}"
    )
    parser.add_argument(
        "--vosk-speaker-ids",
        default=DEFAULT_VOSK_SPEAKER_IDS,
        help=f"Comma-separated list of speaker IDs available in the model. Default: '{DEFAULT_VOSK_SPEAKER_IDS}'"
    )
    parser.add_argument(
        "--default-speaker-id",
        type=int,
        default=DEFAULT_SPEAKER_ID,
        help=f"Default speaker ID to use if none is specified in the request. Default: {DEFAULT_SPEAKER_ID}"
    )
    parser.add_argument(
        "--speech-rate",
        type=float,
        default=DEFAULT_SPEECH_RATE,
        help=f"Default speech rate (speed) for synthesis. Default: {DEFAULT_SPEECH_RATE}"
    )
    args = parser.parse_args()

    # --- Обработка аргументов Vosk ---
    try:
        speaker_ids = [int(sid.strip()) for sid in args.vosk_speaker_ids.split(',')]
        if not speaker_ids:
            raise ValueError("No speaker IDs provided.")
        if args.default_speaker_id not in speaker_ids:
            log.warning(f"Default speaker ID {args.default_speaker_id} is not in the provided list of speaker IDs {speaker_ids}. Using the first available ID ({speaker_ids[0]}) as default.")
            args.default_speaker_id = speaker_ids[0]

        if args.speech_rate <= 0:
            raise ValueError("Speech rate must be positive.")
        log.info(f"Using Vosk model: {args.vosk_model_name}")
        log.info(f"Available speaker IDs: {speaker_ids}")
        log.info(f"Default speaker ID: {args.default_speaker_id}")
        log.info(f"Default speech rate: {args.speech_rate}")

    except ValueError as e:
        log.error(f"Invalid arguments: {e}")
        sys.exit(1)

    # --- Предзагрузка модели Vosk ---
    try:
        log.info("Attempting to preload Vosk model...")
        speech_tts_instance = SpeechTTS(vosk_model_name=args.vosk_model_name)
        log.info("Vosk model preloaded successfully.")
    except RuntimeError as e:
        log.critical(f"Failed to preload Vosk model: {e}", exc_info=True)
        sys.exit(1)

    # --- Подготовка информации Wyoming ---
    
    voices = []
    voice_to_speaker_map = {}

    for speaker_id in speaker_ids:
        if speaker_id in VOICE_MAP:
            voice_description, voice_name_part = VOICE_MAP[speaker_id]
            voice_name = f"vosk_{voice_name_part}"

            voices.append(
                TtsVoice(
                    name=voice_name,
                    description=voice_description,
                    attribution=Attribution(
                        name=VOSK_ATTRIBUTION_NAME,
                        url=VOSK_ATTRIBUTION_URL
                    ),
                    installed=True,
                    version=DEFAULT_VOICE_VERSION,
                    languages=[MODEL_LANGUAGE]
                )
            )
            voice_to_speaker_map[voice_name] = speaker_id
        else:
            log.warning(f"Speaker ID {speaker_id} is not defined in VOICE_MAP and will be ignored.")

    wyoming_info = Info(
        tts=[
            TtsProgram(
                name=PROGRAM_NAME,
                description=PROGRAM_DESCRIPTION,
                attribution=Attribution(
                    name=VOSK_ATTRIBUTION_NAME,
                    url=VOSK_ATTRIBUTION_URL
                ),
                installed=True,
                version=PROGRAM_VERSION,
                voices=voices
            )
        ]
    )

    # --- Подготовка фабрики обработчиков ---
    handler_factory = partial(
        SpeechEventHandler,
        wyoming_info,
        args,
        speech_tts=speech_tts_instance,
        voice_to_speaker_map=voice_to_speaker_map,
        default_speaker_id=args.default_speaker_id,
        default_speech_rate=args.speech_rate,
    )

    # --- Запуск сервера ---
    server = AsyncServer.from_uri(args.uri)
    log.info(f"Server ready and listening at {args.uri}")
    await server.run(handler_factory)

if __name__ == "__main__":
    try:
        with contextlib.suppress(KeyboardInterrupt):
            asyncio.run(main())
    finally:
        log.info("Server shutting down.")

# --- END OF FILE __main__.py ---
