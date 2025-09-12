import os
from argparse import ArgumentParser
import asyncio
import contextlib
import logging
import sys
from functools import partial

from wyoming.info import Attribution, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncServer

from .handler import SpeechEventHandler
from .speech_tts import SpeechTTS

log = logging.getLogger(__name__)

# --- Константы ---
DEFAULT_VOSK_MODEL_NAME = "vosk-model-tts-ru-0.7-multi"
DEFAULT_SPEAKER_ID = 3
DEFAULT_SPEECH_RATE = 1.0
MODEL_LANGUAGE = "ru-RU"
DEFAULT_VOICE_VERSION = "1.0"
VOSK_ATTRIBUTION_NAME = "Vosk"
VOSK_ATTRIBUTION_URL = "https://alphacephei.com/vosk/"
PROGRAM_NAME = "vosk-tts-wyoming"
PROGRAM_DESCRIPTION = "Wyoming server for Vosk TTS"
PROGRAM_VERSION = "1.3"

# Карта гендеров для модели 0.10 (57 голосов)
GENDER_SEQUENCE_0_10 = "mffmfffmmfmfmmmffmffmmfmfmmmmfmmmfmmfmfmmmfmfmfmffmfmfmmm"

VOICE_MAP_LEGACY = {
    0: ("Female 01", "female_01"),
    1: ("Female 02", "female_02"),
    2: ("Female 03", "female_03"),
    3: ("Male 01", "male_01"),
    4: ("Male 02", "male_02"),
}

async def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--uri", default="tcp://0.0.0.0:10205", help="unix:// or tcp://"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging for detailed output.",
    )
    parser.add_argument("--samples-per-chunk", type=int, default=1024)

    parser.add_argument(
        "--vosk-model-name",
        default=None,
        help=f"Name of the Vosk TTS model to download from Hub. Example: {DEFAULT_VOSK_MODEL_NAME}"
    )
    parser.add_argument(
        "--vosk-model-path",
        default=None,
        help="Path to a local directory containing the Vosk TTS model (e.g., for model 0.10)."
    )

    parser.add_argument(
        "--default-speaker-id",
        type=int,
        default=DEFAULT_SPEAKER_ID,
        help=f"Default speaker ID to use if none is specified. Default: {DEFAULT_SPEAKER_ID}"
    )
    parser.add_argument(
        "--speech-rate",
        type=float,
        default=DEFAULT_SPEECH_RATE,
        help=f"Default speech rate (speed) for synthesis. Default: {DEFAULT_SPEECH_RATE}"
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Enable audio streaming on sentence boundaries.",
    )
    parser.add_argument(
        "--use-accentizer",
        action="store_true",
        help="Enable automatic stress marking (accentuation) using ruaccent. Requires `ruaccent` to be installed.",
    )
    args = parser.parse_args()
    
    if not args.vosk_model_name and not args.vosk_model_path:
        args.vosk_model_name = DEFAULT_VOSK_MODEL_NAME
        log.info(f"No model source specified, falling back to default model name: {args.vosk_model_name}")

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
        log.info("Debug logging enabled.")
    else:
        log.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        log.addHandler(handler)
        log.propagate = False
        log.info("Log level set to INFO.")

    if args.use_accentizer:
        log.info("Accentizer (ruaccent) is ENABLED.")
    else:
        log.info("Accentizer (ruaccent) is DISABLED.")

    # Предзагрузка модели Vosk
    try:
        log.info("Attempting to preload Vosk model...")
        speech_tts_instance = SpeechTTS(
            vosk_model_name=args.vosk_model_name,
            vosk_model_path=args.vosk_model_path,
            use_accentizer=args.use_accentizer
        )
        log.info("Vosk model preloaded successfully.")
    except (RuntimeError, ValueError) as e:
        log.critical(f"Failed to preload Vosk model: {e}", exc_info=True)
        sys.exit(1)
        
    num_speakers = speech_tts_instance.num_speakers

    try:
        if not (0 <= args.default_speaker_id < num_speakers):
            log.warning(f"Default speaker ID {args.default_speaker_id} is out of range for this model [0, {num_speakers-1}]. Using ID 0 as default.")
            args.default_speaker_id = 0
        if args.speech_rate <= 0:
            raise ValueError("Speech rate must be positive.")
        log.info(f"Default speaker ID set to: {args.default_speaker_id}")
        log.info(f"Default speech rate set to: {args.speech_rate}")
    except ValueError as e:
        log.error(f"Invalid arguments: {e}")
        sys.exit(1)

    # Подготовка информации для Wyoming
    voices = []
    voice_to_speaker_map = {}
    log.info(f"Generating voice list for {num_speakers} speakers...")

    # Определяем, какую модель мы используем, по количеству голосов
    if num_speakers == len(GENDER_SEQUENCE_0_10):
        log.info("Detected new model with 57 speakers. Generating gender-based names.")
        male_count = 0
        female_count = 0
        for speaker_id in range(num_speakers):
            gender = GENDER_SEQUENCE_0_10[speaker_id]
            if gender == 'm':
                male_count += 1
                voice_description = f"Male {male_count:02d}"
                voice_name = f"male_{male_count}"
            else: # 'f'
                female_count += 1
                voice_description = f"Female {female_count:02d}"
                voice_name = f"female_{female_count}"
            
            voices.append(TtsVoice(name=voice_name, description=voice_description, attribution=Attribution(name=VOSK_ATTRIBUTION_NAME, url=VOSK_ATTRIBUTION_URL), installed=True, version=DEFAULT_VOICE_VERSION, languages=[MODEL_LANGUAGE]))
            voice_to_speaker_map[voice_name] = speaker_id
    else:
        # Логика для старых или неизвестных моделей
        log.info("Detected legacy or unknown model. Generating names from legacy map or generic names.")
        for speaker_id in range(num_speakers):
            if speaker_id in VOICE_MAP_LEGACY:
                voice_description, voice_name = VOICE_MAP_LEGACY[speaker_id]
            else:
                voice_description = f"Speaker {speaker_id}"
                voice_name = f"speaker_{speaker_id}"
                
            voices.append(TtsVoice(name=voice_name, description=voice_description, attribution=Attribution(name=VOSK_ATTRIBUTION_NAME, url=VOSK_ATTRIBUTION_URL), installed=True, version=DEFAULT_VOICE_VERSION, languages=[MODEL_LANGUAGE]))
            voice_to_speaker_map[voice_name] = speaker_id

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
                voices=voices,
                supports_synthesize_streaming=args.streaming,
            )
        ]
    )

    handler_factory = partial(
        SpeechEventHandler,
        wyoming_info,
        args,
        speech_tts_instance,
        voice_to_speaker_map,
        args.default_speaker_id,
        args.speech_rate,
    )

    server = AsyncServer.from_uri(args.uri)
    log.info(f"Server ready and listening at {args.uri}")
    await server.run(handler_factory)


if __name__ == "__main__":
    try:
        with contextlib.suppress(KeyboardInterrupt):
            asyncio.run(main())
    finally:
        log.info("Server shutting down.")
