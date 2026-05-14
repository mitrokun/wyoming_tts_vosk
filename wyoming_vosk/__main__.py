import asyncio
import logging
import sys
from argparse import ArgumentParser
from functools import partial

from wyoming.info import Attribution, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncServer

from .handler import SpeechEventHandler
from .ru_norm import RussianTextNormalizer
from .vosk_engine import VoskEngine
from .vosk_model import VoskModel

log = logging.getLogger(__name__)

# --- Константы для определения голосов ---
# Последовательность полов для модели 0.10 (57 спикеров)
GENDER_SEQ_0_10 = "mffmfffmmfmfmmmffmffmmfmfmmmmfmmmfmmfmfmmmfmfmfmffmfmfmmm"

VOICE_MAP_LEGACY = {
    0: ("Female 01", "female_1"),
    1: ("Female 02", "female_2"),
    2: ("Female 03", "female_3"),
    3: ("Male 01", "male_1"),
    4: ("Male 02", "male_2"),
}


def main() -> None:
    parser = ArgumentParser(description="Vosk TTS Wyoming Server")
    parser.add_argument("--uri", default="tcp://0.0.0.0:10205")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--samples-per-chunk", type=int, default=1024)
    parser.add_argument("--vosk-model-name", default="vosk-model-tts-ru-0.7-multi")
    parser.add_argument("--vosk-model-path", default=None)
    parser.add_argument("--default-speaker-id", type=int, default=1)
    parser.add_argument("--speech-rate", type=float, default=1.0)
    parser.add_argument(
        "--disable-streaming", action="store_true", help="Disable audio streaming"
    )
    parser.add_argument(
        "--min-characters", 
        type=int, 
        default=20, 
        help="Min characters to buffer for the first synthesis request (default: 20)"
    )
    parser.add_argument(
        "--max-characters", 
        type=int, 
        default=200, 
        help="Max character limit for combining sentences after the first request (default: 200)"
    )
    parser.add_argument(
        "--provider",
        choices=[
            "CUDAExecutionProvider",        # NVIDIA
            "TensorRTExecutionProvider",    # NVIDIA RTX
            "ROCMExecutionProvider",        # AMD Linux
            "CoreMLExecutionProvider",      # Apple Mac
            "OpenVINOExecutionProvider",    # Intel
            "DmlExecutionProvider",         # Windows (AMD/Intel/NVIDIA)
            "VulkanExecutionProvider",      # Универсальный GPU
            "CPUExecutionProvider"          # Базовый / CPU
        ],
        default="CPUExecutionProvider",
        help="ONNX execution provider (e.g., CUDAExecutionProvider)"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    log.info("Loading normalizer...")
    normalizer = RussianTextNormalizer()

    log.info("Loading Vosk model...")
    try:
        # Передаем новый ключ provider в модель
        model = VoskModel(
            model_path=args.vosk_model_path,
            model_name=args.vosk_model_name if not args.vosk_model_path else None,
            provider=args.provider
        )
        engine = VoskEngine(model)
    except Exception as e:
        log.critical("Engine initialization failed: %s", e, exc_info=True)
        sys.exit(1)

    voices = []
    voice_map = {}

    num_speakers = engine.num_speakers
    log.info("Generating voice list for %d speakers...", num_speakers)

    # Логика распределения имен
    if num_speakers == len(GENDER_SEQ_0_10):
        log.info("Detected model with 57 speakers. Generating gender-based names.")
        male_count = 0
        female_count = 0
        for speaker_id in range(num_speakers):
            gender = GENDER_SEQ_0_10[speaker_id]
            if gender == "m":
                male_count += 1
                voice_desc = f"Male {male_count:02d}"
                voice_name = f"male_{male_count}"
            else:
                female_count += 1
                voice_desc = f"Female {female_count:02d}"
                voice_name = f"female_{female_count}"

            voices.append(
                TtsVoice(
                    name=voice_name,
                    description=voice_desc,
                    attribution=Attribution(
                        name="Vosk", url="https://alphacephei.com/vosk/"
                    ),
                    installed=True,
                    version="1.0",
                    languages=["ru"],
                )
            )
            voice_map[voice_name] = speaker_id
    else:
        log.info("Detected legacy model. Generating names from map or generic.")
        for speaker_id in range(num_speakers):
            if speaker_id in VOICE_MAP_LEGACY:
                voice_desc, voice_name = VOICE_MAP_LEGACY[speaker_id]
            else:
                voice_desc = f"Speaker {speaker_id}"
                voice_name = f"speaker_{speaker_id}"

            voices.append(
                TtsVoice(
                    name=voice_name,
                    description=voice_desc,
                    attribution=Attribution(
                        name="Vosk", url="https://alphacephei.com/vosk/"
                    ),
                    installed=True,
                    version="1.0",
                    languages=["ru"],
                )
            )
            voice_map[voice_name] = speaker_id

    wyoming_info = Info(
        tts=[
            TtsProgram(
                name="vosk-tts-wyoming",
                description="Vosk TTS for Wyoming",
                attribution=Attribution(
                    name="Vosk", url="https://alphacephei.com/vosk/"
                ),
                installed=True,
                version="2.0",
                voices=voices,
                supports_synthesize_streaming=not args.disable_streaming,
            )
        ]
    )

    handler_factory = partial(
        SpeechEventHandler,
        wyoming_info,
        args,
        engine,
        normalizer,
        voice_map,
        args.default_speaker_id,
        args.speech_rate,
    )

    server = AsyncServer.from_uri(args.uri)
    log.info("Server ready at %s (Streaming: %s)", args.uri, not args.disable_streaming)

    try:
        asyncio.run(server.run(handler_factory))
    except KeyboardInterrupt:
        log.info("Server shutting down.")


if __name__ == "__main__":
    main()