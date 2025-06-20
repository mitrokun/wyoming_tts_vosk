# --- START OF FILE handler.py ---

import logging
import math
import contextlib
import asyncio

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.error import Error
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler
from wyoming.tts import Synthesize

from .speech_tts import SpeechTTS

log = logging.getLogger(__name__)

class SpeechEventHandler(AsyncEventHandler):
    def __init__(
        self,
        wyoming_info: Info,
        cli_args,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        *,
        speech_tts: SpeechTTS,
        voice_to_speaker_map: dict,
        default_speaker_id: int,
        default_speech_rate: float,  # Добавляем параметр
        **kwargs
    ) -> None:
        """Инициализация."""
        super().__init__(reader, writer)
        self.cli_args = cli_args
        self.wyoming_info_event = wyoming_info.event()
        self.speech_tts = speech_tts
        self.voice_to_speaker_map = voice_to_speaker_map
        self.default_speaker_id = default_speaker_id
        self.default_speech_rate = default_speech_rate
        log.debug(f"Handler initialized for new connection. Using shared TTS instance. Default speaker ID: {self.default_speaker_id}, Default speech rate: {self.default_speech_rate}")
        log.debug(f"Voice to speaker map: {self.voice_to_speaker_map}")

    async def handle_event(self, event: Event) -> bool:
        """Обработка события от клиента."""
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            log.debug("Sent info")
            return True

        if Synthesize.is_type(event.type):
            synthesize = Synthesize.from_event(event)
            if not synthesize.text:
                log.warning("Received synthesize request with empty text")
                await self.write_event(Error(text="Text cannot be empty").event())
                return True

            requested_voice_name = synthesize.voice.name if synthesize.voice else None
            speaker_id = self.default_speaker_id
            speech_rate = self.default_speech_rate  # Используем значение по умолчанию

            if requested_voice_name:
                found_speaker_id = self.voice_to_speaker_map.get(requested_voice_name)
                if found_speaker_id is not None:
                    speaker_id = found_speaker_id
                    log.debug(f"Using requested speaker ID {speaker_id} for voice '{requested_voice_name}'")
                else:
                    log.warning(f"Requested voice '{requested_voice_name}' not found. Using default ID {self.default_speaker_id}.")

            # Проверяем, указан ли speech_rate в запросе
            if hasattr(synthesize, 'speech_rate') and synthesize.speech_rate is not None:
                speech_rate = synthesize.speech_rate
                log.debug(f"Using requested speech rate: {speech_rate}")
            else:
                log.debug(f"No speech rate requested. Using default speech rate: {self.default_speech_rate}")

            log.debug(f"Processing in-memory synthesis request: speaker_id={speaker_id}, speech_rate={speech_rate}, text='{synthesize.text[:50]}...'")
            text = " ".join(synthesize.text.strip().splitlines())

            # Получаем байты аудио из памяти
            audio_bytes = await self.speech_tts.synthesize(
                text=text,
                speaker_id=speaker_id,
                speech_rate=speech_rate
            )

            if audio_bytes is None:
                log.error(f"In-memory text synthesis failed for speaker {speaker_id}, speech_rate: {speech_rate}, text: {text[:50]}...")
                try:
                    await self.write_event(Error(text=f"Vosk TTS in-memory synthesis failed for speaker {speaker_id}").event())
                except Exception as e:
                    log.error(f"Failed to send Error event to client: {e}")
                return True

            try:
                rate = self.speech_tts.sample_rate
                width = self.speech_tts.sample_width
                channels = self.speech_tts.channels

                log.debug(f"Streaming in-memory audio: rate={rate}, width={width}, channels={channels}, size={len(audio_bytes)} bytes")

                await self.write_event(
                    AudioStart(
                        rate=rate,
                        width=width,
                        channels=channels,
                    ).event(),
                )

                bytes_per_sample = width * channels
                if bytes_per_sample == 0:
                    log.error("Error: bytes_per_sample is zero. Check audio parameters.")
                    await self.write_event(Error(text="Internal server error: invalid audio parameters").event())
                    return True

                bytes_per_chunk = bytes_per_sample * self.cli_args.samples_per_chunk
                num_chunks = math.ceil(len(audio_bytes) / bytes_per_chunk) if bytes_per_chunk > 0 else 0
                log.debug(f"Chunk size: {bytes_per_chunk}, num chunks: {num_chunks}")

                if bytes_per_chunk > 0:
                    for i in range(num_chunks):
                        offset = i * bytes_per_chunk
                        chunk = audio_bytes[offset : offset + bytes_per_chunk]
                        await self.write_event(
                            AudioChunk(
                                audio=chunk,
                                rate=rate,
                                width=width,
                                channels=channels,
                            ).event(),
                        )
                elif len(audio_bytes) > 0:
                    log.warning("Chunk size is zero, sending all audio in one chunk.")
                    await self.write_event(
                        AudioChunk(
                            audio=audio_bytes,
                            rate=rate,
                            width=width,
                            channels=channels,
                        ).event(),
                    )

                await self.write_event(AudioStop().event())
                log.debug(f"Completed in-memory synthesis request for speaker {speaker_id}, speech_rate: {speech_rate}, text: {text[:50]}...")

            except Exception as e:
                log.error(f"Error streaming in-memory audio: {e}", exc_info=True)
                try:
                    await self.write_event(Error(text=f"Error streaming synthesized audio").event())
                except Exception as e_send:
                    log.error(f"Failed to send Error event to client after streaming failure: {e_send}")

            return True

        log.warning("Unexpected event type: %s", event.type)
        return True

# --- END OF FILE handler.py ---