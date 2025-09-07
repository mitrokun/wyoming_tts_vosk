import logging
import math
import asyncio
from typing import Optional

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.error import Error
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler
from wyoming.tts import (
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
    SynthesizeStopped,
)

from .speech_tts import SpeechTTS
from .sentence_boundary import SentenceBoundaryDetector

log = logging.getLogger(__name__)

class SpeechEventHandler(AsyncEventHandler):
    def __init__(
        self,
        wyoming_info: Info,
        cli_args,
        speech_tts: SpeechTTS,
        voice_to_speaker_map: dict,
        default_speaker_id: int,
        default_speech_rate: float,
        *args,  # Сюда попадут reader и writer от сервера
        **kwargs
    ) -> None:
        """Инициализация."""
        # Передаем "пойманные" reader и writer в родительский класс
        super().__init__(*args, **kwargs)

        # Сохраняем наши зависимости, которые мы передали явно
        self.cli_args = cli_args
        self.wyoming_info_event = wyoming_info.event()
        self.speech_tts = speech_tts
        self.voice_to_speaker_map = voice_to_speaker_map
        self.default_speaker_id = default_speaker_id
        self.default_speech_rate = default_speech_rate

        # Атрибуты для управления потоком
        self.is_streaming: bool = False
        self.sbd: Optional[SentenceBoundaryDetector] = None
        self._synthesize: Optional[Synthesize] = None
        
        log.debug(f"Handler initialized for new connection. Speaker: {self.default_speaker_id}, Rate: {self.default_speech_rate}")

    async def handle_event(self, event: Event) -> bool:
        """Обработка события от клиента."""
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            log.debug("Sent info")
            return True

        try:
            # === Обработка не-потокового запроса ===
            if Synthesize.is_type(event.type):
                # Если мы уже в режиме стриминга, это событие игнорируется
                # (оно нужно для обратной совместимости)
                if self.is_streaming:
                    return True
                
                # Если не в режиме стриминга, обрабатываем как обычно
                synthesize = Synthesize.from_event(event)
                return await self._handle_synthesize(synthesize)

            # (Если сервер запущен без поддержки стриминга, эти блоки не сработают)

            if SynthesizeStart.is_type(event.type):
                # Клиент начинает потоковую передачу
                stream_start = SynthesizeStart.from_event(event)
                self.is_streaming = True
                self.sbd = SentenceBoundaryDetector()
                # Сохраняем голос и другие параметры из стартового события
                self._synthesize = Synthesize(text="", voice=stream_start.voice)
                log.debug(f"Text stream started: voice={stream_start.voice}")
                return True

            if SynthesizeChunk.is_type(event.type):
                # Пришел очередной кусок текста
                assert self._synthesize is not None
                assert self.sbd is not None
                stream_chunk = SynthesizeChunk.from_event(event)

                # Добавляем чанк в детектор и синтезируем каждое готовое предложение
                for sentence in self.sbd.add_chunk(stream_chunk.text):
                    log.debug(f"Synthesizing stream sentence: {sentence}")
                    self._synthesize.text = sentence
                    await self._handle_synthesize(self._synthesize)
                
                return True

            if SynthesizeStop.is_type(event.type):
                # Клиент закончил передавать текст
                assert self._synthesize is not None
                assert self.sbd is not None
                
                # Обрабатываем оставшийся в буфере текст
                final_text = self.sbd.finish()
                if final_text:
                    self._synthesize.text = final_text
                    await self._handle_synthesize(self._synthesize)

                # Сообщаем клиенту, что мы закончили синтез с нашей стороны
                await self.write_event(SynthesizeStopped().event())

                # Сбрасываем состояние
                self.is_streaming = False
                self.sbd = None
                self._synthesize = None

                log.debug("Text stream stopped")
                return True

            log.warning("Unexpected event type: %s", event.type)
            return True

        except Exception as e:
            log.error(f"Error handling event: {e}", exc_info=True)
            await self.write_event(Error(text=str(e), code=e.__class__.__name__).event())
            # Сбрасываем состояние при ошибке
            self.is_streaming = False
            self.sbd = None
            self._synthesize = None
            return False # Возвращаем False, чтобы сервер мог разорвать соединение


    async def _handle_synthesize(self, synthesize: Synthesize) -> bool:
        """
        Основной метод синтеза. Теперь он вызывается как для целых фраз,
        так и для отдельных предложений из потока.
        """
        if not synthesize.text:
            log.warning("Received synthesize request with empty text")
            return True # Просто игнорируем пустые запросы

        requested_voice_name = synthesize.voice.name if synthesize.voice else None
        speaker_id = self.default_speaker_id
        speech_rate = self.default_speech_rate

        if requested_voice_name:
            found_speaker_id = self.voice_to_speaker_map.get(requested_voice_name)
            if found_speaker_id is not None:
                speaker_id = found_speaker_id
            else:
                log.warning(f"Voice '{requested_voice_name}' not found. Using default ID {self.default_speaker_id}.")

        if hasattr(synthesize, 'speech_rate') and synthesize.speech_rate is not None:
            speech_rate = synthesize.speech_rate
        
        text = " ".join(synthesize.text.strip().splitlines())
        log.debug(f"Processing synthesis: speaker={speaker_id}, rate={speech_rate}, text='{text[:50]}...'")

        audio_bytes = await self.speech_tts.synthesize(
            text=text, speaker_id=speaker_id, speech_rate=speech_rate
        )

        if audio_bytes is None:
            log.error(f"Synthesis failed for text: {text[:50]}...")
            await self.write_event(Error(text="TTS synthesis failed").event())
            return True # Не разрываем соединение, просто сообщаем об ошибке

        try:
            rate = self.speech_tts.sample_rate
            width = self.speech_tts.sample_width
            channels = self.speech_tts.channels

            await self.write_event(
                AudioStart(rate=rate, width=width, channels=channels).event()
            )

            bytes_per_sample = width * channels
            bytes_per_chunk = bytes_per_sample * self.cli_args.samples_per_chunk
            
            if bytes_per_chunk > 0:
                for i in range(0, len(audio_bytes), bytes_per_chunk):
                    chunk = audio_bytes[i : i + bytes_per_chunk]
                    await self.write_event(
                        AudioChunk(audio=chunk, rate=rate, width=width, channels=channels).event()
                    )
            elif len(audio_bytes) > 0:
                await self.write_event(
                    AudioChunk(audio=audio_bytes, rate=rate, width=width, channels=channels).event()
                )

            await self.write_event(AudioStop().event())
            log.debug("Completed synthesis request.")
        except Exception as e:
            log.error(f"Error streaming audio: {e}", exc_info=True)

        return True