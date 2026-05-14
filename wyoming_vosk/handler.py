import logging
import asyncio
import time
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

from .vosk_engine import VoskEngine
from .ru_norm import RussianTextNormalizer
from .sentence_boundary import SentenceBoundaryDetector

_LOGGER = logging.getLogger(__name__)

class SpeechEventHandler(AsyncEventHandler):
    def __init__(
        self, 
        wyoming_info: Info, 
        cli_args, 
        engine: VoskEngine, 
        normalizer: RussianTextNormalizer, 
        voice_map: dict, 
        def_speaker: int, 
        def_rate: float, 
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.cli_args = cli_args
        self.wyoming_info_event = wyoming_info.event()
        self.engine = engine
        self.normalizer = normalizer
        self.voice_map = voice_map
        self.def_speaker = def_speaker
        self.def_rate = def_rate

        # Buffering configuration
        self.min_chars = getattr(cli_args, "min_characters", 20)
        self.max_chars = getattr(cli_args, "max_characters", 200)

        # State management
        self.is_streaming = False
        self.sbd: SentenceBoundaryDetector | None = None
        self._synthesize: Synthesize | None = None
        self._audio_started = False
        self._is_first_batch = True
        self._sentence_buffer = ""

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            return True

        try:
            # Single-request synthesis (non-streaming)
            if Synthesize.is_type(event.type):
                if self.is_streaming: 
                    return True
                return await self._handle_single_synthesize(Synthesize.from_event(event))

            # Start of streaming synthesis
            if SynthesizeStart.is_type(event.type):
                if getattr(self.cli_args, "disable_streaming", False):
                    return True
                await self._handle_stream_start(SynthesizeStart.from_event(event))
                return True

            # Incoming text chunks
            if SynthesizeChunk.is_type(event.type):
                if not self.is_streaming or not self.sbd:
                    return True
                for sentence in self.sbd.add_chunk(SynthesizeChunk.from_event(event).text):
                    await self._process_sentence(sentence)
                return True

            # End of stream
            if SynthesizeStop.is_type(event.type):
                if not self.is_streaming:
                    return True
                await self._handle_stream_stop()
                return True

        except Exception as e:
            _LOGGER.error(f"Event handling error: {e}", exc_info=True)
            try:
                await self.write_event(Error(text=str(e), code=e.__class__.__name__).event())
            except:
                pass
            self.is_streaming = False
            return False
        
        return True

    async def _handle_stream_start(self, stream_start: SynthesizeStart):
        self.is_streaming = True
        self.sbd = SentenceBoundaryDetector(emit_break_markers=True)
        self._synthesize = Synthesize(text="", voice=stream_start.voice)
        self._audio_started = False
        self._is_first_batch = True
        self._sentence_buffer = ""

    async def _handle_stream_stop(self):
        assert self.sbd is not None
        final_text = self.sbd.finish()
        if final_text:
            await self._process_sentence(final_text)
        
        await self._flush_buffer()

        if self._audio_started:
            await self.write_event(AudioStop().event())

        await self.write_event(SynthesizeStopped().event())
        self.is_streaming = False

    async def _handle_single_synthesize(self, synthesize: Synthesize) -> bool:
        self._synthesize = synthesize
        self._audio_started = False
        self._is_first_batch = True
        self._sentence_buffer = ""
        
        text = " ".join(synthesize.text.strip().splitlines())
        sbd = SentenceBoundaryDetector(emit_break_markers=True)
        
        for sentence in sbd.add_chunk(text):
            await self._process_sentence(sentence)
            
        final_text = sbd.finish()
        if final_text:
            await self._process_sentence(final_text)
            
        await self._flush_buffer()
        
        if self._audio_started:
            await self.write_event(AudioStop().event())
            
        return True

    async def _process_sentence(self, sentence: str):
        # Ignore structural markers
        if sentence in ("<PARAGRAPH_BREAK>", "<DIALOGUE_BREAK>"):
            return

        sentence = sentence.strip()
        if not sentence:
            return

        # Phase 1: Rapid response for the first batch
        if self._is_first_batch:
            if self._sentence_buffer:
                self._sentence_buffer += " " + sentence
            else:
                self._sentence_buffer = sentence
            
            if len(self._sentence_buffer) >= self.min_chars:
                await self._flush_buffer()
                self._is_first_batch = False
            return

        # Phase 2: Sentence merging for better RTFX efficiency
        current_len = len(self._sentence_buffer)
        new_len = len(sentence)

        if current_len > 0 and (current_len + new_len + 1) > self.max_chars:
            await self._flush_buffer()

        if self._sentence_buffer:
            self._sentence_buffer += " " + sentence
        else:
            self._sentence_buffer = sentence

    async def _flush_buffer(self):
        text_to_synth = self._sentence_buffer.strip()
        self._sentence_buffer = ""
        if text_to_synth:
            await self._synthesize_sentence(text_to_synth)

    async def _synthesize_sentence(self, sentence: str):
        normalized_text = self.normalizer.normalize(sentence)
        if not normalized_text: 
            return

        speaker_id = self.def_speaker
        rate = self.def_rate
        
        if self._synthesize and self._synthesize.voice and self._synthesize.voice.name in self.voice_map:
            speaker_id = self.voice_map[self._synthesize.voice.name]
        
        if self._synthesize and hasattr(self._synthesize, 'speech_rate') and self._synthesize.speech_rate:
            rate = self._synthesize.speech_rate

        _LOGGER.debug(f"Synth: '{normalized_text}'")
        start_time = time.monotonic()

        audio_bytes = await self.engine.synthesize(normalized_text, speaker_id, rate)
        if not audio_bytes: 
            return

        # Performance metrics
        elapsed_time = time.monotonic() - start_time
        audio_duration = len(audio_bytes) / (self.engine.sample_rate * self.engine.sample_width * self.engine.channels)
        rtfx = audio_duration / max(elapsed_time, 1e-6)
        _LOGGER.debug(f"Done: RTFX: {rtfx:.2f}x [{audio_duration:.2f}s / {elapsed_time:.2f}s]")

        try:
            # Initialize audio stream on first chunk
            if not self._audio_started:
                await self.write_event(
                    AudioStart(
                        rate=self.engine.sample_rate, 
                        width=self.engine.sample_width, 
                        channels=self.engine.channels
                    ).event()
                )
                self._audio_started = True
            
            # Stream audio in fixed-size chunks
            chunk_size = self.engine.sample_width * self.engine.channels * self.cli_args.samples_per_chunk
            for i in range(0, len(audio_bytes), chunk_size):
                await self.write_event(
                    AudioChunk(
                        audio=audio_bytes[i:i+chunk_size], 
                        rate=self.engine.sample_rate, 
                        width=self.engine.sample_width, 
                        channels=self.engine.channels
                    ).event()
                )
        except ConnectionError:
            raise
        except Exception as e:
            _LOGGER.error(f"Streaming error: {e}")