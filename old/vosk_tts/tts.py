import aiohttp
import voluptuous as vol
from homeassistant.components.tts import Provider, PLATFORM_SCHEMA, Voice
import homeassistant.helpers.config_validation as cv
import logging
from urllib.parse import quote

_LOGGER = logging.getLogger(__name__)

PLATFORM_SCHEMA = PLATFORM_SCHEMA.extend({
    vol.Required("url"): cv.url,
    vol.Optional("default_voice", default="0"): vol.In(["0", "1", "2", "3", "4"]),
    vol.Optional("default_speech_rate", default=1.0): vol.All(vol.Coerce(float), vol.Range(min=0.3, max=2.0)),
})

def get_engine(hass, config, discovery_info=None):
    return VoskTTSProvider(hass, config["url"], config["default_voice"], config["default_speech_rate"])

class VoskTTSProvider(Provider):
    def __init__(self, hass, url, default_voice, default_speech_rate):
        self.hass = hass
        self._url = url
        self._default_voice = default_voice
        self._default_speech_rate = default_speech_rate
        self.name = "Vosk TTS"

    @property
    def default_language(self):
        return "ru-RU"

    @property
    def supported_languages(self):
        return ["ru-RU"]

    @property
    def supported_options(self):
        return ["voice", "speech_rate"]

    async def async_get_tts_audio(self, message, language, options=None):
        try:
            speaker = self._default_voice
            speech_rate = self._default_speech_rate
            if options:
                if "voice" in options:
                    speaker = options["voice"]
                if "speech_rate" in options:
                    speech_rate = options["speech_rate"]

            params = {"text": message, "speaker": speaker, "speech_rate": speech_rate}
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.get(self._url, params=params) as response:
                    if response.status != 200:
                        _LOGGER.error("Error fetching TTS audio: %s", response.status)
                        return None, None
                    content_type = response.headers.get("Content-Type", "")
                    if "audio/wav" not in content_type:
                        _LOGGER.error("Unsupported audio format: %s", content_type)
                        return None, None
                    audio_data = await response.read()
                    return "wav", audio_data
        except Exception as e:
            _LOGGER.error("Error synthesizing TTS for text '%s' with speaker '%s' and speech_rate '%s': %s", message, speaker, speech_rate, e)
            return None, None
