"""Text-to-speech engine selection and playback helpers."""

import pyttsx3
import os
from kokoro_onnx import Kokoro
import sounddevice as sd
from pathlib import Path

from dotenv import load_dotenv
from helper import debug

load_dotenv()
TTS_ENABLED = os.getenv("TTS_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
TTS_ENGINE = os.getenv("TTS_ENGINE", "pyttsx3").lower()
MODEL_DIR = Path(__file__).resolve().parent / "tts_soundpacks"
KOKORO_MODEL = MODEL_DIR / "kokoro-v1.0.onnx"
KOKORO_VOICES = MODEL_DIR / "voices-v1.0.bin"

class TTS:
    """Speak text with the configured local TTS engine."""

    def __init__(self, engine_name: str | None = None):
        """Initialize the selected engine only when TTS is enabled."""
        self.enabled = TTS_ENABLED
        self.engine_name = (engine_name or TTS_ENGINE).strip().lower()
        self.engine = None

        if not self.enabled:
            debug("[tts] disabled; set TTS_ENABLED=true to enable speech")
            return

        if self.engine_name is None:
            self.engine_name = TTS_ENGINE

        if self.engine_name == "pyttsx3":
            self.engine = pyttsx3.init()
            self.engine.setProperty('voice', 'mb-us1')
            self.engine.setProperty('rate', 150) 
            self.engine.setProperty('volume', 0.9)
        elif self.engine_name == "kokoro":
            try:
                self.engine = Kokoro(
                    str(KOKORO_MODEL),
                    str(KOKORO_VOICES),
                )
            except Exception as exc:
                debug(f"[tts] Failed to load Kokoro model: {exc}")
        else:
            raise ValueError(f"Unsupported TTS engine: {engine_name}")

    def speak(self, text:str):
        """Dispatch non-empty text to the active speech engine."""
        if not self.enabled:
            debug("[tts] disabled; set TTS_ENABLED=true to enable speech")
            return

        if not text or not text.strip():
            debug("[tts] No text provided for speech synthesis.")
            return

        dispatch = {
            "pyttsx3": self._speak_pyttsx3,
            "kokoro": self._speak_kokoro,
        }

        if self.engine_name in dispatch:
            dispatch[self.engine_name](text)
        else:
            debug(f"[tts] No valid TTS engine found for: {self.engine_name}")

    def _speak_pyttsx3(self, text):
        """Speak text synchronously through pyttsx3."""
        if not self.enabled:
            debug("[tts] disabled; set TTS_ENABLED=true to enable speech")
            return

        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as exc:  # pragma: no cover - optional runtime dependency
            debug(f"[tts] speech unavailable: {exc}")

    def _speak_kokoro(self, text):
        """Synthesize and play text through Kokoro."""
        if self.engine is None:
            debug("[tts] Kokoro engine is unavailable")
            return
        try:
            samples, sample_rate = self.engine.create(
                text,
                voice="af_heart",
                speed=1.0,
                lang="en-us",
            )
            sd.play(samples, sample_rate)
            sd.wait()
        except Exception as exc:  # pragma: no cover - optional runtime dependency
            debug(f"[tts] speech unavailable: {exc}")