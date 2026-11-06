import subprocess
import pyttsx3
import os

from dotenv import load_dotenv
from helper import debug

load_dotenv()
TTS_ENABLED = os.getenv("TTS_ENABLED", "false").lower() in {"1", "true", "yes", "on"}

class TTS:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('voice', 'mb-us1')
        self.engine.setProperty('rate', 150) 
        self.engine.setProperty('volume', 0.9)

    def speak(self, text):
        if not TTS_ENABLED:
            debug("[tts] disabled; set TTS_ENABLED=true to enable speech")
            return

        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as exc:  # pragma: no cover - optional runtime dependency
            debug(f"[tts] speech unavailable: {exc}")

    def windows_speak(self,text):
        """Pipes text out of WSL2 and leverages the high-quality native Windows TTS engine."""
        # Clean text to avoid breaking PowerShell arguments
        safe_text = text.replace('"', "'").replace("\n", " ")
        
        # PowerShell command to run on the Windows host machine
        ps_command = f'Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak("{safe_text}")'
        
        # Execute through powershell.exe from inside WSL2
        subprocess.run(["powershell.exe", "-Command", ps_command], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)