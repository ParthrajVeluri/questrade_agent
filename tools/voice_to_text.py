import queue
import time
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel


class VoiceListener:
    def __init__(
        self,
        model_size="small",
        device="cpu",
        compute_type="int8",
        sample_rate=16000,
        silence_threshold=0.02,
        silence_duration=1.0,
        min_speech_duration=0.3,
    ):
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.min_speech_duration = min_speech_duration

    def _rms(self, chunk: np.ndarray) -> float:
        if chunk.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(chunk))))

    def listen_until_silence(self) -> str:
        q = queue.Queue()
        audio_chunks = []
        speech_started = False
        silence_started = None
        speech_started_at = None

        def callback(indata, frames, time_info, status):
            if status:
                print(status, flush=True)
            q.put(indata.copy())

        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=callback,
        )

        with stream:
            print("Listening...")
            stream.start()

            while True:
                chunk = q.get()
                audio = chunk[:, 0].astype(np.float32)

                level = self._rms(audio)
                if level > self.silence_threshold:
                    if not speech_started:
                        speech_started = True
                        speech_started_at = time.monotonic()
                    silence_started = None
                    audio_chunks.append(audio)
                    continue

                if speech_started:
                    if silence_started is None:
                        silence_started = time.monotonic()
                    if time.monotonic() - silence_started >= self.silence_duration:
                        total_speech = (
                            time.monotonic() - speech_started_at
                            if speech_started_at is not None
                            else 0
                        )
                        if total_speech >= self.min_speech_duration:
                            break

                # if we are still silent and haven't started speaking, ignore

            # combine captured audio
            if not audio_chunks:
                return ""

            full_audio = np.concatenate(audio_chunks).astype(np.float32)

            segments, _ = self.model.transcribe(
                full_audio,
                language="en",
                vad_filter=True,
                word_timestamps=False,
            )
            text = " ".join(segment.text for segment in segments).strip()

            print(f"You said: {text}")
            return text