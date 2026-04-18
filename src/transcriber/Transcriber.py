from typing import Protocol
import numpy as np

class Transcriber(Protocol):

    def transcribe(self, audio_16k: np.ndarray, speaker: str) -> list[dict]:
        ...
        