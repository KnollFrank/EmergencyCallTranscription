from faster_whisper import WhisperModel
import numpy as np
from .Model import Model
from .Transcriber import Transcriber

class FasterWhisperTranscriber(Transcriber):

    def __init__(self, model_size: Model, device, compute_type, language, beam_size):
        self.model = WhisperModel(
            model_size,
            device = device,
            compute_type = compute_type)
        self.language = language
        self.beam_size = beam_size

    def transcribe(self, audio_16k: np.ndarray, speaker: str) -> list[dict]:
        segments, _ = self.model.transcribe(
            audio_16k,
            language = self.language,
            vad_filter = False, 
            word_timestamps = True,
            beam_size = self.beam_size)
        return [FasterWhisperTranscriber._convertSegment(segment, speaker) for segment in segments]

    @staticmethod
    def _convertSegment(segment, speaker):
        return {
            "speaker": speaker,
            "start": FasterWhisperTranscriber._round(segment.start),
            "end": FasterWhisperTranscriber._round(segment.end),
            "text": segment.text.strip()
        }

    @staticmethod
    def _round(number):
        return round(number, 2)
