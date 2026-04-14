from faster_whisper import WhisperModel
import numpy as np

class FasterWhisperTranscriber:

    def __init__(self, model_size, device, compute_type, language, beam_size):
        self.model = WhisperModel(
            model_size,
            device = device,
            compute_type = compute_type)
        self.language = language
        self.beam_size = beam_size

    def transcribe(self, audio_16k: np.ndarray, speaker: str):
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
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "text": segment.text.strip()
        }
