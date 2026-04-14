from faster_whisper import WhisperModel
import numpy as np

class WhisperTranscriptionModel:

    def __init__(self, model_size, device, compute_type, language, beam_size):
        self.model = WhisperModel(
            model_size,
            device = device,
            compute_type = compute_type)
        self.language = language
        self.beam_size = beam_size

    def transcribe(self, audio_16k: np.ndarray, speaker: str) -> list[dict]:
        segments_generator, _ = self.model.transcribe(
            audio_16k,
            language = self.language,
            vad_filter = False, 
            word_timestamps = True,
            beam_size = self.beam_size)
        # FK-TODO: extract method
        return [{
            "speaker": speaker,
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": seg.text.strip()
        } for seg in segments_generator]

