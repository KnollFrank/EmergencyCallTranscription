import whisperx
import numpy as np
import soundfile
import tempfile
import os
import gc
from pathlib import Path
from .Model import Model

class WhisperXTranscriber:

    def __init__(self, model_size: Model, device, compute_type, language, batch_size, beam_size, threads):
        self.model = whisperx.load_model(
            model_size,
            device,
            compute_type = compute_type,
            language = language,
            asr_options = {"beam_size": beam_size},
            threads = threads)
        self.device = device
        self.language = language
        self.batch_size = batch_size

    def transcribe(self, audio_16k: np.ndarray, speaker: str) -> list[dict]:
        segments = self._transcribeAudioToSegments(audio_16k)
        return WhisperXTranscriber._convertSegments(segments, speaker)

    def _transcribeAudioToSegments(self, audio_16k):
        audioFile = WhisperXTranscriber._persistAudio(audio_16k)
        try:
            return self._loadAndTranscribeAudio(audioFile)
        finally:
            audioFile.unlink(missing_ok = True)

    @staticmethod
    def _persistAudio(audio_16k: np.ndarray) -> Path:
        with tempfile.NamedTemporaryFile(
            suffix = ".wav",
            delete = False,
            prefix = "asr_in_"
        ) as f:
            soundfile.write(f.name, audio_16k, 16000)
            return Path(f.name)

    def _loadAndTranscribeAudio(self, audioFile: Path):
        audio = whisperx.load_audio(str(audioFile))
        result = self.model.transcribe(
                audio,
                batch_size = self.batch_size,
                language = self.language)
        gc.collect()
        align_model, meta = whisperx.load_align_model(
                language_code = self.language,
                device = self.device)
        result = whisperx.align(result["segments"], align_model, meta, audio, self.device)
        gc.collect()
        return result.get("segments", [])
    
    @staticmethod
    def _convertSegments( segments, speaker):
        return [WhisperXTranscriber._convertSegment(segment, speaker) for segment in segments]

    @staticmethod
    def _convertSegment(segment, speaker):
        return {
            "speaker": speaker,
            "start": WhisperXTranscriber._round(segment["start"]),
            "end": WhisperXTranscriber._round(segment["end"]),
            "text": segment["text"].strip()
        }

    @staticmethod
    def _round(number):
        return round(float(number), 2)
