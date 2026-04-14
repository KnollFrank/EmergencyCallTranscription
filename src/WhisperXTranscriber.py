import whisperx
import numpy as np
import soundfile
import tempfile
import os
import gc

class WhisperXTranscriber:

    def __init__(self, model_size, device, compute_type, language, batch_size, beam_size, threads):
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
        # write audio to a temp file – WhisperX expects a file path
        with tempfile.NamedTemporaryFile(
            suffix = ".wav",
            delete = False,
            prefix = "asr_in_"
        ) as f:
            soundfile.write(f.name, audio_16k, 16000)
            path_tmp = f.name

        try:
            # FK-TODO: extract method
            audio  = whisperx.load_audio(path_tmp)
            result = self.model.transcribe(audio, batch_size = self.batch_size, language = self.language)
            gc.collect()

            # word-level alignment for precise timestamps
            align_model, meta = whisperx.load_align_model(
                language_code = self.language,
                device = self.device)
            result = whisperx.align(result["segments"], align_model, meta, audio, self.device)
            gc.collect()
        finally:
            if os.path.exists(path_tmp):
                os.remove(path_tmp)
        # FK-TODO: extract method
        segments = []
        for seg in result.get("segments", []):
            segments.append({
                "speaker": speaker,
                # FK-TODO: extract method for rounding timestamps
                "start": round(float(seg["start"]), 2),
                "end": round(float(seg["end"]), 2),
                "text": seg["text"].strip(),
            })
        return segments

