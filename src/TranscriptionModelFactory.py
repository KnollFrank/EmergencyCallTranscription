from Engine import Engine
from WhisperTranscriptionModel import WhisperTranscriptionModel
from WhisperXTranscriptionModel import WhisperXTranscriptionModel

class TranscriptionModelFactory:

    @staticmethod
    def createTranscriptionModel(engine: Engine, model_size, language, batch_size):
        language = "de"
        device = "cpu"
        compute_type = "int8"
        beam_size = 5
        match engine:
            case Engine.FASTER_WHISPER:
                return WhisperTranscriptionModel(
                    model_size = model_size,
                    device = device,
                    compute_type = compute_type,
                    language = language,
                    beam_size = beam_size)
            case Engine.WHISPERX:
                return WhisperXTranscriptionModel(
                    model_size = model_size,
                    device = device,
                    compute_type = compute_type,
                    language = language,
                    batch_size = batch_size,
                    beam_size = beam_size,
                    threads = 12)
