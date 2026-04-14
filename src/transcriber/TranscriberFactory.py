from .Engine import Engine
from .FasterWhisperTranscriber import FasterWhisperTranscriber
from .WhisperXTranscriber import WhisperXTranscriber
from .TranscriberTransformer import TranscriberTransformer

class TranscriberFactory:

    @staticmethod
    def createTranscriber(engine: Engine, model_size, language, batch_size):
        return TranscriberTransformer(
            TranscriberFactory._createTranscriber(
                engine,
                model_size,
                language,
                batch_size))
    
    @staticmethod
    def _createTranscriber(engine: Engine, model_size, language, batch_size):
        language = "de"
        device = "cpu"
        compute_type = "int8"
        beam_size = 5
        match engine:
            case Engine.FASTER_WHISPER:
                return FasterWhisperTranscriber(
                    model_size = model_size,
                    device = device,
                    compute_type = compute_type,
                    language = language,
                    beam_size = beam_size)
            case Engine.WHISPERX:
                return WhisperXTranscriber(
                    model_size = model_size,
                    device = device,
                    compute_type = compute_type,
                    language = language,
                    batch_size = batch_size,
                    beam_size = beam_size,
                    threads = 12)
