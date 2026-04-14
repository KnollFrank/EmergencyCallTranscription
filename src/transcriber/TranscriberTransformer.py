import numpy as np

# FK-TODO: rename class
class TranscriberTransformer:

    def __init__(self, transcriber):
        self.transcriber = transcriber

    def transcribe(self, audio_16k: np.ndarray, speaker: str) -> list[dict]:
        return TranscriberTransformer._transformSegments(
            segments = self.transcriber.transcribe(audio_16k, speaker),
            speaker = speaker)

    @staticmethod
    def _transformSegments(segments, speaker: str):
        return [TranscriberTransformer._transformSegment(segment, speaker) for segment in segments]

    @staticmethod
    def _transformSegment(segment, speaker: str):
        return {
            # FK-TODO: add speaker after calling _transformSegments?
            "speaker": speaker,
            # FK-TODO: extract method for rounding timestamps
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "text": segment.text.strip()
        }
