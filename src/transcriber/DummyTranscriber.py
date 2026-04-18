import numpy as np

class DummyTranscriber:

    def transcribe(self, audio_16k: np.ndarray, speaker: str) -> list[dict]:
        match speaker:
            case "Disponent":
                return [
                    {
                        "speaker": speaker,
                        "start": 0,
                        "end": 2,
                        "text": "Feuerwehr Berlin, Notruf, wo genau ist der Unfallort?"
                    }
                ]
            case "Anrufer":
                return [
                    {
                        "speaker": speaker,
                        "start": 3,
                        "end": 5,
                        "text": "Torstraße, Ecke Friedrichstraße. Hier ist ein schwerer Unfall. Auto gegen Radfahrer. Kommen Sie schnell."
                    }
                ]
            case _:
                raise ValueError
