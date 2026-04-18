import unittest

from src.common.TranscriptSimplifier import TranscriptSimplifier

class TranscriptSimplifierTest(unittest.TestCase):

    def test_merge_empty_list(self):
        """Testet das Zusammenführen einer leeren Liste von Segmenten."""
        self.assertEqual(TranscriptSimplifier.mergeConsecutiveSegments([]), [])

    def test_merge_single_segment(self):
        """Testet mit einem einzelnen Segment, das unverändert bleiben sollte."""
        segments = [{"start": 0, "end": 5, "speaker": "Disponent", "text": "Notruf, Feuerwehr und Rettungsdienst."}]
        # Die Methode kopiert das Segment, daher werden die Werte verglichen, nicht die Identität
        self.assertEqual(TranscriptSimplifier.mergeConsecutiveSegments(segments), segments)

    def test_no_merge_needed_alternating_speakers(self):
        """Testet mit abwechselnden Sprechern, bei denen keine Zusammenführung stattfinden sollte."""
        segments = [
            {"start": 0, "end": 5, "speaker": "Disponent", "text": "Notruf."},
            {"start": 6, "end": 8, "speaker": "Anrufer", "text": "Hallo?"},
            {"start": 9, "end": 12, "speaker": "Disponent", "text": "Was ist passiert?"}
        ]
        self.assertEqual(TranscriptSimplifier.mergeConsecutiveSegments(segments), segments)

    def test_merge_two_consecutive_segments(self):
        """Testet das Zusammenführen von zwei aufeinanderfolgenden Segmenten desselben Sprechers."""
        segments = [
            {"start": 0, "end": 5, "speaker": "Disponent", "text": "Notruf, Feuerwehr und Rettungsdienst."},
            {"start": 5, "end": 10, "speaker": "Disponent", "text": "Wo genau ist der Notfallort?"},
            {"start": 11, "end": 15, "speaker": "Anrufer", "text": "In der Hauptstraße 1."}
        ]
        expected = [
            {"start": 0, "end": 10, "speaker": "Disponent", "text": "Notruf, Feuerwehr und Rettungsdienst. Wo genau ist der Notfallort?"},
            {"start": 11, "end": 15, "speaker": "Anrufer", "text": "In der Hauptstraße 1."}
        ]
        self.assertEqual(TranscriptSimplifier.mergeConsecutiveSegments(segments), expected)

    def test_merge_multiple_consecutive_segments(self):
        """Testet das Zusammenführen von mehr als zwei aufeinanderfolgenden Segmenten."""
        segments = [
            {"start": 0, "end": 2, "speaker": "Anrufer", "text": "Ich brauche"},
            {"start": 2, "end": 4, "speaker": "Anrufer", "text": "einen Krankenwagen."},
            {"start": 4, "end": 6, "speaker": "Anrufer", "text": "Schnell!"}
        ]
        expected = [
            {"start": 0, "end": 6, "speaker": "Anrufer", "text": "Ich brauche einen Krankenwagen. Schnell!"}
        ]
        self.assertEqual(TranscriptSimplifier.mergeConsecutiveSegments(segments), expected)

    def test_merge_all_segments_from_same_speaker(self):
        """Testet den Fall, dass alle Segmente vom selben Sprecher stammen."""
        segments = [
            {"start": 0, "end": 5, "speaker": "Disponent", "text": "Erste Zeile."},
            {"start": 6, "end": 10, "speaker": "Disponent", "text": "Zweite Zeile."},
            {"start": 11, "end": 15, "speaker": "Disponent", "text": "Dritte Zeile."}
        ]
        expected = [
            {"start": 0, "end": 15, "speaker": "Disponent", "text": "Erste Zeile. Zweite Zeile. Dritte Zeile."}
        ]
        self.assertEqual(TranscriptSimplifier.mergeConsecutiveSegments(segments), expected)

if __name__ == '__main__':
    unittest.main()