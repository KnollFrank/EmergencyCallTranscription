import unittest

from src.common.TranscriptSimplifier import TranscriptSimplifier

class TranscriptSimplifierTest(unittest.TestCase):

    def test_merge_empty_list(self):
        # Given
        segments = []

        # When
        actual_merged_segments = TranscriptSimplifier.mergeConsecutiveSegments(segments)

        # Then
        self.assertEqual(actual_merged_segments, [])

    def test_merge_single_segment(self):
        # Given
        segments = [{"start": 0, "end": 5, "speaker": "Disponent", "text": "Notruf, Feuerwehr und Rettungsdienst."}]

        # When
        actual_merged_segments = TranscriptSimplifier.mergeConsecutiveSegments(segments)

        # Then
        self.assertEqual(actual_merged_segments, segments)

    def test_no_merge_needed_alternating_speakers(self):
        # Given
        segments = [
            {"start": 0, "end": 5, "speaker": "Disponent", "text": "Notruf."},
            {"start": 6, "end": 8, "speaker": "Anrufer", "text": "Hallo?"},
            {"start": 9, "end": 12, "speaker": "Disponent", "text": "Was ist passiert?"}
        ]

        # When
        actual_merged_segments = TranscriptSimplifier.mergeConsecutiveSegments(segments)

        # Then
        self.assertEqual(actual_merged_segments, segments)

    def test_merge_two_consecutive_segments(self):
        # Given
        segments = [
            {"start": 0, "end": 5, "speaker": "Disponent", "text": "Notruf, Feuerwehr und Rettungsdienst."},
            {"start": 5, "end": 10, "speaker": "Disponent", "text": "Wo genau ist der Notfallort?"},
            {"start": 11, "end": 15, "speaker": "Anrufer", "text": "In der Hauptstraße 1."}
        ]

        # When
        actual_merged_segments = TranscriptSimplifier.mergeConsecutiveSegments(segments)

        # Then
        self.assertEqual(
            actual_merged_segments,
            [
                {"start": 0, "end": 10, "speaker": "Disponent", "text": "Notruf, Feuerwehr und Rettungsdienst. Wo genau ist der Notfallort?"},
                {"start": 11, "end": 15, "speaker": "Anrufer", "text": "In der Hauptstraße 1."}
            ])

    def test_merge_multiple_consecutive_segments(self):
        # Given
        segments = [
            {"start": 0, "end": 2, "speaker": "Anrufer", "text": "Ich brauche"},
            {"start": 2, "end": 4, "speaker": "Anrufer", "text": "einen Krankenwagen."},
            {"start": 4, "end": 6, "speaker": "Anrufer", "text": "Schnell!"}
        ]

        # When
        actual_merged_segments = TranscriptSimplifier.mergeConsecutiveSegments(segments)

        # Then
        self.assertEqual(
            actual_merged_segments,
            [{"start": 0, "end": 6, "speaker": "Anrufer", "text": "Ich brauche einen Krankenwagen. Schnell!"}])

    def test_merge_all_segments_from_same_speaker(self):
        # Given
        segments = [
            {"start": 0, "end": 5, "speaker": "Disponent", "text": "Erste Zeile."},
            {"start": 6, "end": 10, "speaker": "Disponent", "text": "Zweite Zeile."},
            {"start": 11, "end": 15, "speaker": "Disponent", "text": "Dritte Zeile."}
        ]

        # When
        actual_merged_segments = TranscriptSimplifier.mergeConsecutiveSegments(segments)

        # Then
        self.assertEqual(
            actual_merged_segments,
            [{"start": 0, "end": 15, "speaker": "Disponent", "text": "Erste Zeile. Zweite Zeile. Dritte Zeile."}])

if __name__ == '__main__':
    unittest.main()