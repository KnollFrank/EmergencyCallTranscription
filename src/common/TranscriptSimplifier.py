class TranscriptSimplifier:

    @staticmethod
    def mergeConsecutiveSegments(segments: list[dict]) -> list[dict]:
        if not segments:
            return []
        merged = []
        current = segments[0].copy()
        for segment in segments[1:]:
            if TranscriptSimplifier.isConsecutive(current, segment):
                TranscriptSimplifier.mergeSrcIntoDst(src = segment, dst = current)
            else:
                merged.append(current)
                current = segment.copy()
        merged.append(current)
        return merged

    @staticmethod
    def isConsecutive(segment1, segment2):
        return segment1["speaker"] == segment2["speaker"]

    @staticmethod
    def mergeSrcIntoDst(src, dst):
        dst["end"] = src["end"]
        dst["text"] += " " + src["text"].strip()
