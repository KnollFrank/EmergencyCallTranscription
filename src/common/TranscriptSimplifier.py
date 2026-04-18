class TranscriptSimplifier:

    # FK-TODO: add unit test
    @staticmethod
    def mergeConsecutiveSegments(segments: list[dict]) -> list[dict]:
        """
        Merges consecutive segments of the same speaker.
        """
        if not segments:
            return []
            
        merged = []
        current = segments[0].copy()
        
        for seg in segments[1:]:
            if seg["speaker"] == current["speaker"]:
                current["end"] = seg["end"]
                current["text"] += " " + seg["text"].strip()
            else:
                merged.append(current)
                current = seg.copy()
        
        merged.append(current)
        return merged
