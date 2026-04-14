class Anonymizer:
    
    def __init__(self, language, analyzerEngine, anonymizerEngine, operators):
        self.language = language
        self.analyzer = analyzerEngine
        self.anonymizer = anonymizerEngine
        self.operators = operators
    
    def anonymize(self, text: str) -> tuple[str, list[str]]:
        """
        Detect and replace PII in text using Presidio.
        Returns (anonymized text, sorted list of detected entity types).
        The original text is never stored or logged.
        """
        found = self.analyzer.analyze(
            text = text,
            language = self.language,
            entities = list(self.operators.keys()))
        anon = self.anonymizer.anonymize(
            text = text,
            analyzer_results = found,
            operators = self.operators)
        return anon.text, Anonymizer._getTypes(found)
    
    @staticmethod
    def _getTypes(found):
        return sorted({e.entity_type for e in found})
