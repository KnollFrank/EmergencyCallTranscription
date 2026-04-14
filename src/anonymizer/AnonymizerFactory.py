from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from .Anonymizer import Anonymizer

class AnonymizerFactory:

    @staticmethod
    def createAnonymizer(language):
        return Anonymizer(
            language = language,
            analyzerEngine = AnonymizerFactory._createAnalyzerEngine(language),
            anonymizerEngine = AnonymizerEngine(),
            operators = {
                "PERSON":        OperatorConfig("replace", {"new_value": "<PERSON>"}),
                "LOCATION":      OperatorConfig("replace", {"new_value": "<ORT>"}),
                "PHONE_NUMBER":  OperatorConfig("replace", {"new_value": "<TELEFON>"}),
                "DATE_TIME":     OperatorConfig("replace", {"new_value": "<DATUM>"}),
                "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "<EMAIL>"}),
                "IBAN_CODE":     OperatorConfig("replace", {"new_value": "<IBAN>"}),
                "NRP":           OperatorConfig("replace", {"new_value": "<KENNZEICHEN>"}),
            })

    @staticmethod
    def _createAnalyzerEngine(language):
        return AnalyzerEngine(
            nlp_engine = AnonymizerFactory._createNlpEngine(language),
            supported_languages = [language])

    @staticmethod
    def _createNlpEngine(language):
        nlpEngineProvider = NlpEngineProvider(
            nlp_configuration = {
                "nlp_engine_name": "spacy",
                "models": [
                    {
                        "lang_code": language,
                        "model_name": "de_core_news_lg"
                    }],
            })
        return nlpEngineProvider.create_engine()
    