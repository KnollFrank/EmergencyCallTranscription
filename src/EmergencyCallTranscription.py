import logging

from transcriber.Engine import Engine
from transcriber.Model import Model
from transcriber.TranscriberFactory import TranscriberFactory
from anonymizer.AnonymizerFactory import AnonymizerFactory
from ui.GradioUI import GradioUI

logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(message)s",
    datefmt = "%H:%M:%S",
)
log = logging.getLogger(__name__)

def launchUI():
    language = "de"
    gradioUI = GradioUI(
        transcriberFactory = (
            lambda engine: TranscriberFactory.createTranscriber(
                engine = engine,
                model_size = Model.largeV3,
                language = language,
                batch_size = 4)),
        anonymizer = AnonymizerFactory.createAnonymizer(language))
    gradioUI.launch(server_name = "127.0.0.1", server_port = 7860)


if __name__ == "__main__":
    launchUI()
