from typing import NamedTuple
import numpy as np

class AudioPair(NamedTuple):
    dispatcherAudio: np.ndarray
    callerAudio: np.ndarray
