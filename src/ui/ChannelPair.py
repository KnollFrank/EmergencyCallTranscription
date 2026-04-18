from typing import NamedTuple
from .Channel import Channel

class ChannelPair(NamedTuple):
    dispatcherChannel: Channel
    callerChannel: Channel
