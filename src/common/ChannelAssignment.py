from enum import StrEnum
from .Channel import Channel
from .ChannelPair import ChannelPair

class ChannelAssignment(StrEnum):
    DISPATCHER_LEFT_CALLER_RIGHT = "Disponent (links) / Anrufer (rechts)"
    DISPATCHER_RIGHT_CALLER_LEFT = "Anrufer (links) / Disponent (rechts)"

    def getChannelPair(self) -> ChannelPair:
        match self:
            case ChannelAssignment.DISPATCHER_LEFT_CALLER_RIGHT:
                return ChannelPair(
                    dispatcherChannel = Channel.LEFT,
                    callerChannel = Channel.RIGHT)
            case ChannelAssignment.DISPATCHER_RIGHT_CALLER_LEFT:
                return ChannelPair(
                    dispatcherChannel = Channel.RIGHT,
                    callerChannel = Channel.LEFT)