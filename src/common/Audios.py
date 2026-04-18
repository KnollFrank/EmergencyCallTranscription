import librosa
from .ChannelAssignment import ChannelAssignment
from .AudioPair import AudioPair
from .Channel import Channel

class Audios:

    @staticmethod
    def isolateAndResampleChannelsTo16kHz(audio_path, channel_assignment: ChannelAssignment) -> AudioPair:
        audio, sr = librosa.load(audio_path, sr = None, mono = False)
        Audios.assertStereoAnd8kHz(audio, sr)
        
        def resampleChannelTo16kHz(channel: Channel):
            return librosa.resample(
                audio[channel].astype("float32"),
                orig_sr = sr,
                target_sr = 16000)
        
        channelPair = channel_assignment.getChannelPair()
        return AudioPair(
            dispatcherAudio = resampleChannelTo16kHz(channelPair.dispatcherChannel),
            callerAudio = resampleChannelTo16kHz(channelPair.callerChannel))

    @staticmethod
    def assertStereoAnd8kHz(audio, sr):
        if not (audio.ndim == 2 and audio.shape[0] == 2):
            raise ValueError(f"Die Datei muss 2 Kanäle (Stereo) haben, hat aber {audio.shape[0] if audio.ndim == 2 else 1}.")
        if sr != 8000:
            raise ValueError(f"Die Datei muss eine Abtastrate von 8 kHz haben, hat aber {sr / 1000:.1f} kHz.")
