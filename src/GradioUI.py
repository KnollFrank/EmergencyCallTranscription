import gradio
import librosa
import numpy as np
from transcriber.Engine import Engine

class GradioUI:

    def __init__(self, transcriber, anonymizer, engine: Engine):
        self.transcriber = transcriber
        self.anonymizer = anonymizer
        self.engine = engine

    def launch(self, server_name, server_port):
        self._createUI().launch(
            server_name = server_name,
            server_port = server_port,
            share = False,
            show_error = True,
            theme = gradio.themes.Soft(),
            css = ".footer { font-size: 0.8em; color: #888; } #status-box { border: 1px solid #ddd; }")

    def _createUI(self):
        with gradio.Blocks(
            title = "Notruf-Transkription",
            theme = gradio.themes.Soft()
        ) as ui:
            gradio.Markdown(f"# Notruf-Transkription & Anonymisierung (Engine: `{self.engine}`)")
            with gradio.Row():
                with gradio.Column(scale = 1, min_width = 280):
                    audio_input = gradio.Audio(
                        label = "Notruf-WAV hochladen",
                        type = "filepath",
                        sources = ["upload"])
                    # FK-TODO: die Kanalzuweisung im UI und im Code einstellbar machen
                    gradio.Markdown("""
                    ---
                    **Kanalzuweisung (fest):**
                    - Kanal 0 links  → Disponent
                    - Kanal 1 rechts → Anrufer
                    """)
                    status_out = gradio.Textbox(
                        label = "Status & Fortschritt",
                        lines = 3,
                        interactive = False,
                        elem_id = "status-box")
                with gradio.Column(scale = 2):
                    COPY_BUTTON = { "buttons": ["copy"] }
                    lines = 20
                    with gradio.Tab("📄 Gesprächsprotokoll"):
                        # FK-TODO: das Gesprächsprotokoll soll editierbar sein und die Anonymisierung über einen Button gestartet werden.
                        roh_out = gradio.Textbox(
                            label = "Gesprächsprotokoll",
                            lines = lines,
                            placeholder = (
                                "[00.00s – 03.21s]  Anrufer:\n"
                                "    Notruf, hier ist ein Unfall auf der Hauptstraße!\n\n"
                                "[03.50s – 05.10s]  Disponent:\n"
                                "    Wo genau ist der Unfall?\n\n"
                                "[05.30s – 08.40s]  Anrufer:\n"
                                "    Hauptstraße 12, vor dem Supermarkt ..."),
                            **COPY_BUTTON)
                    with gradio.Tab("🔒 Anonymisiertes Gesprächsprotokoll"):
                        anon_out = gradio.Textbox(
                            label = "Anonymisiertes Gesprächsprotokoll",
                            lines = lines,
                            placeholder = (
                                "[00.00s – 03.21s]  Anrufer:\n"
                                "    Notruf, hier ist ein Unfall auf der <ORT>!\n\n"
                                "[03.50s – 05.10s]  Disponent:\n"
                                "    Wo genau ist der Unfall?\n\n"
                                "[05.30s – 08.40s]  Anrufer:\n"
                                "    <ORT>, vor dem Supermarkt ..."),
                            **COPY_BUTTON)
            outputs = [roh_out, anon_out, status_out]
            audio_input.upload(
                fn = self._process_call,
                inputs = [audio_input],
                outputs = outputs,
                show_progress_on = [status_out])
            audio_input.clear(
                fn = lambda: ("", "", ""),
                outputs = outputs)
        return ui
    
    def _process_call(self, audio_path, progress = gradio.Progress()):
        """
        Gradio generator callback.
        Transcribes BOTH channels separately and merges them into a dialogue.
        Yields: (raw_text, anon_text, status)
        """
        if audio_path is None:
            yield "", "", "⚠️ Bitte zuerst eine WAV-Datei hochladen."
            return

        progress(0.1, desc="🚀 Starte Verarbeitung...")
        yield "", "", f"🔊  Isoliere Kanäle (Engine: {self.engine}) ..."
        try:
            audio_dispatcher = GradioUI._extract_channel(audio_path, channel_idx=0)
            audio_caller = GradioUI._extract_channel(audio_path, channel_idx=1)
        except Exception as e:
            yield "", "", f"❌ Fehler: {e}"
            return

        progress(0.2, desc = f"📝 Transkribiere Disponent...")
        seg_dispatcher = self.transcriber.transcribe(audio_dispatcher, speaker = "Disponent")

        progress(0.5, desc=f"📝 Transkribiere Anrufer...")
        seg_caller = self.transcriber.transcribe(audio_caller, speaker = "Anrufer")

        progress(0.8, desc = "🔗 Führe Dialog zusammen ...")
        segments = GradioUI._merge_dialogue(seg_dispatcher, seg_caller)
        raw_text = GradioUI._dialogue_to_text(segments, anon = False)

        progress(0.9, desc="🔒 Anonymisiere...")
        all_types: set[str] = set()
        for seg in segments:
            anon_text, types = self.anonymizer.anonymize(seg["text"])
            seg["text_anon"] = anon_text
            all_types.update(types)

        anon_formatted = GradioUI._dialogue_to_text(segments, anon = True)

        progress(1.0, desc = "✅ Abgeschlossen")
        yield raw_text, anon_formatted, f"✅  Fertig ({self.engine})"

    @staticmethod
    def _extract_channel(audio_path: str, channel_idx: int) -> np.ndarray:
        """
        Load a stereo WAV (8 kHz), extract the requested channel,
        and upsample to 16 kHz (required by Whisper).
        Returns a float32 mono numpy array at 16 kHz.
        channel_idx: 0 = left (dispatcher), 1 = right (caller)
        """
        audio, sr = librosa.load(audio_path, sr=None, mono=False)

        # FK-TODO: extract method
        if audio.ndim == 1:
            mono = audio
        else:
            mono = audio[channel_idx]

        # FK-TODO: extract constant or parameter for 16000 in all places or rename method 
        return librosa.resample(mono.astype("float32"), orig_sr=sr, target_sr=16000)

    @staticmethod
    def _merge_dialogue(segments_dispatcher: list[dict], segments_caller: list[dict]) -> list[dict]:
        """
        Merge segments from both channels into a single chronological list,
        sorted by start time.
        """
        return sorted(segments_dispatcher + segments_caller, key = lambda s: s["start"])

    @staticmethod
    def _dialogue_to_text(segments: list[dict], anon: bool = False) -> str:
        """
        Format the merged dialogue as a readable transcript with timestamps.
        Speaker changes are separated by a blank line for readability.
        Example output:
            [00.00s – 03.21s]  Anrufer:
                Emergency, there is an accident on the main road!

            [03.50s – 05.80s]  Disponent:
                Where exactly? What house number?
        """
        lines = []
        last_speaker = None

        for seg in segments:
            text    = seg.get("text_anon", seg["text"]) if anon else seg["text"]
            speaker = seg["speaker"]

            # blank line on speaker change for readability
            if last_speaker is not None and speaker != last_speaker:
                lines.append("")

            lines.append(f"[{seg['start']:06.2f}s – {seg['end']:06.2f}s]  {speaker}:")
            lines.append(f"    {text}")
            last_speaker = speaker

        return "\n".join(lines)
