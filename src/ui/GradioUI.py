import os
import gradio
import librosa
import numpy as np
from transcriber.Engine import Engine
from transcriber.TranscriberFactory import TranscriberFactory
from transcriber.Model import Model

class GradioUI:

    def __init__(self, transcriber, anonymizer, engine: Engine):
        self.transcribers = {engine.value: transcriber}
        self.anonymizer = anonymizer
        self.engine = engine.value

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
            gradio.Markdown("# Notruf-Transkription & Anonymisierung")

            with gradio.Group():
                gradio.Markdown("### Schritt 1: Eingabe & Einstellungen")
                with gradio.Row():
                    with gradio.Column(scale = 1):
                        audio_input = gradio.Audio(
                            label = "Notruf-WAV hochladen",
                            type = "filepath",
                            sources = ["upload"])
                        filename_out = gradio.Textbox(
                            label = "Ausgewählte Datei",
                            interactive = False,
                            lines = 1
                        )
                    with gradio.Column(scale = 1):
                        engine_dropdown = gradio.Dropdown(
                            choices = [e.value for e in Engine], 
                            value = self.engine, 
                            label = "Transkriptions-Engine"
                        )
                        channel_assignment = gradio.Radio(
                            choices = ["Disponent (links) / Anrufer (rechts)", "Anrufer (links) / Disponent (rechts)"],
                            value = "Disponent (links) / Anrufer (rechts)",
                            label = "Kanalzuordnung (Kanal 0 / Kanal 1)"
                        )

                transcribe_btn = gradio.Button(
                    value = "Transkription starten",
                    variant = "primary"
                )
                status_out = gradio.Textbox(
                    label = "Status & Fortschritt",
                    lines = 3,
                    interactive = False,
                    elem_id = "status-box")

            with gradio.Group():
                gradio.Markdown("### Schritt 2: Transkription & Korrektur")
                audio_playback = gradio.Audio(
                    label = "Audio-Wiedergabe",
                    type = "filepath",
                    interactive = False
                )
                headers = ["Zeitstempel", "Rolle", "Gesprächsinhalt"]
                roh_out = gradio.Dataframe(
                    headers = headers,
                    datatype = ["str", "str", "str"],
                    col_count = (3, "fixed"),
                    label = "📄 Gesprächsprotokoll",
                    interactive = True)

                anon_btn = gradio.Button(
                    value = "Anonymisierung starten ↓", 
                    variant = "primary")

            with gradio.Group():
                gradio.Markdown("### Schritt 3: Anonymisiertes Ergebnis")
                anon_out = gradio.Dataframe(
                    headers = headers,
                    datatype = ["str", "str", "str"],
                    col_count = (3, "fixed"),
                    label = "🔒 Anonymisiertes Gesprächsprotokoll",
                    interactive = True)

            audio_input.change(
                fn = lambda path: (os.path.basename(path) if path else "", path),
                inputs = [audio_input],
                outputs = [filename_out, audio_playback])

            transcribe_btn.click(
                fn = self._transcribe,
                inputs = [audio_input, engine_dropdown, channel_assignment],
                outputs = [roh_out, anon_out, status_out],
                show_progress_on = [status_out])

            anon_btn.click(
                fn = self._anonymize,
                inputs = [roh_out],
                outputs = [anon_out])

            audio_input.clear(
                fn = lambda: (None, None, "", None),
                outputs = [roh_out, anon_out, status_out, audio_playback])
        return ui    
    def _get_transcriber(self, engine_value: str):
        if engine_value not in self.transcribers:
            self.transcribers[engine_value] = TranscriberFactory.createTranscriber(
                engine = Engine(engine_value),
                model_size = Model.largeV3,
                language = "de",
                batch_size = 4
            )
        return self.transcribers[engine_value]

    def _transcribe(self, audio_path, engine_name, channel_assignment, progress = gradio.Progress()):
        """
        Transcribes audio and yields the formatted dataframe rows.
        """
        if audio_path is None:
            yield None, None, "⚠️ Bitte zuerst eine WAV-Datei hochladen."
            return
            
        transcriber = self._get_transcriber(engine_name)

        progress(0.1, desc="🚀 Starte Verarbeitung...")
        yield None, None, f"🔊 Isoliere Kanäle (Engine: {engine_name}) ..."
        
        if channel_assignment == "Disponent (links) / Anrufer (rechts)":
            disp_idx, caller_idx = 0, 1
        else:
            disp_idx, caller_idx = 1, 0

        try:
            audio_dispatcher = GradioUI._extract_channel(audio_path, channel_idx=disp_idx)
            audio_caller = GradioUI._extract_channel(audio_path, channel_idx=caller_idx)
        except Exception as e:
            yield None, None, f"❌ Fehler: {e}"
            return

        progress(0.2, desc = f"📝 Transkribiere Disponent...")
        seg_dispatcher = transcriber.transcribe(audio_dispatcher, speaker = "Disponent")

        progress(0.5, desc=f"📝 Transkribiere Anrufer...")
        seg_caller = transcriber.transcribe(audio_caller, speaker = "Anrufer")

        progress(0.8, desc = "🔗 Führe Dialog zusammen ...")
        segments = GradioUI._merge_dialogue(seg_dispatcher, seg_caller)
        
        progress(1.0, desc = "✅ Transcription complete")
        yield GradioUI._getTableData(segments), None, f"✅ Transcription finished ({engine_name}). You can now edit the text in the table."

    @staticmethod
    def _merge_consecutive_segments(segments: list[dict]) -> list[dict]:
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

    @staticmethod
    def _getTableData(segments):
        def format_time(seconds):
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes:02d}:{secs:02d}"

        def getTableRow(segment):
            start_time = format_time(segment['start'])
            end_time = format_time(segment['end'])
            timestamp = f"{start_time} – {end_time}"
            return [timestamp, segment["speaker"], segment["text"]]
        
        return [getTableRow(segment) for segment in segments]

    def _anonymize(self, table_data):
        """
        Anonymizes ONLY the 'Gesprächsinhalt' (third) column of the provided table data.
        """
        if table_data is None or len(table_data) == 0:
            return None
        return self._anonymizeRows(GradioUI._getRows(table_data))

    @staticmethod
    def _getRows(table_data):
        return table_data.values if hasattr(table_data, "values") else table_data

    def _anonymizeRows(self, rows):
        return [self._anonymizeRow(row) for row in rows]

    def _anonymizeRow(self, row):
        anon_text, _ = self.anonymizer.anonymize(str(row[2]))
        return [row[0], row[1], anon_text]

    @staticmethod
    def _extract_channel(audio_path: str, channel_idx: int, target_sr: int = 16000) -> np.ndarray:
        """
        Extracts channel and resamples to target_sr.
        """
        audio, sr = librosa.load(audio_path, sr=None, mono=False)
        mono = audio if audio.ndim == 1 else audio[channel_idx]
        return librosa.resample(mono.astype("float32"), orig_sr=sr, target_sr=target_sr)

    @staticmethod
    def _merge_dialogue(segments_dispatcher: list[dict], segments_caller: list[dict]) -> list[dict]:
        """
        Sorts all segments chronologically by start time and merges consecutive segments of the same speaker.
        """
        sorted_segments = sorted(segments_dispatcher + segments_caller, key = lambda s: s["start"])
        return GradioUI._merge_consecutive_segments(sorted_segments)
