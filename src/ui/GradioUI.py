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
            # FK-TODO: die engine soll im UI einstellbar sein
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
                    
                    anon_out = gradio.Dataframe(
                        headers = headers,
                        datatype = ["str", "str", "str"],
                        col_count = (3, "fixed"),
                        label = "🔒 Anonymisiertes Gesprächsprotokoll",
                        interactive = False)

            audio_input.upload(
                fn = self._transcribe,
                inputs = [audio_input],
                outputs = [roh_out, anon_out, status_out],
                show_progress_on = [status_out])
            
            anon_btn.click(
                fn = self._anonymize,
                inputs = [roh_out],
                outputs = [anon_out])
            
            audio_input.clear(
                fn = lambda: (None, None, ""),
                outputs = [roh_out, anon_out, status_out])
        return ui
    
    def _transcribe(self, audio_path, progress = gradio.Progress()):
        """
        Transcribes audio and yields the formatted dataframe rows.
        """
        if audio_path is None:
            yield None, None, "⚠️ Bitte zuerst eine WAV-Datei hochladen."
            return

        progress(0.1, desc="🚀 Starte Verarbeitung...")
        yield None, None, f"🔊 Isoliere Kanäle (Engine: {self.engine}) ..."
        try:
            audio_dispatcher = GradioUI._extract_channel(audio_path, channel_idx=0)
            audio_caller = GradioUI._extract_channel(audio_path, channel_idx=1)
        except Exception as e:
            yield None, None, f"❌ Fehler: {e}"
            return

        progress(0.2, desc = f"📝 Transkribiere Disponent...")
        # FK-TODO: wrap self.transcriber.transcribe() with a method which merges consecutive segments of the same speaker and returns a list of dicts with keys "start", "end" (adapted because of merge), "speaker", "text"
        seg_dispatcher = self.transcriber.transcribe(audio_dispatcher, speaker = "Disponent")

        progress(0.5, desc=f"📝 Transkribiere Anrufer...")
        seg_caller = self.transcriber.transcribe(audio_caller, speaker = "Anrufer")

        progress(0.8, desc = "🔗 Führe Dialog zusammen ...")
        segments = GradioUI._merge_dialogue(seg_dispatcher, seg_caller)
        
        # FK-TODO: extract method
        table_data = []
        for seg in segments:
            time_str = f"{seg['start']:06.2f}s – {seg['end']:06.2f}s"
            table_data.append([time_str, seg["speaker"], seg["text"]])

        progress(1.0, desc = "✅ Transcription complete")
        yield table_data, None, f"✅ Transcription finished ({self.engine}). You can now edit the text in the table."

    def _anonymize(self, table_data):
        """
        Anonymizes ONLY the 'Gesprächsinhalt' (third) column of the provided table data.
        """
        if table_data is None or len(table_data) == 0:
            return None
        
        # FK-TODO: extract method
        # Handle Gradio's Dataframe format (can be list of lists or pandas DF)
        rows = table_data.values if hasattr(table_data, "values") else table_data
        processed_rows = []
        
        for row in rows:
            # Check row length to avoid index errors and process only the content column (index 2)
            if len(row) >= 3:
                time_val = row[0]
                role_val = row[1]
                text_val = str(row[2])
                
                if text_val.strip():
                    anon_text, _ = self.anonymizer.anonymize(text_val)
                    processed_rows.append([time_val, role_val, anon_text])
                else:
                    processed_rows.append([time_val, role_val, text_val])
            else:
                processed_rows.append(row)
                
        return processed_rows

    @staticmethod
    def _extract_channel(audio_path: str, channel_idx: int) -> np.ndarray:
        """
        Extracts channel and resamples to 16kHz.
        """
        audio, sr = librosa.load(audio_path, sr=None, mono=False)
        mono = audio if audio.ndim == 1 else audio[channel_idx]
        # FK-TODO: extract constant or parameter for 16000 in all places or rename method? 
        return librosa.resample(mono.astype("float32"), orig_sr=sr, target_sr=16000)

    @staticmethod
    def _merge_dialogue(segments_dispatcher: list[dict], segments_caller: list[dict]) -> list[dict]:
        """
        Sorts all segments chronologically by start time.
        """
        return sorted(segments_dispatcher + segments_caller, key = lambda s: s["start"])