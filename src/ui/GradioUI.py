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
                    COPY_BUTTON = { "buttons": ["copy"] }
                    lines = 12
                    roh_out = gradio.Textbox(
                        label = "📄 Gesprächsprotokoll",
                        lines = lines,
                        interactive = True,
                        placeholder = "Das Transkript erscheint hier nach dem Upload...",
                        **COPY_BUTTON)
                    anon_btn = gradio.Button(
                        value = "Anonymisierung starten ↓", 
                        variant = "primary")
                    anon_out = gradio.Textbox(
                        label = "🔒 Anonymisiertes Gesprächsprotokoll",
                        lines = lines,
                        interactive = False,
                        placeholder = "Klicken Sie auf den Button oben, um PII zu maskieren.",
                        **COPY_BUTTON)
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
                fn = lambda: ("", "", ""),
                outputs = [roh_out, anon_out, status_out])
        return ui
    
    def _transcribe(self, audio_path, progress = gradio.Progress()):
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

        progress(1.0, desc = "✅ Transkription abgeschlossen")
        yield raw_text, "", f"✅  Transkription fertig ({self.engine}). Sie können den Text nun bearbeiten."

    def _anonymize(self, text: str):
        if not text or text.strip() == "":
            return "⚠️ Kein Text zum Anonymisieren vorhanden."
        
        anon_text, _ = self.anonymizer.anonymize(text)
        return anon_text

    @staticmethod
    def _extract_channel(audio_path: str, channel_idx: int) -> np.ndarray:
        audio, sr = librosa.load(audio_path, sr=None, mono=False)
        mono = audio if audio.ndim == 1 else audio[channel_idx]
        # FK-TODO: extract constant or parameter for 16000 in all places or rename method? 
        return librosa.resample(mono.astype("float32"), orig_sr=sr, target_sr=16000)

    @staticmethod
    def _merge_dialogue(segments_dispatcher: list[dict], segments_caller: list[dict]) -> list[dict]:
        return sorted(segments_dispatcher + segments_caller, key = lambda s: s["start"])

    @staticmethod
    def _dialogue_to_text(segments: list[dict], anon: bool = False) -> str:
        lines = []
        last_speaker = None

        for seg in segments:
            text    = seg.get("text_anon", seg["text"]) if anon else seg["text"]
            speaker = seg["speaker"]

            if last_speaker is not None and speaker != last_speaker:
                lines.append("")

            lines.append(f"[{seg['start']:06.2f}s – {seg['end']:06.2f}s]  {speaker}:")
            lines.append(f"    {text}")
            last_speaker = speaker

        return "\n".join(lines)
