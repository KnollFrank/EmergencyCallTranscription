"""
Emergency Call Transcription & GDPR Anonymisation
==================================================
Models:    WhisperX large-v3 (CPU/INT8) + Presidio
Input:     Stereo WAV, 8 kHz
             Channel 0 (left)  = dispatcher
             Channel 1 (right) = caller
Output:    True dialogue – both channels transcribed separately
           and merged in chronological order:
             [00.00s] Caller:     Emergency, there is an accident ...
             [02.10s] Dispatcher: Where exactly is the accident?
             [04.30s] Caller:     On Hauptstrasse 12 ...
Privacy:   All processing runs locally, no cloud access,
           raw transcript is never written to disk.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
import numpy as np

import gradio
import librosa

from transcriber.Engine import Engine
from transcriber.TranscriberFactory import TranscriberFactory
from anonymizer.AnonymizerFactory import AnonymizerFactory

ENGINE: Engine = Engine.FASTER_WHISPER

# ─────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────
LANGUAGE = "de"
BATCH_SIZE = 4
BEAM_SIZE = 5
# FK-TODO: add enum for MODEL_SIZE
MODEL_SIZE = "large-v3"

# Channel assignment (fixed per project spec):
#   index 0 = left  = dispatcher
#   index 1 = right = caller
SPEAKERS = {
    0: "Disponent",
    1: "Anrufer",
}

# Output directory for anonymized JSON transcripts
EXPORT_DIR = Path.home() / "notruf-protokolle"
EXPORT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────
# LOAD MODELS (once at startup)
# ─────────────────────────────────────────────────────────
log.info(f"Loading Engine: {ENGINE} ({MODEL_SIZE if 'MODEL_SIZE' in locals() else 'large-v3'}) ...")

transcriber = TranscriberFactory.createTranscriber(
    engine = ENGINE,
    model_size = MODEL_SIZE,
    language = LANGUAGE,
    batch_size = BATCH_SIZE)

log.info("ASR Model ready.")

log.info("Loading Presidio + spaCy de_core_news_lg ...")
anonymizer = AnonymizerFactory.createAnonymizer(LANGUAGE)
log.info("Presidio ready.")
log.info("App ready → http://127.0.0.1:7860")

# ─────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────
def extract_channel(audio_path: str, channel_idx: int) -> np.ndarray:
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


def merge_dialogue(
    segments_dispatcher: list[dict],
    segments_caller: list[dict],
) -> list[dict]:
    """
    Merge segments from both channels into a single chronological list,
    sorted by start time.
    """
    return sorted(segments_dispatcher + segments_caller, key = lambda s: s["start"])

def dialogue_to_text(segments: list[dict], anon: bool = False) -> str:
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

# ─────────────────────────────────────────────────────────
# MAIN CALLBACK (Gradio)
# ─────────────────────────────────────────────────────────

def process_call(audio_path, progress = gradio.Progress()):
    """
    Gradio generator callback.
    Transcribes BOTH channels separately and merges them into a dialogue.
    Yields: (raw_text, anon_text, status)
    """
    if audio_path is None:
        yield "", "", "⚠️ Bitte zuerst eine WAV-Datei hochladen."
        return

    progress(0.1, desc="🚀 Starte Verarbeitung...")
    yield "", "", f"🔊  Isoliere Kanäle (Engine: {ENGINE}) ..."
    try:
        audio_dispatcher = extract_channel(audio_path, channel_idx=0)
        audio_caller = extract_channel(audio_path, channel_idx=1)
    except Exception as e:
        yield "", "", f"❌ Fehler: {e}"
        return

    duration_s = round(len(audio_caller) / 16000)

    progress(0.2, desc = f"📝 Transkribiere Disponent...")
    seg_dispatcher = transcriber.transcribe(audio_dispatcher, speaker = "Disponent")

    progress(0.5, desc=f"📝 Transkribiere Anrufer...")
    seg_caller = transcriber.transcribe(audio_caller, speaker = "Anrufer")

    progress(0.8, desc = "🔗 Führe Dialog zusammen ...")
    segments = merge_dialogue(seg_dispatcher, seg_caller)
    raw_text = dialogue_to_text(segments, anon = False)

    progress(0.9, desc="🔒 Anonymisiere...")
    all_types: set[str] = set()
    for seg in segments:
        anon_text, types = anonymizer.anonymize(seg["text"])
        seg["text_anon"] = anon_text
        all_types.update(types)

    anon_formatted = dialogue_to_text(segments, anon = True)

    # export anonymized JSON ─
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_path = EXPORT_DIR / f"notruf_{timestamp}_{ENGINE}.json"

    export_data = {
        "meta": {
            "timestamp":           datetime.now().isoformat(),
            "audio_duration_s":    duration_s,
            "engine":              ENGINE,
            "model_asr":           "large-v3",
            "anonymized":          True,
            "pii_types":           sorted(all_types)
        },
        "dialogue": [
            {
                "speaker": seg["speaker"],
                "start":   seg["start"],
                "end":     seg["end"],
                "text":    seg["text_anon"],
            }
            for seg in segments
        ],
    }

    # FK-TODO: remove export?
    with open(export_path, "w", encoding = "utf-8") as f:
        json.dump(export_data, f, ensure_ascii = False, indent=2)

    progress(1.0, desc = "✅ Abgeschlossen")
    status = (
        f"✅  Fertig ({ENGINE}) | "
        f"Export → {export_path.name}")
    yield raw_text, anon_formatted, status

# ─────────────────────────────────────────────────────────
# GRADIO UI
# ─────────────────────────────────────────────────────────
LAUNCH_KWARGS = {
    "server_name": "127.0.0.1",
    "server_port": 7860,
    "share": False,
    "show_error": True,
    "theme": gradio.themes.Soft(),
    "css": ".footer { font-size: 0.8em; color: #888; } #status-box { border: 1px solid #ddd; }",
}

with gradio.Blocks(
    title = "Notruf-Transkription",
    theme = gradio.themes.Soft()
) as demo:
    gradio.Markdown(f"# Notruf-Transkription & Anonymisierung (Engine: `{ENGINE}`)")
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
        fn = process_call,
        inputs = [audio_input],
        outputs = outputs,
        show_progress_on = [status_out])
    audio_input.clear(
        fn = lambda: ("", "", ""),
        outputs = outputs)

if __name__ == "__main__":
    demo.launch(**LAUNCH_KWARGS)