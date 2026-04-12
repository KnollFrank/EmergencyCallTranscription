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

import gc
import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
import numpy as np

import gradio
import librosa
import soundfile
import whisperx
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

# Gradio compatibility helpers
GRADIO_MAJOR   = int(gradio.__version__.split(".")[0])
GRADIO_6       = GRADIO_MAJOR >= 6
COPY_BUTTON    = {"buttons": ["copy"]} if GRADIO_6 else {"show_copy_button": True}
NO_COPY_BUTTON = {"buttons": []}       if GRADIO_6 else {"show_copy_button": False}

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
DEVICE       = "cpu"
COMPUTE_TYPE = "int8"
LANGUAGE     = "de"
BATCH_SIZE   = 4
CPU_THREADS  = 12
BEAM_SIZE    = 5

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

# PII replacement rules for Presidio
PII_OPERATORS = {
    "PERSON":        OperatorConfig("replace", {"new_value": "<PERSON>"}),
    "LOCATION":      OperatorConfig("replace", {"new_value": "<ORT>"}),
    "PHONE_NUMBER":  OperatorConfig("replace", {"new_value": "<TELEFON>"}),
    "DATE_TIME":     OperatorConfig("replace", {"new_value": "<DATUM>"}),
    "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "<EMAIL>"}),
    "IBAN_CODE":     OperatorConfig("replace", {"new_value": "<IBAN>"}),
    "NRP":           OperatorConfig("replace", {"new_value": "<KENNZEICHEN>"}),
}

# ─────────────────────────────────────────────────────────
# LOAD MODELS (once at startup)
# ─────────────────────────────────────────────────────────
log.info("Loading WhisperX large-v3 (CPU/INT8) ...")
asr_model = whisperx.load_model(
    "large-v3",
    DEVICE,
    compute_type=COMPUTE_TYPE,
    language=LANGUAGE,
    asr_options={"beam_size": BEAM_SIZE},
    threads=CPU_THREADS,
)
log.info("WhisperX ready.")

log.info("Loading Presidio + spaCy de_core_news_lg ...")
nlp_config = {
    "nlp_engine_name": "spacy",
    "models": [{"lang_code": LANGUAGE, "model_name": "de_core_news_lg"}],
}
pii_analyzer = AnalyzerEngine(
    nlp_engine = NlpEngineProvider(nlp_configuration = nlp_config).create_engine(),
    supported_languages = [LANGUAGE])
pii_anonymizer = AnonymizerEngine()
log.info("Presidio ready.")
log.info("App ready → http://127.0.0.1:7860")

# ─────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────

def extract_channel(audio_path: str, channel_idx: int) -> np.ndarray:
    """
    Load a stereo WAV (8 kHz), extract the requested channel,
    and upsample to 16 kHz (required by WhisperX).
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

def transcribe(audio_16k: np.ndarray, speaker: str) -> list[dict]:
    """
    Transcribe a 16 kHz mono signal with WhisperX.
    Returns a list of segment dicts:
        [{"sprecher": str, "start": float, "end": float, "text": str}]
    """
    # FK-TODO: extract method
    # write audio to a temp file – WhisperX expects a file path
    with tempfile.NamedTemporaryFile(
        suffix = ".wav",
        delete = False,
        prefix = "asr_in_"
    ) as f:
        soundfile.write(f.name, audio_16k, 16000)
        path_tmp = f.name

    try:
        # FK-TODO: extract method
        audio  = whisperx.load_audio(path_tmp)
        result = asr_model.transcribe(audio, batch_size = BATCH_SIZE, language = LANGUAGE)
        gc.collect()

        # word-level alignment for precise timestamps
        align_model, meta = whisperx.load_align_model(
            language_code = LANGUAGE,
            device = DEVICE)
        result = whisperx.align(result["segments"], align_model, meta, audio, DEVICE)
        gc.collect()
    finally:
        os.remove(path_tmp)

    # FK-TODO: extract method
    segments = []
    for seg in result.get("segments", []):
        segments.append({
            # FK-TODO: rename "sprecher" to "speaker" in all places
            "sprecher": speaker,
            "start":    round(float(seg["start"]), 2),
            "end":      round(float(seg["end"]),   2),
            "text":     seg["text"].strip(),
        })
    return segments

def anonymize_text(text: str) -> tuple[str, list[str]]:
    """
    Detect and replace PII in text using Presidio.
    Returns (anonymized text, sorted list of detected entity types).
    The original text is never stored or logged.
    """
    found = pii_analyzer.analyze(
        text = text,
        language = LANGUAGE,
        entities = list(PII_OPERATORS.keys()))
    anon = pii_anonymizer.anonymize(
        text = text,
        analyzer_results = found,
        operators = PII_OPERATORS)
    # FK-TODO: extract method getTypes(found)
    types = sorted({e.entity_type for e in found})
    return anon.text, types

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
        speaker = seg["sprecher"]

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

# FK-TODO: entferne Parameter anonymize_active und die Checkbox aus der UI, da Anonymisierung immer aktiv sein soll. Alle Stellen im Code anpassen.
def process_call(audio_path, anonymize_active):
    """
    Gradio generator callback.
    Transcribes BOTH channels separately and merges them into a dialogue.
    Yields: (raw_text, anon_text, pii_report, status)
    """
    if audio_path is None:
        yield "", "", "", "⚠️  Bitte zuerst eine WAV-Datei hochladen."
        return

    # ── Step 1: extract both channels ────────────────────
    yield "", "", "", "🔊  Kanäle extrahieren (8 kHz → 16 kHz) ..."
    try:
        audio_dispatcher = extract_channel(audio_path, channel_idx=0)
        audio_caller = extract_channel(audio_path, channel_idx=1)
    except Exception as e:
        yield "", "", "", f"❌  Fehler beim Kanal-Extrahieren: {e}"
        return

    duration_s = round(len(audio_caller) / 16000)

    # ── Step 2: transcribe dispatcher channel ────────────
    yield "", "", "", f"📝  WhisperX transkribiert Disponenten ({duration_s} s) – bitte warten ..."
    try:
        seg_dispatcher = transcribe(audio_dispatcher, speaker="Disponent")
    except Exception as e:
        yield "", "", "", f"❌  Fehler bei Transkription (Disponent): {e}"
        return

    # ── Step 3: transcribe caller channel ────────────────
    yield "", "", "", f"📝  WhisperX transkribiert Anrufer ({duration_s} s) – bitte warten ..."
    try:
        seg_caller = transcribe(audio_caller, speaker="Anrufer")
    except Exception as e:
        yield "", "", "", f"❌  Fehler bei Transkription (Anrufer): {e}"
        return

    # ── Step 4: merge into chronological dialogue ─────────
    yield "", "", "", "🔗  Führe Dialog zusammen ..."
    segments = merge_dialogue(seg_dispatcher, seg_caller)
    raw_text = dialogue_to_text(segments, anon = False)

    if not anonymize_active:
        # FK-TODO: Segmente für Anrufer und Disponent sind uninteressant, also entfernen.
        status = (
            f"✅  Fertig | Anrufer: {len(seg_caller)} Segmente, "
            f"Disponent: {len(seg_dispatcher)} Segmente | "
            "Anonymisierung deaktiviert"
        )
        yield raw_text, "(Anonymisierung nicht aktiviert)", "–", status
        return

    # ── Step 5: anonymize every segment ──────────────────
    yield raw_text, "", "", "🔒  Presidio anonymisiert ..."

    all_types: set[str] = set()
    for seg in segments:
        anon_text, types = anonymize_text(seg["text"])
        seg["text_anon"] = anon_text
        all_types.update(types)

    anon_formatted = dialogue_to_text(segments, anon = True)
    pii_report = ", ".join(sorted(all_types)) if all_types else "Keine PII erkannt"

    # ── Step 6: export anonymized JSON (raw text excluded) ─
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_path = EXPORT_DIR / f"notruf_{timestamp}.json"

    export_data = {
        "meta": {
            "timestamp":           datetime.now().isoformat(),
            "audio_duration_s":    duration_s,
            "model_asr":           "whisperx-large-v3",
            "anonymized":          True,
            "pii_types":           sorted(all_types),
            # FK-TODO: Segmente für Anrufer und Disponent sind uninteressant, also entfernen.
            "segments_caller":     len(seg_caller),
            "segments_dispatcher": len(seg_dispatcher),
        },
        # NOTE: no raw text field in the export
        "dialogue": [
            {
                "speaker": seg["sprecher"],
                "start":   seg["start"],
                "end":     seg["end"],
                "text":    seg["text_anon"],
            }
            for seg in segments
        ],
    }

    with open(export_path, "w", encoding = "utf-8") as f:
        json.dump(export_data, f, ensure_ascii = False, indent=2)

    status = (
        f"✅  Fertig | "
        # FK-TODO: Segmente für Anrufer und Disponent sind uninteressant, also entfernen.
        f"Anrufer: {len(seg_caller)} Segmente, "
        f"Disponent: {len(seg_dispatcher)} Segmente | "
        f"PII: {pii_report} | "
        f"Export → {export_path}"
    )
    yield raw_text, anon_formatted, pii_report, status


# ─────────────────────────────────────────────────────────
# GRADIO UI
# ─────────────────────────────────────────────────────────

BLOCK_KWARGS = {
    "title": "Notruf-Transkription",
    "theme": gradio.themes.Soft(),
    "css": ".footer { font-size: 0.8em; color: #888; }",
}
LAUNCH_KWARGS = {
    "server_name": "127.0.0.1",
    "server_port": 7860,
    "share": False,
    "show_error": True,
}
if GRADIO_6:
    BLOCK_KWARGS = {}
    # FK-TODO: use BLOCK_KWARGS from definition
    LAUNCH_KWARGS.update(
        theme = gradio.themes.Soft(),
        css = ".footer { font-size: 0.8em; color: #888; }")

with gradio.Blocks(**BLOCK_KWARGS) as demo:
    
    gradio.Markdown("# Notruf-Transkription & Anonymisierung")

    with gradio.Row():

        # ── left column: input ────────────────────────────
        with gradio.Column(scale = 1, min_width = 280):

            audio_input = gradio.Audio(
                label = "Notruf-WAV hochladen (Stereo, 8 kHz)",
                type = "filepath",
                sources = ["upload"])

            anon_toggle = gradio.Checkbox(
                value = True,
                label = "DSGVO-Anonymisierung (Presidio)",
                info = "Erkennt und ersetzt Namen, Orte, Telefonnummern etc.")

            with gradio.Row():
                start_btn = gradio.Button("▶  Verarbeiten", variant = "primary")
                clear_btn = gradio.ClearButton(
                    components =[audio_input],
                    value = "🗑  Reset",
                )

            gradio.Markdown("""
            ---
            **Kanalzuweisung (fest):**
            - Kanal 0 links  → Disponent
            - Kanal 1 rechts → Anrufer

            **Export:** `~/notruf-protokolle/notruf_*.json`
            *(nur anonymisierter Dialog)*
            """)

        # ── right column: output ──────────────────────────
        with gradio.Column(scale = 2):

            with gradio.Tab("📄 Rohtranskript (Dialog)"):
                roh_out = gradio.Textbox(
                    label = "Gesprächsprotokoll mit Zeitstempeln",
                    lines = 20,
                    placeholder = (
                        "[00.00s – 03.21s]  Anrufer:\n"
                        "    Notruf, hier ist ein Unfall auf der Hauptstraße!\n\n"
                        "[03.50s – 05.10s]  Disponent:\n"
                        "    Wo genau ist der Unfall?\n\n"
                        "[05.30s – 08.40s]  Anrufer:\n"
                        "    Hauptstraße 12, vor dem Supermarkt ..."),
                    **COPY_BUTTON)
                gradio.Markdown(
                    "⚠️  *Dieses Rohtranskript enthält personenbezogene Daten und wird nicht gespeichert.*",
                    elem_classes = ["footer"])

            with gradio.Tab("🔒 Anonymisiert (Dialog)"):
                anon_out = gradio.Textbox(
                    label = "Anonymisiertes Gesprächsprotokoll",
                    lines = 20,
                    placeholder = (
                        "[00.00s – 03.21s]  Anrufer:\n"
                        "    Notruf, hier ist ein Unfall auf der <ORT>!\n\n"
                        "[03.50s – 05.10s]  Disponent:\n"
                        "    Wo genau ist der Unfall?\n\n"
                        "[05.30s – 08.40s]  Anrufer:\n"
                        "    <ORT>, vor dem Supermarkt ..."
                    ),
                    **COPY_BUTTON)

            # FK-TODO: remove PII-Bericht?
            with gradio.Tab("ℹ️ PII-Bericht"):
                pii_out = gradio.Textbox(
                    label = "Erkannte PII-Kategorien",
                    lines = 3,
                    placeholder = "PERSON, LOCATION, PHONE_NUMBER")
                gradio.Markdown("""
                **Platzhalter-Legende:**
                `<PERSON>` · `<ORT>` · `<TELEFON>` · `<DATUM>` · `<EMAIL>` · `<IBAN>` · `<KENNZEICHEN>`
                """)

            status_out = gradio.Textbox(
                label = "Status",
                lines = 2,
                interactive = False,
                **NO_COPY_BUTTON)

    gradio.Markdown("""
    ---
    <div class="footer">
    ⚠️ <strong>Datenschutzhinweis:</strong>
    Alle Verarbeitungen laufen ausschließlich lokal auf diesem Rechner.
    Keine Audio- oder Textdaten werden an externe Server übermittelt (<code>share=False</code>).
    Der Rohtext (mit personenbezogenen Daten) wird nur im Arbeitsspeicher gehalten
    und nach der Anonymisierung nicht gespeichert.
    Nur das anonymisierte JSON-Protokoll wird in <code>~/notruf-protokolle/</code> abgelegt.
    Für den Produktiveinsatz in Behörden ist eine
    Datenschutz-Folgenabschätzung (DSFA, Art. 35 DSGVO) erforderlich.
    </div>
    """)

    # connect callback – channel selection removed, both channels always processed
    start_btn.click(
        fn = process_call,
        inputs = [audio_input, anon_toggle],
        outputs = [roh_out, anon_out, pii_out, status_out])

# ─────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(**LAUNCH_KWARGS)
