# Emergency Call Transcription & GDPR Anonymisation

Local, GDPR-compliant pipeline for transcribing and anonymising
emergency call recordings.

**Hardware:** Dell XPS 9320 · Intel i7-1360P · 32 GB RAM · Ubuntu 24.04 LTS  
**Mode:** Fully local, no CUDA, no cloud access

---

## Project structure

```
notruf-app/
├── src/
│   └── notruf_app.py      # Gradio web UI
├── requirements.txt       # all Python dependencies
├── setup_und_start.sh     # setup & launch script
└── README.md
```

---

## Quick start

```bash
cd notruf-app
chmod +x setup_und_start.sh
bash setup_und_start.sh
```

The script:
1. Checks for Python 3.12, ffmpeg, git
2. Creates a venv under `./venv/`
3. Runs `pip install -r requirements.txt`
4. Downloads the German spaCy model
5. Starts the app at http://127.0.0.1:7860

**First start:** WhisperX downloads its models from Hugging Face (~1.5 GB),
then caches them locally under `~/.cache`.

---

## Manual start (after installation)

```bash
cd notruf-app
source venv/bin/activate
python src/notruf_app.py
```

---

## Input format

| Parameter      | Value          |
|----------------|----------------|
| Format         | WAV (PCM)      |
| Sample rate    | 8 kHz          |
| Channels       | Stereo         |
| Channel 0 (left)  | Dispatcher  |
| Channel 1 (right) | Caller      |

---

## Pipeline

```
WAV (stereo, 8 kHz)
       │
       ▼
Channel extraction (librosa)
8 kHz → 16 kHz resampling
       │  (both channels in parallel)
       ▼
WhisperX large-v3 (CPU/INT8, German)
+ word-level alignment (timestamps)
       │
       ▼
Presidio (spaCy de_core_news_lg)
GDPR anonymisation
       │
       ▼
JSON export (~/notruf-protokolle/)
[raw transcript never saved to disk]
```

---

## Estimated duration (1:47 min audio, i7-1360P)

| Step                   | Duration     |
|------------------------|--------------|
| Channel + resampling   | ~5 sec       |
| WhisperX × 2 channels  | ~2–4 min     |
| Presidio               | ~5 sec       |
| **Total**              | **~2–4 min** |

---

## PII placeholders (Presidio)

| Placeholder      | Example                  |
|------------------|--------------------------|
| `<PERSON>`       | Max Mustermann           |
| `<ORT>`          | Stuttgarter Straße 12    |
| `<TELEFON>`      | 0711-123456              |
| `<DATUM>`        | 14. März, gestern Abend  |
| `<EMAIL>`        | name@beispiel.de         |
| `<IBAN>`         | DE89 3704 0044 0532      |
| `<KENNZEICHEN>`  | LB-XY 123                |

---

## Export format

Anonymised transcripts are saved to `~/notruf-protokolle/notruf_YYYYMMDD_HHMMSS.json`:

```json
{
  "meta": {
    "timestamp": "2025-04-11T14:32:01",
    "audio_duration_s": 107,
    "model_asr": "whisperx-large-v3",
    "anonymised": true,
    "pii_types": ["LOCATION", "PERSON", "PHONE_NUMBER"]
  },
  "dialogue": [
    { "speaker": "Anrufer",   "start": 0.0,  "end": 3.21, "text": "Unfall auf der <ORT>." },
    { "speaker": "Disponent", "start": 3.50, "end": 5.10, "text": "Wo genau?" }
  ]
}
```

---

## GDPR notes

- **Local processing:** no data leaves the machine
- **Data minimisation:** raw transcript is never written to disk
- **Purpose limitation:** export contains anonymised text only
- **Cleanup:** temporary audio files deleted immediately after use
- **Production use:** a DPIA (Art. 35 GDPR) is required

---

## Troubleshooting

**`No module named 'whisperx'`**
```bash
source venv/bin/activate
pip install -r requirements.txt
```

**spaCy model missing**
```bash
source venv/bin/activate
python -m spacy download de_core_news_lg
```

**Out of memory during transcription**
```bash
# Increase swap (one-time):
sudo swapoff /swapfile
sudo fallocate -l 8G /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
```

**Change model (speed vs. accuracy)**

Edit the `load_model` call in `src/notruf_app.py`:
```python
asr_model = whisperx.load_model("medium", ...)      # faster, slightly less accurate
asr_model = whisperx.load_model("large-v3", ...)    # default
```
