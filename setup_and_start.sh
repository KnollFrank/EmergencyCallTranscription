#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────
# setup_und_start.sh
# Sets up the Python venv and starts the app.
# First run:       bash setup_und_start.sh
# Subsequent runs: bash setup_und_start.sh  (skips installation)
# ─────────────────────────────────────────────────────────

set -euo pipefail

VENV_DIR="$(pwd)/venv"
APP="src/EmergencyCallTranscription.py"

echo ""
echo "════════════════════════════════════════════════"
echo "  Notruf-Transkription – Setup & Start"
echo "════════════════════════════════════════════════"
echo ""

# ── Require Python 3.12 (ctranslate2 has no wheels for 3.13) ──
PYTHON=""
for cmd in python3.12 python3; do
    if command -v "$cmd" &>/dev/null; then
        VER=$("$cmd" -c "import sys; print(sys.version_info[:2])")
        if [[ "$VER" == "(3, 12)" ]]; then
            PYTHON="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "ERROR: Python 3.12 is required."
    echo "  sudo apt install python3.12 python3.12-venv python3.12-dev"
    exit 1
fi
echo "▶  Python: $($PYTHON --version) ✓"

# ── Check system dependencies ─────────────────────────────
echo "▶  Checking system dependencies ..."
for dep in ffmpeg git; do
    if ! command -v "$dep" &>/dev/null; then
        echo "   $dep missing – installing."
        sudo apt-get update -qq
        sudo apt-get install -y "$dep"
    else
        echo "   $dep ✓"
    fi
done

# ── Create venv ───────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "▶  Creating venv with Python 3.12 ..."
    $PYTHON -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
echo "   venv: $VIRTUAL_ENV ✓"

# ── Install all packages ──────────────────────────────────
MARKER="$VENV_DIR/.packages_installed"

if [ ! -f "$MARKER" ]; then
    echo ""
    echo "▶  Installing dependencies ..."
    echo ""

    pip install --upgrade pip

    echo ""
    echo "── pip install -r requirements.txt ─────────────────────"
    echo "   All packages including PyTorch CPU and WhisperX."
    echo ""
    pip install -r requirements.txt

    echo ""
    echo "── German spaCy language model ──────────────────────────"
    echo ""
    python -m spacy download de_core_news_lg

    echo ""
    echo "── Installation check ───────────────────────────────────"
    python - <<'EOF'
import sys
ok = True
checks = {
    "whisperx":          "import whisperx",
    "torch (CPU)":       "import torch",
    "gradio":            "import gradio",
    "presidio_analyzer": "from presidio_analyzer import AnalyzerEngine",
    "spacy de model":    "import spacy; spacy.load('de_core_news_lg')",
    "librosa":           "import librosa",
    "soundfile":         "import soundfile",
}
for name, stmt in checks.items():
    try:
        exec(stmt)
        print(f"   {name:<24} ok")
    except Exception as e:
        print(f"   {name:<24} ERROR: {e}")
        ok = False
if not ok:
    print("\n   Errors found – setup aborted.")
    sys.exit(1)
else:
    print("\n   All packages present.")
EOF

    touch "$MARKER"
    echo ""
    echo "════════════════════════════════════════════════"
    echo "  Installation complete."
    echo "════════════════════════════════════════════════"
else
    echo "   Dependencies already installed ✓"
fi

# ── Start app ─────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════"
echo "  Starting app ..."
echo "  → http://127.0.0.1:7860"
echo ""
echo "  First start: models will be downloaded"
echo "    WhisperX large-v3"
echo "  (cached locally under ~/.cache afterwards)"
echo ""
echo "  Stop: Ctrl+C"
echo "════════════════════════════════════════════════"
echo ""

python "$APP"
