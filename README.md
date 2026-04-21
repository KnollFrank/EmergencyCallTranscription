# Notruf-Transkription & Anonymisierung

Lokale, DSGVO-konforme Pipeline zur Transkription und Anonymisierung von Notruf-Aufzeichnungen. 

Die Anwendung bietet eine Web-Oberfläche (Gradio), über die Notrufe (Stereo-WAV, 8kHz) hochgeladen, durch verschiedene KI-Modelle transkribiert und anschließend mittels Presidio (NLP) anonymisiert werden können. 

**Hardware-Referenz:** Dell XPS 9320 · Intel i7-1360P · 32 GB RAM · Ubuntu 24.04 LTS  
**Modus:** Vollständig lokal, keine Cloud-Anbindung notwendig, CPU-kompatibel.

## Schnellstart

**Hinweis:** Das Setup-Skript (`setup_and_start.sh`) ist ausschließlich für Linux-Systeme geeignet.

```bash
cd EmergencyCallTranscription
chmod +x setup_and_start.sh
bash setup_and_start.sh
```

Das Skript:
1. Prüft die Systemvoraussetzungen (Python 3.12, ffmpeg, git)
2. Erstellt eine virtuelle Umgebung unter `./venv/`
3. Installiert alle Pakete aus der `requirements.txt`
4. Lädt das deutsche spaCy-Modell herunter
5. Startet die Web-Oberfläche unter http://127.0.0.1:7860

**Beim ersten Start:** Die Modelle (WhisperX / Faster-Whisper) werden von Hugging Face heruntergeladen (ca. 1.5 GB bis 3 GB je nach Modell) und lokal unter `~/.cache` gespeichert.

## Manueller Start (nach Installation)

```bash
cd EmergencyCallTranscription
source venv/bin/activate
python src/EmergencyCallTranscription.py
```

## Bedienung der Web-Oberfläche

Die UI ist in drei Schritte unterteilt:

### Schritt 1: Eingabe & Einstellungen
* **Notruf-WAV hochladen:** Die Datei muss im WAV-Format (PCM), Stereo (2 Kanäle) und mit einer Abtastrate von 8 kHz vorliegen.
* **Transkriptions-Engine:** Auswahl zwischen `WhisperX` (sehr genaue Zeitstempel) und `Faster-Whisper` (schneller, ressourcenschonender).
* **Kanalzuordnung:** Auswahl, welcher Kanal (links/rechts) dem Disponenten bzw. dem Anrufer zugeordnet ist.

### Schritt 2: Transkription & Korrektur
Nach dem Klick auf "Transkription starten" wird das Audio verarbeitet. Das Ergebnis wird als interaktive Tabelle (Gesprächsprotokoll) dargestellt, in der manuelle Korrekturen am erkannten Text vorgenommen werden können.

### Schritt 3: Anonymisiertes Ergebnis
Ein Klick auf "Anonymisierung starten" leitet den (ggf. korrigierten) Text durch die Presidio-NLP-Pipeline. Das Resultat ist eine zweite Tabelle, in der sensible Daten wie Namen, Orte und Telefonnummern durch Platzhalter wie `<PERSON>`, `<LOCATION>` ersetzt wurden.

## Architektur & Pipeline

```text
WAV (Stereo, 8 kHz)
       │
       ▼
Kanal-Trennung & Resampling (librosa)
8 kHz → 16 kHz (für Whisper-Modelle)
       │
       ▼
Transkriptions-Engine (WhisperX / Faster-Whisper)
Erkennung pro Kanal & Zusammenführung des Dialogs
       │
       ▼
Manuelle Korrektur-Möglichkeit (Gradio Dataframe)
       │
       ▼
Microsoft Presidio (spaCy de_core_news_lg)
Erkennung & Ersetzung personenbezogener Daten (PII)
```

## DSGVO-Hinweise

- **Lokale Verarbeitung:** Es verlassen keine Daten den Rechner.
- **Datensparsamkeit:** Rohtranskripte werden nur temporär im RAM/Browser gehalten und standardmäßig nicht auf der Festplatte gespeichert.
- **Zweckbindung:** Das Tool ist zur Erstellung anonymisierter Protokolle für die weitere (ungefährliche) Verarbeitung gedacht.
- **Machbarkeitsstudie:** Dieses Tool ist lediglich ein Proof of Concept (PoC) / eine Machbarkeitsstudie und nicht für den produktiven Einsatz in einer Leitstelle vorgesehen.
