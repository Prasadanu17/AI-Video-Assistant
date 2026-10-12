import os
import requests
from pydub import AudioSegment
import whisper

# Sarvam's sync STT-translate API rejects audio longer than 30s.
# We slice each chunk into 25s pieces (with a 5s safety margin) before sending.
SARVAM_PIECE_SECONDS = 25

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"
SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")

_model = None

# Comprehensive Language Mapping & Alias Normalization
LANGUAGE_CONFIG = {
    # English
    "english": {"engine": "whisper", "whisper_code": "en", "default_task": "transcribe", "display_name": "English"},
    "english transcription": {"engine": "whisper", "whisper_code": "en", "default_task": "transcribe", "display_name": "English"},
    "en": {"engine": "whisper", "whisper_code": "en", "default_task": "transcribe", "display_name": "English"},

    # Hindi
    "hindi": {"engine": "whisper", "whisper_code": "hi", "default_task": "transcribe", "display_name": "Hindi"},
    "hi": {"engine": "whisper", "whisper_code": "hi", "default_task": "transcribe", "display_name": "Hindi"},

    # Nepali Transcription (Devanagari script)
    "nepali": {"engine": "whisper", "whisper_code": "ne", "default_task": "transcribe", "display_name": "Nepali (Devanagari)"},
    "nepali transcription": {"engine": "whisper", "whisper_code": "ne", "default_task": "transcribe", "display_name": "Nepali (Devanagari)"},
    "ne": {"engine": "whisper", "whisper_code": "ne", "default_task": "transcribe", "display_name": "Nepali (Devanagari)"},

    # Nepali-to-English Translation
    "nepali-to-english": {"engine": "whisper", "whisper_code": "ne", "default_task": "translate", "display_name": "Nepali → English Translation"},
    "nepali_to_english": {"engine": "whisper", "whisper_code": "ne", "default_task": "translate", "display_name": "Nepali → English Translation"},
    "nepali speech → english text": {"engine": "whisper", "whisper_code": "ne", "default_task": "translate", "display_name": "Nepali → English Translation"},
    "nepali speech -> english text": {"engine": "whisper", "whisper_code": "ne", "default_task": "translate", "display_name": "Nepali → English Translation"},
    "nepali-en": {"engine": "whisper", "whisper_code": "ne", "default_task": "translate", "display_name": "Nepali → English Translation"},
    "ne-en": {"engine": "whisper", "whisper_code": "ne", "default_task": "translate", "display_name": "Nepali → English Translation"},
    "ne-to-en": {"engine": "whisper", "whisper_code": "ne", "default_task": "translate", "display_name": "Nepali → English Translation"},
    "nepali_translate": {"engine": "whisper", "whisper_code": "ne", "default_task": "translate", "display_name": "Nepali → English Translation"},

    # Auto-detection
    "auto": {"engine": "whisper", "whisper_code": None, "default_task": "transcribe", "display_name": "Auto-Detect"},
    "auto-detect": {"engine": "whisper", "whisper_code": None, "default_task": "transcribe", "display_name": "Auto-Detect"},
    "autodetect": {"engine": "whisper", "whisper_code": None, "default_task": "transcribe", "display_name": "Auto-Detect"},
    "automatic language detection": {"engine": "whisper", "whisper_code": None, "default_task": "transcribe", "display_name": "Auto-Detect"},

    # Hinglish (Sarvam AI cloud translation)
    "hinglish": {"engine": "sarvam", "whisper_code": None, "default_task": "transcribe", "display_name": "Hinglish (Sarvam AI)"},
    "hinglish transcription using the existing sarvam integration": {"engine": "sarvam", "whisper_code": None, "default_task": "transcribe", "display_name": "Hinglish (Sarvam AI)"},
    "sarvam": {"engine": "sarvam", "whisper_code": None, "default_task": "transcribe", "display_name": "Hinglish (Sarvam AI)"},
}


def resolve_language_mode(language: str = "english", task: str = None) -> dict:
    """
    Normalize user input into a validated engine, whisper_code, and task configuration.
    
    Args:
        language: Language name or alias ('english', 'hindi', 'nepali', 'nepali-to-english', 'auto', 'hinglish').
        task: Explicit task override ('transcribe' or 'translate').
        
    Returns:
        dict with keys: engine, whisper_code, task, display_name, mode_key.
    """
    if not language:
        normalized = "english"
    else:
        normalized = str(language).strip().lower().replace("→", "->")
        # Direct lookup first
        if normalized in LANGUAGE_CONFIG:
            config = LANGUAGE_CONFIG[normalized]
            resolved_task = task if task in ("transcribe", "translate") else config["default_task"]
            return {
                "engine": config["engine"],
                "whisper_code": config["whisper_code"],
                "task": resolved_task,
                "display_name": config["display_name"],
                "mode_key": normalized,
            }

        # Normalize spaces and underscores to dashes
        slug = normalized.replace(" ", "-").replace("_", "-")
        if slug in LANGUAGE_CONFIG:
            config = LANGUAGE_CONFIG[slug]
            resolved_task = task if task in ("transcribe", "translate") else config["default_task"]
            return {
                "engine": config["engine"],
                "whisper_code": config["whisper_code"],
                "task": resolved_task,
                "display_name": config["display_name"],
                "mode_key": slug,
            }

        # Check standard Whisper tokenizer languages
        clean_code = slug.replace("-", "")
        import whisper.tokenizer as tok
        if clean_code in tok.LANGUAGES:
            config = {
                "engine": "whisper",
                "whisper_code": clean_code,
                "default_task": "transcribe",
                "display_name": tok.LANGUAGES[clean_code].capitalize(),
            }
        else:
            print(f"[!] Warning: Unknown language '{language}'. Defaulting to auto-detection.")
            config = {
                "engine": "whisper",
                "whisper_code": None,
                "default_task": "transcribe",
                "display_name": "Auto-Detect",
            }

    # Override task if explicitly provided
    resolved_task = task if task in ("transcribe", "translate") else config["default_task"]

    return {
        "engine": config["engine"],
        "whisper_code": config["whisper_code"],
        "task": resolved_task,
        "display_name": config["display_name"],
        "mode_key": normalized,
    }


def validate_whisper_model(model, language: str = None, task: str = None):
    """
    Validate that the loaded Whisper model supports multilingual processing.
    Raises ValueError if an English-only model (.en) is used for non-English transcription or translation.
    """
    is_multilingual = getattr(model, "is_multilingual", True)
    if not is_multilingual:
        if (language and language != "en") or task == "translate":
            raise ValueError(
                f"The loaded Whisper model '{WHISPER_MODEL}' is an English-only model (.en) "
                f"and does not support language='{language}' or task='{task}'. "
                "Please configure a multilingual Whisper model (e.g., 'small', 'base', 'medium') in WHISPER_MODEL."
            )


def load_model():
    """Load and cache the local Whisper model."""
    global _model
    if _model is None:
        print(f"Loading Whisper model: {WHISPER_MODEL} ...")
        _model = whisper.load_model(WHISPER_MODEL)
        print("Whisper model loaded.")
    return _model


def transcribe_chunk_whisper(
    chunk_path: str,
    language: str = None,
    task: str = "transcribe",
    return_segments: bool = False,
) -> str | dict:
    """
    Transcribe or translate an audio chunk using local Whisper.
    
    Args:
        chunk_path: Path to the WAV audio chunk.
        language: Whisper language code ('en', 'hi', 'ne', or None for auto-detect).
        task: 'transcribe' (default) or 'translate' (translates to English).
        return_segments: Whether to return full segments with timestamps.
    """
    if not os.path.exists(chunk_path):
        raise FileNotFoundError(f"Audio chunk file not found: {chunk_path}")

    # Check for empty / zero-byte chunk
    if os.path.getsize(chunk_path) == 0:
        print(f"[!] Warning: Audio chunk {chunk_path} is 0 bytes. Skipping.")
        return {"text": "", "segments": []} if return_segments else ""

    model = load_model()
    validate_whisper_model(model, language=language, task=task)

    transcribe_kwargs = {"task": task}
    if language is not None:
        transcribe_kwargs["language"] = language

    try:
        result = model.transcribe(chunk_path, **transcribe_kwargs)
        text = result.get("text", "").strip()
        if return_segments:
            return {"text": text, "segments": result.get("segments", [])}
        return text
    except Exception as e:
        print(f"[!] Error during Whisper transcription for {chunk_path}: {e}")
        # Return empty text safely so one bad chunk does not drop the entire run
        if return_segments:
            return {"text": "", "segments": []}
        return ""


def _send_to_sarvam(piece_path: str) -> str:
    """Send one <=30s WAV file to Sarvam and return the English transcript."""
    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY is not set in environment / .env")

    headers = {"api-subscription-key": SARVAM_API_KEY}

    with open(piece_path, "rb") as f:
        files = {"file": (os.path.basename(piece_path), f, "audio/wav")}
        data = {"model": SARVAM_MODEL, "with_diarization": "false"}
        response = requests.post(
            SARVAM_STT_TRANSLATE_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=120,
        )

    if not response.ok:
        print(f"\n[!] Sarvam returned {response.status_code}")
        print(f"Response body: {response.text}\n")
        response.raise_for_status()

    return response.json().get("transcript", "")


def transcribe_chunk_sarvam(chunk_path: str) -> str:
    """
    Sarvam sync API only accepts <=30s audio. We split this chunk into
    25-second pieces, send each separately, and join the transcripts.
    """
    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY is not set in environment / .env")

    audio = AudioSegment.from_wav(chunk_path)
    piece_ms = SARVAM_PIECE_SECONDS * 1000

    full_text = ""
    total_pieces = (len(audio) + piece_ms - 1) // piece_ms

    for i, start in enumerate(range(0, len(audio), piece_ms)):
        piece = audio[start : start + piece_ms]
        piece_path = f"{chunk_path}_sv_{i}.wav"
        piece.export(piece_path, format="wav")

        try:
            print(f"  -> Sarvam piece {i + 1}/{total_pieces} ...")
            full_text += _send_to_sarvam(piece_path) + " "
        finally:
            if os.path.exists(piece_path):
                try:
                    os.remove(piece_path)
                except Exception:
                    pass

    return full_text.strip()


def transcribe_chunk(
    chunk_path: str,
    language: str = "english",
    task: str = None,
    return_segments: bool = False,
) -> str | dict:
    """
    Route one chunk to Whisper or Sarvam depending on the language/mode configuration.
    """
    cfg = resolve_language_mode(language=language, task=task)
    if cfg["engine"] == "sarvam":
        text = transcribe_chunk_sarvam(chunk_path)
        return {"text": text, "segments": []} if return_segments else text

    return transcribe_chunk_whisper(
        chunk_path,
        language=cfg["whisper_code"],
        task=cfg["task"],
        return_segments=return_segments,
    )


def transcribe_all(
    chunks: list,
    language: str = "english",
    task: str = None,
    return_original: bool = False,
    chunk_duration_seconds: float = 180.0,
) -> str | dict:
    """
    Transcribe all audio chunks sequentially with the specified language mode and task.

    Args:
        chunks: List of WAV chunk file paths.
        language: Language name or mode key.
        task: Explicit task ('transcribe' or 'translate').
        return_original: If True and translating from Nepali, also generates original Devanagari transcript.
        chunk_duration_seconds: Estimated duration per chunk for timestamp offsets.

    Returns:
        str if return_original is False;
        dict with {"transcript", "original_transcript", "segments"} if return_original is True.
    """
    if not chunks:
        print("[!] No audio chunks provided to transcribe.")
        if return_original:
            return {"transcript": "", "original_transcript": "", "segments": []}
        return ""

    cfg = resolve_language_mode(language=language, task=task)

    if cfg["engine"] == "sarvam":
        print(f"Using Sarvam AI for transcription ({cfg['display_name']}).")
    else:
        lang_label = cfg["whisper_code"] if cfg["whisper_code"] else "auto"
        print(
            f"Using Whisper for {cfg['display_name']} "
            f"(code: '{lang_label}', task: '{cfg['task']}')."
        )

    full_transcript = ""
    original_transcript = "" if return_original else None
    all_segments = []

    is_translating_nepali = (cfg["whisper_code"] == "ne" and cfg["task"] == "translate")

    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i + 1}/{len(chunks)}...")
        chunk_offset = i * chunk_duration_seconds

        # 1. Primary pass (Translation or Transcription)
        res = transcribe_chunk(chunk, language=language, task=task, return_segments=True)
        text = res["text"] if isinstance(res, dict) else str(res)
        segments = res.get("segments", []) if isinstance(res, dict) else []

        if text and text.strip():
            full_transcript += text.strip() + " "

        for seg in segments:
            all_segments.append({
                "start": chunk_offset + seg.get("start", 0.0),
                "end": chunk_offset + seg.get("end", 0.0),
                "text": seg.get("text", "").strip(),
            })

        # 2. Dual pass for original Nepali Devanagari text if requested
        if return_original and is_translating_nepali:
            print(f"  -> Transcribing original Nepali (Devanagari) for chunk {i + 1}...")
            orig_text = transcribe_chunk_whisper(chunk, language="ne", task="transcribe")
            if orig_text and orig_text.strip():
                original_transcript += orig_text.strip() + " "

    print("Transcription complete.")
    cleaned_transcript = full_transcript.strip()

    if return_original:
        return {
            "transcript": cleaned_transcript,
            "original_transcript": (original_transcript.strip() if original_transcript else None),
            "segments": all_segments,
            "display_language": cfg["display_name"],
            "is_translated": cfg["task"] == "translate",
        }

    return cleaned_transcript


def translate_nepali_to_english(chunks: list) -> str:
    """
    Transcribe and translate Nepali audio chunks directly to English.
    """
    return transcribe_all(chunks, language="nepali-to-english")


def transcribe_nepali(chunks: list) -> str:
    """
    Transcribe Nepali audio chunks in original Nepali (Devanagari script).
    """
    return transcribe_all(chunks, language="nepali")


def transcribe_nepali_dual(chunks: list) -> dict:
    """
    Transcribe Nepali audio chunks to both English translation and original Devanagari transcript.
    Returns: dict with {"transcript": <English>, "original_transcript": <Nepali>, "segments": [...]}
    """
    return transcribe_all(chunks, language="nepali-to-english", return_original=True)