import os
import threading
import requests
from pydub import AudioSegment
import whisper
import numpy as np

# Sarvam's sync STT-translate API rejects audio longer than 30s.
# We slice each chunk into 25s pieces (with a 5s safety margin) before sending.
SARVAM_PIECE_SECONDS = 25

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"
SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")

_model = None
_faster_model = None
_openai_model = None
_model_lock = threading.Lock()

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
    is_multilingual = getattr(model, "is_multilingual", None)
    if is_multilingual is None:
        if hasattr(model, "model") and hasattr(model.model, "is_multilingual"):
            is_multilingual = model.model.is_multilingual
        else:
            model_name = getattr(model, "model_size_or_path", WHISPER_MODEL)
            is_multilingual = not str(model_name).endswith(".en")

    if not is_multilingual:
        if (language and language != "en") or task == "translate":
            raise ValueError(
                f"The loaded Whisper model '{WHISPER_MODEL}' is an English-only model (.en) "
                f"and does not support language='{language}' or task='{task}'. "
                "Please configure a multilingual Whisper model (e.g., 'small', 'base', 'medium') in WHISPER_MODEL."
            )


def get_device_and_compute():
    """Detect available hardware accelerator and optimal compute type for Whisper."""
    env_device = os.getenv("WHISPER_DEVICE", "").strip().lower()
    env_compute = os.getenv("WHISPER_COMPUTE_TYPE", "").strip().lower()

    if env_device in ("cuda", "cpu"):
        device = env_device
    else:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    if env_compute:
        compute_type = env_compute
    else:
        compute_type = "float16" if device == "cuda" else "int8"

    return device, compute_type


def load_faster_whisper_model(model_name: str = None):
    """Load and cache the faster-whisper model (thread-safe singleton)."""
    global _faster_model
    target_model = model_name or WHISPER_MODEL
    with _model_lock:
        if _faster_model is None:
            from faster_whisper import WhisperModel
            device, compute_type = get_device_and_compute()
            cpu_threads = min(os.cpu_count() or 4, 8)
            print(f"Loading faster-whisper model '{target_model}' on {device} ({compute_type}, {cpu_threads} threads)...")
            _faster_model = WhisperModel(
                target_model,
                device=device,
                compute_type=compute_type,
                cpu_threads=cpu_threads,
            )
            print(f"faster-whisper model '{target_model}' loaded successfully.")
    return _faster_model


def load_openai_whisper_model(model_name: str = None):
    """Load and cache the fallback OpenAI Whisper model (thread-safe singleton)."""
    global _openai_model
    target_model = model_name or WHISPER_MODEL
    with _model_lock:
        if _openai_model is None:
            print(f"Loading fallback OpenAI Whisper model: {target_model} ...")
            _openai_model = whisper.load_model(target_model)
            print("OpenAI Whisper model loaded.")
    return _openai_model


def load_model():
    """
    Load and cache the active local Whisper model.
    Prioritizes faster-whisper with INT8 CPU or float16 GPU acceleration,
    falling back to standard openai-whisper if faster-whisper is unavailable.
    """
    global _model
    use_faster = os.getenv("USE_FASTER_WHISPER", "true").strip().lower() in ("1", "true", "yes")
    if use_faster:
        try:
            _model = load_faster_whisper_model()
            return _model
        except Exception as e:
            print(f"[!] Warning: Failed to load faster-whisper ({e}). Falling back to OpenAI Whisper.")
    _model = load_openai_whisper_model()
    return _model


def transcribe_chunk_whisper(
    chunk_path: str | np.ndarray,
    language: str = None,
    task: str = "transcribe",
    return_segments: bool = False,
) -> str | dict:
    """
    Transcribe or translate an audio chunk using faster-whisper or fallback Whisper.
    
    Args:
        chunk_path: Path to the WAV audio chunk, or 1D float32 NumPy audio array.
        language: Whisper language code ('en', 'hi', 'ne', or None for auto-detect).
        task: 'transcribe' (default) or 'translate' (translates to English).
        return_segments: Whether to return full segments with timestamps.
    """
    if isinstance(chunk_path, np.ndarray):
        if chunk_path.size == 0:
            print("[!] Warning: Audio chunk numpy array is empty. Skipping.")
            return {"text": "", "segments": []} if return_segments else ""
    elif isinstance(chunk_path, str):
        if not os.path.exists(chunk_path):
            raise FileNotFoundError(f"Audio chunk file not found: {chunk_path}")
        if os.path.getsize(chunk_path) == 0:
            print(f"[!] Warning: Audio chunk {chunk_path} is 0 bytes. Skipping.")
            return {"text": "", "segments": []} if return_segments else ""
    else:
        raise TypeError(f"chunk_path must be a file path string or numpy array, got {type(chunk_path)}")

    model = load_model()
    validate_whisper_model(model, language=language, task=task)

    transcribe_kwargs = {"task": task}
    if language is not None:
        transcribe_kwargs["language"] = language

    # Configure beam size and robust decoding parameters for faster-whisper
    if type(model).__name__ == "WhisperModel":
        transcribe_kwargs["beam_size"] = int(os.getenv("WHISPER_BEAM_SIZE", "2"))
        transcribe_kwargs["condition_on_previous_text"] = False
        use_vad = os.getenv("WHISPER_VAD_FILTER", "true").strip().lower() in ("1", "true", "yes")
        if use_vad:
            transcribe_kwargs["vad_filter"] = True
            transcribe_kwargs["vad_parameters"] = dict(min_silence_duration_ms=400)

    try:
        try:
            raw_result = model.transcribe(chunk_path, **transcribe_kwargs)
        except Exception as vad_err:
            if transcribe_kwargs.get("vad_filter"):
                print(f"[!] VAD notice: {vad_err}. Retrying chunk without VAD...")
                transcribe_kwargs.pop("vad_filter", None)
                transcribe_kwargs.pop("vad_parameters", None)
                raw_result = model.transcribe(chunk_path, **transcribe_kwargs)
            else:
                raise


        # 1. Faster-whisper output format: (segments_generator, info)
        if isinstance(raw_result, tuple) and len(raw_result) == 2:
            segments_iter, _ = raw_result
            text_parts = []
            segments = []
            for seg in segments_iter:
                seg_text = getattr(seg, "text", "").strip()
                if seg_text:
                    text_parts.append(seg_text)
                segments.append({
                    "start": float(getattr(seg, "start", 0.0)),
                    "end": float(getattr(seg, "end", 0.0)),
                    "text": seg_text,
                })
            text = " ".join(text_parts).strip()
            if return_segments:
                return {"text": text, "segments": segments}
            return text

        # 2. OpenAI Whisper / Mock dict format: {"text": "...", "segments": [...]}
        if isinstance(raw_result, dict):
            text = raw_result.get("text", "").strip()
            if return_segments:
                return {"text": text, "segments": raw_result.get("segments", [])}
            return text

        # 3. String format
        text = str(raw_result).strip()
        if return_segments:
            return {"text": text, "segments": []}
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
    chunk_path: str | np.ndarray,
    language: str = "english",
    task: str = None,
    return_segments: bool = False,
) -> str | dict:
    """
    Route one chunk to Whisper or Sarvam depending on the language/mode configuration.
    """
    cfg = resolve_language_mode(language=language, task=task)
    if cfg["engine"] == "sarvam":
        if isinstance(chunk_path, np.ndarray):
            raise NotImplementedError("Sarvam API requires WAV file path, not in-memory numpy array.")
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
    progress_callback = None,
) -> str | dict:
    """
    Transcribe all audio chunks sequentially with the specified language mode and task.

    Args:
        chunks: List of WAV chunk file paths.
        language: Language name or mode key.
        task: Explicit task ('transcribe' or 'translate').
        return_original: If True and translating from Nepali, also generates original Devanagari transcript.
        chunk_duration_seconds: Estimated duration per chunk for timestamp offsets.
        progress_callback: Optional callable(current_chunk_idx, total_chunks, message) for UI progress indicators.

    Returns:
        str if return_original is False;
        dict with {"transcript", "original_transcript", "segments"} if return_original is True.
    """
    import time
    t_start = time.perf_counter()
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

    total_chunks = len(chunks)
    for i, chunk in enumerate(chunks):
        t_chunk_start = time.perf_counter()
        msg = f"Transcribing chunk {i + 1}/{total_chunks}..."
        print(f"[*] {msg}")
        if progress_callback:
            try:
                progress_callback(i + 1, total_chunks, msg)
            except Exception:
                pass

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

        chunk_elapsed = time.perf_counter() - t_chunk_start
        print(f"[*] Chunk {i + 1}/{total_chunks} completed in {chunk_elapsed:.2f}s")

    total_elapsed = time.perf_counter() - t_start
    print(f"[+] Transcription complete in {total_elapsed:.2f}s across {total_chunks} chunk(s).")
    cleaned_transcript = full_transcript.strip()

    if return_original:
        return {
            "transcript": cleaned_transcript,
            "original_transcript": (original_transcript.strip() if original_transcript else None),
            "segments": all_segments,
            "display_language": cfg["display_name"],
            "is_translated": cfg["task"] == "translate",
            "transcription_time": total_elapsed,
        }

    return cleaned_transcript


def translate_nepali_to_english(chunks: list, **kwargs) -> str:
    """
    Transcribe and translate Nepali audio chunks directly to English.
    """
    return transcribe_all(chunks, language="nepali-to-english", **kwargs)


def transcribe_nepali(chunks: list, **kwargs) -> str:
    """
    Transcribe Nepali audio chunks in original Nepali (Devanagari script).
    """
    return transcribe_all(chunks, language="nepali", **kwargs)


def transcribe_nepali_dual(chunks: list, **kwargs) -> dict:
    """
    Transcribe Nepali audio chunks to both English translation and original Devanagari transcript.
    Returns: dict with {"transcript": <English>, "original_transcript": <Nepali>, "segments": [...]}
    """
    return transcribe_all(chunks, language="nepali-to-english", return_original=True, **kwargs)