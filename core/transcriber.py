import os
import sys
# Ensure UTF-8 console output and graceful encoding fallback on Windows
for _s in ('stdout', 'stderr'):
    _stream = getattr(sys, _s, None)
    if _stream and hasattr(_stream, 'reconfigure'):
        try:
            _stream.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

_builtin_print = print

def print(*args, **kwargs):
    try:
        _builtin_print(*args, **kwargs)
    except UnicodeEncodeError:
        _enc = getattr(sys.stdout, 'encoding', None) or 'ascii'
        _safe_args = [
            str(a).replace('→', '->').encode(_enc, errors='replace').decode(_enc, errors='replace')
            if isinstance(a, str) else a
            for a in args
        ]
        _builtin_print(*_safe_args, **kwargs)
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

# --------------------------------------------------------------------------
# Processing Modes & Engine Presets (Phase 1)
# --------------------------------------------------------------------------
PROCESSING_MODES = {
    "fast": {
        "model_name": os.getenv("WHISPER_FAST_MODEL", "tiny"),
        "beam_size": 1,
        "vad_threshold": 0.40,
        "description": "Fast Mode: Lightweight multilingual model (tiny) prioritizing minimum CPU latency.",
    },
    "balanced": {
        "model_name": os.getenv("WHISPER_BALANCED_MODEL", "small"),
        "beam_size": 1,
        "vad_threshold": 0.35,
        "description": "Balanced Mode: Standard multilingual model (small) balancing speed and accuracy.",
    },
    "high_accuracy": {
        "model_name": os.getenv("WHISPER_ACCURATE_MODEL", "medium"),
        "beam_size": 3,
        "vad_threshold": 0.30,
        "description": "High Accuracy Mode: High-capacity model (medium) for complex accents, jargon, and code-switching.",
    },
}

_current_processing_mode = os.getenv("PROCESSING_MODE", "balanced").strip().lower()
if _current_processing_mode not in PROCESSING_MODES:
    _current_processing_mode = "balanced"

_model = None
_faster_model = None
_faster_model_registry = {}
_openai_model_registry = {}
_model_lock = threading.RLock()


def unload_all_models():
    """Unload all cached models from memory to free RAM."""
    global _faster_model_registry, _openai_model_registry, _model, _faster_model
    with _model_lock:
        _faster_model_registry.clear()
        _openai_model_registry.clear()
        _faster_model = None
        _model = None
    import gc
    gc.collect()


def get_processing_mode() -> str:
    """Return the active processing mode ('fast', 'balanced', or 'high_accuracy')."""
    return _current_processing_mode


def set_processing_mode(mode: str) -> str:
    """Set the active processing mode ('fast', 'balanced', or 'high_accuracy')."""
    global _current_processing_mode
    m = str(mode).strip().lower().replace(" ", "_").replace("-", "_")
    if m in PROCESSING_MODES:
        _current_processing_mode = m
        print(f"[Transcriber] Switched processing mode to '{m}' (model: {PROCESSING_MODES[m]['model_name']}).")
    else:
        print(f"[!] Unknown processing mode '{mode}'. Preserving '{_current_processing_mode}'.")
    return _current_processing_mode


def get_active_model_name(mode: str = None) -> str:
    """Return model name for the specified or current processing mode."""
    m = (mode or _current_processing_mode).strip().lower().replace(" ", "_").replace("-", "_")
    cfg = PROCESSING_MODES.get(m, PROCESSING_MODES["balanced"])
    return cfg["model_name"]


# --------------------------------------------------------------------------
# Multilingual Lexical Markers for Contextual Language Detection (Phase 2)
# --------------------------------------------------------------------------
NEPALI_LEXICAL_MARKERS = {
    "छ", "छन्", "छैन", "थियो", "थिए", "भयो", "भने", "भनेर", "गर्यो", "गरेको",
    "गर्नुहोस्", "हुनुहुन्छ", "तर", "लाई", "मा", "बाट", "सँग", "पनि", "गर्छु",
    "हुँदैन", "हुन्छ", "भएको", "गर्ने", "गर्नेछ", "रहेको", "धेरै", "अलि", "कस्तो",
    "हामी", "तपाईं", "तिमी", "उनी", "यस", "त्यस", "कुरा", "काम", "समय", "नेपाल",
    "धन्यवाद", "राम्रो", "हो", "होइन", "के", "कसरी", "कहिले", "कहाँ"
}

HINDI_LEXICAL_MARKERS = {
    "है", "हैं", "था", "थी", "थे", "हुआ", "किया", "कीजिए", "होगा", "होंगे",
    "लेकिन", "को", "में", "से", "भी", "करता", "करेंगे", "नहीं", "सकता", "सकते",
    "गया", "गए", "गई", "वाले", "बात", "आप", "हम", "यह", "वह", "इस", "उस",
    "समय", "काम", "बहुत", "भारत", "चाहिए", "धन्यवाद", "अच्छा", "हाँ", "क्या"
}


def analyze_segment_language(text: str, whisper_code: str = None, prob: float = 0.0) -> dict:
    """
    Contextual language detection distinguishing English, Hindi, Nepali, and code-switching.
    Works at segment boundaries to avoid misidentifying short technical terms or proper nouns.
    """
    import re
    if not text or not text.strip():
        return {
            "primary_code": whisper_code or "en",
            "display_name": "English",
            "is_code_switched": False,
            "script": "Latin",
        }

    clean_text = text.strip()
    devanagari_chars = len(re.findall(r"[\u0900-\u097F]", clean_text))
    latin_chars = len(re.findall(r"[a-zA-Z]", clean_text))
    total_alpha = devanagari_chars + latin_chars

    clean_words = set(re.findall(r"[\w\u0900-\u097F]+", clean_text.lower()))
    nepali_hits = len(clean_words & NEPALI_LEXICAL_MARKERS)
    hindi_hits = len(clean_words & HINDI_LEXICAL_MARKERS)

    has_devanagari = devanagari_chars > 0
    has_latin = latin_chars > 0

    # 1. Code-switching detection (substantial presence of both Latin and Devanagari)
    if has_devanagari and has_latin and total_alpha > 0:
        deva_ratio = devanagari_chars / total_alpha
        latin_ratio = latin_chars / total_alpha
        if deva_ratio > 0.12 and latin_ratio > 0.12:
            if nepali_hits >= hindi_hits and nepali_hits > 0:
                return {
                    "primary_code": "en+ne",
                    "display_name": "English + Nepali",
                    "is_code_switched": True,
                    "script": "Mixed",
                }
            elif hindi_hits > nepali_hits:
                return {
                    "primary_code": "en+hi",
                    "display_name": "English + Hindi",
                    "is_code_switched": True,
                    "script": "Mixed",
                }
            else:
                return {
                    "primary_code": "multilingual",
                    "display_name": "Multilingual (Code-Switching)",
                    "is_code_switched": True,
                    "script": "Mixed",
                }

    # 2. Devanagari script evaluation (Nepali vs Hindi)
    if has_devanagari:
        if nepali_hits > hindi_hits:
            code = "ne"
            name = "Nepali (Devanagari)"
        elif hindi_hits > nepali_hits:
            code = "hi"
            name = "Hindi"
        else:
            if whisper_code in ("ne", "hi"):
                code = whisper_code
                name = "Nepali (Devanagari)" if code == "ne" else "Hindi"
            else:
                code = "ne" if nepali_hits > 0 else "hi"
                name = "Nepali (Devanagari)" if code == "ne" else "Hindi"

        return {
            "primary_code": code,
            "display_name": name,
            "is_code_switched": False,
            "script": "Devanagari",
        }

    # 3. Pure Latin / English
    return {
        "primary_code": "en",
        "display_name": "English",
        "is_code_switched": False,
        "script": "Latin",
    }


# Comprehensive Language Mapping & Alias Normalization
LANGUAGE_CONFIG = {
    # Default Automatic Language Detection (Recommended default for all meetings)
    "auto": {"engine": "whisper", "whisper_code": None, "default_task": "transcribe", "display_name": "Auto-Detect (English, Hindi, Nepali, Mixed)"},
    "auto-detect": {"engine": "whisper", "whisper_code": None, "default_task": "transcribe", "display_name": "Auto-Detect (English, Hindi, Nepali, Mixed)"},
    "autodetect": {"engine": "whisper", "whisper_code": None, "default_task": "transcribe", "display_name": "Auto-Detect (English, Hindi, Nepali, Mixed)"},
    "automatic language detection": {"engine": "whisper", "whisper_code": None, "default_task": "transcribe", "display_name": "Auto-Detect (English, Hindi, Nepali, Mixed)"},

    # Multilingual & Code-Switching
    "multilingual": {"engine": "whisper", "whisper_code": None, "default_task": "transcribe", "display_name": "Multilingual (Code-Switching)"},
    "code-switching": {"engine": "whisper", "whisper_code": None, "default_task": "transcribe", "display_name": "Multilingual (Code-Switching)"},
    "mixed": {"engine": "whisper", "whisper_code": None, "default_task": "transcribe", "display_name": "Multilingual (Code-Switching)"},
    "en+ne": {"engine": "whisper", "whisper_code": None, "default_task": "transcribe", "display_name": "English + Nepali"},
    "en+hi": {"engine": "whisper", "whisper_code": None, "default_task": "transcribe", "display_name": "English + Hindi"},
    "hi+ne": {"engine": "whisper", "whisper_code": None, "default_task": "transcribe", "display_name": "Hindi + Nepali"},
    "en+hi+ne": {"engine": "whisper", "whisper_code": None, "default_task": "transcribe", "display_name": "English + Hindi + Nepali"},

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
    "nepali-to-english": {"engine": "whisper", "whisper_code": "ne", "default_task": "translate", "display_name": "Nepali -> English Translation"},
    "nepali_to_english": {"engine": "whisper", "whisper_code": "ne", "default_task": "translate", "display_name": "Nepali -> English Translation"},
    "nepali speech → english text": {"engine": "whisper", "whisper_code": "ne", "default_task": "translate", "display_name": "Nepali -> English Translation"},
    "nepali speech -> english text": {"engine": "whisper", "whisper_code": "ne", "default_task": "translate", "display_name": "Nepali -> English Translation"},
    "nepali-en": {"engine": "whisper", "whisper_code": "ne", "default_task": "translate", "display_name": "Nepali -> English Translation"},
    "ne-en": {"engine": "whisper", "whisper_code": "ne", "default_task": "translate", "display_name": "Nepali -> English Translation"},
    "ne-to-en": {"engine": "whisper", "whisper_code": "ne", "default_task": "translate", "display_name": "Nepali -> English Translation"},
    "nepali_translate": {"engine": "whisper", "whisper_code": "ne", "default_task": "translate", "display_name": "Nepali -> English Translation"},

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
    """Load and cache the faster-whisper model (thread-safe multi-model singleton registry)."""
    global _faster_model_registry, _faster_model
    target_model = model_name or get_active_model_name()
    with _model_lock:
        if target_model not in _faster_model_registry:
            from faster_whisper import WhisperModel
            device, compute_type = get_device_and_compute()
            cpu_threads = min(os.cpu_count() or 4, 8)
            print(f"Loading faster-whisper model '{target_model}' on {device} ({compute_type}, {cpu_threads} threads)...")
            instance = WhisperModel(
                target_model,
                device=device,
                compute_type=compute_type,
                cpu_threads=cpu_threads,
            )
            _faster_model_registry[target_model] = instance
            _faster_model = instance
            print(f"faster-whisper model '{target_model}' loaded successfully.")
    return _faster_model_registry[target_model]


def load_openai_whisper_model(model_name: str = None):
    """Load and cache the fallback OpenAI Whisper model (thread-safe multi-model singleton registry)."""
    global _openai_model_registry
    target_model = model_name or get_active_model_name()
    with _model_lock:
        if target_model not in _openai_model_registry:
            print(f"Loading fallback OpenAI Whisper model: {target_model} ...")
            _openai_model_registry[target_model] = whisper.load_model(target_model)
            print(f"OpenAI Whisper model '{target_model}' loaded.")
    return _openai_model_registry[target_model]


def load_model(model_name: str = None, processing_mode: str = None):
    """
    Load and cache the active local Whisper model for current processing mode or target model.
    Prioritizes faster-whisper with INT8 CPU or float16 GPU acceleration,
    falling back to standard openai-whisper if faster-whisper is unavailable.
    """
    target = model_name or get_active_model_name(processing_mode)
    use_faster = os.getenv("USE_FASTER_WHISPER", "true").strip().lower() in ("1", "true", "yes")
    if use_faster:
        try:
            return load_faster_whisper_model(target)
        except Exception as e:
            print(f"[!] Warning: Failed to load faster-whisper ({e}). Falling back to OpenAI Whisper.")
    return load_openai_whisper_model(target)


def transcribe_chunk_whisper(
    chunk_path: str | np.ndarray,
    language: str = None,
    task: str = "transcribe",
    return_segments: bool = False,
    initial_prompt: str = None,
    processing_mode: str = None,
) -> str | dict:
    """
    Transcribe or translate an audio chunk using faster-whisper or fallback Whisper.
    Supports Fast, Balanced, and High Accuracy processing modes, dynamic language detection,
    and dual verbatim + English translation generation.
    
    Args:
        chunk_path: Path to the WAV audio chunk, or 1D float32 NumPy audio array.
        language: Whisper language code ('en', 'hi', 'ne', or None for auto-detect).
        task: 'transcribe' (default), 'translate' (translates to English), or 'auto' (dual stream).
        return_segments: Whether to return full segments with timestamps.
        initial_prompt: Optional context prompt to prime continuous vocabulary and grammar.
        processing_mode: 'fast' (tiny), 'balanced' (small), or 'high_accuracy' (medium).
    """
    if isinstance(chunk_path, np.ndarray):
        if chunk_path.size == 0:
            print("[!] Warning: Audio chunk numpy array is empty. Skipping.")
            return {"text": "", "original_text": "", "english_text": "", "segments": []} if return_segments else ""
    elif isinstance(chunk_path, str):
        if not os.path.exists(chunk_path):
            raise FileNotFoundError(f"Audio chunk file not found: {chunk_path}")
        if os.path.getsize(chunk_path) == 0:
            print(f"[!] Warning: Audio chunk {chunk_path} is 0 bytes. Skipping.")
            return {"text": "", "original_text": "", "english_text": "", "segments": []} if return_segments else ""
    else:
        raise TypeError(f"chunk_path must be a file path string or numpy array, got {type(chunk_path)}")

    # Resolve language mode if an alias or descriptive name was passed
    resolved_lang = language
    resolved_task = task or "transcribe"
    if language is not None:
        try:
            mode_cfg = resolve_language_mode(language, task=task)
            resolved_lang = mode_cfg.get("whisper_code")
            resolved_task = mode_cfg.get("task", resolved_task)
        except Exception:
            pass

    active_mode = (processing_mode or get_processing_mode()).strip().lower()
    mode_spec = PROCESSING_MODES.get(active_mode, PROCESSING_MODES["balanced"])

    model = load_model(model_name=mode_spec["model_name"])
    validate_whisper_model(model, language=resolved_lang, task=resolved_task)

    transcribe_kwargs = {"task": resolved_task if resolved_task != "auto" else "transcribe"}
    if resolved_lang is not None:
        transcribe_kwargs["language"] = resolved_lang

    # Configure beam size and robust decoding parameters for faster-whisper
    if type(model).__name__ == "WhisperModel":
        env_beam = os.getenv("WHISPER_BEAM_SIZE")
        transcribe_kwargs["beam_size"] = int(env_beam) if env_beam else mode_spec.get("beam_size", 1)
        transcribe_kwargs["condition_on_previous_text"] = False
        use_vad = os.getenv("WHISPER_VAD_FILTER", "true").strip().lower() in ("1", "true", "yes")
        if use_vad:
            transcribe_kwargs["vad_filter"] = True
            vad_thresh = float(os.getenv("WHISPER_VAD_THRESHOLD", str(mode_spec.get("vad_threshold", 0.35))))
            transcribe_kwargs["vad_parameters"] = dict(
                threshold=vad_thresh,
                min_silence_duration_ms=int(os.getenv("WHISPER_MIN_SILENCE_MS", "400")),
                speech_pad_ms=int(os.getenv("WHISPER_SPEECH_PAD_MS", "250")),
            )
        transcribe_kwargs["no_speech_threshold"] = float(os.getenv("WHISPER_NO_SPEECH_THRESH", "0.45"))
        transcribe_kwargs["compression_ratio_threshold"] = 2.4
        if initial_prompt:
            transcribe_kwargs["initial_prompt"] = initial_prompt

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
            segments_iter, info = raw_result
            segments_list = list(segments_iter)

            # Sensitive VAD recovery: if VAD produced no segments on audible audio, retry without VAD
            if not segments_list and transcribe_kwargs.get("vad_filter"):
                has_audible_energy = False
                if isinstance(chunk_path, np.ndarray) and len(chunk_path) > 0:
                    arr_rms = float(np.sqrt(np.mean(chunk_path.astype(np.float32) ** 2)))
                    if arr_rms > 1.5:
                        arr_rms /= 32768.0
                    has_audible_energy = arr_rms >= 0.0012
                if has_audible_energy:
                    retry_kwargs = dict(transcribe_kwargs)
                    retry_kwargs.pop("vad_filter", None)
                    retry_kwargs.pop("vad_parameters", None)
                    try:
                        raw_retry = model.transcribe(chunk_path, **retry_kwargs)
                        if isinstance(raw_retry, tuple) and len(raw_retry) == 2:
                            segments_list = list(raw_retry[0])
                    except Exception:
                        pass

            text_parts = []
            segments = []
            for seg in segments_list:
                seg_text = getattr(seg, "text", "").strip()
                if seg_text:
                    text_parts.append(seg_text)
                segments.append({
                    "start": float(getattr(seg, "start", 0.0)),
                    "end": float(getattr(seg, "end", 0.0)),
                    "text": seg_text,
                })
            text = " ".join(text_parts).strip()

            detected_code = getattr(info, "language", resolved_lang or "en")
            prob = float(getattr(info, "language_probability", 1.0))
            lang_analysis = analyze_segment_language(text, whisper_code=detected_code, prob=prob)

            original_text = text
            english_text = text

            # Dynamic dual translation: If non-English detected and task is auto or translate
            needs_translation = (
                lang_analysis["primary_code"] not in ("en",)
                or lang_analysis["is_code_switched"]
                or resolved_task == "translate"
            )

            if needs_translation and text and resolved_task != "transcribe":
                # Translate chunk into English
                tr_kwargs = dict(transcribe_kwargs)
                tr_kwargs["task"] = "translate"
                if resolved_lang:
                    tr_kwargs["language"] = resolved_lang
                try:
                    raw_tr = model.transcribe(chunk_path, **tr_kwargs)
                    if isinstance(raw_tr, tuple) and len(raw_tr) == 2:
                        tr_segs = list(raw_tr[0])
                        tr_txt = " ".join(s.text.strip() for s in tr_segs if s.text.strip()).strip()
                        if tr_txt:
                            english_text = tr_txt
                        # Align segment-level english text if segment count matches
                        if len(tr_segs) == len(segments):
                            for idx, s in enumerate(segments):
                                s["english_text"] = tr_segs[idx].text.strip()
                        elif tr_segs:
                            # Map full translation across segments
                            for s in segments:
                                s["english_text"] = english_text
                except Exception as tr_err:
                    print(f"[!] Translation pass notice: {tr_err}")
                    english_text = original_text

            if return_segments:
                primary_text = english_text if resolved_task == "translate" else original_text
                return {
                    "text": primary_text,
                    "original_text": original_text,
                    "english_text": english_text,
                    "detected_language": lang_analysis["display_name"],
                    "language_code": lang_analysis["primary_code"],
                    "is_code_switched": lang_analysis["is_code_switched"],
                    "segments": segments,
                }
            return english_text if resolved_task == "translate" else original_text

        # 2. OpenAI Whisper / Mock dict format: {"text": "...", "segments": [...]}
        if isinstance(raw_result, dict):
            text = raw_result.get("text", "").strip()
            if return_segments:
                return {
                    "text": text,
                    "original_text": text,
                    "english_text": text,
                    "detected_language": "Auto",
                    "language_code": "en",
                    "is_code_switched": False,
                    "segments": raw_result.get("segments", []),
                }
            return text

        # 3. String format
        text = str(raw_result).strip()
        if return_segments:
            return {
                "text": text,
                "original_text": text,
                "english_text": text,
                "detected_language": "Auto",
                "language_code": "en",
                "is_code_switched": False,
                "segments": [],
            }
        return text

    except Exception as e:
        print(f"[!] Error during Whisper transcription for {chunk_path}: {e}")
        if return_segments:
            return {"text": "", "original_text": "", "english_text": "", "segments": []}
        return ""

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
    language: str = "auto",
    task: str = None,
    return_segments: bool = False,
    initial_prompt: str = None,
    processing_mode: str = None,
) -> str | dict:
    """
    Route one chunk to Whisper or Sarvam depending on the language/mode configuration.
    Supports processing mode selection ('fast', 'balanced', 'high_accuracy').
    """
    cfg = resolve_language_mode(language=language, task=task)
    if cfg["engine"] == "sarvam":
        if isinstance(chunk_path, np.ndarray):
            raise NotImplementedError("Sarvam API requires WAV file path, not in-memory numpy array.")
        text = transcribe_chunk_sarvam(chunk_path)
        return {
            "text": text,
            "original_text": text,
            "english_text": text,
            "detected_language": "Hinglish (Sarvam)",
            "language_code": "hi",
            "segments": [],
        } if return_segments else text

    return transcribe_chunk_whisper(
        chunk_path,
        language=cfg["whisper_code"],
        task=cfg["task"],
        return_segments=return_segments,
        initial_prompt=initial_prompt,
        processing_mode=processing_mode,
    )


def transcribe_all(
    chunks: list,
    language: str = "auto",
    task: str = None,
    return_original: bool = True,
    chunk_duration_seconds: float = 180.0,
    progress_callback = None,
    processing_mode: str = None,
) -> str | dict:
    """
    Transcribe all audio chunks sequentially with dynamic multilingual detection,
    code-switching support, and aligned dual transcripts (verbatim original + English translation).

    Args:
        chunks: List of WAV chunk file paths.
        language: Language name or mode key (defaults to 'auto' for automatic language detection).
        task: Explicit task ('transcribe' or 'translate').
        return_original: If True, returns full dict with English transcript, original transcript, and segments.
        chunk_duration_seconds: Estimated duration per chunk for timestamp offsets.
        progress_callback: Optional callable(current_chunk_idx, total_chunks, message) for UI progress indicators.
        processing_mode: Processing preset ('fast', 'balanced', or 'high_accuracy').

    Returns:
        str if return_original is False;
        dict with {"transcript", "original_transcript", "segments", "display_language", ...} if return_original is True.
    """
    import time
    t_start = time.perf_counter()
    if not chunks:
        print("[!] No audio chunks provided to transcribe.")
        if return_original:
            return {
                "transcript": "",
                "original_transcript": "",
                "segments": [],
                "display_language": "Auto-Detect",
                "is_translated": False,
                "transcription_time": 0.0,
            }
        return ""

    if isinstance(chunks, str):
        chunks = [chunks]

    cfg = resolve_language_mode(language=language, task=task)
    active_mode = (processing_mode or get_processing_mode()).strip().lower()

    if cfg["engine"] == "sarvam":
        print(f"Using Sarvam AI for transcription ({cfg['display_name']}).")
    else:
        lang_label = cfg["whisper_code"] if cfg["whisper_code"] else "auto"
        print(
            f"Using Whisper for {cfg['display_name']} "
            f"(code: '{lang_label}', task: '{cfg['task']}', mode: '{active_mode}')."
        )

    full_english = []
    full_original = []
    all_segments = []
    detected_languages = []
    any_translated = False

    total_chunks = len(chunks)
    for i, chunk in enumerate(chunks):
        t_chunk_start = time.perf_counter()
        msg = f"Transcribing chunk {i + 1}/{total_chunks} ({active_mode} mode)..."
        print(f"[*] {msg}")
        if progress_callback:
            try:
                progress_callback(i + 1, total_chunks, msg)
            except Exception:
                pass

        chunk_offset = i * chunk_duration_seconds

        res = transcribe_chunk(
            chunk,
            language=language,
            task=task,
            return_segments=True,
            processing_mode=active_mode,
        )

        if isinstance(res, dict):
            orig_txt = res.get("original_text", res.get("text", "")).strip()
            eng_txt = res.get("english_text", res.get("text", "")).strip()
            det_lang = res.get("detected_language", cfg["display_name"])
            segs = res.get("segments", [])
            if res.get("is_code_switched") or (orig_txt != eng_txt):
                any_translated = True
        else:
            orig_txt = str(res).strip()
            eng_txt = orig_txt
            det_lang = cfg["display_name"]
            segs = []

        if det_lang and det_lang not in detected_languages:
            detected_languages.append(det_lang)

        if orig_txt:
            full_original.append(orig_txt)
        if eng_txt:
            full_english.append(eng_txt)

        for seg in segs:
            s_start = chunk_offset + seg.get("start", 0.0)
            s_end = chunk_offset + seg.get("end", 0.0)
            s_orig = seg.get("text", "").strip()
            s_eng = seg.get("english_text", s_orig).strip()
            all_segments.append({
                "start": s_start,
                "end": s_end,
                "text": s_orig,
                "english_text": s_eng,
                "language": det_lang,
            })

        chunk_elapsed = time.perf_counter() - t_chunk_start
        print(f"[*] Chunk {i + 1}/{total_chunks} completed in {chunk_elapsed:.2f}s ({det_lang})")

    total_elapsed = time.perf_counter() - t_start
    print(f"[+] Transcription complete in {total_elapsed:.2f}s across {total_chunks} chunk(s).")

    english_transcript = " ".join(full_english).strip()
    original_transcript = " ".join(full_original).strip()

    overall_lang = ", ".join(detected_languages) if detected_languages else cfg["display_name"]

    if return_original:
        return {
            "transcript": english_transcript or original_transcript,
            "original_transcript": original_transcript,
            "segments": all_segments,
            "display_language": overall_lang,
            "detected_languages": detected_languages,
            "is_translated": any_translated or cfg["task"] == "translate",
            "transcription_time": total_elapsed,
            "processing_mode": active_mode,
        }

    return english_transcript or original_transcript


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