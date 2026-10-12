import os
import shutil
import yt_dlp
from pydub import AudioSegment


# Folder for downloaded audio files
DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def download_youtube_audio(url: str) -> str:
    """
    Download audio from a YouTube URL and convert it to WAV.

    Browser cookies are disabled by default.
    Optional environment variables:
        YTDLP_COOKIE_FILE
        YTDLP_COOKIES_FROM_BROWSER
    """

    output_path = os.path.join(
        DOWNLOAD_DIR,
        "%(title)s_%(id)s.%(ext)s"
    )

    ydl_opts = {
        # Download the best available audio
        "format": "bestaudio/best",

        # Save files in the downloads folder with unique ID
        "outtmpl": output_path,

        # Do not download playlists
        "noplaylist": True,

        # Display download progress
        "quiet": False,

        # Retry on network failures
        "retries": 10,
        "fragment_retries": 10,

        # Use mobile clients to circumvent 403 Forbidden errors
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "web", "ios"]
            }
        },

        # Convert downloaded audio to WAV
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],
    }

    # Use cookies only when explicitly configured.
    cookie_file = os.getenv("YTDLP_COOKIE_FILE", "").strip()
    cookie_browser = os.getenv("YTDLP_COOKIES_FROM_BROWSER", "").strip()

    if cookie_file:
        ydl_opts["cookiefile"] = os.path.expandvars(os.path.expanduser(cookie_file))
    elif cookie_browser:
        ydl_opts["cookiesfrombrowser"] = (cookie_browser,)

    # Configure an available JavaScript runtime if present
    for runtime in ("deno", "node"):
        if shutil.which(runtime):
            ydl_opts["js_runtimes"] = {runtime: {}}
            break

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        # Check if the WAV already exists locally before downloading
        try:
            info = ydl.extract_info(url, download=False)
            filename = ydl.prepare_filename(info)
            base, _ = os.path.splitext(filename)
            wav_path = base + ".wav"

            if os.path.exists(wav_path) and os.path.getsize(wav_path) > 1024:
                print(f"\n[i] Found existing downloaded WAV: {wav_path}. Skipping download.\n")
                return wav_path
        except Exception:
            # If metadata check fails, proceed to normal download attempt
            pass

        # Perform download
        try:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            base, _ = os.path.splitext(filename)
            wav_path = base + ".wav"
        except Exception as e:
            raise RuntimeError(f"Failed to download audio from YouTube URL '{url}': {e}")

        if not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
            raise FileNotFoundError(f"WAV file not found or is empty after download: {wav_path}")

        print("\nDownload successful!")
        print(f"Saved to: {wav_path}\n")
        return wav_path


def convert_to_wav(input_path: str, force_reconvert: bool = False) -> str:
    """
    Convert a local audio or video file to 16 kHz mono WAV.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if os.path.getsize(input_path) == 0:
        raise ValueError(f"Supplied input file is empty (0 bytes): {input_path}")

    output_path = os.path.splitext(input_path)[0] + "_converted.wav"

    if not force_reconvert and os.path.exists(output_path) and os.path.getsize(output_path) > 1024:
        print(f"[i] Using existing converted audio: {output_path}")
        return output_path

    try:
        audio = AudioSegment.from_file(input_path)
    except Exception as e:
        raise ValueError(
            f"Unsupported or corrupted media file '{input_path}'. "
            f"Could not decode audio: {e}"
        )

    if len(audio) == 0:
        raise ValueError(f"Extracted audio from '{input_path}' is empty (0ms duration).")

    # Convert to mono and 16 kHz
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)

    audio.export(output_path, format="wav")
    print(f"Converted audio saved to: {output_path}")
    return output_path


def chunk_audio(wav_path: str, chunk_minutes: int = 3, force_rechunk: bool = False) -> list:
    """
    Split a WAV file into smaller chunks.
    Default chunk duration: 3 minutes.
    """
    if chunk_minutes <= 0:
        raise ValueError("chunk_minutes must be greater than zero.")

    if not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
        raise ValueError(f"WAV audio file is missing or empty: {wav_path}")

    base = os.path.splitext(wav_path)[0]

    # Check if chunks already exist and are valid
    existing_chunk_0 = f"{base}_chunk_0.wav"
    if not force_rechunk and os.path.exists(existing_chunk_0) and os.path.getsize(existing_chunk_0) > 0:
        existing_chunks = []
        i = 0
        while True:
            chunk_file = f"{base}_chunk_{i}.wav"
            if os.path.exists(chunk_file) and os.path.getsize(chunk_file) > 0:
                existing_chunks.append(chunk_file)
                i += 1
            else:
                break
        if existing_chunks:
            print(f"[i] Found {len(existing_chunks)} existing audio chunk(s). Reusing.")
            return existing_chunks

    try:
        audio = AudioSegment.from_wav(wav_path)
    except Exception as e:
        raise ValueError(f"Failed to read WAV file '{wav_path}': {e}")

    if len(audio) == 0:
        raise ValueError(f"Audio file '{wav_path}' contains no audio data (0ms duration).")

    chunk_ms = chunk_minutes * 60 * 1000
    chunks = []

    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start:start + chunk_ms]
        chunk_path = f"{base}_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)
        print(f"Created audio chunk: {chunk_path}")

    return chunks


def process_input(source: str, force_reprocess: bool = False) -> list:
    """
    Process a YouTube URL or a local audio/video file.
    Returns a list of WAV audio chunk paths in sequential order.
    """
    if source.startswith(("http://", "https://")):
        print("Detected YouTube URL. Downloading...")
        wav_path = download_youtube_audio(source)
    else:
        if not os.path.isfile(source):
            raise FileNotFoundError(f"Input file not found: {source}")

        print("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source, force_reconvert=force_reprocess)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path, force_rechunk=force_reprocess)
    print(f"Audio ready - {len(chunks)} chunk(s) created.")
    return chunks
