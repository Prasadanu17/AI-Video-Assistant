import os
import time
import whisper
import torch

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")  # tiny, base, small

_model = None


def load_model():
    global _model

    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        print("=" * 50)
        print(f"Loading Whisper model : {WHISPER_MODEL}")
        print(f"Running on           : {device}")
        print("=" * 50)

        _model = whisper.load_model(WHISPER_MODEL, device=device)

        print("Whisper model loaded successfully.\n")

    return _model


def transcribe_chunk(chunk_path: str, translate: bool = False):
    model = load_model()

    task = "translate" if translate else "transcribe"

    print(f"Processing: {os.path.basename(chunk_path)}")

    start = time.time()

    result = model.transcribe(
        chunk_path,
        task=task,
        language="hi",
        fp16=torch.cuda.is_available(),
        verbose=True,
    )

    end = time.time()

    print(f"Finished in {end-start:.2f} seconds\n")

    return result["text"]


def transcribe_all(chunks, translate=False):
    transcript = []

    total = len(chunks)

    print(f"\nTotal Chunks : {total}\n")

    for i, chunk in enumerate(chunks, start=1):
        print(f"========== Chunk {i}/{total} ==========")

        text = transcribe_chunk(chunk, translate)

        transcript.append(text)

    print("\nTranscription Completed Successfully!")

    return "\n\n".join(transcript)