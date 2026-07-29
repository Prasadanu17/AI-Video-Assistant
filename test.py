from utils.audio_precessor import process_input
from core.transcriber import transcribe_all

source = "https://youtu.be/dWQTiqjdaAs?si=qEX-MCAHNIlMQ6ij"

chunks = process_input(source)

print(transcribe_all(chunks))