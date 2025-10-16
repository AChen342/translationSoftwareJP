import whisper
from pydub import AudioSegment

audioFile = "samples/neurosama.wav"

model = whisper.load_model("base")
result = model.transcribe(audioFile)

print(result["text"])