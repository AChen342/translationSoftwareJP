import whisper
from pydub import AudioSegment

#audioFile = "samples/asmr.wav"

model = whisper.load_model("medium")
# result = model.transcribe(audioFile, language="ja", task="transcribe")

#print(result["text"])