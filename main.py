import re
import os
from pydub import AudioSegment
import whisper
import json

# 1. Ask user for audio file name
# 2. If not .wav convert to correct format
# 3. Transcribe audio
# 4. Save transcription to audio file

# transcribes audio
def speechToText(audioFile):
    model = whisper.load_model("medium")
    result = model.transcribe(audioFile, language="en", task="transcribe", verbose=False)

    # saves segments into json file to be translated later on
    with open("segments.json", "w", encoding="utf-8") as f:
        json.dump(result["segments"], f, ensure_ascii=False, indent=2)

    # Load raw Whisper result
    with open("segments.json", "r", encoding="utf-8") as f:
        raw_segments = json.load(f)

    # Simplify
    clean_segments = [
        {
            "id": seg["id"],
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip()
        }
        for seg in raw_segments
    ]

    # Save clean JSON
    with open("segments.json", "w", encoding="utf-8") as f:
        json.dump(clean_segments, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(clean_segments)} cleaned segments.")


# function used to convert audio files to .wav
def convertToWav(audioFile):
    audioFileName = audioFile.split(".")[0]

    source = AudioSegment.from_file("samples/" + audioFile)
    source = source.set_channels(1).set_frame_rate(16000)

    newAudioFileName = audioFileName + ".wav"
    source.export("samples/" + newAudioFileName, format="wav")

    return newAudioFileName

def main():
    invalidFileName = True
    while invalidFileName:
        print("Enter full name of audio file (extension included): ")
        audioFile = input()

        # search for audio file in samples folder
        for root, dirs, files in os.walk("samples/"):
            if audioFile in files:
                print("Audio file located.")
                invalidFileName = False

    # using regex to verify file extension
    validExtension = ".wav"

    # convert audio file to .wav
    if re.search(validExtension, audioFile) == None:
        audioFile = convertToWav(audioFile)
        print("Conversion completed.")

    # transcribe audio file
    speechToText("samples/" + audioFile)
    print("Audio transcribed.")


if __name__== "__main__":
    main()