import re
import os
from pydub import AudioSegment
import whisper
import json
import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# 1. Ask user for audio file name
# 2. If not .wav convert to correct format
# 3. Transcribe audio
# 4. Save transcription to json file
# 5. Translate transcriptions

# utility function to view transcription with time stamp
def viewTranscription():
    with open("segments.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    for d in data:
        start = d['start']
        end = d['end']
        text = d['text']
        print(f"[{start:.2f}-{end:.2f}]: {text}")
    
def en_jpTranslate():
    # tokenizer breaks down sentences into subwords (tokens)
    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
    model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
    tokenizer.src_lang = "en"

    with open("segments.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    jp_translation = []
    for d in data:
        text = d['text']

        encoded = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            generated_tokens = model.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id("ja"))
    
        translated = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        segment = {
            "id": d['id'],
            "start" : d['start'],
            "end" : d['end'],
            "text" : translated[0]
        }

        jp_translation.append(segment)

    with open("jp_translation.json", "w", encoding="utf-8") as f:
        json.dump(jp_translation, f, ensure_ascii=False, indent=2)
    
    print("Japanese translation completed.")

# transcribes audio
def speechToText(audioFile):
    model = whisper.load_model("medium")
    result = model.transcribe(audioFile, language="en", task="transcribe", verbose=False)

    # first clean segments to have only necessary data
    # id, start time, end time, and text
    clean_segments = [
        {
            "id": seg["id"],
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip()
        }
        for seg in result['segments']
    ]
    print("Segments have been cleaned.")

    # saves segments into json file to be translated later on
    with open("segments.json", "w", encoding="utf-8") as f:
        json.dump(clean_segments, f, ensure_ascii=False, indent=2)
    print("Transcription saved.")
    
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

    #translate
    en_jpTranslate()


if __name__== "__main__":
    main()