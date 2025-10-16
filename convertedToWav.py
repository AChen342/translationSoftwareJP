from pydub import AudioSegment

audioFile = "mp3/neurosama.mp3"

# convert from mp3 to wav
source = AudioSegment.from_file(audioFile)
source = source.set_channels(1).set_frame_rate(16000)
source.export("samples/neurosama.wav", format="wav")