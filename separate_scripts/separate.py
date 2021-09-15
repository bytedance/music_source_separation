import time
import soundfile
import librosa
from bytesep.inference import SeparatorWrapper

sample_rate = 44100		# Must be 44100 when using the downloaded checkpoints.
device = "cuda"		# "cuda" | "cpu"
source_type = "accompaniment"	# "vocals" | "accompaniment"

# Load audio.
audio_path = "resources/vocals_accompaniment_10s.mp3"
audio, fs = librosa.load(audio_path, sr=sample_rate, mono=False)

assert audio.ndim == 2

# separator
separator = SeparatorWrapper(
	source_type=source_type, 
	model=None, 
	checkpoint_path=None, 
	device='cpu'
)

t1 = time.time()

# Separate.
sep_wav = separator.separate(audio)

sep_time = time.time() - t1

# Write out audio
sep_audio_path = 'sep_{}.wav'.format(source_type)


soundfile.write(file=sep_audio_path, data=sep_wav.T, samplerate=sample_rate)

print("Write out to {}".format(sep_audio_path))
print("Time: {:.3f}".format(sep_time))
