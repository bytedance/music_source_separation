import soundfile
import torch
import librosa
import augment
import time

audio, fs = librosa.load('resources/vocals_accompaniment_10s.mp3', sr=44100, mono=True)

t1 = time.time()
# y = librosa.effects.pitch_shift(y=audio, sr=44100, n_steps=5, bins_per_octave=12, res_type='linear')
y = librosa.resample(
    y=audio,
    orig_sr=44100,
    target_sr=32000,
    res_type='linear',
    axis=-1,
)
print(time.time() - t1)
soundfile.write(file='_zz.wav', data=y, samplerate=44100)

# a1 = torch.Tensor(audio)

# import numpy as np
# # random_pitch_shift = lambda: np.random.randint(-100, +100)
# # the pitch will be changed by a shift somewhere between (-100, +100)


# # input signal properties
# src_info = {'rate': 44100}

# # output signal properties
# target_info = {'channels': 1,
#                'length': 0, # not known beforehand
#                'rate': 44100}

# y = augment.EffectChain().pitch(700).rate(44100).apply(a1, src_info=src_info, target_info=target_info)

# soundfile.write(file='_zz.wav', data=y.T, samplerate=44100)

# from IPython import embed; embed(using=False); os._exit(0)
