import argparse
import time

import librosa
import soundfile

from bytesep.inference import SeparatorWrapper

sample_rate = 44100  # Must be 44100 when using the downloaded checkpoints.


def separate(args):

    audio_path = args.audio_path
    source_type = args.source_type
    device = "cuda"  # "cuda" | "cpu"

    # Load audio.
    audio, fs = librosa.load(audio_path, sr=sample_rate, mono=False)

    if audio.ndim == 1:
        audio = audio[None, :]
        # (2, segment_samples)

    # separator
    separator = SeparatorWrapper(
        source_type=source_type,
        model=None,
        checkpoint_path=None,
        device=device,
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--audio_path',
        type=str,
        default="resources/vocals_accompaniment_10s.mp3",
        help="Audio path",
    )
    parser.add_argument(
        '--source_type',
        type=str,
        choices=['vocals', 'accompaniment'],
        default="accompaniment",
        help="Source type to be separated.",
    )

    args = parser.parse_args()

    separate(args)
