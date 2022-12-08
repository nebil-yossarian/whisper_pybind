import build.whisperbind as whisperbind
import sys
from scipy.io import wavfile
import soundfile as sf
import tcod.sdl.audio
import numpy as np

class Whisper():

    def __init__(self, model = None, params = None, print_ = False):
        if model is None:
            model_path = '/mnt/c/Users/nebilibrahim/Documents/Python Scripts/whisper_pybind/whisper.cpp/models/ggml-base.en.bin'
        else:
            model_path = f'/mnt/c/Users/nebilibrahim/Documents/Python Scripts/whisper_pybind/whisper.cpp/models/ggml-{model}.en.bin'


        self.params = whisperbind.whisper_full_default_params(whisperbind.WHISPER_SAMPLING_GREEDY)
        self.params.print_progress = bool(print_)
        print(dir(self.params))
        
        # print("Before", self.context.buf_memory())
        self.context = whisperbind.whisper_init(model_path)

        if not self.context:
            print("Context is null")
            sys.exit(0)
        print(f"Context: {self.context}")
        

    def full(self, samples):
        samples_input = samples
        if samples_input.ndim == 2:
            samples_input = samples_input[:,0]
        n_samples = len(samples_input)
        for i in range(30):
            print(samples_input[i])
        return whisperbind.whisper_full(self.context, self.params, samples_input, n_samples)

    def __del__(self):
        if self.params:
            del self.params
        if self.context:
            del self.context
        return True

def normalize(data):
    return data.astype('float32') / 32768.0
        

audio_path = "/mnt/c/Users/nebilibrahim/Documents/Python Scripts/Whisper_Testing/whisper/tests/jfk.wav"
sr, audio = wavfile.read(audio_path)
audio = normalize(audio)
# device = tcod.sdl.audio.open() 
# print("Audio", audio_old.shape)
# audio = device.convert(audio_old, sr)
# print("Audio", audio.shape, sr)
# print("Audios equal", audio == audio_old)
# sf.write(audio_path.rstrip("flac") + "wav", audio, sr, 'PCM_16')
# audio_path = "/mnt/c/Users/nebilibrahim/Documents/Python Scripts/Whisper_Testing/whisper/tests/jfk.wav"
# sample_rate, data = wavfile.read(audio_path)
# print(data, data.shape)
x = Whisper(params = 'default', print_ = True)
print("SpeedUp: ", x.params.speed_up)
# print(audio.T[0,:][].shape)
t = x.full(audio)
print("Hello World!")
# sys.exit(0
# print("Hello", dict(x.context))
# print("Result:", t)