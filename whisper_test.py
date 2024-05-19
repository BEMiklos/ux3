#!/usr/bin/env python3
import whisper
from openai import OpenAI
import os
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import soundfile as sf
from nemo.collections.tts.models import FastPitchModel
from nemo.collections.tts.models import HifiGanModel

# Parameters
Model = 'medium'     # Whisper model size (tiny, base, small, medium, large)
English = True      # Use English-only model?
Translate = False   # Translate non-English to English?
SampleRate = 44100  # Stream device recording frequency
BlockSize = 30      # Block size in milliseconds
Threshold = 0.1     # Minimum volume threshold to activate listening
Vocals = [50, 1000] # Frequency range to detect sounds that could be speech
EndBlocks = 40      # Number of blocks to wait before sending to Whisper
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

class StreamHandler:
    def __init__(self):
        self.running = True
        self.padding = 0
        self.prevblock = self.buffer = np.zeros((0, 1))
        self.fileready = False
        self.model = whisper.load_model(f'{Model}{".en" if English else ""}')
        self.spec_generator = FastPitchModel.from_pretrained("tts_en_fastpitch").eval()
        self.vocoder = HifiGanModel.from_pretrained(model_name="tts_en_hifigan").eval()

    def callback(self, indata, frames, time, status):
        if not any(indata):
            return
        freq = np.argmax(np.abs(np.fft.rfft(indata[:, 0]))) * SampleRate / frames
        if np.sqrt(np.mean(indata**2)) > Threshold and Vocals[0] <= freq <= Vocals[1]:
            if self.padding < 1:
                self.buffer = self.prevblock.copy()
            self.buffer = np.concatenate((self.buffer, indata))
            self.padding = EndBlocks
        else:
            self.padding -= 1
            if self.padding > 1:
                self.buffer = np.concatenate((self.buffer, indata))
            elif self.padding < 1 < self.buffer.shape[0] > SampleRate:
                self.fileready = True
                write('input_audio.wav', SampleRate, self.buffer)
                self.buffer = np.zeros((0, 1))
            elif self.padding < 1 < self.buffer.shape[0] < SampleRate:
                self.buffer = np.zeros((0, 1))

    def process(self):
        if self.fileready:
            result = self.model.transcribe('input_audio.wav', fp16=False, language='en' if English else '', task='translate' if Translate else 'transcribe')
            print(f"Transcription: {result['text']}")
            self.generate_response(result['text'])
            self.fileready = False

    def generate_response(self, text):
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpfull AI assistant, answer shortly in 1 or 2 sentences in the style of the Wikipedia."},
                {"role": "user", "content": f"Answere this: {text}"},
            ]
            )
        response = completion.choices[0].message.content
        print(f"Response: {response}")
        parsed = self.spec_generator.parse(response)
        spectrogram = self.spec_generator.generate_spectrogram(tokens=parsed)
        audio = self.vocoder.convert_spectrogram_to_audio(spec=spectrogram)
        mono_audio = audio.mean(dim=0)  # Convert stereo to mono
        sf.write(file='response.wav', data=mono_audio.to('cpu').detach().numpy(), samplerate=22050)
        sd.play(data=mono_audio.to('cpu').detach().numpy(), samplerate=22050)

    def listen(self):
        print("Listening.. (Ctrl+C to Quit)")
        with sd.InputStream(channels=1, callback=self.callback, blocksize=int(SampleRate * BlockSize / 1000), samplerate=SampleRate):
            while self.running:
                self.process()

def main():
    try:
        handler = StreamHandler()
        handler.listen()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        print("Quitting..")

if __name__ == '__main__':
    main()
