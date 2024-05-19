#!/usr/bin/env python3

import re
import os
import whisper
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import soundfile as sf
from nemo.collections.tts.models import FastPitchModel, HifiGanModel
from openai import OpenAI
from transformers import pipeline

# Parameters
MODEL = 'medium'
ENGLISH_ONLY = True
TRANSLATE = False
SAMPLE_RATE = 44100
BLOCK_SIZE = 30
VOLUME_THRESHOLD = 0.1
VOCAL_RANGE = [50, 1000]
END_BLOCKS = 40
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Load pre-trained sentiment analysis pipeline
sentiment_pipeline = pipeline('sentiment-analysis')

class StreamHandler:
    def __init__(self):
        self.running = True
        self.padding = 0
        self.prevblock = np.zeros((0, 1))
        self.buffer = np.zeros((0, 1))
        self.fileready = False
        self.model = whisper.load_model(f'{MODEL}{".en" if ENGLISH_ONLY else ""}')
        self.spec_generator = FastPitchModel.from_pretrained("tts_en_fastpitch").eval()
        self.vocoder = HifiGanModel.from_pretrained(model_name="tts_en_hifigan").eval()

    def callback(self, indata, frames, time, status):
        if not any(indata):
            return

        freq = np.argmax(np.abs(np.fft.rfft(indata[:, 0]))) * SAMPLE_RATE / frames

        if np.sqrt(np.mean(indata**2)) > VOLUME_THRESHOLD and VOCAL_RANGE[0] <= freq <= VOCAL_RANGE[1]:
            if self.padding < 1:
                self.buffer = self.prevblock.copy()
            self.buffer = np.concatenate((self.buffer, indata))
            self.padding = END_BLOCKS
        else:
            self.padding -= 1
            if self.padding > 1:
                self.buffer = np.concatenate((self.buffer, indata))
            elif self.padding < 1 and self.buffer.shape[0] > SAMPLE_RATE:
                self.fileready = True
                write('input_audio.wav', SAMPLE_RATE, self.buffer)
                self.buffer = np.zeros((0, 1))
            elif self.padding < 1 and self.buffer.shape[0] < SAMPLE_RATE:
                self.buffer = np.zeros((0, 1))

        self.prevblock = indata.copy()

    def process(self):
        if self.fileready:
            result = self.model.transcribe('input_audio.wav', fp16=False, language='en' if ENGLISH_ONLY else '', task='translate' if TRANSLATE else 'transcribe')
            print(f"Transcription: {result['text']}")
            sentiment, score = self.detect_sentiment(result['text'])
            print(f"Sentiment: {sentiment}, Confidence: {score}")
            self.generate_response(result['text'])
            self.fileready = False

    def detect_sentiment(self, text):
        result = sentiment_pipeline(text)[0]
        return result['label'], result['score']

    def extract_appointment_details(self, text):
        match = re.search(r'Appointment to the doctor: (.*?), further message: (.*)', text)
        if match:
            return match.group(1), match.group(2)
        return None, None

    def generate_response(self, text):
        try:
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an assistant that schedules appointments with a doctor. If you find a date, respond exactly with 'Appointment to the doctor: {date}, further message: {message}'."},
                    {"role": "user", "content": f"Schedule an appointment with the doctor. {text}"},
                ]
            )
            response = completion.choices[0].message.content
            print(f"Response: {response}")

            date, message = self.extract_appointment_details(response)
            if (date != None) & (message != None):
                print(f"Appointment confirmed: Date: {date}, Message: {message}")
                self.running = False  # End the conversation

            parsed = self.spec_generator.parse(response)
            spectrogram = self.spec_generator.generate_spectrogram(tokens=parsed)
            audio = self.vocoder.convert_spectrogram_to_audio(spec=spectrogram)
            mono_audio = audio.mean(dim=0)  # Convert stereo to mono
            sf.write(file='response.wav', data=mono_audio.to('cpu').detach().numpy(), samplerate=22050)
            sd.play(data=mono_audio.to('cpu').detach().numpy(), samplerate=22050)
        except Exception as e:
            print(f"Error in generating response: {e}")

    def listen(self):
        print("Listening.. (Ctrl+C to Quit)")
        with sd.InputStream(channels=1, callback=self.callback, blocksize=int(SAMPLE_RATE * BLOCK_SIZE / 1000), samplerate=SAMPLE_RATE):
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
