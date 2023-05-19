import openai
from collections import deque
import soundfile as sf
import sounddevice as sd
import queue
import numpy as np
import os
import webrtcvad
import whisper

openai.api_key = os.environ["OPENAI_API_KEY"]

class WhisperTranslator():
    def __init__(self):
        self.filename = "data/temp_recordings/output.wav"
        if not os.path.exists(os.path.dirname(self.filename)):
            os.makedirs(os.path.dirname(self.filename))

        self.sample_rate = 48000
        self.channels = 1
        self.device = 0

        # We record 30 ms of audio at a time
        self.block_duration = 30 
        self.blocksize = int(self.sample_rate * self.block_duration / 1000)

        # We process 50 chunks of 30 ms each to determine if someone is talking
        self.speech_chunk_size = 50

        # We record at least 4.5 seconds of audio at the beginning
        self.minimum_recorded_time = 150 # (150 * 30 ms = 4.5 seconds)

        # If the rolling mean of the speech queue is below this threshold, we stop recording
        self.silence_threshold = .15

        # Queue to store speech chunks
        self.speech_queue = deque(maxlen = int(self.speech_chunk_size))
        self.q = queue.Queue()

        # Voice Activity Detection
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(3)

    def record(self):
        def callback(indata, frames, time, status):
            self.q.put(indata.copy())

        print('Starting Recording')
        iters = 0
        with sf.SoundFile(self.filename, mode='w', samplerate=self.sample_rate, channels = self.channels) as f:
            with sd.InputStream(samplerate=self.sample_rate, channels=self.channels, device=self.device, blocksize=self.blocksize, callback=callback):
                while True:
                    data = self.q.get()
                    data = np.array(data * 32767, dtype=np.int16)

                    # Check if someone is talking in the chunk (Voice Activity Detection)
                    is_speech = self.vad.is_speech(data.tobytes(), self.sample_rate)
                    self.speech_queue.append(is_speech)
                    rolling_mean = sum(self.speech_queue) / self.speech_chunk_size

                    if iters > self.minimum_recorded_time:
                        if rolling_mean < self.silence_threshold:
                            print('Recording Ended - no voice activity in 1.5 seconds')
                            break
                    
                    if iters > 300:
                        break
                    
                    f.write(data)
                    iters += 1
 
    def translate(self):
        transcript = 'default'
        try:
            with open(self.filename, 'rb') as f:
                result = openai.Audio.transcribe('whisper-1', f)
                transcript = result["text"]
                raise
        except Exception as e_cloud:
            print('Error occured while inferencing Whisper from OpenAI CLOUD client: \n', e_cloud)

            try:
                whisper_model = whisper.load_model("base", device="cuda")
                audio = whisper.load_audio(self.filename)
                audio = whisper.pad_or_trim(audio)

                # make log-Mel spectrogram and move to the same device as the model
                mel = whisper.log_mel_spectrogram(audio).to(whisper_model.device)

                # detect the spoken language
                _, probs = whisper_model.detect_language(mel)
                print(f"Detected language: {max(probs, key=probs.get)}")

                # decode the audio
                options = whisper.DecodingOptions()
                result = whisper.decode(whisper_model, mel, options)

                # get the transcript out of whisper's decoded result
                transcript = result.text
            except Exception as e_local:
                print('Error occured while inferencing Whisper from OpenAI LOCAL client: \n', e_local)
        return transcript

if __name__ == "__main__":
    wt = WhisperTranslator()
    wt.record()
    print("Done recording")
    translation = wt.translate()
    print(translation)
