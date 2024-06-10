# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import queue
from collections import deque

import numpy as np
import openai
import sounddevice as sd
import soundfile as sf
import webrtcvad
import whisper


class WhisperTranslator:
    def __init__(self):
        print("\n=====================================")
        print("Initializing Whisper Translator")
        self.filename = "data/temp_recordings/output.wav"
        if not os.path.exists(os.path.dirname(self.filename)):
            os.makedirs(os.path.dirname(self.filename))

        self.sample_rate = 48000
        self.channels = 1
        self.device = self.identify_device("USB Microphone")

        # We record 30 ms of audio at a time
        self.block_duration = 30
        self.blocksize = int(self.sample_rate * self.block_duration / 1000)

        # We process 50 chunks of 30 ms each to determine if someone is talking
        self.speech_chunk_size = 50

        # We record at least 4.5 seconds of audio at the beginning
        self.minimum_recorded_time = 150  # (150 * 30 ms = 4.5 seconds)

        # If the rolling mean of the speech queue is below this threshold, we stop recording
        self.silence_threshold = 0.15

        # Queue to store speech chunks
        self.speech_queue = deque(maxlen=int(self.speech_chunk_size))
        self.q = queue.Queue()

        # Voice Activity Detection
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(3)
        self.whisper = whisper.load_model("base", device="cuda:1", in_memory=True)
        print("=====================================\n")

    def record(self):
        """
        Records audio from the microphone, translates it to text and returns the text
        """

        def callback(indata, frames, time, status):
            self.q.put(indata.copy())

        print("Starting Recording")
        iters = 0
        with sf.SoundFile(
            self.filename, mode="w", samplerate=self.sample_rate, channels=self.channels
        ) as f:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                device=self.device,
                blocksize=self.blocksize,
                callback=callback,
            ):
                while True:
                    data = self.q.get()
                    data = np.array(data * 32767, dtype=np.int16)

                    # Check if someone is talking in the chunk (Voice Activity Detection)
                    is_speech = self.vad.is_speech(data.tobytes(), self.sample_rate)
                    self.speech_queue.append(is_speech)
                    rolling_mean = sum(self.speech_queue) / self.speech_chunk_size

                    if iters > self.minimum_recorded_time:
                        if rolling_mean < self.silence_threshold:
                            print("Recording Ended - no voice activity in 1.5 seconds")
                            break

                    f.write(data)
                    iters += 1
        print("Done Recording")

    def translate(self, online=False):
        """
        Translates the audio to text using Whisper first from OPENAI CLOUD client and if it fails, then from locally downloaded model
        """
        transcript = "default"
        if online:
            openai.api_key = os.environ["OPENAI_API_KEY"]
            print("online whisper model")
            try:
                with open(self.filename, "rb") as f:
                    result = openai.Audio.transcribe("whisper-1", f)
                    transcript = result["text"]
            except Exception as e_cloud:
                print(
                    "Error occured while inferencing Whisper from OpenAI CLOUD client: \n",
                    e_cloud,
                )
        else:
            print("offline whisper model")
            try:
                audio = whisper.load_audio(self.filename)
                audio = whisper.pad_or_trim(audio)

                # make log-Mel spectrogram and move to the same device as the model
                mel = whisper.log_mel_spectrogram(audio).to(self.whisper.device)

                # detect the spoken language
                _, probs = self.whisper.detect_language(mel)
                print(f"Detected language: {max(probs, key=probs.get)}")

                # decode the audio
                options = whisper.DecodingOptions()
                result = whisper.decode(self.whisper, mel, options)

                # get the transcript out of whisper's decoded result
                transcript = result.text
            except Exception as e_local:
                print(
                    "Error occured while inferencing Whisper from OpenAI LOCAL client: \n",
                    e_local,
                )
        return transcript

    def identify_device(self, device_name="USB Microphone"):
        """
        Identify the device number of the USB Microphone and returns it
        """
        device_list = sd.query_devices()
        devices = [
            (i, x["name"])
            for i, x in enumerate(device_list)
            if device_name in x["name"]
        ]
        if len(devices) == 0:
            print("USB Microphone not found. Using default device")
            device_id = 0
        else:
            print("Found following devices with name USB Microphone:\n", devices)
            if len(devices) > 1:
                print("Using first device from the list")
            device_id = devices[0][0]
        return device_id


if __name__ == "__main__":
    wt = WhisperTranslator()
    wt.record()
    translation = wt.translate()
    print(translation)
