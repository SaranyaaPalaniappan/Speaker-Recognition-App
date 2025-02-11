import tkinter
import tkinter.messagebox
import customtkinter
import joblib
import librosa
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import speech_recognition as Sr
import os
import webbrowser
import wave
from dataclasses import dataclass, asdict
import pyaudio
from pydub import AudioSegment
from pydub.silence import split_on_silence
from textblob import TextBlob

customtkinter.set_appearance_mode("System") 
customtkinter.set_default_color_theme("green")

@dataclass
class StreamParams:
    format: int = pyaudio.paInt16
    channels: int = 2
    rate: int = 44100
    frames_per_buffer: int = 1024
    input: bool = True
    output: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

class Recorder:
    def __init__(self, stream_params: StreamParams) -> None:
        self.stream_params = stream_params
        self._pyaudio = None
        frames_per_buffer: int = 1024
        self._stream = None
        self._wav_file = None

    def record(self, duration: int, save_path: str) -> None:
        print("Start recording...")
        self._create_recording_resources(save_path)
        self._write_wav_file_reading_from_stream(duration)
        self._close_recording_resources()
        print("Stop recording")

    def _create_recording_resources(self, save_path: str) -> None:
        self._pyaudio = pyaudio.PyAudio()
        self._stream = self._pyaudio.open(**self.stream_params.to_dict())
        self._create_wav_file(save_path)

    def _create_wav_file(self, save_path: str):
        self._wav_file = wave.open(save_path, "wb")
        self._wav_file.setnchannels(self.stream_params.channels)
        self._wav_file.setsampwidth(self._pyaudio.get_sample_size(self.stream_params.format))
        self._wav_file.setframerate(self.stream_params.rate)

    def _write_wav_file_reading_from_stream(self, duration: int) -> None:
        for _ in range(int(self.stream_params.rate * duration / self.stream_params.frames_per_buffer)):
            audio_data = self._stream.read(self.stream_params.frames_per_buffer)
            self._wav_file.writeframes(audio_data)

    def _close_recording_resources(self) -> None:
        self._wav_file.close()
        self._stream.close()
        self._pyaudio.terminate()

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.title("Voice Vault")
        self.geometry(f"{1100}x{580}")

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)
        self.grid_columnconfigure(3, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=1)

        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)

        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="VoiceVault", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.sidebar_button_1 = customtkinter.CTkButton(self.sidebar_frame, text='Get started!', command=self.start_identification)
        self.sidebar_button_1.grid(row=1, column=0, padx=20, pady=10)

        self.sidebar_button_3 = customtkinter.CTkButton(self.sidebar_frame, text='History', command=self.open_input_dialog_event)
        self.sidebar_button_3.grid(row=3, column=0, padx=20, pady=10)

        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 10))
        
        self.scaling_label = customtkinter.CTkLabel(self.sidebar_frame, text="UI Scaling:", anchor="w")
        self.scaling_label.grid(row=7, column=0, padx=20, pady=(10, 0))
        
        self.scaling_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["80%", "90%", "100%", "110%", "120%"],
                                                               command=self.change_scaling_event)
        self.scaling_optionemenu.grid(row=8, column=0, padx=20, pady=(10, 20))

        self.textbox = customtkinter.CTkTextbox(self, width=700, height=500)
        self.textbox.grid(row=0, column=1, padx=(20, 0), pady=(20, 0), sticky="nsew")
        
        # Centering the text in the textbox
        self.textbox.insert("10.0", "Welcome to VoiceVault!\n\n" +
                                    "An application to identify the speaker!\n\n")


    # Add more customization and widgets here
        self.setup_ui()

    def setup_ui(self):

        # Example of adding a button in the main area
        self.button = customtkinter.CTkButton(self, text="Start Speaker Identification", command=self.start_identification)
        self.button.grid(row=1, column=1, columnspan=2, padx=10, pady=10, sticky="nsew")

    def open_input_dialog_event(self):
        with open(r'C:\Users\Saranyaa P\OneDrive\Desktop\history.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                self.textbox.insert("10.0", line.rstrip())
                self.textbox.insert("10.0", '\n')

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)

    def start_identification(self):
        # Record audio
        stream_params = StreamParams()
        recorder = Recorder(stream_params)
        recorder.record(5, "audio.wav")

        file_path = r'C:\Users\Saranyaa P\Downloads\audio.wav'
        file_name = file_path.split('/')[-1]
        audio_format = "wav"

        gmm_speaker1 = joblib.load(r'C:\Users\Saranyaa P\Downloads\saranyaa (1).pkl')
        gmm_speaker2 = joblib.load(r'C:\Users\Saranyaa P\Downloads\divya (1).pkl')
        gmm_speaker3 = joblib.load(r'C:\Users\Saranyaa P\Downloads\ananth (1).pkl')
        

        y, sr = librosa.load(file_path)
        y, _ = librosa.effects.trim(y)
        df = pd.DataFrame()
        mfc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=512, hop_length=256)
        df = pd.concat([df, pd.DataFrame(np.ndarray.tolist(mfc.T))], ignore_index=True)

        df_filtered = df[(df != 0).all(1)]
        w = StandardScaler()
        W = w.fit_transform(df_filtered)
        W = pd.DataFrame(W)

        s = a = d = 0
        for i in range(len(W)):
            score_speaker1 = gmm_speaker1.score(W.values[i].reshape(1, -1))
            score_speaker2 = gmm_speaker2.score(W.values[i].reshape(1, -1))
            score_speaker3 = gmm_speaker3.score(W.values[i].reshape(1, -1))
            if score_speaker1 > score_speaker2 and score_speaker1 > score_speaker3:
                s += 1
            elif score_speaker2 > score_speaker3 and score_speaker2 > score_speaker1:
                d +=1
            else:
                a +=1

        if s > d and s > a:
            speaker = 'Saranyaa Palaniappan'
        elif d > a and d > s:
            speaker = 'Divya'
        else:
            speaker = 'Ananthakrishnan'
    

        print(s,d,a)

        self.textbox.insert("20.0", f'Hi!!!\n Welcome {speaker}\n\n')
        with open(r'C:\Users\Saranyaa P\OneDrive\Desktop\history.txt', 'a') as f:
            f.write(f'{speaker}\n\n')

if __name__ == "__main__":
    app = App()
    app.mainloop()


        
        

