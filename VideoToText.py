import csv
import glob
import os
import pickle
import time
from pathlib import Path

import moviepy.editor as mp
import speech_recognition as sr
from google.cloud import speech_v1 as speech


class VideoToText:

    def __init__(
        self,
        audio_format="wav",
        language="de-DE",
        path=Path("Stimuli").absolute().resolve(),
    ):
        self.language = language
        self.audio_format = audio_format
        self.path = path

    def extract_text(
        self,
        path=Path("Stimuli").absolute().resolve(),
    ):
        text_dic = {}
        for video_file_path in glob.glob(os.path.join(path, "*.mpg")):
            file_name = video_file_path.split("\\")[-1]
            audio_file_name = self.video_to_audio(video_file_path, file_name)

            text = self.speech_to_text(audio_file_name)

            text_dic[file_name] = text
        with open("SpeechToTextDic.pickle", "wb") as output:
            pickle.dump(text_dic, output)
        return text_dic

    def video_to_text(self):
        text_dic = None
        try:
            text_dic = pickle.load(open("SpeechToTextDic.pickle", "rb"))
        except FileNotFoundError as err:
            print(
                "There is no saved speech to text data. SpeechToText function will be executed.",
                err,
            )

        if text_dic is None:
            text_dic = self.extract_text()
        with open("SpeechToText.csv", "w", newline="") as csv_file:
            writer = csv.writer(csv_file, delimiter=";", quoting=csv.QUOTE_ALL)
            for key, value in text_dic.items():
                writer.writerow([key, value])

    def speech_to_text(self, audio_file_name):
        retry_count = 3
        r = sr.Recognizer()
        for j in range(retry_count):
            try:
                with sr.AudioFile(audio_file_name) as source:
                    data = r.record(source)
                # , show_all=True
                text = r.recognize_google(data, language=self.language)
                break
            except Exception as err:
                print("Some error happened while doing speech to text", err)
                time.sleep(3)
                continue
        return text

    def video_to_audio(self, video_file_path, file_name):
        retry_count = 3
        for i in range(retry_count):
            try:
                video = mp.VideoFileClip(video_file_path)
                audio_file = video.audio
                audio_file_name = f"{file_name.split('.')[0]}.{self.audio_format}"
                audio_file.write_audiofile(audio_file_name)
                break
            except Exception as err:
                print(
                    "Some error happened while converting video to Audio file",
                    err,
                    video_file_path,
                )
                time.sleep(3)
                continue
        return audio_file_name
