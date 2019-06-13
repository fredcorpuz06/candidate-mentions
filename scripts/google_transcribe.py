# set GOOGLE_APPLICATION_CREDENTIALS=config/qac239-corpuz.json

import io
import os

# Imports the Google Cloud client library
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="../config/qac239-corpuz.json"

def transcribe_file(speech_file):
    """Transcribe the given audio file.
    
    Args:
        speech_file: File location of audio file. Must be in FLAC format
        with bitrate = 128 kb/ + set sampling rate to 48000 Hz + 
        audio channels = 1        

    Returns:
        A list of text strings where each each text string (result) is
        for a consecutive portion of the audio.
    """
    client = speech.SpeechClient()
    content = "".encode()
    # need to put a try here + lift up the error
    try:
        with io.open(speech_file, 'rb') as audio_file:
            content = audio_file.read()
    except Exception as e:
        print(f'--- {speech_file} had {e}')
        return "INVALID AUDIO FILE PATH"

    audio = types.RecognitionAudio(content=content)
    config = types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.FLAC,
        sample_rate_hertz=48000,
        language_code='en-US')
    print(f"Transcribing {speech_file}")

    try:
        response = client.recognize(config, audio)
        # Each result is for a consecutive portion of the audio. Iterate through
        # them to get the transcripts for the entire audio file.
        text_rez = []
        for result in response.results:
            # The first alternative is the most likely one for this portion.
            # print(u'Transcript: {}'.format(result.alternatives[0].transcript))
            text_rez.append(result.alternatives[0].transcript)

        return " ".join(text_rez)
    except Exception as e:
        print(f'--- {speech_file} had {e}')
        return "NEEDS LONG-RUNNING"        

def main():
    rez = transcribe_file(
        "../data/tv-ads-kantar/GOV_CA_EASTIN_WE_GO_HIGH.flac")
    print(rez)

if __name__ == "__main__":
    main()