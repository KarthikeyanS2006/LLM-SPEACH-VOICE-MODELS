import ollama
from TTS.api import TTS
import sounddevice as sd
import numpy as np

# Load the English-only Tacotron2 model
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=True, gpu=False)

def speak_text(text):
    try:
        # Generate audio from text
        audio = tts.tts(text=text)
        # Play audio
        sd.play(audio, samplerate=22050)
        sd.wait()
    except Exception as e:
        print(f"Error in TTS: {e}")

# Stream Ollama model output and speak live
def stream_and_speak(model_name, prompt):
    print("Streaming response from Ollama...")
    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    buffer = ""
    for chunk in response:
        content = chunk['message']['content']
        buffer += content
        # Speak when a sentence ends
        if content.endswith((".", "!", "?")):
            speak_text(content)
            buffer = ""
    # Speak any remaining text
    if buffer:
        speak_text(buffer)

# Example usage
stream_and_speak("emma", "hi")
