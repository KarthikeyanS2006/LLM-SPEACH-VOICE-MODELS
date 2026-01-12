import ollama
from TTS.api import TTS
import sounddevice as sd
import numpy as np

# Load TTS model (female voice, multilingual)
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False, gpu=False)

def speak_text(text, language="en", speaker="EN-FEMALE"):
    # Generate audio from text
    try:
        audio = tts.tts(text=text, speaker=speaker, language=language)
        sd.play(audio, samplerate=22050)
        sd.wait()
    except Exception as e:
        print(f"Error in TTS: {e}")

# Stream Ollama model output and speak live
def stream_and_speak(model_name, prompt, language="en", speaker="EN-FEMALE"):
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
            speak_text(content, language, speaker)
            buffer = ""
    # Speak any remaining text
    if buffer:
        speak_text(buffer, language, speaker)

# Example usage
stream_and_speak("emma", "hi", language="en", speaker="EN-FEMALE")
