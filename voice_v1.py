import ollama
from TTS.api import TTS
import sounddevice as sd
import numpy as np

# Load model (GPU is highly recommended for XTTS if available)
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True, gpu=False)

def speak_text(text):
    if not text.strip():
        return
    try:
        # XTTS v2 outputs at 24kHz
        audio = tts.tts(text=text, speaker="Claribel Dervla", language="en")
        sd.play(np.array(audio), samplerate=24000)
        sd.wait()
    except Exception as e:
        print(f"Error in TTS: {e}")

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
        print(content, end="", flush=True) # See text as it generates
        buffer += content
        
        # Speak when a natural pause is reached
        if any(char in content for char in (".", "!", "?", "\n")):
            speak_text(buffer.strip())
            buffer = ""
            
    if buffer.strip():
        speak_text(buffer.strip())

stream_and_speak("emma", "Hello! Tell me a very short joke.")