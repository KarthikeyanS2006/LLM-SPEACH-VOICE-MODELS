import ollama
from TTS.api import TTS
import sounddevice as sd
import numpy as np
import threading
import queue
import re

# 1. Load Model (CPU mode)
print("Loading Emma's voice...")
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True, gpu=False)

# 2. Setup a Queue for synchronized audio playback
audio_queue = queue.Queue()

def play_audio_worker():
    """ Keeps the audio player alive and playing sequences from the queue """
    while True:
        try:
            audio_data = audio_queue.get()
            if audio_data is None: break
            
            # Final check before playing
            if len(audio_data) > 0:
                sd.play(audio_data, samplerate=24000)
                sd.wait()
            audio_queue.task_done()
        except Exception as e:
            print(f"\n[Playback Error] {e}")

# Start background player thread
threading.Thread(target=play_audio_worker, daemon=True).start()

def speak_text(text):
    # SAFETY: Remove non-alphanumeric junk and check length
    clean_text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text).strip()
    
    if len(clean_text) < 2: # Skip if it's just a dot or a space
        return
        
    try:
        # Generate the audio
        audio = tts.tts(text=clean_text, speaker="Claribel Dervla", language="en")
        audio_queue.put(np.array(audio))
    except Exception as e:
        # This catches the 'index out of range' error without crashing the script
        print(f"\n[TTS Skip] Text '{clean_text}' was too short or invalid: {e}")

def stream_and_speak(model_name, prompt):
    print("\nEmma is thinking...")
    try:
        response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}], stream=True)
        
        buffer = ""
        for chunk in response:
            content = chunk['message']['content']
            print(content, end="", flush=True)
            buffer += content
            
            # Speak when a sentence or line ends
            if any(char in content for char in (".", "!", "?", "\n")):
                sentence = buffer.strip()
                if sentence:
                    # Threading here makes the LLM keep typing while voice generates
                    threading.Thread(target=speak_text, args=(sentence,), daemon=True).start()
                buffer = ""

        # Final speak for any remaining text
        if buffer.strip():
            speak_text(buffer.strip())
            
    except Exception as e:
        print(f"\n[Ollama Error] {e}")

# Continuous Chat Loop
if __name__ == "__main__":
    print("Ready! Type 'exit' to stop.")
    while True:
        user_msg = input("\n\nYou: ")
        if user_msg.lower() in ["exit", "quit", "bye"]:
            break
        stream_and_speak("qwen2.5-coder:3b", user_msg)