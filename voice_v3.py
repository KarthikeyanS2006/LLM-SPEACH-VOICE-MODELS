import ollama
from TTS.api import TTS
import sounddevice as sd
import numpy as np
import threading
import queue
import re

# --- CONFIGURATION ---
MODEL_NAME = "emma"  # Change to "emma" or your preferred model
SPEAKER_NAME = "Gracie Wise" # XTTS v2 built-in speaker
SAMPLE_RATE = 24000              # Matches XTTS v2 default output
# ---------------------

print("Initializing Emma's voice engine...")
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False, gpu=False)

audio_queue = queue.Queue()

def play_audio_worker():
    """Background thread to handle smooth audio playback without lag."""
    while True:
        try:
            audio_data = audio_queue.get()
            if audio_data is None: break
            
            if len(audio_data) > 0:
                sd.play(audio_data, samplerate=SAMPLE_RATE)
                sd.wait()
            audio_queue.task_done()
        except Exception as e:
            print(f"\n[Playback Error] {e}")

# Start the performance-optimized background thread
threading.Thread(target=play_audio_worker, daemon=True).start()

def speak_text(text):
    """Sanitizes text and generates audio in a thread-safe way."""
    # Customization: Clean text to prevent 'index out of range' errors
    clean_text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text).strip()
    
    if len(clean_text) < 2: 
        return
        
    try:
        # Generate the waveform
        audio = tts.tts(text=clean_text, speaker=SPEAKER_NAME, language="en")
        audio_queue.put(np.array(audio))
    except Exception as e:
        print(f"\n[TTS Error] Skipping fragment: {e}")

def stream_and_speak(prompt):
    """Streams from Ollama and triggers speech on natural pauses."""
    print(f"\n--- Speaking with {MODEL_NAME} ---")
    try:
        response = ollama.chat(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}], stream=True)
        
        buffer = ""
        for chunk in response:
            content = chunk['message']['content']
            print(content, end="", flush=True)
            buffer += content
            
            # Sentence-based triggering for human-like flow
            if any(char in content for char in (".", "!", "?", "\n")):
                sentence = buffer.strip()
                if sentence:
                    # Threading ensures the LLM keeps typing while the voice works
                    threading.Thread(target=speak_text, args=(sentence,), daemon=True).start()
                buffer = ""

        if buffer.strip():
            speak_text(buffer.strip())
            
    except Exception as e:
        print(f"\n[Ollama Error] Check if Ollama is running: {e}")

if __name__ == "__main__":
    print("\nSYSTEM READY.")
    print("Model Selection: Set to", MODEL_NAME)
    print("Performance: Background threading active.")
    
    while True:
        user_msg = input("\n\nYou: ")
        if user_msg.lower() in ["exit", "quit", "bye"]:
            print("Goodbye!")
            break
        stream_and_speak(user_msg)