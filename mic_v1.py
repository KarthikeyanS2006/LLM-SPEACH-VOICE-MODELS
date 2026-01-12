import ollama
import speech_recognition as sr
from TTS.api import TTS
import sounddevice as sd
import numpy as np
import threading
import queue
import re

# --- CONFIGURATION ---
MODEL_NAME = "emma"             # Your local Ollama model
ENGLISH_SPEAKER = "Ana Florence" # XTTS v2 Female voice
SAMPLE_RATE = 24000             # Native XTTS v2 sample rate
# ---------------------

print("Loading Emma's voice engine (XTTS v2)...")
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False, gpu=False)

audio_queue = queue.Queue()

def play_audio_worker():
    """Background thread: Plays audio chunks smoothly."""
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

# Start audio playback thread
threading.Thread(target=play_audio_worker, daemon=True).start()

def clean_text_for_tts(text):
    """Cleans AI output for TTS."""
    text = re.sub(r'\(.*?\)', '', text)  # Remove (actions)
    text = re.sub(r'\*.*?\*', '', text)  # Remove *actions*
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove emojis
    text = re.sub(r'[^a-zA-Z0-9\s.,!?\']', '', text)
    return text.strip()

def speak_text(text):
    """Generates audio and adds to playback queue."""
    clean_text = clean_text_for_tts(text)
    if len(clean_text) < 2:
        return
        
    try:
        audio = tts.tts(text=clean_text, speaker=ENGLISH_SPEAKER, language="en")
        audio_queue.put(np.array(audio))
    except Exception:
        pass

def listen_to_user():
    """Listens to microphone and converts speech to text."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Listening...")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
    
    try:
        text = recognizer.recognize_google(audio)
        print(f"✅ You said: {text}")
        return text
    except sr.UnknownValueError:
        print("❌ Could not understand audio")
        return ""
    except sr.RequestError as e:
        print(f"❌ Service error: {e}")
        return ""

def stream_and_speak(prompt):
    """Streams LLM response and speaks it."""
    print(f"\n🤖 {MODEL_NAME} is thinking...")
    
    messages = [
        {"role": "system", "content": "You are Emma. Be playful and helpful. Speak naturally for voice output."},
        {"role": "user", "content": prompt}
    ]
    
    try:
        response = ollama.chat(model=MODEL_NAME, messages=messages, stream=True)
        
        buffer = ""
        for chunk in response:
            content = chunk['message']['content']
            print(content, end="", flush=True)
            buffer += content
            
            if any(char in content for char in (".", "!", "?", "\n")):
                sentence = buffer.strip()
                if sentence:
                    threading.Thread(target=speak_text, args=(sentence,), daemon=True).start()
                buffer = ""
                
        if buffer.strip():
            speak_text(buffer.strip())
            
    except Exception as e:
        print(f"\n[Ollama Error] {e}")

# === MAIN VOICE CHAT LOOP ===
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🔊🔊 EMMA FULL VOICE CHAT (SPEAK & LISTEN)")
    print("💡 Say 'exit', 'quit', or 'bye' to stop")
    print("💡 Speak clearly into your microphone")
    print("="*50 + "\n")
    
    while True:
        # 1. LISTEN to user
        user_input = listen_to_user()
        
        if not user_input:
            continue
            
        # 2. Check for exit commands
        if any(word in user_input.lower() for word in ["exit", "quit", "bye"]):
            speak_text("Goodbye, darling! It was lovely chatting with you!")
            break
            
        # 3. RESPOND with voice
        stream_and_speak(user_input)
        
    print("👋 Chat ended. Goodbye!")
