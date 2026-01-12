import ollama
from TTS.api import TTS
import sounddevice as sd
import numpy as np
import threading
import queue
import re


# --- CONFIGURATION ---
MODEL_NAME = "emma"             # Your local Ollama model
ENGLISH_SPEAKER = "Ana Florence" # Built-in XTTS v2 English Female voice
TAMIL_SPEAKER = "TA-FEMALE"      # Built-in XTTS v2 Tamil Female voice (check available speakers)
SAMPLE_RATE = 24000              # Native XTTS v2 sample rate
# ---------------------


print("Loading Emma's voice engine (XTTS v2)...")
# Note: gpu=True is much faster if you have an NVIDIA card
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True, gpu=False)


audio_queue = queue.Queue()


def play_audio_worker():
    """Background thread: Plays audio chunks smoothly in the order they arrive."""
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


# Start the audio playback thread
threading.Thread(target=play_audio_worker, daemon=True).start()


def clean_text_for_tts(text):
    """
    Cleans the AI output so the TTS doesn't crash on emojis or actions.
    """
    # 1. Remove text inside parentheses (e.g., "(Giggles)", "(Eyes sparkling)")
    text = re.sub(r'\(.*?\)', '', text)
    
    # 2. Remove asterisks (e.g., "*smiles*")
    text = re.sub(r'\*.*?\*', '', text)
    
    # 3. Remove emojis and non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    
    # 4. Remove code blocks or weird symbols
    text = re.sub(r'[^a-zA-Z0-9\s.,!?\']', '', text)
    
    return text.strip()


def speak_text(text, language="en"):
    """Generates audio from cleaned text and adds it to the playback queue."""
    clean_text = clean_text_for_tts(text)
    
    # Skip if nothing is left to say
    if len(clean_text) < 2:
        return
        
    try:
        # Choose speaker and language
        speaker = ENGLISH_SPEAKER if language == "en" else TAMIL_SPEAKER
        # Generate waveform
        audio = tts.tts(text=clean_text, speaker=speaker, language=language)
        audio_queue.put(np.array(audio))
    except Exception as e:
        # This catches the 'index out of range' error silently
        pass


def stream_and_speak(prompt, language="en"):
    """Streams from Ollama and sends sentences to the TTS thread."""
    print(f"\n--- Chatting with {MODEL_NAME} ---")
    
    # System message helps Emma be more 'TTS friendly'
    messages = [
        {"role": "system", "content": "You are Emma. Be playful and helpful. Avoid using emojis or writing actions in parentheses as I cannot hear them."},
        {"role": "user", "content": prompt}
    ]
    
    try:
        response = ollama.chat(model=MODEL_NAME, messages=messages, stream=True)
        
        buffer = ""
        for chunk in response:
            content = chunk['message']['content']
            print(content, end="", flush=True)
            buffer += content
            
            # Trigger speech when a sentence ends
            if any(char in content for char in (".", "!", "?", "\n")):
                sentence = buffer.strip()
                if sentence:
                    # Threading allows audio to generate while text is still streaming
                    threading.Thread(target=speak_text, args=(sentence, language), daemon=True).start()
                buffer = ""


        # Final check for remaining text
        if buffer.strip():
            speak_text(buffer.strip(), language)
            
    except Exception as e:
        print(f"\n[Ollama Error] Ensure Ollama is running: {e}")


if __name__ == "__main__":
    print("\n" + "="*30)
    print("EMMA LIVE VOICE CHAT")
    print("Speaker:", ENGLISH_SPEAKER)
    print("Model:", MODEL_NAME)
    print("Type 'exit' to quit.")
    print("="*30 + "\n")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Goodbye, darling!")
            break
        # To switch to Tamil, use language="ta"
        stream_and_speak(user_input, language="en")  # Change to "ta" for Tamil
