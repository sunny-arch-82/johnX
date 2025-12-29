"""
JOHN X — Transparent Context-Aware Voice Assistant
Runs 100% locally using Whisper + zero-shot transformer + FAISS memory
"""

import os, webbrowser, asyncio, numpy as np
import pyttsx3, gradio as gr, whisper, spacy, faiss
from transformers import pipeline
from sentence_transformers import SentenceTransformer

# ---------- INIT MODELS ----------
print("Loading models (first run may take a minute)...")
stt_model = whisper.load_model("base")
intent_model = pipeline("zero-shot-classification",
                        model="facebook/bart-large-mnli")
emb_model = SentenceTransformer("all-MiniLM-L6-v2")
nlp = spacy.load("en_core_web_sm")

# ---------- MEMORY ----------
index = faiss.IndexFlatL2(384)
memory_texts = []

def remember(text):
    vec = emb_model.encode([text]).astype("float32")
    index.add(np.array(vec))
    memory_texts.append(text)

def recall(query, topk=1):
    if len(memory_texts) == 0:
        return []
    qv = emb_model.encode([query]).astype("float32")
    D, I = index.search(qv, topk)
    return [memory_texts[i] for i in I[0]]

# ---------- SPEECH ----------
engine = pyttsx3.init()
def speak(msg):
    engine.say(msg)
    engine.runAndWait()

# ---------- INTENTS ----------
intents = [
    "open youtube", "search web", "tell joke", "get weather",
    "remember note", "recall note", "exit"
]

def explain_reason(intent):
    reasons = {
        "open youtube": "because you asked to open a media website",
        "search web": "to help you find information online",
        "tell joke": "to make you smile",
        "get weather": "to show current weather information",
        "remember note": "to store your note safely in memory",
        "recall note": "to retrieve your saved notes",
        "exit": "to end the assistant session"
    }
    return reasons.get(intent, "based on your request")

# ---------- ACTIONS ----------
def execute(intent, text):
    if intent == "open youtube":
        webbrowser.open("https://youtube.com")
        return "Opened YouTube."
    elif intent == "search web":
        q = text.replace("search", "").strip()
        webbrowser.open(f"https://google.com/search?q={q}")
        return f"Searching for {q}."
    elif intent == "tell joke":
        return "Why did the deep network go broke? It ran out of cache!"
    elif intent == "get weather":
        return "I cannot access live APIs right now, but you can say: 'search weather in Boulder' to check online."
    elif intent == "remember note":
        remember(text)
        return "Okay, I saved that note."
    elif intent == "recall note":
        recalled = recall(text)
        return f"You told me earlier: {recalled[0]}" if recalled else "I don't recall anything similar."
    elif intent == "exit":
        speak("Goodbye!")
        os._exit(0)
    else:
        return "I'm not sure what to do."

# ---------- MAIN LOGIC ----------
import gc
import time

def johnx_core(audio_path):
    start = time.time()
    if not audio_path:
        return "No audio file received."

    # ----- 1️⃣ Transcribe Speech -----
    try:
        result = stt_model.transcribe(audio_path)
        user_text = result.get("text", "").strip()
    except Exception as e:
        return f"❌ Transcription failed: {e}"

    if not user_text:
        return "Didn't catch that. Please try again."

    print(f"\nUser said: {user_text}")

    # ----- 2️⃣ Intent Detection -----
    try:
        pred = intent_model(user_text, candidate_labels=intents)
        intent = pred["labels"][0]
        conf = pred["scores"][0]
    except Exception as e:
        return f"❌ Intent model error: {e}"

    reason = explain_reason(intent)

    # ----- 3️⃣ Execute Action -----
    response = execute(intent, user_text)
    elapsed = time.time() - start

    # ----- 4️⃣ Respond & Cleanup -----
    reply_text = f"{response}\nIntent: {intent} ({conf:.2f})\nReason: {reason}\nProcessed in {elapsed:.1f}s."
    print(reply_text)
    speak(reply_text)

    # Free up RAM for next round
    gc.collect()

    return reply_text

    # NLP intent
    pred = intent_model(user_text, candidate_labels=intents)
    intent = pred["labels"][0]
    conf = pred["scores"][0]
    reason = explain_reason(intent)

    # Action
    response = execute(intent, user_text)
    log = f"[Intent: {intent} ({conf:.2f}) | Reason: {reason}]"
    print(log)

    # Memory
    remember(user_text)

    # Speak result
    speak_text = f"{response}. I did that {reason}."
    asyncio.run(async_speak_edge(speak_text))

    return f"You said: {user_text}\n{response}\n{log}"

# ---------- EDGE-TTS (async smoother voice) ----------
async def async_speak_edge(text):
    try:
        import edge_tts
        communicate = edge_tts.Communicate(text, "en-US-JennyNeural")
        await communicate.stream()
    except Exception:
        # fallback to pyttsx3 if Edge-TTS not available
        speak(text)

# ---------- GRADIO UI ----------
ui = gr.Interface(
    fn=johnx_core,
    inputs=gr.Audio(sources=["microphone"], type="filepath"),
    outputs="text",
    title="JOHN X — Transparent Voice Assistant",
    description="Speaks, remembers, and explains why it does things. 100% local."
)

ui.launch(inbrowser=True)

if __name__ == "__main__":
    ui.launch()

