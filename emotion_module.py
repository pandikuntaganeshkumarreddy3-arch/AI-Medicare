"""
emotion_module.py
-----------------
Detects emotions from user text using a local HuggingFace model.
Maps detected emotions to conversational tones for medical responses.

Model used: j-hartmann/emotion-english-distilroberta-base
Runs fully offline after first download. No paid API required.
"""

from transformers import pipeline

# ──────────────────────────────────────────────
# 1.  Load the emotion-detection pipeline once
#     (downloaded to ~/.cache/huggingface on first run)
# ──────────────────────────────────────────────
print("[emotion_module] Loading emotion detection model …")

emotion_classifier = pipeline(
    task="text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=1,           # return only the highest-scoring label
    truncation=True,   # safely handle long inputs
)

print("[emotion_module] Model loaded successfully.")


# ──────────────────────────────────────────────
# 2.  Emotion → Tone mapping table
# ──────────────────────────────────────────────
EMOTION_TONE_MAP: dict[str, str] = {
    "fear":     "reassuring",
    "sadness":  "reassuring",
    "anger":    "calm",
    "joy":      "friendly",
    "neutral":  "informative",
    "disgust":  "calm",       # extra label the model may return
    "surprise": "informative",
}

# Tone → Opening sentence used by the response generator
TONE_INTRO_MAP: dict[str, str] = {
    "reassuring":  "I understand your concern and I'll try to provide helpful information.",
    "calm":        "I hear you. Let me calmly walk you through the relevant medical information.",
    "friendly":    "Great to hear from you! Here is some useful medical information.",
    "informative": "Here is the medical information you requested.",
    "supportive":  "I'm sorry you're experiencing discomfort. Let me explain what might help.",
}


# ──────────────────────────────────────────────
# 3.  Public functions
# ──────────────────────────────────────────────

def detect_emotion(text: str) -> str:
    """
    Analyse *text* and return the dominant emotion label as a lowercase string.

    Possible return values (from the model):
        "anger" | "disgust" | "fear" | "joy" | "neutral" | "sadness" | "surprise"

    Example
    -------
    >>> detect_emotion("I am really scared about my diagnosis.")
    'fear'
    """
    if not text or not text.strip():
        return "neutral"

    results = emotion_classifier(text)

    # pipeline with top_k=1 returns: [[{"label": "fear", "score": 0.97}]]
    top_result = results[0][0]
    emotion = top_result["label"].lower()

    print(f"[emotion_module] Detected emotion: '{emotion}' "
          f"(confidence: {top_result['score']:.2%})")

    return emotion


def map_emotion_to_tone(emotion: str) -> str:
    """
    Map an emotion string to a conversational tone string.

    Parameters
    ----------
    emotion : str
        A lowercase emotion label (e.g. "fear", "joy").

    Returns
    -------
    str
        A tone label such as "reassuring", "calm", "friendly", or "informative".

    Example
    -------
    >>> map_emotion_to_tone("fear")
    'reassuring'
    >>> map_emotion_to_tone("joy")
    'friendly'
    """
    tone = EMOTION_TONE_MAP.get(emotion.lower(), "informative")
    print(f"[emotion_module] Emotion '{emotion}' → tone '{tone}'")
    return tone


def get_tone_intro(tone: str) -> str:
    """
    Return the opening sentence for the given tone.

    Parameters
    ----------
    tone : str
        A tone label returned by map_emotion_to_tone().

    Returns
    -------
    str
        A warm, context-appropriate opening sentence.

    Example
    -------
    >>> get_tone_intro("reassuring")
    "I understand your concern and I'll try to provide helpful information."
    """
    return TONE_INTRO_MAP.get(tone, TONE_INTRO_MAP["informative"])


# ──────────────────────────────────────────────
# 4.  Quick self-test (run: python emotion_module.py)
# ──────────────────────────────────────────────
if __name__ == "__main__":
    test_sentences = [
        "I am really scared about my diagnosis.",
        "This medication is making me so angry!",
        "I feel much better today, thank you!",
        "What is the dosage for paracetamol?",
        "I feel so sad and hopeless about my condition.",
    ]

    print("\n" + "=" * 55)
    print("  Emotion Module — Self Test")
    print("=" * 55)

    for sentence in test_sentences:
        emotion = detect_emotion(sentence)
        tone    = map_emotion_to_tone(emotion)
        intro   = get_tone_intro(tone)

        print(f"\nInput   : {sentence}")
        print(f"Emotion : {emotion}")
        print(f"Tone    : {tone}")
        print(f"Intro   : {intro}")

    print("\n" + "=" * 55)
    