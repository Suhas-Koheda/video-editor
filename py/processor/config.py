import os

# "english" or "multilingual"
MODEL_MODE = os.environ.get("ANTIGRAVITY_MODEL_MODE", "english")

def set_model_mode(mode):
    global MODEL_MODE
    MODEL_MODE = mode
    os.environ["ANTIGRAVITY_MODEL_MODE"] = mode

def get_model_mode():
    return MODEL_MODE

def get_whisper_model():
    return "tiny"

def get_gliner_model():
    if MODEL_MODE == "english":
        return "urchade/gliner_small-v2.1"
    return "urchade/gliner_multi-v2.1"

def get_sentence_transformer_model():
    if MODEL_MODE == "english":
        return "all-MiniLM-L6-v2"
    return "paraphrase-multilingual-MiniLM-L12-v2"
