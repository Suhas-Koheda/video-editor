import os


MODEL_MODE = os.environ.get("ANTIGRAVITY_MODEL_MODE", "english")

def set_model_mode(mode):
    global MODEL_MODE
    MODEL_MODE = mode
    os.environ["ANTIGRAVITY_MODEL_MODE"] = mode

def get_model_mode():
    return MODEL_MODE

# STT Settings
# Options: whisper (English only), sarvam (Best for local Indian languages)
STT_ENGINE = os.environ.get("ANTIGRAVITY_STT_ENGINE", "sarvam")

def set_stt_engine(engine):
    global STT_ENGINE
    STT_ENGINE = engine
    os.environ["ANTIGRAVITY_STT_ENGINE"] = engine

def get_stt_engine():
    # If in english mode, whisper is fine. If multilingual, use sarvam
    if MODEL_MODE == "english":
        return "whisper"
    return STT_ENGINE

def get_whisper_model():
    return "tiny" 

def get_sarvam_config():
    # Using the user's provided Sarvam AI API key
    return {
        "api_key": os.environ.get("SARVAM_API_KEY", "sk_4xz879a5_5p8lxlf8VCPUzDLgB7fPdUxw")
    }

def get_gliner_model():
    if MODEL_MODE == "english":
        return "urchade/gliner_small-v2.1"
    return "urchade/gliner_multi-v2.1"

def get_sentence_transformer_model():
    # 'all-MiniLM-L6-v2' is ~80MB (English-heavy but fast)
    # 'paraphrase-multilingual-MiniLM-L12-v2' is ~471MB (Better for multi-lang)
    if MODEL_MODE == "english":
        return "all-MiniLM-L6-v2"
    
    # Using the smaller v2 model for multilingual as well by default to avoid large 471MB download
    # Change this back to 'paraphrase-multilingual-MiniLM-L12-v2' for better non-English accuracy
    return "all-MiniLM-L6-v2" 


# Translation Settings (Local Mode Only)
def get_translation_config():
    return {
        "model": "facebook/nllb-200-distilled-600M",
        "mode": "local"
    }

