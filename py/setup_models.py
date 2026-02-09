import os
import sys
from huggingface_hub import snapshot_download
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer
from processor.config import get_gliner_model, get_whisper_model, get_sentence_transformer_model, set_model_mode

CACHE_DIR = os.path.join(os.getcwd(), ".model_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

os.environ["HF_HOME"] = CACHE_DIR
os.environ["SENTENCE_TRANSFORMERS_HOME"] = CACHE_DIR

def download_models(mode="english"):
    set_model_mode(mode)
    print(f"--- ANTIGRAVITY MODEL PRE-DOWNLOADER ({mode.upper()} MODE) ---")
    
    gliner_path = get_gliner_model()
    print(f"\n[1/4] Downloading GLiNER ({gliner_path})...")
    try:
        from gliner import GLiNER
        GLiNER.from_pretrained(gliner_path, cache_dir=CACHE_DIR)
        print("✓ GLiNER Done.")
    except Exception as e:
        print(f"✗ GLiNER Failed: {e}")

    whisper_path = get_whisper_model()
    print(f"\n[2/4] Downloading Whisper '{whisper_path}' (Speech to Text)...")
    try:
        WhisperModel(whisper_path, device="cpu", compute_type="int8", download_root=os.path.join(CACHE_DIR, "whisper"))
        print("✓ Whisper Done.")
    except Exception as e:
        print(f"✗ Whisper Failed: {e}")

    st_path = get_sentence_transformer_model()
    print(f"\n[3/4] Downloading Sentence Transformer ({st_path})...")
    try:
        SentenceTransformer(st_path, cache_folder=CACHE_DIR)
        print("✓ Sentence Transformer Done.")
    except Exception as e:
        print(f"✗ Sentence Transformer Failed: {e}")

    print("\n[4/4] Downloading NLTK data (POS Tagging)...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)
        print("✓ NLTK Done.")
    except Exception as e:
        print(f"✗ NLTK Failed: {e}")

    print(f"\n--- ALL {mode.upper()} MODELS CACHED IN ./.model_cache ---")

if __name__ == "__main__":
    mode = "english"
    if len(sys.argv) > 1:
        if sys.argv[1].lower() in ["english", "multilingual"]:
            mode = sys.argv[1].lower()
    else:
        print("Select model mode:")
        print("1. English (Default)")
        print("2. Multilingual")
        choice = input("Enter Choice (1/2): ").strip()
        if choice == "2":
            mode = "multilingual"
            
    download_models(mode)
