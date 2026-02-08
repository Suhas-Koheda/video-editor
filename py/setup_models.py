import os
from huggingface_hub import snapshot_download
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer

CACHE_DIR = os.path.join(os.getcwd(), ".model_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

os.environ["HF_HOME"] = CACHE_DIR
os.environ["SENTENCE_TRANSFORMERS_HOME"] = CACHE_DIR

def download_models():
    print("--- ANTIGRAVITY MODEL PRE-DOWNLOADER ---")
    
    print("\n[1/4] Downloading Multilingual GLiNER (NLP Engine)...")
    try:
        from gliner import GLiNER
        GLiNER.from_pretrained("urchade/gliner_multi", cache_dir=CACHE_DIR)
        print("✓ GLiNER Done.")
    except Exception as e:
        print(f"✗ GLiNER Failed: {e}")

    print("\n[2/4] Downloading Whisper 'base' (Speech to Text)...")
    try:
        WhisperModel("base", device="cpu", compute_type="int8", download_root=os.path.join(CACHE_DIR, "whisper"))
        print("✓ Whisper Done.")
    except Exception as e:
        print(f"✗ Whisper Failed: {e}")

    print("\n[3/4] Downloading Multilingual Sentence Transformer (Retrieval Engine)...")
    try:
        SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', cache_folder=CACHE_DIR)
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

    print("\n--- ALL MODELS CACHED IN ./.model_cache ---")

if __name__ == "__main__":
    download_models()
