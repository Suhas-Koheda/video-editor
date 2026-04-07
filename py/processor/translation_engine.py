import os
import torch
import requests
from processor.config import get_translation_config, get_sarvam_config

# Cache for local models
_local_translation_model = None
_local_tokenizer = None

CACHE_DIR = os.path.join(os.getcwd(), ".model_cache", "translation")
os.makedirs(CACHE_DIR, exist_ok=True)

def _get_local_model():
    """
    Lazily loads the open-source Facebook NLLB translation model.
    """
    global _local_translation_model, _local_tokenizer
    if _local_translation_model is None:
        print("[Translate] Loading Local Facebook NLLB-200-distilled-600M...")
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        
        cfg = get_translation_config()
        model_name = cfg["model"]
        _local_tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
        _local_translation_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=CACHE_DIR)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            _local_translation_model = _local_translation_model.to("cuda")
            
    return _local_translation_model, _local_tokenizer

import time

# simple cache
_translation_cache = {}

def translate_text(text, source_lang, target_lang="en"):
    """
    Translates text using either Sarvam AI (API) or Local NLLB.
    """
    cache_key = f"{text}:{source_lang}:{target_lang}"
    if cache_key in _translation_cache:
        return _translation_cache[cache_key]

    # 1. Try Sarvam AI first (Fastest & No large download)
    sarvam_result = _translate_sarvam(text, source_lang, target_lang)
    
    result = sarvam_result if sarvam_result else _translate_local_nllb(text, source_lang, target_lang)
    
    _translation_cache[cache_key] = result
    return result

def _translate_sarvam(text, source_lang, target_lang="en", retries=3):
    config = get_sarvam_config()
    api_key = config.get("api_key")
    if not api_key: return None

    # Mapping for Sarvam
    sarvam_langs = {
        "hi": "hi-IN", "te": "te-IN", "ta": "ta-IN", "kn": "kn-IN",
        "ml": "ml-IN", "mr": "mr-IN", "bn": "bn-IN", "gu": "gu-IN",
        "pa": "pa-IN", "as": "as-IN", "or": "or-IN", "en": "en-IN"
    }
    
    src_code = sarvam_langs.get(source_lang, "hi-IN")
    tgt_code = sarvam_langs.get(target_lang, "en-IN")
    
    if src_code == tgt_code: return text

    url = "https://api.sarvam.ai/translate"
    
    payload = {
        "input": text,
        "source_language_code": src_code,
        "target_language_code": tgt_code,
        "model": "mayura:v1"
    }
    headers = {
        "api-subscription-key": api_key,
        "Content-Type": "application/json"
    }
    
    for attempt in range(retries):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            if response.status_code == 200:
                return response.json().get("translated_text", "")
            elif response.status_code == 429: # Rate limit
                wait = (2 ** attempt) + 1
                print(f"[Translate] Rate limited (429). Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"[Translate] Sarvam AI Failed ({response.status_code}): {response.text}")
                break
        except Exception as e:
            print(f"[Translate] Sarvam AI Error attempt {attempt+1}: {e}")
            time.sleep(1)
            
    return None

def _translate_local_nllb(text, source_lang, target_lang="en"):
    """
    Local Facebook Translation (using NLLB).
    """
    try:
        print(f"[Translate] Falling back to Local NLLB...")
        model, tokenizer = _get_local_model()
        
        # Mapping standard codes to NLLB codes
        lang_map = {
            "hi": "hin_Deva", "ta": "tam_Taml", "te": "tel_Telu", "kn": "kan_Knda",
            "ml": "mal_Mlym", "mr": "mar_Deva", "bn": "ben_Beng", "gu": "guj_Gujr",
            "pa": "pan_Guru", "as": "asm_Beng", "or": "ory_Orya", "en": "eng_Latn"
        }
        
        src_code = lang_map.get(source_lang, "hin_Deva")
        tgt_code = lang_map.get(target_lang, "eng_Latn")
        
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        translated_tokens = model.generate(
            **inputs, 
            forced_bos_token_id=tokenizer.lang_code_to_id[tgt_code], 
            max_length=256
        )
        
        return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    except Exception as e:
        print(f"[Translate] Local Translation Failed: {e}")
        return text
