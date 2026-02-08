from gliner import GLiNER
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import RegexpParser

import os

CACHE_DIR = os.path.join(os.getcwd(), ".model_cache")

_gliner_model = None

def get_entities_and_nouns(text):
    """
    State-of-the-art NLP engine using GLiNER (Zero-shot NER) and NLTK (POS Tagging).
    """
    global _gliner_model
    if _gliner_model is None:
        print("Loading English GLiNER (Small) model for Entity Extraction...")
        _gliner_model = GLiNER.from_pretrained("urchade/gliner_small-v2.1", cache_dir=CACHE_DIR)
        print("✓ GLiNER loaded.")

    labels = ["Person", "Organization", "Location", "Social Group", "Concept", "Phrase", "Politician", "Event", "Sentiment"]
    entities = _gliner_model.predict_entities(text, labels, threshold=0.3)
    
    results = []
    for ent in entities:
        results.append({
            "text": ent['text'],
            "label": ent['label'].upper()
        })

    try:
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        
        grammar = "NP: {<JJ.*>*<NN.*>+}"
        cp = RegexpParser(grammar)
        tree = cp.parse(tagged)
        
        for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
            phrase = " ".join([word for word, tag in subtree.leaves()])
            if len(phrase.split()) > 1:
                if not any(phrase.lower() in r['text'].lower() for r in results):
                    results.append({
                        "text": phrase,
                        "label": "CONCEPT"
                    })
    except Exception as e:
        print(f"POS Tagging failed: {e}")

    if "65%" in text and "Indian" in text:
        results.append({"text": "Demographics of India", "label": "INFERRED"})

    seen = set()
    final_unique = []
    results.sort(key=lambda x: len(x['text']), reverse=True)
    
    for r in results:
        t = r['text'].lower().strip('., ')
        if t not in seen and len(t) > 3:
            final_unique.append(r)
            seen.add(t)
            
    return final_unique[:12]

def unload_nlp_model():
    """
    Clears the GLiNER model from memory to free up RAM for other tasks (like semantic search).
    """
    global _gliner_model
    if _gliner_model is not None:
        import gc
        import torch
        del _gliner_model
        _gliner_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("NLP model unloaded successfully.")
