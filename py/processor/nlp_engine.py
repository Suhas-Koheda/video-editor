from gliner import GLiNER
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import RegexpParser
import os
from processor.config import get_gliner_model

CACHE_DIR = os.path.join(os.getcwd(), ".model_cache")

_gliner_model = None

def get_entities_and_nouns(text):
    """
    State-of-the-art NLP engine using GLiNER (Zero-shot NER) and NLTK (POS Tagging).
    """
    global _gliner_model
    if _gliner_model is None:
        model_path = get_gliner_model()
        _gliner_model = GLiNER.from_pretrained(model_path, cache_dir=CACHE_DIR)

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
        
    seen = set()
    final_unique = []
    results.sort(key=lambda x: len(x['text']), reverse=True)

    for r in results:
        t = r['text'].lower().strip('., ')
        if t not in seen and len(t) > 3:
            final_unique.append(r)
            seen.add(t)

    return final_unique[:12]

def normalize_entity(text):
    """
    Normalize entity text: lowercase, strip punctuation, and trim spaces.
    """
    import string
    text = text.lower().strip()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return " ".join(text.split())

def build_global_entity_stats(segments):
    """
    Iterates over all segments to extract entities and build global statistics.
    """
    global_stats = {}
    for i, seg in enumerate(segments):
        # Use existing entities if present to avoid redundant GLiNER calls
        if 'entities' in seg and seg['entities']:
            local_entities = seg['entities']
        else:
            text = seg.get('translated_text', seg['text'])
            local_entities = get_entities_and_nouns(text)
        
        for ent in local_entities:
            norm_text = normalize_entity(ent['text'])
            if norm_text not in global_stats:
                global_stats[norm_text] = {
                    "text": ent['text'], # keep original form for display
                    "count": 0,
                    "total_span_length": 0,
                    "first_seen_position": i,
                    "label": ent['label']
                }
            
            global_stats[norm_text]["count"] += 1
            global_stats[norm_text]["total_span_length"] += len(ent['text'])
            
    return global_stats

def compute_global_scores(global_stats):
    """
    Computes importance scores for entities based on frequency, length, and position.
    """
    for norm_text, stats in global_stats.items():
        count = stats["count"]
        span = stats["total_span_length"]
        first_seen = stats["first_seen_position"]
        
        # score = (count * 2.0) + (span * 0.3) - (first_seen * 0.1)
        score = (count * 2.0) + (span * 0.3) - (first_seen * 0.1)
        stats["score"] = score
        
    return global_stats

def get_sliding_context(segments, index, window_size=1):
    """
    Concatenates text from neighboring segments to provide context.
    """
    start = max(0, index - window_size)
    end = min(len(segments), index + window_size + 1)
    
    context_parts = []
    for i in range(start, end):
        text = segments[i].get('translated_text', segments[i]['text'])
        context_parts.append(text)
        
    return " ".join(context_parts)

def rank_entities_for_segment(segment_text, local_entities, global_stats, context_text):
    """
    Ranks entities for a specific segment based on local presence and global importance.
    """
    ranked = []
    
    context_text_norm = context_text.lower()
    
    for ent in local_entities:
        norm_text = normalize_entity(ent['text'])
        stats = global_stats.get(norm_text, {})
        
        # Base score from global stats
        base_score = stats.get("score", 0)
        
        # Boost if entity appears in context_text
        if norm_text in context_text_norm:
            base_score += 5.0
            
        # Boost longer phrases (more words)
        word_count = len(ent['text'].split())
        if word_count > 1:
            base_score += word_count * 2.0
            
        # Penalize very short/common words (redundant with normalize_entity and get_entities_and_nouns but safe)
        if len(norm_text) < 4:
            base_score -= 10.0
            
        # Keep all original keys but add/update score
        ent_copy = ent.copy()
        ent_copy["score"] = base_score
        ranked.append(ent_copy)
        
    # Sort by score descending and return top 5
    ranked.sort(key=lambda x: x.get('score', 0), reverse=True)
    return ranked[:5]

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
        pass

