import wikipedia
from sentence_transformers import SentenceTransformer, util
import torch

_embedder = None

def agentic_search(segment_text, entity_text):
    """
    Uses Contextual Reasoning to find the most relevant Wikipedia article.
    Heuristics:
    1. Direct Search.
    2. Context-aware search (if Indians/65% mentioned, search for Demographics).
    3. Semantic Ranking: Ranks all results against the original spoken text.
    """
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer('all-MiniLM-L6-v2')

    try:
        # 1. Broad Search queries
        queries = [entity_text]
        if "65%" in segment_text or "youth" in segment_text.lower():
             if "india" in segment_text.lower():
                 queries.append("Demographics of India")
                 queries.append("Youth in India")
        
        raw_results = []
        for q in queries:
            raw_results.extend(wikipedia.search(q, results=3))
        
        raw_results = list(set(raw_results)) # Unique titles
        
        # 2. Semantic Ranking: Find which Wikipedia title fits the spoken context best
        context_embedding = _embedder.encode(segment_text, convert_to_tensor=True)
        title_embeddings = _embedder.encode(raw_results, convert_to_tensor=True)
        
        hits = util.semantic_search(context_embedding, title_embeddings, top_k=5)[0]
        
        candidates = []
        for hit in hits:
            title = raw_results[hit['corpus_id']]
            candidates.append({
                "title": title,
                "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                "score": hit['score']
            })
            
        return candidates
    except Exception as e:
        print(f"Agentic Search Error: {e}")
        return []

def get_wiki_page_data(title):
    try:
        page = wikipedia.page(title, auto_suggest=False)
        return {
            "title": page.title,
            "url": page.url,
            "summary": page.summary[:300] + "..."
        }
    except:
        return None