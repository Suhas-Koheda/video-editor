import wikipedia
from sentence_transformers import SentenceTransformer, util
from ddgs import DDGS
import torch
import os
from processor.config import get_sentence_transformer_model, get_model_mode

CACHE_DIR = os.path.join(os.getcwd(), ".model_cache")

_embedder = None

def agentic_search(segment_text, entity_text, search_type="all", language="en"):
    """
    Multilingual Agentic Search. Adjusts source (Wiki/News) language
    based on the detected language of the video.
    """
    global _embedder
    if _embedder is None:
        try:
            model_name = get_sentence_transformer_model()
            _embedder = SentenceTransformer(model_name, cache_folder=CACHE_DIR)
        except Exception as e:
            print(f"Warning: Could not load semantic ranker: {e}. Falling back to basic search.")
            _embedder = "FAILED"

    # Respect chosen mode
    if get_model_mode() == "english":
        language = "en"
    
    wikipedia.set_lang(language)

    results = []
    
    if search_type in ["all", "wiki"]:
        try:
            wiki_queries = [entity_text]
            if language in ["en", "hi", "te", "ta", "mr", "bn"]:
                if "youth" in segment_text.lower() or "population" in segment_text.lower():
                     wiki_queries.append("Demographics of India")

            for q in wiki_queries:
                search_results = wikipedia.search(q, results=3)
                for title in search_results:
                    results.append({
                        "title": f"[{language.upper()} Wiki] {title}",
                        "url": f"https://{language}.wikipedia.org/wiki/{title.replace(' ', '_')}",
                        "source": "Wikipedia"
                    })
        except: pass

    if search_type in ["all", "news"]:
        try:
            with DDGS() as ddgs:
                news_query = f"{entity_text} news" 
                news_results = list(ddgs.text(news_query, max_results=5))
                for r in news_results:
                    results.append({
                        "title": f"[News] {r['title']}",
                        "url": r['href'],
                        "source": "News"
                    })
        except: pass

    if not results: return []
    if _embedder == "FAILED": return results
    
    try:
        context_embedding = _embedder.encode(segment_text, convert_to_tensor=True)
        titles = [r['title'] for r in results]
        title_embeddings = _embedder.encode(titles, convert_to_tensor=True)
        
        hits = util.semantic_search(context_embedding, title_embeddings, top_k=8)[0]
        
        candidates = []
        for hit in hits:
            res = results[hit['corpus_id']]
            res['score'] = hit['score']
            candidates.append(res)
        return candidates
    except Exception as e:
        print(f"Semantic ranking failed: {e}")
        return results 

def get_wiki_page_data(title):
    try:
        clean_title = title.split("] ")[-1] if "] " in title else title
        page = wikipedia.page(clean_title, auto_suggest=False)
        return {"title": page.title, "url": page.url}
    except:
        return None

def search_wikipedia_candidates(query, language="en"):
    """
    Search wikipedia for candidates.
    """
    return agentic_search("", query, search_type="wiki", language=language)