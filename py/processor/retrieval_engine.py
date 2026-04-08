import wikipedia
from sentence_transformers import SentenceTransformer, util
from ddgs import DDGS
import torch
import os
from processor.config import get_sentence_transformer_model, get_model_mode

CACHE_DIR = os.path.join(os.getcwd(), ".model_cache")

_embedder = None

def agentic_search(segment_text, entity_text, search_type="all", language="en", context_text=None, global_entity_scores=None):
    """
    Multilingual Agentic Search. Adjusts source (Wiki/News) language
    based on the detected language of the video.
    Now supports Sliding Window Context and Global Entity Importance scoring.
    """
    global _embedder
    if _embedder is None:
        try:
            model_name = get_sentence_transformer_model()
            _embedder = SentenceTransformer(model_name, cache_folder=CACHE_DIR)
        except Exception as e:
            print(f"Warning: Could not load semantic ranker: {e}. Falling back to basic search.")
            _embedder = "FAILED"

    if get_model_mode() == "english":
        language = "en"

    wikipedia.set_lang(language)

    results = []

    if search_type in ["all", "wiki"]:
        try:
            wiki_queries = [entity_text]
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
        # FEATURE 1: SLIDING WINDOW CONTEXT
        full_context = segment_text
        if context_text:
            full_context = f"{context_text} {segment_text}"
        
        context_embedding = _embedder.encode(full_context, convert_to_tensor=True)

        # FEATURE 4: IMPROVED TITLE PROCESSING (Clean prefixes and use lowercase)
        processed_titles = []
        for r in results:
            t = r['title']
            if "] " in t:
                # Remove "[EN Wiki] " or "[News] " prefix
                t = t.split("] ", 1)[1]
            processed_titles.append(t.lower())

        title_embeddings = _embedder.encode(processed_titles, convert_to_tensor=True)

        hits = util.semantic_search(context_embedding, title_embeddings, top_k=8)[0]

        candidates = []
        for hit in hits:
            res = results[hit['corpus_id']]
            semantic_score = float(hit['score'])
            
            # FEATURE 3: ENTITY-AWARE BOOSTING
            boosts = 0.0
            
            # 1. Entity Match Boost (Entity appears in result title)
            if entity_text.lower() in res['title'].lower():
                boosts += 2.0
            
            # 2. Global Importance Boost
            if global_entity_scores:
                # Normalize key search for robustness
                norm_entity = entity_text.lower().strip()
                global_score = global_entity_scores.get(norm_entity, 0.0)
                boosts += float(global_score) * 0.3
            
            # 3. Context Presence Boost
            if context_text and entity_text.lower() in context_text.lower():
                boosts += 1.5

            # FEATURE 5: OUTPUT STRUCTURE
            res['semantic_score'] = semantic_score
            res['final_score'] = semantic_score + boosts
            candidates.append(res)
        
        # Sort by final_score descending
        candidates.sort(key=lambda x: x['final_score'], reverse=True)
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

def unload_search_model():
    """
    Clears the SentenceTransformer from memory.
    """
    global _embedder
    if _embedder is not None:
        import gc
        del _embedder
        _embedder = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
