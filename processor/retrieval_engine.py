import wikipedia
from sentence_transformers import SentenceTransformer, util
from duckduckgo_search import DDGS
import torch

_embedder = None

def agentic_search(segment_text, entity_text, search_type="all", language="en"):
    """
    Multilingual Agentic Search. Adjusts source (Wiki/News) language
    based on the detected language of the video.
    """
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    # Set Wikipedia language (e.g., 'hi' for Hindi, 'te' for Telugu)
    try:
        wikipedia.set_lang(language)
    except:
        wikipedia.set_lang("en")

    results = []
    
    # 1. Wikipedia Search
    if search_type in ["all", "wiki"]:
        try:
            wiki_queries = [entity_text]
            # Contextual injection for Indian context
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

    # 2. News Search (Multilingual via DuckDuckGo)
    if search_type in ["all", "news"]:
        try:
            with DDGS() as ddgs:
                # Search for original query + news site hints in local language
                news_query = f"{entity_text} news" 
                news_results = list(ddgs.text(news_query, max_results=5))
                for r in news_results:
                    results.append({
                        "title": f"[News] {r['title']}",
                        "url": r['href'],
                        "source": "News"
                    })
        except: pass

    # 3. Semantic Ranking
    if not results: return []
    
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
    except:
        return results 

def get_wiki_page_data(title):
    try:
        # Title might have prefix like "[HI Wiki]"
        clean_title = title.split("] ")[-1] if "] " in title else title
        page = wikipedia.page(clean_title, auto_suggest=False)
        return {"title": page.title, "url": page.url}
    except:
        return None

def search_wikipedia_candidates(query, language="en"):
    return agentic_search("", query, search_type="wiki", language=language)