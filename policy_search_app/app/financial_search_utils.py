import joblib
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Optional, Dict

FIN_VECTORIZER_PATH = "app/models/financial_news_vectorizer.pkl"
FIN_MATRIX_PATH = "app/models/financial_news_tfidf_matrix.pkl"

try:
    fin_vectorizer = joblib.load(FIN_VECTORIZER_PATH)
except Exception:
    fin_vectorizer = None

try:
    fin_data = joblib.load(FIN_MATRIX_PATH)
    fin_tfidf_matrix = fin_data.get("matrix")
    fin_df = fin_data.get("df")
except Exception:
    fin_tfidf_matrix = None
    fin_df = None

def search_financial_news(query: str, top_k: int = 3) -> List[Dict]:
    """
    Semantic search over financial news dataset using TF-IDF & cosine similarity.
    Returns top_k results with columns:
    Date, Headline, Market_Event, Sector, Sentiment, Impact_Level, Related_Company, Similarity
    """
    if not fin_vectorizer or fin_tfidf_matrix is None or fin_df is None or fin_df.empty:
        return []
    try:
        query_vec = fin_vectorizer.transform([str(query).lower()])
        sims = cosine_similarity(query_vec, fin_tfidf_matrix).flatten()
        df_copy = fin_df.copy()
        df_copy["Similarity"] = sims
        cols = ["Date","Headline","Market_Event","Sector","Sentiment","Impact_Level","Related_Company","Similarity"]
        top = df_copy.sort_values("Similarity", ascending=False).head(int(top_k))
        results = []
        for _, row in top.iterrows():
            results.append({
                "Date": row.get("Date", ""),
                "Headline": row.get("Headline", ""),
                "Market_Event": row.get("Market_Event", ""),
                "Sector": row.get("Sector", ""),
                "Sentiment": row.get("Sentiment", ""),
                "Impact_Level": row.get("Impact_Level", ""),
                "Related_Company": row.get("Related_Company", ""),
                "Similarity": float(row.get("Similarity", 0.0)),
            })
        return results
    except Exception:
        return []
