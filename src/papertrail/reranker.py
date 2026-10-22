from sentence_transformers import CrossEncoder

RERANKER_MODEL_NAME="cross-encoder/ms-marco-MiniLM-L-6-v2"

def load_reranker()-> CrossEncoder:
    return CrossEncoder(RERANKER_MODEL_NAME)

def rerank(
        query:str,
        candidates:list[dict],
        model:CrossEncoder,
        top_k:int=5,
)-> list[dict]:
    if not query.strip():
        raise ValueError("Query cannot be empty")
    if top_k<=0:
        raise ValueError("top_k must be positive.")

    if not candidates:
        return []
    pairs=[
        (query,candidate["text"])
        for candidate in candidates
    ]

    scores=model.predict(
        pairs,
        show_progress_bar=False
    )

    results=[]

    for candidate,score in zip(candidates,scores):
        item=candidate.copy()
        item["rerank_score"]=float(score)
        results.append(item)

    results.sort(
        key=lambda item: item["rerank_score"],
        reverse=True,
    )

    return results[:top_k]