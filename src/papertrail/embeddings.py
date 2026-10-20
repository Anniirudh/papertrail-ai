from sentence_transformers import SentenceTransformer

MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"

def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)

def generate_embeddings(
        text:list[str],
        model:SentenceTransformer,
):
    embeddings=model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return embeddings