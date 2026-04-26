"""
retriever.py
------------
Given a query text (e.g. a new patient's summary), retrieves
the top-k most similar patients from ChromaDB.
"""

import ollama
import chromadb

from config import CHROMA_DIR, CHROMA_COLLECTION, EMBED_MODEL, TOP_K
from vector_store import get_chroma_client, get_or_create_collection


def retrieve_similar_patients(
    query_text: str,
    top_k: int = TOP_K,
) -> list[dict]:
    """
    Retrieve the top-k most similar patient summaries from ChromaDB.

    Parameters
    ----------
    query_text : plain-English description of the patient to look up
    top_k      : number of results to return

    Returns
    -------
    List of dicts:
        {
            "patient_id": str,
            "summary":    str,
            "distance":   float  (lower = more similar)
        }
    """
    # Embed the query
    try:
        response = ollama.embeddings(model=EMBED_MODEL, prompt=query_text)
        query_emb = response["embedding"]
    except Exception as e:
        raise RuntimeError(
            f"Could not embed query with '{EMBED_MODEL}'.\n"
            f"Make sure Ollama is running and the model is pulled:\n"
            f"  ollama pull {EMBED_MODEL}\n"
            f"Original error: {e}"
        )

    # Query ChromaDB
    client     = get_chroma_client()
    collection = get_or_create_collection(client)

    if collection.count() == 0:
        raise RuntimeError(
            "ChromaDB collection is empty.\n"
            "Please run:  cd src && python vector_store.py"
        )

    results = collection.query(
        query_embeddings=[query_emb],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        output.append({
            "patient_id": meta.get("patient_id", "unknown"),
            "summary":    doc,
            "distance":   round(dist, 4),
        })

    return output


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_query = (
        "55-year-old female with diabetes, high HbA1c of 9.2%, "
        "BMI 32, and multiple encounter history."
    )
    results = retrieve_similar_patients(test_query, top_k=3)
    print(f"\nTop {len(results)} similar patients:\n")
    for i, r in enumerate(results, 1):
        print(f"[{i}] Patient ID: {r['patient_id']}")
        print(f"    Distance : {r['distance']}")
        print(f"    Summary  : {r['summary'][:200]}…\n")
