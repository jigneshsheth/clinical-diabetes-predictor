"""
vector_store.py
---------------
Embeds patient summaries using the nomic-embed-text Ollama model
and stores them in a local ChromaDB collection.

Run this after building summaries:
    cd src && python vector_store.py
"""

import ollama
import chromadb
from chromadb.config import Settings

from config import CHROMA_DIR, CHROMA_COLLECTION, EMBED_MODEL
from load_data             import load_all_csv
from preprocess            import merge_all
from build_rag_documents   import build_all_summaries


def get_chroma_client() -> chromadb.Client:
    """Return a persistent ChromaDB client pointing to ./chroma_db/."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client


def get_or_create_collection(client: chromadb.Client):
    """Return (or create) the patient summaries collection."""
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"description": "Patient clinical summaries for RAG"},
    )
    return collection


def embed_text(text: str) -> list[float]:
    """
    Generate an embedding for a single text string using Ollama.
    Make sure you have pulled the model first:
        ollama pull nomic-embed-text
    """
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return response["embedding"]


def build_and_store_embeddings(summaries_df, batch_size: int = 50):
    """
    Embed all patient summaries and upsert them into ChromaDB.

    Parameters
    ----------
    summaries_df : pd.DataFrame with columns PATIENT, SUMMARY
    batch_size   : how many documents to upsert at once
    """
    client     = get_chroma_client()
    collection = get_or_create_collection(client)

    # Check if collection is already populated
    existing = collection.count()
    if existing > 0:
        print(f"[vector_store] Collection already has {existing} documents.")
        ans = input("Re-embed? This will overwrite existing data. [y/N] ").strip().lower()
        if ans != "y":
            print("[vector_store] Skipping embedding - using existing data.")
            return collection

        # Delete existing and recreate
        client.delete_collection(CHROMA_COLLECTION)
        collection = get_or_create_collection(client)

    total = len(summaries_df)
    print(f"[vector_store] Embedding {total:,} patient summaries with '{EMBED_MODEL}' ...")
    print("  (This may take several minutes on first run)")

    ids       = []
    embeddings = []
    documents  = []
    metadatas  = []

    for i, (_, row) in enumerate(summaries_df.iterrows()):
        patient_id = str(row["PATIENT"])
        summary    = str(row["SUMMARY"])

        # Generate embedding
        try:
            emb = embed_text(summary)
        except Exception as e:
            print(f"  WARNING: Could not embed patient {patient_id}: {e}")
            continue

        ids.append(patient_id)
        embeddings.append(emb)
        documents.append(summary)
        metadatas.append({"patient_id": patient_id})

        # Upsert in batches
        if len(ids) == batch_size or i == total - 1:
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            print(f"  Stored {min(i+1, total)}/{total} patients ...", end="\r")
            ids, embeddings, documents, metadatas = [], [], [], []

    print(f"\n[vector_store] Done. Collection now has {collection.count()} documents.")
    return collection


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    data      = load_all_csv()
    merged    = merge_all(data)
    summaries = build_all_summaries(merged)
    build_and_store_embeddings(summaries)
