import os
import pickle
from typing import List, Dict

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from groq import Groq

# Configure constants
INDEX_PATH = "faiss.index"
DOCS_PATH = "docs.pkl"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Initialize Groq client (keep existing usage; recommend using env var in production)
client = Groq(api_key=os.getenv("GROQ_API_KEY", "your api key"))


def extract_text_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            # Best-effort: skip pages that error
            pages.append("")
    return "\n".join(pages)


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    if not text:
        return []
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if c]


def load_and_chunk_documents(path: str = ".", chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
    """Load .txt and .pdf files from `path` (file or directory) and return list of chunks.

    Each document chunk is a dict: {"text": ..., "source": "path"}
    """
    files = []
    if os.path.isfile(path):
        files = [path]
    elif os.path.isdir(path) or path in (".", None):
        files = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith((".txt", ".pdf"))]
    else:
        # treat as single path fallback
        files = [path]

    documents = []
    for f in files:
        try:
            if f.lower().endswith(".pdf"):
                text = extract_text_from_pdf(f)
            else:
                with open(f, "r", encoding="utf-8") as fh:
                    text = fh.read()
        except Exception:
            continue

        for chunk in chunk_text(text, chunk_size=chunk_size, overlap=overlap):
            documents.append({"text": chunk, "source": os.path.basename(f)})

    return documents


def build_faiss_index(documents: List[Dict], model_name: str = EMBEDDING_MODEL, index_path: str = INDEX_PATH, docs_path: str = DOCS_PATH):
    model = SentenceTransformer(model_name)
    texts = [d["text"] for d in documents]

    if len(texts) == 0:
        dim = model.get_sentence_embedding_dimension()
        index = faiss.IndexFlatIP(dim)
        return index, model, documents

    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    # Normalize embeddings for cosine similarity with IndexFlatIP
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Save index and documents for reuse
    try:
        faiss.write_index(index, index_path)
        with open(docs_path, "wb") as fh:
            pickle.dump(documents, fh)
    except Exception:
        pass

    return index, model, documents


def load_faiss(index_path: str = INDEX_PATH, docs_path: str = DOCS_PATH):
    if not os.path.exists(index_path) or not os.path.exists(docs_path):
        return None, None
    try:
        index = faiss.read_index(index_path)
        with open(docs_path, "rb") as fh:
            documents = pickle.load(fh)
        model = SentenceTransformer(EMBEDDING_MODEL)
        return index, model, documents
    except Exception:
        return None, None


def retrieve_similar_chunks(query: str, index, model, documents: List[Dict], k: int = 3):
    if index is None or model is None or len(documents) == 0:
        return []

    q_emb = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k)

    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(documents):
            continue
        results.append({"score": float(dist), "text": documents[idx]["text"], "source": documents[idx].get("source")})
    return results


def build_rag_prompt(question: str, retrieved_chunks: List[Dict]):
    context = "\n\n".join([f"Source: {c.get('source', '')}\n{c['text']}" for c in retrieved_chunks])
    prompt = f"""
You are a Library Management Assistant.

Use ONLY the following library information to answer the question. If the information is not in the provided context, say you don't have that information.

Library Information:
{context}

Question:
{question}

Answer:
"""
    return prompt


def main():
    print("📚 Library RAG Chatbot - Initializing...\n")

    # Try loading existing index
    index_model_docs = load_faiss()
    if index_model_docs[0] is not None:
        index, model, documents = index_model_docs
        print(f"✅ Loaded FAISS index with {len(documents)} chunks\n")
    else:
        # Scan current directory for .txt and .pdf files
        documents = load_and_chunk_documents(".")
        print(f"✅ Loaded {len(documents)} document chunks\n")
        index, model, documents = build_faiss_index(documents)

    print("\n" + "="*70)
    print("📚 Library RAG Chatbot Started!")
    print("="*70)
    print("Type 'quit' to exit\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ("quit", "exit"):
                print("\nChatbot: Goodbye!")
                break
            if not user_input:
                continue

            print("\n🔎 Searching knowledge base...")
            retrieved = retrieve_similar_chunks(user_input, index, model, documents, k=3)
            rag_prompt = build_rag_prompt(user_input, retrieved)

            print("⏳ Generating response...\n")
            completion = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[{"role": "user", "content": rag_prompt}],
            )
            answer = completion.choices[0].message.content
            print(f"Chatbot: {answer}\n")

        except KeyboardInterrupt:
            print("\n\nChatbot: Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}\n")


if __name__ == "__main__":

    main()
