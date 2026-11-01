"""
RAG Chatbot Starter
-------------------
Chat with your own documents using Retrieval Augmented Generation (RAG):

  1. Ingest: split .md / .txt files from ./data into chunks and store their
     OpenAI embeddings in a local ChromaDB vector database.
  2. Chat:   for every question, retrieve the most relevant chunks and let the
     LLM answer ONLY from that context, with sources.

Usage:
    python app.py ingest     # (re)build the knowledge base from ./data
    python app.py            # start chatting

Author: Adarsh Shukla (https://github.com/adarsh0008)
"""
import glob
import os
import sys

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DATA_DIR = os.getenv("DATA_DIR", "data")
CHROMA_DIR = os.getenv("CHROMA_DIR", ".chroma")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
TOP_K = int(os.getenv("TOP_K", "4"))

SYSTEM_PROMPT = (
    "You are a helpful assistant for a business. Answer the user's question "
    "using ONLY the context provided. If the answer is not in the context, say "
    "you don't know and suggest contacting the team. Be concise and friendly."
)

llm = OpenAI()  # reads OPENAI_API_KEY from the environment
embedder = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.environ["OPENAI_API_KEY"], model_name=EMBED_MODEL
)
db = chromadb.PersistentClient(path=CHROMA_DIR)
collection = db.get_or_create_collection("knowledge_base", embedding_function=embedder)


def chunk_text(text: str, size: int = 800, overlap: int = 150) -> list[str]:
    """Split text into overlapping chunks, preferring paragraph boundaries."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= size:
            current = f"{current}\n\n{para}" if current else para
            continue
        if current:
            chunks.append(current)
            current = current[-overlap:] + "\n\n" + para if overlap else para
        else:
            current = para
        # very long paragraphs are split hard
        while len(current) > size:
            chunks.append(current[:size])
            current = current[size - overlap:]
    if current:
        chunks.append(current)
    return chunks


def ingest() -> None:
    files = sorted(glob.glob(os.path.join(DATA_DIR, "**", "*.md"), recursive=True)
                   + glob.glob(os.path.join(DATA_DIR, "**", "*.txt"), recursive=True))
    if not files:
        print(f"No .md or .txt files found in '{DATA_DIR}/'.")
        return

    total = 0
    for path in files:
        with open(path, encoding="utf-8") as f:
            chunks = chunk_text(f.read())
        if not chunks:
            continue
        collection.upsert(
            ids=[f"{path}:{i}" for i in range(len(chunks))],
            documents=chunks,
            metadatas=[{"source": path, "chunk": i} for i in range(len(chunks))],
        )
        total += len(chunks)
        print(f"  indexed {len(chunks):>3} chunks from {path}")
    print(f"Done. {total} chunks in the knowledge base.")


def retrieve(question: str, k: int = TOP_K) -> list[tuple[str, str]]:
    res = collection.query(query_texts=[question], n_results=k)
    docs = res["documents"][0] if res["documents"] else []
    metas = res["metadatas"][0] if res["metadatas"] else []
    return [(doc, meta["source"]) for doc, meta in zip(docs, metas)]


def answer(question: str, history: list[dict]) -> tuple[str, list[str]]:
    hits = retrieve(question)
    context = "\n\n---\n\n".join(f"[Source: {src}]\n{doc}" for doc, src in hits)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history[-6:],  # keep the last few turns for follow-up questions
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    resp = llm.chat.completions.create(model=CHAT_MODEL, messages=messages, temperature=0.2)
    sources = sorted({src for _, src in hits})
    return resp.choices[0].message.content, sources


def chat() -> None:
    if collection.count() == 0:
        print("Knowledge base is empty. Run:  python app.py ingest")
        return
    print("RAG chatbot ready. Ask a question (type 'exit' to quit).\n")
    history: list[dict] = []
    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        reply, sources = answer(question, history)
        print(f"\nBot: {reply}")
        if sources:
            print("Sources: " + ", ".join(sources))
        print()
        history += [{"role": "user", "content": question}, {"role": "assistant", "content": reply}]


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ingest":
        ingest()
    else:
        chat()
