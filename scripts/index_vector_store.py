from pathlib import Path
import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# =========================
# Configuration
# =========================

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / ".vector"
INDEX_FILE = OUTPUT_DIR / "index.faiss"
METADATA_FILE = OUTPUT_DIR / "metadata.pkl"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Files that are useful for code/documentation search
ALLOWED_EXTENSIONS = {
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".css",
    ".json",
    ".md",
    ".yml",
    ".yaml",
}

# Directories that should never be indexed
IGNORED_DIRECTORIES = {
    "node_modules",
    ".next",
    ".git",
    ".vector",
    "dist",
    "build",
    "coverage",
}


# =========================
# File collection
# =========================

def should_ignore(path: Path) -> bool:
    return any(part in IGNORED_DIRECTORIES for part in path.parts)


def collect_files() -> list[Path]:
    files = []

    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue

        if should_ignore(path):
            continue

        if path.suffix.lower() not in ALLOWED_EXTENSIONS:
            continue

        files.append(path)

    return sorted(files)


# =========================
# Text chunking
# =========================

def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200):
    """
    Split text into overlapping chunks.

    This gives the RAG system smaller pieces of code/docs
    to search instead of embedding entire files.
    """

    if not text.strip():
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


# =========================
# Main indexing process
# =========================

def main():
    print("🔎 Collecting project files...")

    files = collect_files()

    if not files:
        print("❌ No supported files found.")
        return

    print(f"📁 Found {len(files)} files.")

    documents = []
    metadata = []

    for file_path in files:
        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            print(f"⚠️ Skipping non-UTF8 file: {file_path}")
            continue
        except Exception as exc:
            print(f"⚠️ Could not read {file_path}: {exc}")
            continue

        relative_path = file_path.relative_to(ROOT)

        chunks = chunk_text(text)

        for chunk_number, chunk in enumerate(chunks):
            documents.append(chunk)

            metadata.append(
                {
                    "file": str(relative_path),
                    "chunk": chunk_number,
                    "text": chunk,
                }
            )

    if not documents:
        print("❌ No text could be indexed.")
        return

    print(f"🧩 Created {len(documents)} chunks.")

    print(f"🤖 Loading embedding model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    print("🧠 Creating embeddings...")

    embeddings = model.encode(
        documents,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(embeddings, dtype="float32")

    dimension = embeddings.shape[1]

    print(f"📐 Embedding dimension: {dimension}")

    # Inner Product + normalized vectors ≈ cosine similarity
    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("💾 Saving FAISS index...")

    faiss.write_index(index, str(INDEX_FILE))

    with METADATA_FILE.open("wb") as file:
        pickle.dump(metadata, file)

    print()
    print("✅ Vector store created successfully!")
    print(f"   Index:    {INDEX_FILE}")
    print(f"   Metadata: {METADATA_FILE}")
    print(f"   Vectors:  {index.ntotal}")


if __name__ == "__main__":
    main()