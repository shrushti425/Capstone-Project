from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP
from pathlib import Path
from utils.pdf_loader import load_pdf
from utils.docx_loader import load_docx
from utils.ocr_loader import load_image_ocr
from utils.text_cleaner import clean_text
from embeddings import embed_texts
from vector_store import add_documents


def chunk_text(text: str) -> list:
    """
    Split a long text into overlapping chunks.

    Why RecursiveCharacterTextSplitter?
    It tries to split on paragraphs first, then sentences,
    then words — so chunks stay semantically meaningful.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,       # Max characters per chunk
        chunk_overlap=CHUNK_OVERLAP, # Overlap to avoid losing context
        separators=["\n\n", "\n", ". ", " ", ""]  # Try these in order
    )

    chunks = splitter.split_text(text)
    return chunks

def load_document(file_path: str) -> str:
    """Route file to the correct loader based on extension."""
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        return load_pdf(file_path)
    elif ext == ".docx":
        return load_docx(file_path)
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    elif ext in {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}:
        return load_image_ocr(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def ingest_document(file_path: str) -> int:
    """
    Full ingestion pipeline for a single document.

    Returns the number of chunks stored.
    """
    print(f"Ingesting: {file_path}")

    # 1. Load raw text from file
    raw_text = load_document(file_path)

    if not raw_text.strip():
        print("Warning: No text extracted from document.")
        return 0

    # 2. Clean the text
    cleaned = clean_text(raw_text)

    # 3. Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks = splitter.split_text(cleaned)
    print(f"Split into {len(chunks)} chunks")

    # 4. Generate embeddings for all chunks
    print("Generating embeddings...")
    embeddings = embed_texts(chunks)

    # 5. Store in ChromaDB
    add_documents(chunks, embeddings)
    print(f"Done! {len(chunks)} chunks stored.")

    return len(chunks)
