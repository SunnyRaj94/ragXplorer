import os
from pathlib import Path
from typing import List, Dict

# import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
import docx2txt


def read_text_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def read_pdf_file(file_path: str) -> str:
    reader = PdfReader(file_path)
    return "\n".join(
        page.extract_text() for page in reader.pages if page.extract_text()
    )


def load_documents_from_folder(
    folder_path: str, extensions: List[str] = [".txt", ".md", ".pdf"]
) -> List[Dict]:
    """Loads text content from all supported files in a folder."""
    docs = []
    for ext in extensions:
        for file in Path(folder_path).rglob(f"*{ext}"):
            if ext == ".pdf":
                content = read_text_file(str(file))
            else:
                content = read_pdf_file(str(file))
            docs.append(
                {"text": content, "metadata": {"source": str(file), "extension": ext}}
            )
    return docs


def scrape_url_firecrawl(url: str, api_key: str) -> Dict:
    """
    Placeholder for FireCrawl API-based web scraping.
    You need an actual API key from FireCrawl.ai.
    """
    endpoint = "https://api.firecrawl.dev/v1/scrape"
    headers = {"x-api-key": api_key, "Content-Type": "application/json"}
    payload = {"url": url, "options": {"extract_html": False}}
    try:
        response = requests.post(endpoint, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return {"text": data.get("text", ""), "metadata": {"source": url}}
    except Exception as e:
        print(f"[Error] Failed to scrape {url}: {e}")
        return {}


def extract_text_from_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def extract_text_from_docx(file_path: str) -> str:
    return docx2txt.process(file_path)


def extract_text_from_url(url: str) -> str:
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")

        # Remove scripts/styles
        for tag in soup(["script", "style"]):
            tag.decompose()

        return soup.get_text(separator="\n", strip=True)
    except Exception as e:
        raise ValueError(f"Failed to extract text from URL: {url} — {e}")


def extract_text(source: str) -> str:
    source_lower = source.lower()

    if source_lower.startswith("http://") or source_lower.startswith("https://"):
        return extract_text_from_url(source)
    elif source_lower.endswith(".pdf"):
        return extract_text_from_pdf(source)
    elif source_lower.endswith(".docx"):
        return extract_text_from_docx(source)
    elif source_lower.endswith(".txt"):
        with open(source, "r", encoding="utf-8") as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file format or input source: {source}")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """
    Splits text into overlapping chunks for embedding.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def ingest_documents(
    folder_path: str, chunk_size: int = 300, overlap: int = 50
) -> List[Dict]:
    """
    Ingests all .txt and .pdf files from a folder and returns list of chunks with metadata.
    """
    docs = []
    for filename in os.listdir(folder_path):
        if not filename.endswith((".txt", ".pdf")):
            continue
        file_path = os.path.join(folder_path, filename)
        try:
            content = extract_text(file_path)
            chunks = chunk_text(content, chunk_size, overlap)
            for i, chunk in enumerate(chunks):
                docs.append(
                    {"text": chunk, "metadata": {"source": filename, "chunk_id": i}}
                )
        except Exception as e:
            print(f"❌ Failed to read {filename}: {e}")
    return docs
