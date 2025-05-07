from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_texts(
    texts: List[str], chunk_size: int = 500, chunk_overlap: int = 100
) -> List[str]:
    """
    Split long texts into smaller chunks using recursive character splitting.

    Args:
        texts (List[str]): List of input documents as strings.
        chunk_size (int): Max characters per chunk.
        chunk_overlap (int): Overlap between chunks.

    Returns:
        List[str]: List of text chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    all_chunks = []
    for doc in texts:
        chunks = splitter.split_text(doc)
        all_chunks.extend(chunks)
    return all_chunks
