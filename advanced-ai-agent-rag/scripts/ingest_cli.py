"""
CLI to bulk-ingest every supported document in a directory.

Usage:
    python scripts/ingest_cli.py ./data/documents
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion import ingest_directory  # noqa: E402
from src.vectorstore import get_vectorstore  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Bulk-ingest documents into the vector store.")
    parser.add_argument("directory", help="Directory containing .pdf/.txt/.md/.docx files")
    args = parser.parse_args()

    directory = Path(args.directory)
    if not directory.is_dir():
        print(f"Not a directory: {directory}")
        sys.exit(1)

    print(f"Scanning {directory} ...")
    chunks = ingest_directory(directory)
    print(f"Found {len(chunks)} chunks. Embedding + indexing...")

    store = get_vectorstore()
    added = store.add_chunks(chunks)
    print(f"Done. {added} chunks indexed. Total in store: {store.count()}")


if __name__ == "__main__":
    main()
