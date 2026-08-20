from __future__ import annotations

import argparse
from pathlib import Path

import chromadb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = PROJECT_ROOT / "runtime" / "chroma"
COLLECTION_NAME = "week1_smoke_test"


def create_client() -> chromadb.PersistentClient:
    """Connect to the local persistent ChromaDB database."""
    DATABASE_PATH.mkdir(parents=True, exist_ok=True)

    return chromadb.PersistentClient(
        path=str(DATABASE_PATH),
    )


def seed_database() -> None:
    """Create the collection, insert vectors and run a query."""
    client = create_client()

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
    )

    # Use upsert so that running the script multiple times does not
    # create duplicate ID errors.
    collection.upsert(
        ids=[
            "python_document",
            "cooking_document",
            "travel_document",
        ],
        documents=[
            "Python is a programming language.",
            "This document contains a pasta recipe.",
            "This document describes a trip to Japan.",
        ],
        metadatas=[
            {"category": "programming"},
            {"category": "cooking"},
            {"category": "travel"},
        ],
        embeddings=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
    )

    result = collection.query(
        query_embeddings=[
            [0.95, 0.05, 0.0],
        ],
        n_results=2,
    )

    first_result_id = result["ids"][0][0]

    print(f"Database path: {DATABASE_PATH}")
    print(f"Collection name: {collection.name}")
    print(f"Number of records: {collection.count()}")
    print(f"Returned IDs: {result['ids'][0]}")
    print(f"Best match: {first_result_id}")

    if first_result_id != "python_document":
        raise RuntimeError(
            "ChromaDB query returned an unexpected first result."
        )

    print("ChromaDB seed and query check passed.")


def verify_persistence() -> None:
    """Open the existing database without inserting data."""
    client = create_client()

    # get_collection will fail if the previous process did not
    # persist the collection.
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
    )

    if collection.count() != 3:
        raise RuntimeError(
            f"Expected 3 records, but found {collection.count()}."
        )

    stored_record = collection.get(
        ids=["python_document"],
    )

    result = collection.query(
        query_embeddings=[
            [0.95, 0.05, 0.0],
        ],
        n_results=1,
    )

    if stored_record["ids"] != ["python_document"]:
        raise RuntimeError("The persisted record was not found.")

    if result["ids"][0][0] != "python_document":
        raise RuntimeError("The persisted vector query failed.")

    print(f"Database path: {DATABASE_PATH}")
    print(f"Existing collection: {collection.name}")
    print(f"Persisted record count: {collection.count()}")
    print(f"Loaded record: {stored_record['ids'][0]}")
    print(f"Query result: {result['ids'][0][0]}")
    print("ChromaDB persistence check passed.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Week 1 ChromaDB validation.",
    )
    parser.add_argument(
        "command",
        choices=["seed", "verify"],
        help="Seed the database or verify persisted data.",
    )

    args = parser.parse_args()

    if args.command == "seed":
        seed_database()
    else:
        verify_persistence()


if __name__ == "__main__":
    main()
