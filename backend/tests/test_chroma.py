import chromadb


def test_chroma_persistence_and_query(tmp_path) -> None:
    database_path = tmp_path / "chroma"

    first_client = chromadb.PersistentClient(
        path=str(database_path),
    )
    first_collection = first_client.get_or_create_collection(
        name="test_collection",
        embedding_function=None,
    )

    first_collection.add(
        ids=["document_1"],
        documents=["Python programming document"],
        embeddings=[[1.0, 0.0, 0.0]],
    )

    # Create another client to simulate reopening the database.
    second_client = chromadb.PersistentClient(
        path=str(database_path),
    )
    second_collection = second_client.get_collection(
        name="test_collection",
        embedding_function=None,
    )

    assert second_collection.count() == 1

    result = second_collection.query(
        query_embeddings=[[0.9, 0.1, 0.0]],
        n_results=1,
    )

    assert result["ids"][0][0] == "document_1"
