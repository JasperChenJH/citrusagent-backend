from dataclasses import asdict

from src.citrus_agent.pojo.knowledge import DocumentEntity, DocumentIngestResult, SearchResult
from src.citrus_agent.rag.rag_api import RAGApi


class FakeIngestor:
    def __init__(self) -> None:
        self.last_document = None
        self.last_replace_existing = None

    def ingest_document(self, document, replace_existing=True):
        self.last_document = document
        self.last_replace_existing = replace_existing
        return DocumentIngestResult(
            document_id=document.id,
            kb_id=document.kb_id,
            success=True,
            chunk_count=2,
            error_message="",
        )


class FakeRetriever:
    def __init__(self) -> None:
        self.last_filters = None

    def search(self, query_text, top_k=None, filters=None):
        self.last_filters = filters
        self.last_top_k = top_k
        return [
            SearchResult(
                chunk_id="chunk-1",
                content="砂糖橘溃疡病应加强排水。",
                score=0.9,
                payload={"kb_id": 3, "document_id": 12},
            )
        ]


class FakeQdrantStore:
    def __init__(self) -> None:
        self.deleted_document_id = None
        self.deleted_kb_id = None

    def delete_by_document_id(self, document_id: int) -> None:
        self.deleted_document_id = document_id

    def delete_by_kb_id(self, kb_id: int) -> None:
        self.deleted_kb_id = kb_id


def build_document_entity() -> DocumentEntity:
    return DocumentEntity(
        id=12,
        kb_id=3,
        original_filename="砂糖橘溃疡病.md",
        stored_path="knowledge_raw/砂糖橘溃疡病.md",
        file_size=100,
        file_hash="hash",
        mime_type="text/markdown",
        status="pending",
    )


def test_rag_api_ingest_document_for_backend() -> None:
    ingestor = FakeIngestor()
    api = RAGApi(ingestor=ingestor, retriever=FakeRetriever(), qdrant_store=FakeQdrantStore())

    result = api.ingest_document(build_document_entity(), replace_existing=True)

    assert result.document_id == 12
    assert result.kb_id == 3
    assert result.success is True
    assert result.chunk_count == 2
    assert ingestor.last_document.id == 12
    assert ingestor.last_replace_existing is True


def test_rag_api_search_uses_kb_filter() -> None:
    retriever = FakeRetriever()
    api = RAGApi(ingestor=FakeIngestor(), retriever=retriever, qdrant_store=FakeQdrantStore())

    result = api.search(kb_id=3, query_text="砂糖橘溃疡病怎么防", top_k=5)

    assert retriever.last_filters == {"kb_id": 3}
    assert retriever.last_top_k == 5
    assert result.kb_id == 3
    assert result.query_text == "砂糖橘溃疡病怎么防"
    assert result.top_k == 5
    assert result.score == 0.9
    assert result.chunks[0].content == "砂糖橘溃疡病应加强排水。"


def test_rag_api_search_without_kb_filter() -> None:
    retriever = FakeRetriever()
    api = RAGApi(ingestor=FakeIngestor(), retriever=retriever, qdrant_store=FakeQdrantStore())

    result = api.search(query_text="砂糖橘溃疡病怎么防", top_k=5)

    assert retriever.last_filters is None
    assert retriever.last_top_k == 5
    assert result.kb_id is None
    assert result.query_text == "砂糖橘溃疡病怎么防"
    assert result.top_k == 5
    assert result.score == 0.9
    assert result.chunks[0].content == "砂糖橘溃疡病应加强排水。"


def test_rag_api_delete_document_vectors() -> None:
    store = FakeQdrantStore()
    api = RAGApi(ingestor=FakeIngestor(), retriever=FakeRetriever(), qdrant_store=store)

    assert api.delete_document_vectors(document_id=12) is True
    assert store.deleted_document_id == 12


def test_rag_api_delete_knowledge_base_vectors() -> None:
    store = FakeQdrantStore()
    api = RAGApi(ingestor=FakeIngestor(), retriever=FakeRetriever(), qdrant_store=store)

    assert api.delete_knowledge_base_vectors(kb_id=3) is True
    assert store.deleted_kb_id == 3
    assert store.deleted_document_id is None


def test_document_ingest_result_can_use_asdict() -> None:
    result = DocumentIngestResult(
        document_id=12,
        kb_id=3,
        success=True,
        chunk_count=2,
        error_message="",
    )

    assert asdict(result) == {
        "document_id": 12,
        "kb_id": 3,
        "success": True,
        "chunk_count": 2,
        "error_message": "",
    }


def test_document_search_result_can_use_asdict() -> None:
    api = RAGApi(ingestor=FakeIngestor(), retriever=FakeRetriever(), qdrant_store=FakeQdrantStore())

    result = api.search(kb_id=3, query_text="砂糖橘溃疡病怎么防", top_k=5)

    data = asdict(result)
    assert data["kb_id"] == 3
    assert data["query_text"] == "砂糖橘溃疡病怎么防"
    assert data["top_k"] == 5
    assert data["score"] == 0.9
    assert data["chunks"][0]["content"] == "砂糖橘溃疡病应加强排水。"
