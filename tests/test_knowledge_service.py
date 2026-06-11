from pathlib import Path

from src.citrus_agent.pojo.knowledge import DocumentEntity, DocumentIngestResult
from src.citrus_agent.services.knowledge_service import KnowledgeIngestor
from src.citrus_agent.vectorstores.embeddings import FixedEmbeddingProvider


class FakeQdrantStore:
    def __init__(self) -> None:
        self.deleted_document_ids: list[int] = []
        self.chunks = []

    def build_point_id(self, chunk_id: str) -> str:
        return f"point-{chunk_id}"

    def delete_by_document_id(self, document_id: int) -> None:
        self.deleted_document_ids.append(document_id)

    def upsert_chunks(self, chunks) -> None:
        self.chunks.extend(chunks)


def test_ingest_document_returns_result_and_minimal_payload(tmp_path: Path) -> None:
    file_path = tmp_path / "砂糖橘溃疡病.md"
    file_path.write_text(
        "砂糖橘溃疡病在雨季容易发生，需要清理病叶并加强排水。",
        encoding="utf-8",
    )
    document = DocumentEntity(
        id=12,
        kb_id=3,
        original_filename="砂糖橘溃疡病.md",
        stored_path=str(file_path),
        file_size=file_path.stat().st_size,
        file_hash="fake-hash",
        mime_type="text/markdown",
        status="pending",
    )
    store = FakeQdrantStore()
    ingestor = KnowledgeIngestor(
        embedding_provider=FixedEmbeddingProvider(vector_size=4),
        qdrant_store=store,
        chunk_size=20,
        chunk_overlap=5,
    )

    result = ingestor.ingest_document(document)

    assert isinstance(result, DocumentIngestResult)
    assert result.document_id == 12
    assert result.kb_id == 3
    assert result.success is True
    assert result.chunk_count >= 1
    assert result.error_message == ""
    assert store.deleted_document_ids == [12]

    payload = store.chunks[0].payload.to_dict()
    assert set(payload.keys()) == {
        "chunk_id",
        "kb_id",
        "document_id",
        "file_name",
        "page",
        "chunk_index",
        "text",
    }
    assert payload["kb_id"] == 3
    assert payload["document_id"] == 12
    assert payload["file_name"] == "砂糖橘溃疡病.md"
    assert payload["chunk_index"] == 0
    assert "砂糖橘溃疡病" in payload["text"]


def test_ingest_document_returns_failure_result() -> None:
    document = DocumentEntity(
        id=99,
        kb_id=3,
        original_filename="missing.md",
        stored_path="not-exists/missing.md",
        file_size=0,
        file_hash="fake-hash",
    )
    ingestor = KnowledgeIngestor(
        embedding_provider=FixedEmbeddingProvider(vector_size=4),
        qdrant_store=FakeQdrantStore(),
    )

    result = ingestor.ingest_document(document)

    assert isinstance(result, DocumentIngestResult)
    assert result.document_id == 99
    assert result.kb_id == 3
    assert result.success is False
    assert result.chunk_count == 0
    assert "文件不存在" in result.error_message
