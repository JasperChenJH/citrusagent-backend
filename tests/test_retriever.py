from src.citrus_agent.rag.retriever import CitrusRetriever
from src.citrus_agent.vectorstores.embeddings import FixedEmbeddingProvider


class FakeSearchStore:
    def __init__(self) -> None:
        self.last_filters = None

    def search(self, query_vector, top_k=30, filters=None):
        self.last_filters = filters
        return [
            {
                "id": "1",
                "score": 0.70,
                "payload": {
                    "chunk_id": "chunk-1",
                    "kb_id": 3,
                    "document_id": 12,
                    "file_name": "砂糖橘溃疡病.md",
                    "page": None,
                    "chunk_index": 0,
                    "text": "砂糖橘溃疡病应结合雨季排水、清园和药剂防治。",
                },
            }
        ]


def test_retriever_reads_text_from_minimal_payload_and_passes_kb_filter() -> None:
    store = FakeSearchStore()
    retriever = CitrusRetriever(
        embedding_provider=FixedEmbeddingProvider(vector_size=4),
        qdrant_store=store,
        top_k=10,
    )

    results = retriever.search("砂糖橘溃疡病怎么防", filters={"kb_id": 3})

    assert store.last_filters == {"kb_id": 3}
    assert results[0].chunk_id == "chunk-1"
    assert results[0].content == "砂糖橘溃疡病应结合雨季排水、清园和药剂防治。"
    assert results[0].score == 0.70
    assert results[0].payload["document_id"] == 12
