from src.citrus_agent.vectorstores.qdrant import QdrantStore
from src.citrus_agent.pojo.knowledge import ChunkPayload, KnowledgeChunk, SparseEmbedding


class FakeQdrantClient:
    """假的 Qdrant client，只记录 delete 调用参数。"""

    def __init__(self) -> None:
        self.collection_name = ""
        self.points_selector = None
        self.points = []

    def delete(self, collection_name, points_selector) -> None:
        self.collection_name = collection_name
        self.points_selector = points_selector

    def upsert(self, collection_name, points) -> None:
        self.collection_name = collection_name
        self.points.extend(points)


def build_store_with_fake_client() -> tuple[QdrantStore, FakeQdrantClient]:
    """构造不连接真实 Qdrant 的 QdrantStore。"""

    fake_client = FakeQdrantClient()
    store = QdrantStore.__new__(QdrantStore)
    store.collection_name = "orange_knowledge"
    store.client = fake_client
    store.ensure_collection = lambda: None
    store.ensure_hybrid_collection = lambda: None
    store.dense_vector_name = "dense"
    store.sparse_vector_name = "sparse"
    return store, fake_client


def build_store_with_collection_flags() -> tuple[QdrantStore, FakeQdrantClient, dict[str, bool]]:
    """构造可以记录 collection 创建入口的 QdrantStore。"""

    store, fake_client = build_store_with_fake_client()
    flags = {"dense": False, "hybrid": False}
    store.ensure_collection = lambda: flags.__setitem__("dense", True)
    store.ensure_hybrid_collection = lambda: flags.__setitem__("hybrid", True)
    return store, fake_client, flags


def test_delete_by_kb_id_builds_kb_filter() -> None:
    """删除知识库向量时，必须按 payload.kb_id 过滤。"""

    store, fake_client = build_store_with_fake_client()

    store.delete_by_kb_id(kb_id=3)

    assert fake_client.collection_name == "orange_knowledge"
    condition = fake_client.points_selector.filter.must[0]
    assert condition.key == "kb_id"
    assert condition.match.value == 3


def test_delete_by_document_id_from_hybrid_uses_hybrid_collection() -> None:
    """hybrid 删除旧 chunk 时不能创建 dense-only collection。"""

    store, fake_client, flags = build_store_with_collection_flags()

    store.delete_by_document_id_from_hybrid(document_id=12)

    assert flags["hybrid"] is True
    assert flags["dense"] is False
    condition = fake_client.points_selector.filter.must[0]
    assert condition.key == "document_id"
    assert condition.match.value == 12


def test_delete_by_kb_id_from_hybrid_uses_hybrid_collection() -> None:
    """hybrid 删除知识库时不能创建 dense-only collection。"""

    store, fake_client, flags = build_store_with_collection_flags()

    store.delete_by_kb_id_from_hybrid(kb_id=3)

    assert flags["hybrid"] is True
    assert flags["dense"] is False
    condition = fake_client.points_selector.filter.must[0]
    assert condition.key == "kb_id"
    assert condition.match.value == 3


def test_upsert_hybrid_chunks_writes_dense_sparse_and_payload() -> None:
    """hybrid 入库时必须同时写入 dense、sparse 和 payload。"""

    store, fake_client = build_store_with_fake_client()
    chunk = KnowledgeChunk(
        point_id="point-1",
        text="砂糖橘溃疡病应加强排水。",
        vector=[],
        dense_vector=[1.0, 0.0, 0.0, 0.0],
        sparse_vector=SparseEmbedding(indices=[10, 20], values=[0.7, 0.3]),
        payload=ChunkPayload(
            chunk_id="chunk-1",
            kb_id=3,
            document_id=12,
            file_name="砂糖橘溃疡病.md",
            page=None,
            chunk_index=0,
            text="砂糖橘溃疡病应加强排水。",
        ),
    )

    store.upsert_hybrid_chunks([chunk])

    assert fake_client.collection_name == "orange_knowledge"
    point = fake_client.points[0]
    assert point.vector["dense"] == [1.0, 0.0, 0.0, 0.0]
    assert point.vector["sparse"].indices == [10, 20]
    assert point.vector["sparse"].values == [0.7, 0.3]
    assert point.payload["document_id"] == 12
    assert point.payload["text"] == "砂糖橘溃疡病应加强排水。"
