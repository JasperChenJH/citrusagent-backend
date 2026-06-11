from src.citrus_agent.vectorstores.qdrant import QdrantStore


class FakeQdrantClient:
    """假的 Qdrant client，只记录 delete 调用参数。"""

    def __init__(self) -> None:
        self.collection_name = ""
        self.points_selector = None

    def delete(self, collection_name, points_selector) -> None:
        self.collection_name = collection_name
        self.points_selector = points_selector


def build_store_with_fake_client() -> tuple[QdrantStore, FakeQdrantClient]:
    """构造不连接真实 Qdrant 的 QdrantStore。"""

    fake_client = FakeQdrantClient()
    store = QdrantStore.__new__(QdrantStore)
    store.collection_name = "orange_knowledge"
    store.client = fake_client
    store.ensure_collection = lambda: None
    return store, fake_client


def test_delete_by_kb_id_builds_kb_filter() -> None:
    """删除知识库向量时，必须按 payload.kb_id 过滤。"""

    store, fake_client = build_store_with_fake_client()

    store.delete_by_kb_id(kb_id=3)

    assert fake_client.collection_name == "orange_knowledge"
    condition = fake_client.points_selector.filter.must[0]
    assert condition.key == "kb_id"
    assert condition.match.value == 3
