from pathlib import Path

from src.citrus_agent.services.document_parser import DocumentParser


def test_parse_markdown_file(tmp_path: Path) -> None:
    file_path = tmp_path / "砂糖橘溃疡病.md"
    file_path.write_text("# 砂糖橘溃疡病\n雨季应加强排水和病叶清理。", encoding="utf-8")

    document = DocumentParser().parse_file(file_path)

    assert document.source_type == "md"
    assert document.title == "砂糖橘溃疡病"
    assert "雨季应加强排水" in document.content
    assert document.pages


def test_parse_csv_file(tmp_path: Path) -> None:
    file_path = tmp_path / "施肥方案.csv"
    file_path.write_text("品种,时期,措施\n沃柑,膨果期,补充钾肥\n", encoding="utf-8-sig")

    document = DocumentParser().parse_file(file_path)

    assert document.source_type == "csv"
    assert "沃柑 | 膨果期 | 补充钾肥" in document.content


def test_parse_documents_returns_unified_document_fields(tmp_path: Path) -> None:
    """测试不同文本类文档都会解析成统一的 ParsedDocument。

    这个测试不依赖 Qdrant 和 embedding，只验证文档解析这一步：
    文件 -> title/source_type/source_uri/content/pages。
    """

    txt_path = tmp_path / "橘子黄叶处理.txt"
    txt_path.write_text("橘子黄叶可能和缺素、积水有关。\n需要先检查根系和排水。", encoding="utf-8")

    md_path = tmp_path / "砂糖橘病虫害.md"
    md_path.write_text("# 砂糖橘病虫害\n红蜘蛛高发时要注意叶背检查。", encoding="utf-8")

    csv_path = tmp_path / "采收记录.csv"
    csv_path.write_text("品种,地区,说明\n脐橙,赣南,成熟期分批采收\n", encoding="utf-8-sig")

    parser = DocumentParser()
    documents = [parser.parse_file(path) for path in [txt_path, md_path, csv_path]]

    assert [document.source_type for document in documents] == ["txt", "md", "csv"]
    assert documents[0].title == "橘子黄叶处理"
    assert documents[1].title == "砂糖橘病虫害"
    assert documents[2].title == "采收记录"

    for document in documents:
        assert document.source_uri
        assert document.content.strip()
        assert document.pages
        assert document.metadata["file_name"]

    assert "橘子黄叶可能和缺素" in documents[0].content
    assert "红蜘蛛高发时要注意叶背检查" in documents[1].content
    assert "脐橙 | 赣南 | 成熟期分批采收" in documents[2].content
