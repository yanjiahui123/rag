from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class DagsterParseSplitSourceTests(unittest.TestCase):
    def test_asset_graph_uses_explicit_parse_save_split_nodes(self):
        source = (ROOT / "rag_service" / "dagster" / "assets" / "init_knowledge_base_asset.py").read_text(
            encoding="utf-8"
        )

        self.assert_graph_uses_explicit_parse_save_split_nodes(source)

    def test_reload_graph_uses_explicit_parse_save_split_nodes(self):
        source = (ROOT / "rag_service" / "dagster" / "assets" / "reload_knowledge_base_asset.py").read_text(
            encoding="utf-8"
        )

        self.assert_graph_uses_explicit_parse_save_split_nodes(source)

    def test_update_graph_uses_explicit_parse_save_split_nodes(self):
        source = (ROOT / "rag_service" / "dagster" / "assets" / "updated_knowledge_base_asset.py").read_text(
            encoding="utf-8"
        )

        self.assert_graph_uses_explicit_parse_save_split_nodes(source)

    def assert_graph_uses_explicit_parse_save_split_nodes(self, source: str):
        compact_source = "".join(source.split())
        self.assertIn("map(parse_original_documents)", compact_source)
        self.assertIn("map(save_parsed_documents)", compact_source)
        self.assertIn("map(split_parsed_documents)", compact_source)
        self.assertNotIn("map(load_original_documents)", compact_source)

    def test_common_ops_define_parse_save_split_ops(self):
        source = (ROOT / "rag_service" / "dagster" / "dagster_common_op.py").read_text(encoding="utf-8")

        self.assertIn("def parse_original_documents(", source)
        self.assertIn("def save_parsed_documents(", source)
        self.assertIn("def split_parsed_documents(", source)
        self.assertIn("ParsedDocument", source)

    def test_common_ops_persist_and_cleanup_structured_docx_artifacts(self):
        source = (ROOT / "rag_service" / "dagster" / "dagster_common_op.py").read_text(encoding="utf-8")

        self.assertIn("persist_structured_docx_artifacts", source)
        self.assertIn("persist_structured_excel_artifacts", source)
        self.assertIn("persist_structured_html_artifacts", source)
        self.assertIn("persist_structured_markdown_artifacts", source)
        self.assertIn("persist_parsed_markdown_artifact", source)
        self.assertIn("merge_structured_docx_metadata", source)
        self.assertIn("extract_structured_docx_artifact_prefix", source)
        self.assertIn("extract_structured_excel_artifact_prefix", source)
        self.assertIn("extract_structured_html_artifact_prefix", source)
        self.assertIn("extract_structured_markdown_artifact_prefix", source)
        self.assertIn("extract_parsed_markdown_artifact_prefix", source)
        self.assertIn("extract_structured_image_object_keys", source)
        self.assertIn("_delete_structured_image_objects", source)
        self.assertIn("delete_object(image_object_key)", source)
        self.assertIn("is_safe_structured_artifact_prefix", source)
        self.assertIn("delete_download_key", source)


if __name__ == "__main__":
    unittest.main()
