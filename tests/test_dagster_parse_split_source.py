from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class DagsterParseSplitSourceTests(unittest.TestCase):
    def test_asset_graph_uses_explicit_parse_save_split_nodes(self):
        source = (ROOT / "rag_service" / "dagster" / "assets" / "init_knowledge_base_asset.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("parse_original_documents", source)
        self.assertIn("save_parsed_documents", source)
        self.assertIn("split_parsed_documents", source)
        self.assertNotIn(".map(\n        load_original_documents\n    )", source)

    def test_common_ops_define_parse_save_split_ops(self):
        source = (ROOT / "rag_service" / "dagster" / "dagster_common_op.py").read_text(encoding="utf-8")

        self.assertIn("def parse_original_documents(", source)
        self.assertIn("def save_parsed_documents(", source)
        self.assertIn("def split_parsed_documents(", source)


if __name__ == "__main__":
    unittest.main()
