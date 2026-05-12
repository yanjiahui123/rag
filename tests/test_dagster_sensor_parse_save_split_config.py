from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class DagsterSensorParseSaveSplitConfigTests(unittest.TestCase):
    def test_reload_sensor_config_matches_parse_save_split_graph(self):
        source = (ROOT / "rag_service" / "dagster" / "sensors" / "reload_knowledge_base_asset_sensor.py").read_text(
            encoding="utf-8"
        )

        self.assert_sensor_config_uses_parse_save_split(source)
        self.assertIn("save_document_metadata.name", source)

    def test_update_sensor_config_matches_parse_save_split_graph(self):
        source = (ROOT / "rag_service" / "dagster" / "sensors" / "update_knowledge_base_asset_sensor.py").read_text(
            encoding="utf-8"
        )

        self.assert_sensor_config_uses_parse_save_split(source)
        self.assertIn("fetch_updated_original_document_set.name", source)

    def assert_sensor_config_uses_parse_save_split(self, source: str):
        self.assertIn("parse_original_documents.name", source)
        self.assertIn("save_parsed_documents.name", source)
        self.assertIn("split_parsed_documents.name", source)
        self.assertNotIn("load_original_documents.name", source)


if __name__ == "__main__":
    unittest.main()
