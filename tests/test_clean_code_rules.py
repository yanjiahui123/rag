import ast
from pathlib import Path
import unittest
import warnings


ROOT = Path(__file__).resolve().parents[1]
RULE_FILE = ROOT / "docs" / "CLEAN_CODE.md"
MODEL_FILES = [
    ROOT / "rag_service" / "document_loaders" / "parsed_blocks.py",
    ROOT / "rag_service" / "document_loaders" / "table" / "models.py",
    ROOT / "rag_service" / "document_loaders" / "table" / "docx_parser.py",
]
SOURCE_FILES = list((ROOT / "rag_service").rglob("*.py"))


class CleanCodeRuleTests(unittest.TestCase):
    def test_clean_code_rule_file_documents_required_principles(self):
        self.assertTrue(RULE_FILE.exists())
        content = RULE_FILE.read_text(encoding="utf-8")

        self.assertIn("BaseModel", content)
        self.assertIn("50", content)

    def test_new_models_use_basemodel_instead_of_dataclass(self):
        for path in MODEL_FILES:
            content = path.read_text(encoding="utf-8")
            self.assertIn("BaseModel", content, path.as_posix())
            self.assertNotIn("dataclass", content, path.as_posix())

    def test_project_functions_are_under_50_lines(self):
        for path in SOURCE_FILES:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    length = node.end_lineno - node.lineno + 1
                    self.assertLessEqual(length, 50, f"{path.as_posix()}::{node.name} has {length} lines")


if __name__ == "__main__":
    unittest.main()
