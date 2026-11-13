"""Exercise missing coverage and invalid evidence references, not generated wording."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from review_coverage import inspect, render


class CoverageTest(unittest.TestCase):
    def setUp(self):
        self.manifest = {
            "snapshot": {"source": "h", "base": "b", "target": "t", "rules": "r"},
            "rules": ["R1", "R2"], "files": ["file.cpp"],
        }
        self.record = {
            "snapshot": deepcopy(self.manifest["snapshot"]),
            "subjects": [{"id": "S1", "path": "file.cpp", "symbol": "Function"}],
            "excluded_files": {},
            "evidence": [{"id": "E1", "locations": ["file.cpp:Function"], "fact": "Shared source fact"}],
            "checks": [
                {"id": "C1", "rule": "R1", "subjects": ["S1"], "evidence": ["E1"],
                 "status": "checked", "basis": "First independent judgment"},
                {"id": "C2", "rule": "R2", "subjects": ["S1"], "evidence": ["E1"],
                 "status": "checked", "basis": "Second independent judgment"},
            ],
        }

    def test_shared_evidence_supports_both_views(self):
        errors, gaps = inspect(self.manifest, self.record)
        self.assertEqual((errors, gaps), ([], []))
        view = render(self.manifest, self.record, errors, gaps)
        self.assertIn("| R1 | C1 |", view)
        self.assertIn("| S1: file.cpp:Function | C1, C2 |", view)

    def test_missing_rule_and_unchecked_changed_object_remain_gaps(self):
        self.record["checks"].pop()
        self.record["subjects"].append({"id": "S2", "path": "file.cpp", "symbol": "OtherFunction"})
        errors, gaps = inspect(self.manifest, self.record)
        self.assertFalse(errors)
        self.assertEqual(len(gaps), 2)

    def test_new_file_cannot_be_hidden_by_old_coverage(self):
        self.manifest["files"].append("new.cpp")
        self.assertTrue(inspect(self.manifest, self.record)[1])

    def test_unknown_evidence_and_empty_judgment_are_invalid(self):
        self.record["checks"][0].update(evidence=["missing"], basis="")
        self.assertEqual(len(inspect(self.manifest, self.record)[0]), 2)

    def test_version_mismatch_is_invalid(self):
        self.record["snapshot"]["rules"] = "old-rules"
        self.assertTrue(inspect(self.manifest, self.record)[0])

    def test_duplicate_ids_are_invalid(self):
        self.record["evidence"].append(deepcopy(self.record["evidence"][0]))
        with self.assertRaises(ValueError):
            inspect(self.manifest, self.record)

    def test_exclusion_requires_a_reason_and_cannot_cover_checked_file(self):
        self.record["excluded_files"] = {"file.cpp": "out of scope"}
        self.assertTrue(inspect(self.manifest, self.record)[0])
        self.record["subjects"] = []
        self.record["excluded_files"] = {"file.cpp": ""}
        self.assertTrue(inspect(self.manifest, self.record)[0])

    def test_partial_cli_keeps_gaps_and_final_cli_rejects_them(self):
        self.record["checks"][0]["status"] = "pending"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(json.dumps(self.manifest))
            (root / "record.json").write_text(json.dumps(self.record))
            cmd = [sys.executable, str(Path(__file__).with_name("review_coverage.py")),
                   str(root / "manifest.json"), str(root / "record.json")]
            final = subprocess.run(cmd, capture_output=True, text=True)
            partial = subprocess.run([*cmd, "--partial"], capture_output=True, text=True)
        self.assertEqual(final.returncode, 1)
        self.assertEqual(partial.returncode, 0)
        self.assertIn("C1: pending", partial.stdout)


if __name__ == "__main__":
    unittest.main()
