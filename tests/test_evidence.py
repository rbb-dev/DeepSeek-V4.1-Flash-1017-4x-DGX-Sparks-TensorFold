#!/usr/bin/env python3
"""The README's published numbers against their evidence folder (evidence/README.md): its performance tables are, row
for row, what tools/readme_tables.py rebuilds from the folder the README names, and that folder's tables.md is the same
output, so a published number cannot drift from the files it was computed from. Standard library only, no Spark.

  python3 -m unittest tests.test_evidence
"""
import json
import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class Evidence(unittest.TestCase):
    def setUp(self):
        self.readme = (ROOT / "README.md").read_text()
        found = re.findall(r"\(evidence/(native-[0-9a-f]{12})/\)", self.readme)
        self.assertTrue(found, "the README names no evidence folder")
        self.folder = ROOT / "evidence" / found[0]

    def test_the_readme_shows_its_evidence_tables(self):
        f = self.folder
        p = subprocess.run([sys.executable, "-I", str(ROOT / "tools" / "readme_tables.py"), "--main", str(f / "recipe-main.json"),
                            "--turbo", str(f / "recipe-turbo.json"), "--code", str(f / "code.jsonl")],
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(p.stdout.strip(), (f / "tables.md").read_text().strip())
        rows = [l for l in p.stdout.splitlines() if l.startswith("| ") and not l.startswith("|---")]
        self.assertGreaterEqual(len(rows), 9 + 5 + 5)
        for row in rows:
            self.assertIn("\n" + row + "\n", self.readme)

    def test_the_manifest_names_the_folder_s_image(self):
        m = json.loads((self.folder / "MANIFEST.json").read_text())
        self.assertEqual(m["image"], "tensorfold-dsv41:" + self.folder.name)
        for key in ("image_id", "base_image", "kernel_kit", "model", "software", "settings"):
            self.assertIn(key, m)


if __name__ == "__main__":
    unittest.main()
