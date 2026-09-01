import unittest

from jieyi.domain.models import SegmentKind
from jieyi.ingestion.plaintext import parse_text, segments_from_text


class IngestionTests(unittest.TestCase):
    def test_markdown_structure_is_preserved(self):
        text = """# Introduction

An ordinary paragraph.

> A cited passage.

[^1]: A note.
"""
        blocks = parse_text(text, "markdown")
        self.assertEqual(
            [item.kind for item in blocks],
            [
                SegmentKind.HEADING,
                SegmentKind.PARAGRAPH,
                SegmentKind.BLOCKQUOTE,
                SegmentKind.FOOTNOTE,
            ],
        )
        self.assertEqual(blocks[1].heading_path, "Introduction")
        self.assertEqual(blocks[2].text, "A cited passage.")

    def test_unrelated_insert_does_not_change_existing_stable_keys(self):
        before = segments_from_text("doc_fixed", "Alpha.\n\nBeta.", "txt")
        after = segments_from_text("doc_fixed", "Inserted.\n\nAlpha.\n\nBeta.", "txt")
        keys_before = {item.source_text: item.stable_key for item in before}
        keys_after = {item.source_text: item.stable_key for item in after}
        self.assertEqual(keys_before["Alpha."], keys_after["Alpha."])
        self.assertEqual(keys_before["Beta."], keys_after["Beta."])


if __name__ == "__main__":
    unittest.main()

