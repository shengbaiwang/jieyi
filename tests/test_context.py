import unittest
from dataclasses import replace

from jieyi.context.compiler import compile_neighbor_context
from jieyi.domain.models import Segment, SegmentKind


class NeighborContextTests(unittest.TestCase):
    def setUp(self):
        self.current = Segment(
            id="current", document_id="document", stable_key="current", ordinal=1,
            kind=SegmentKind.PARAGRAPH, source_text="Only this is the translation target.",
            heading_path="Chapter / Section",
        )
        self.before = replace(
            self.current, id="before", ordinal=0, source_text="Earlier source.",
        )
        self.after = replace(
            self.current, id="after", ordinal=2, source_text="Later source.",
        )

    def test_budget_keeps_both_sides_and_the_text_closest_to_current_segment(self):
        before = replace(self.before, source_text="X" * 10_000 + "previous ending")
        after = replace(self.after, source_text="following beginning" + "Y" * 10_000)
        context = compile_neighbor_context(self.current, [before, after], max_chars=800)
        self.assertLessEqual(len(context), 800)
        self.assertIn("Chapter / Section", context)
        self.assertIn("# PREVIOUS SOURCE", context)
        self.assertIn("# FOLLOWING SOURCE", context)
        self.assertIn("previous ending", context)
        self.assertIn("following beginning", context)
        self.assertNotIn(self.current.source_text, context)
        self.assertTrue(context.endswith("# END REFERENCE CONTEXT"))
        self.assertEqual(compile_neighbor_context(self.current, [before], max_chars=10), "")

    def test_review_translation_fallbacks_and_document_isolation(self):
        for field in ("accepted_translation", "reviewed_translation",
                      "edited_translation", "machine_translation"):
            with self.subTest(field=field):
                before = replace(self.before, **{field: "已有译文"})
                foreign = replace(self.after, document_id="another-document", source_text="foreign")
                context = compile_neighbor_context(
                    self.current, [before, self.current, foreign],
                    max_chars=2_000, include_translations=True,
                )
                self.assertIn("已有译文", context)
                self.assertNotIn("foreign", context)
                self.assertNotIn(self.current.source_text, context)


if __name__ == "__main__":
    unittest.main()
