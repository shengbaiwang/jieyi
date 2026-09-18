import json
import unittest

from jieyi.domain.models import Segment, SegmentKind
from jieyi.workflow.cohesion import (
    build_cohesion_messages,
    parse_cohesion_revisions,
    plan_cohesion_groups,
)


def _segment(ordinal, text="Sentence.", kind=SegmentKind.PARAGRAPH, heading="H"):
    return Segment(
        id=f"seg-{ordinal}",
        document_id="document",
        stable_key=f"key-{ordinal}",
        ordinal=ordinal,
        kind=kind,
        source_text=text,
        heading_path=heading,
    )


class CohesionPlanningTests(unittest.TestCase):
    def test_groups_cap_size_and_drop_singletons(self):
        segments = [_segment(index) for index in range(9)]

        groups = plan_cohesion_groups(segments)

        self.assertEqual(
            [[item.ordinal for item in group.segments] for group in groups],
            [[0, 1, 2, 3, 4, 5, 6, 7]],
        )
        self.assertEqual(plan_cohesion_groups(segments[:1]), [])

    def test_groups_cut_at_headings_and_heading_paths(self):
        segments = [
            _segment(0, "Title", SegmentKind.HEADING, ""),
            _segment(1, "Body.", heading="Title"),
            _segment(2, "Next", SegmentKind.HEADING, "Title / Next"),
            _segment(3, "More.", heading="Title / Next"),
        ]

        groups = plan_cohesion_groups(segments)

        self.assertEqual(
            [[item.ordinal for item in group.segments] for group in groups],
            [[2, 3]],
        )

    def test_groups_cut_when_group_chars_overflow(self):
        segments = [_segment(0, "x" * 5_900), _segment(1, "y" * 500)]

        self.assertEqual(plan_cohesion_groups(segments), [])


class CohesionMessagesTests(unittest.TestCase):
    def test_messages_carry_sources_and_drafts_with_strict_contract(self):
        messages = build_cohesion_messages(
            context="# PROJECT\nTarget language: zh-CN",
            sources={"a": "Hi.", "b": "Bye."},
            drafts={"a": "你好。", "b": "再见。"},
            ordered_ids=["a", "b"],
        )

        self.assertEqual(len(messages), 2)
        self.assertIn("cohesion only", messages[0]["content"])
        self.assertIn("[[JY_PH_0000]]", messages[0]["content"])
        self.assertIn("exactly the supplied segment IDs", messages[0]["content"])
        self.assertIn("# PROJECT", messages[0]["content"])
        payload = json.loads(messages[1]["content"])
        self.assertEqual(
            payload,
            {"segments": [
                {"id": "a", "source": "Hi.", "draft": "你好。"},
                {"id": "b", "source": "Bye.", "draft": "再见。"},
            ]},
        )

    def test_parse_accepts_dict_list_and_translations_shapes(self):
        expected = {"a", "b"}
        direct = parse_cohesion_revisions(
            json.dumps({"a": "你好！", "b": "再见！"}), expected,
        )
        listed = parse_cohesion_revisions(
            json.dumps([{"id": "a", "text": "你好！"}, {"id": "b", "text": "再见！"}]),
            expected,
        )
        wrapped = parse_cohesion_revisions(
            json.dumps({"translations": {"a": "你好！", "b": "再见！"}}), expected,
        )
        fenced = parse_cohesion_revisions(
            "```json\n" + json.dumps({"a": "你好！", "b": "再见！"}) + "\n```",
            expected,
        )

        for parsed in (direct, listed, wrapped, fenced):
            self.assertEqual(parsed, {"a": "你好！", "b": "再见！"})

    def test_parse_rejects_bad_json_and_id_mismatch(self):
        expected = {"a", "b"}
        with self.assertRaises(ValueError):
            parse_cohesion_revisions("not json", expected)
        with self.assertRaises(ValueError):
            parse_cohesion_revisions(json.dumps({"a": "x"}), expected)
        with self.assertRaises(ValueError):
            parse_cohesion_revisions(json.dumps({"a": "x", "zzz": "y"}), expected)
        with self.assertRaises(ValueError):
            parse_cohesion_revisions(json.dumps({"a": "  ", "b": "y"}), expected)


if __name__ == "__main__":
    unittest.main()
