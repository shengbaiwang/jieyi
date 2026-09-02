import unittest
from dataclasses import replace

from jieyi.domain.models import (
    CandidateStage,
    Document,
    Project,
    Segment,
    SegmentKind,
    TranslationRequest,
)
from jieyi.prompting import build_system_prompt
from jieyi.workflow.provider_responses import parse_review_response


class PromptingTests(unittest.TestCase):
    def setUp(self):
        self.project = Project(
            id="project",
            name="Book",
            source_lang="en",
            target_lang="zh-CN",
        )
        self.document = Document(
            id="document",
            project_id=self.project.id,
            title="Book",
            source_format="txt",
            source_hash="hash",
        )
        self.segment = Segment(
            id="segment",
            document_id=self.document.id,
            stable_key="segment-0",
            ordinal=0,
            kind=SegmentKind.PARAGRAPH,
            source_text="Ordinary source.",
        )

    def request(self, source_text: str) -> TranslationRequest:
        return TranslationRequest(
            project=self.project,
            document=self.document,
            segment=replace(self.segment, source_text=source_text),
            context="",
            task=CandidateStage.DRAFT,
        )

    def test_prompt_does_not_show_a_fake_required_token_when_source_has_none(self):
        prompt = build_system_prompt(self.request("Ordinary source."))

        self.assertIn("contains no placeholder tokens", prompt)
        self.assertNotIn("[[JY_PH_0000]]", prompt)

    def test_prompt_lists_only_the_tokens_present_in_the_masked_source(self):
        prompt = build_system_prompt(
            self.request("A [[JY_PH_0003]] B [[JY_PH_0012]]")
        )

        self.assertIn("[[JY_PH_0003]], [[JY_PH_0012]]", prompt)
        self.assertIn("Do not create any other placeholder tokens", prompt)

    def test_review_prompt_requires_an_independent_source_comparison(self):
        request = replace(
            self.request("Original claim."),
            task=CandidateStage.REVIEW,
            existing_translation="现有译文。",
        )

        prompt = build_system_prompt(request)

        self.assertIn("Independently compare the source and current translation", prompt)
        self.assertIn("do not assume the draft is correct", prompt)
        self.assertIn("JY_REVIEW_ISSUES:", prompt)
        self.assertIn("Omit this appendix when no human decision is needed", prompt)

    def test_review_response_separates_translation_and_human_questions(self):
        translation, findings = parse_review_response(
            "修订后的译文。\n\nJY_REVIEW_ISSUES:\n"
            "- 此处代词指代可能有两种解释。\n"
            "• 专名缺少可核实的标准译名。"
        )

        self.assertEqual(translation, "修订后的译文。")
        self.assertEqual(
            findings,
            ("此处代词指代可能有两种解释。", "专名缺少可核实的标准译名。"),
        )
        self.assertEqual(parse_review_response("普通审校译文。"), ("普通审校译文。", ()))

    def test_review_without_draft_explicitly_requests_a_fresh_translation(self):
        request = replace(
            self.request("Sensitive source."),
            task=CandidateStage.REVIEW,
            existing_translation=None,
            issue_summary="Draft model content filter.",
        )

        prompt = build_system_prompt(request)

        self.assertIn("No draft translation is available", prompt)
        self.assertIn("Translate the source faithfully now", prompt)


if __name__ == "__main__":
    unittest.main()
