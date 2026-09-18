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
from jieyi.prompting import (
    build_messages,
    build_system_prompt,
    downgrade_to_minimal_prompt,
    is_minimal_prompt_request,
    minimal_prompt_eligible,
)


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

    def test_minimal_prompt_eligibility_rules(self):
        self.assertTrue(minimal_prompt_eligible(
            source_text="Hello world, this is a short plain sentence.",
            task=CandidateStage.DRAFT, has_terms=False,
            has_atom_boundaries=False, max_chars=120,
        ))
        # Placeholders, terms, structure, multi-paragraph, and overlong
        # sources must stay on the full contract.
        self.assertFalse(minimal_prompt_eligible(
            source_text="See [[JY_PH_0000]] now.", task=CandidateStage.DRAFT,
            has_terms=False, has_atom_boundaries=False, max_chars=120,
        ))
        self.assertFalse(minimal_prompt_eligible(
            source_text="Short sentence.", task=CandidateStage.DRAFT,
            has_terms=True, has_atom_boundaries=False, max_chars=120,
        ))
        self.assertFalse(minimal_prompt_eligible(
            source_text="Short sentence.", task=CandidateStage.DRAFT,
            has_terms=False, has_atom_boundaries=True, max_chars=120,
        ))
        self.assertFalse(minimal_prompt_eligible(
            source_text="First.\n\nSecond.", task=CandidateStage.DRAFT,
            has_terms=False, has_atom_boundaries=False, max_chars=120,
        ))
        self.assertFalse(minimal_prompt_eligible(
            source_text="x" * 121, task=CandidateStage.DRAFT,
            has_terms=False, has_atom_boundaries=False, max_chars=120,
        ))
        self.assertFalse(minimal_prompt_eligible(
            source_text="Short sentence.", task=CandidateStage.REPAIR,
            has_terms=False, has_atom_boundaries=False, max_chars=120,
        ))

    def test_minimal_prompt_renders_immersive_style_messages(self):
        request = downgrade_to_minimal_prompt(
            self.request("Hello world, this is a short plain sentence."),
            has_terms=False, enabled=True, max_chars=120,
        )

        self.assertTrue(request.minimal_prompt)
        self.assertTrue(is_minimal_prompt_request(request))
        self.assertEqual(request.context, "")
        self.assertEqual(request.segment_context, "")
        system_prompt = build_system_prompt(request)
        self.assertIn("Translate to zh-CN (output translation only)", system_prompt)
        self.assertNotIn("placeholder", system_prompt.casefold())
        messages = build_messages(request)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[1]["content"], "Hello world, this is a short plain sentence.")

    def test_minimal_prompt_never_activates_with_neighbor_context(self):
        request = downgrade_to_minimal_prompt(
            self.request("Hello world, this is a short plain sentence."),
            has_terms=False, enabled=True, max_chars=120,
        )
        with_context = replace(request, segment_context="# REFERENCE CONTEXT\nneighbor")

        self.assertFalse(is_minimal_prompt_request(with_context))
        self.assertIn("Translate the source faithfully", build_system_prompt(with_context))

    def test_minimal_prompt_respects_opt_out(self):
        request = downgrade_to_minimal_prompt(
            self.request("Hello world, this is a short plain sentence."),
            has_terms=False, enabled=False, max_chars=120,
        )

        self.assertFalse(request.minimal_prompt)
        self.assertFalse(is_minimal_prompt_request(request))


if __name__ == "__main__":
    unittest.main()
