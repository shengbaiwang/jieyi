import unittest

from jieyi.domain.models import IssueSeverity, TermEntry, TermStatus
from jieyi.quality import run_deterministic_checks


class QualityTests(unittest.TestCase):
    def test_detects_footnotes_and_terms(self):
        term = TermEntry(
            id="term_1",
            project_id="proj_1",
            source="agency",
            target="能动性",
            status=TermStatus.APPROVED,
            forbidden_targets=("代理性",),
        )
        issues = run_deterministic_checks(
            "Agency rose by 12% (Smith 2020) [^3].",
            "代理性有所上升（Smith 2021）。",
            [term],
        )
        codes = {item.code for item in issues}
        self.assertIn("footnote_mismatch", codes)
        self.assertIn("approved_term_missing", codes)
        self.assertIn("forbidden_term_used", codes)

    def test_clean_translation_has_no_deterministic_issue(self):
        issues = run_deterministic_checks(
            "Agency rose by 12% (Smith 2020) [^3].",
            "能动性提高了 12% (Smith 2020) [^3]。",
            [
                TermEntry(
                    id="term_1",
                    project_id="proj_1",
                    source="agency",
                    target="能动性",
                    status=TermStatus.APPROVED,
                )
            ],
        )
        self.assertEqual(issues, [])

    def test_ignores_number_and_parenthesis_differences(self):
        issues = run_deterministic_checks(
            "Revenue fell from 12% to 9% (Smith 2020) [12].",
            "收入从12%下降到8%（Smith 2021 [13]。",
        )
        self.assertEqual(issues, [])

    def test_western_terms_use_token_boundaries(self):
        term = TermEntry(
            id="term_1",
            project_id="proj_1",
            source="agency",
            target="能动性",
            status=TermStatus.APPROVED,
        )
        issues = run_deterministic_checks("Interagency work matters.", "跨部门工作很重要。", [term])
        self.assertNotIn("approved_term_missing", {item.code for item in issues})

    def test_aliases_are_active_source_forms_not_passive_metadata(self):
        term = TermEntry(
            id="term_1",
            project_id="proj_1",
            source="artificial intelligence",
            target="人工智能",
            aliases=("AI", "machine intelligence"),
            status=TermStatus.APPROVED,
        )
        issues = run_deterministic_checks(
            "AI changes the workflow.",
            "智能技术改变了工作流程。",
            [term],
        )
        issue = next(item for item in issues if item.code == "approved_term_missing")
        self.assertEqual(issue.details["matched_forms"], ["AI"])
        self.assertEqual(issue.severity, IssueSeverity.ERROR)

    def test_context_keywords_select_one_sense_for_an_ambiguous_form(self):
        terms = [
            TermEntry(
                id="financial",
                project_id="proj_1",
                source="bank",
                target="银行",
                sense="金融机构",
                context_keywords=("loan", "credit"),
                status=TermStatus.APPROVED,
            ),
            TermEntry(
                id="river",
                project_id="proj_1",
                source="bank",
                target="河岸",
                sense="河流边缘",
                context_keywords=("river", "flood"),
                status=TermStatus.APPROVED,
            ),
        ]
        issues = run_deterministic_checks(
            "The bank approved the loan.",
            "河岸批准了贷款。",
            terms,
        )
        codes = {item.code for item in issues}
        self.assertIn("approved_term_missing", codes)
        self.assertNotIn("ambiguous_term_unresolved", codes)
        issue = next(item for item in issues if item.code == "approved_term_missing")
        self.assertEqual(issue.details["target"], "银行")

    def test_unresolved_sense_is_warned_without_forcing_conflicting_targets(self):
        terms = [
            TermEntry(
                id="financial",
                project_id="proj_1",
                source="bank",
                target="银行",
                sense="金融机构",
                context_keywords=("loan",),
                status=TermStatus.APPROVED,
            ),
            TermEntry(
                id="river",
                project_id="proj_1",
                source="bank",
                target="河岸",
                sense="河流边缘",
                context_keywords=("river",),
                status=TermStatus.APPROVED,
            ),
        ]
        issues = run_deterministic_checks("The bank was old.", "它很古老。", terms)
        codes = {item.code for item in issues}
        self.assertIn("ambiguous_term_unresolved", codes)
        self.assertNotIn("approved_term_missing", codes)


if __name__ == "__main__":
    unittest.main()
